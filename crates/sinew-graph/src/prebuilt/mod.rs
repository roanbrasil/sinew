//! Ready-made graphs: the tool-calling [`react_agent`] (and its configurable
//! [`ReactAgent`] builder), a [`Supervisor`] that routes work among agents,
//! and a [`Swarm`] of agents that hand off to each other.

mod handoff;
mod loop_guard;
mod supervisor;
mod swarm;

use std::{future::Future, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{
    context::ContextPolicy, ChatModel, ChatRequest, Message, Role, StreamEvent, StructuredOutput,
    ToolBox, ToolSpec,
};
use sinew_trace::{traced_chat, traced_chat_stream};

pub use handoff::{create_handoff_tool, Handoff};
pub use loop_guard::{LoopAction, LoopCheck, LoopGuard, RepeatedCall, LOOP_STOP_REPLY};
pub use supervisor::{create_supervisor, NamedAgent, OutputMode, Supervisor};
pub use swarm::{create_swarm, Swarm, SwarmState};

use crate::{
    graph::BoxFut, Command, CompiledGraph, GraphError, NodeContext, NodeResult, StateGraph, END,
};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentState {
    pub messages: Vec<Message>,
    /// Set by agents built with [`ReactAgent::response_format`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_response: Option<Value>,
}

impl AgentState {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            structured_response: None,
        }
    }
    pub fn last(&self) -> Option<&Message> {
        self.messages.last()
    }
}

/// Classic tool-calling loop: `agent` calls the model; if it asked for tools,
/// `tools` runs them and control returns to `agent`; otherwise the run ends.
///
/// When the graph is consumed with [`CompiledGraph::stream`], the `agent`
/// node calls [`ChatModel::chat_stream`] and forwards every text delta as a
/// [`GraphEvent::Token`](crate::GraphEvent::Token); with `invoke`/`run` it
/// makes a plain `chat` call. See [`ReactAgent`] for more options.
pub fn react_agent(
    model: Arc<dyn ChatModel>,
    tools: ToolBox,
) -> Result<CompiledGraph<AgentState>, GraphError> {
    ReactAgent::new(model, tools).compile()
}

type InputHook = Arc<dyn Fn(&AgentState) -> Vec<Message> + Send + Sync>;
type HookNode = Arc<dyn Fn(AgentState, NodeContext) -> BoxFut<AgentState> + Send + Sync>;

/// What an agent answers when it runs out of iterations.
pub const MAX_ITERATIONS_REPLY: &str = "Sorry, need more steps to process this request.";

/// Builder for a configurable ReAct agent (LangGraph's
/// `create_react_agent` options).
///
/// Graph: `[pre-model input hook] agent -> [post_model_hook] -> tools ->
/// agent ... -> [structured_response] -> END`.
pub struct ReactAgent {
    model: Arc<dyn ChatModel>,
    tools: ToolBox,
    system_prompt: Option<String>,
    pre_model_hook: Option<InputHook>,
    post_model_hook: Option<HookNode>,
    response_format: Option<StructuredOutput>,
    max_iterations: Option<usize>,
    handoffs: Vec<Handoff>,
    loop_guard: Option<LoopGuard>,
    unknown_outcome: crate::durable::UnknownOutcome,
    context_policy: Option<ContextPolicy>,
    persist_context: bool,
}

impl ReactAgent {
    pub fn new(model: Arc<dyn ChatModel>, tools: ToolBox) -> Self {
        Self {
            model,
            tools,
            system_prompt: None,
            pre_model_hook: None,
            post_model_hook: None,
            response_format: None,
            max_iterations: None,
            handoffs: Vec::new(),
            loop_guard: None,
            unknown_outcome: crate::durable::UnknownOutcome::default(),
            context_policy: None,
            persist_context: false,
        }
    }

    /// Prepended as a system message to every model call (not stored in the
    /// state).
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Builds the messages sent to the model from the state, e.g. to trim or
    /// summarize a long history. The state itself is not changed.
    pub fn pre_model_hook(
        mut self,
        f: impl Fn(&AgentState) -> Vec<Message> + Send + Sync + 'static,
    ) -> Self {
        self.pre_model_hook = Some(Arc::new(f));
        self
    }

    /// Fits the messages sent to the model into a token budget before every
    /// model call (after [`pre_model_hook`](Self::pre_model_hook), if any):
    /// clearing old tool results, summarizing or trimming, as the policy
    /// says. The [`system_prompt`](Self::system_prompt) is part of the
    /// messages the policy sees (and always kept); it is added unless a
    /// system message with the same text is already there. The budget is
    /// reduced by the estimated size of the tool schemas.
    ///
    /// The state is not changed unless [`persist_context`](Self::persist_context)
    /// is set, so checkpoints keep the full history. When the policy changes
    /// the messages, a [`GraphEvent::Custom`](crate::GraphEvent::Custom) with
    /// a `context_policy` field reports the token counts and strategies.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use sinew_core::context::ContextPolicy;
    /// use sinew_core::{mock::ScriptedModel, ChatModel, ToolBox};
    /// use sinew_graph::prebuilt::ReactAgent;
    ///
    /// let model: Arc<dyn ChatModel> = Arc::new(ScriptedModel::new([]));
    /// let agent = ReactAgent::new(model.clone(), ToolBox::new())
    ///     .context_policy(ContextPolicy::summarizing(16_000, model))
    ///     .compile()
    ///     .unwrap();
    /// # let _ = agent;
    /// ```
    pub fn context_policy(mut self, policy: ContextPolicy) -> Self {
        self.context_policy = Some(policy);
        self
    }

    /// Whether the output of the [`context_policy`](Self::context_policy)
    /// replaces [`AgentState::messages`] (default `false`). With `true` the
    /// stored history shrinks too: a summary message stays in the state and
    /// is updated on later turns, and dropped messages are gone from
    /// checkpoints. With a [`pre_model_hook`](Self::pre_model_hook), its
    /// output is stored as well.
    pub fn persist_context(mut self, yes: bool) -> Self {
        self.persist_context = yes;
        self
    }

    /// A node that runs after every model call, before tools: guardrails,
    /// validation, or human review (it may call
    /// [`NodeContext::interrupt`]). It can edit the state, e.g. remove tool
    /// calls it rejects.
    pub fn post_model_hook<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(AgentState, NodeContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = NodeResult<AgentState>> + Send + 'static,
    {
        self.post_model_hook = Some(Arc::new(move |s, ctx| Box::pin(f(s, ctx))));
        self
    }

    /// After the loop ends, asks the model for a final answer matching this
    /// schema and stores it in [`AgentState::structured_response`].
    pub fn response_format(mut self, format: StructuredOutput) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Stops after `n` model calls in one run (one invocation of the agent,
    /// e.g. one turn, or one handoff to it inside a supervisor), answering
    /// [`MAX_ITERATIONS_REPLY`] instead of calling the model again.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Catches the agent repeating the same tool call or failing over and
    /// over: the repeated call is answered with a nudge, the run stops, or a
    /// human is asked, depending on [`LoopGuard::action`]. Each detection is
    /// also emitted as a [`GraphEvent::Custom`](crate::GraphEvent::Custom)
    /// with a `loop_detected` field.
    pub fn loop_guard(mut self, guard: LoopGuard) -> Self {
        self.loop_guard = Some(guard);
        self
    }

    /// With a checkpointer that keeps a tool ledger (the in-memory, SQLite
    /// and Postgres ones do), tool calls are durable: after a crash or an
    /// error mid-turn, calls that already finished are not run again on
    /// resume. This sets what happens to a side-effecting call
    /// (`Tool::side_effects`) that started but has no recorded result
    /// (default [`UnknownOutcome::Fail`](crate::durable::UnknownOutcome::Fail)).
    pub fn unknown_tool_outcome(mut self, policy: crate::durable::UnknownOutcome) -> Self {
        self.unknown_outcome = policy;
        self
    }

    /// Adds a handoff tool: when the model calls it, the agent leaves its
    /// graph and the parent (a [`Swarm`] or any graph that has this agent as
    /// a subgraph) continues at `handoff.agent`.
    pub fn handoff(mut self, handoff: Handoff) -> Self {
        self.handoffs.push(handoff);
        self
    }

    /// The agent graph, not yet compiled (to add nodes or policies).
    pub fn graph(self) -> StateGraph<AgentState> {
        let mut specs = self.tools.specs();
        specs.extend(self.handoffs.iter().map(Handoff::spec));
        let model = self.model.clone();
        let system = self.system_prompt.clone();
        let hook = self.pre_model_hook.clone();
        let policy = self.context_policy.clone();
        let persist = self.persist_context;
        // Room for what is added after the policy: system prompt and tools.
        let reserved = policy.as_ref().map_or(0, |p| {
            let c = p.token_counter();
            specs
                .iter()
                .map(|t| {
                    c.count_text(&t.name)
                        + c.count_text(&t.description)
                        + c.count_text(&t.parameters.to_string())
                })
                .sum()
        });
        let max = self.max_iterations;
        // Supersteps per loop round: agent, [post_model_hook], tools.
        let cycle = if self.post_model_hook.is_some() { 3 } else { 2 };
        let mut g = StateGraph::<AgentState>::new();

        g.add_node_with_context("agent", move |mut s: AgentState, ctx| {
            let model = model.clone();
            let messages = match &hook {
                Some(h) => h(&s),
                None => s.messages.clone(),
            };
            let system = system.clone();
            let specs = specs.clone();
            let policy = policy.clone();
            // `ctx.step` restarts at 0 for every run (and every subgraph
            // invocation), so this counts this run's model calls.
            let exhausted = max.is_some_and(|max| ctx.step / cycle >= max);
            async move {
                if exhausted {
                    s.messages.push(Message::assistant(MAX_ITERATIONS_REPLY));
                    return Ok(s);
                }
                let mut messages = messages;
                match &policy {
                    None => {
                        if let Some(p) = &system {
                            if messages.first().is_none_or(|m| m.role != Role::System) {
                                messages.insert(0, Message::system(p.clone()));
                            }
                        }
                    }
                    Some(policy) => {
                        // The prompt goes in before the policy runs, so it is
                        // counted and kept like any system message. A summary
                        // message may lead the history, so the prompt is
                        // added unless it is already there.
                        let added = match &system {
                            Some(p)
                                if !messages
                                    .iter()
                                    .any(|m| m.role == Role::System && m.content == *p) =>
                            {
                                messages.insert(0, Message::system(p.clone()));
                                true
                            }
                            _ => false,
                        };
                        let report = policy.apply_with_report(messages, reserved).await?;
                        if report.changed() {
                            ctx.emit(json!({ "context_policy": {
                                "tokens_before": report.tokens_before,
                                "tokens_after": report.tokens_after,
                                "budget": report.budget,
                                "applied": report.applied,
                            }}));
                            if persist {
                                let mut stored = report.messages.clone();
                                if added {
                                    let p = system.as_deref().unwrap_or_default();
                                    if let Some(i) = stored
                                        .iter()
                                        .position(|m| m.role == Role::System && m.content == p)
                                    {
                                        stored.remove(i);
                                    }
                                }
                                s.messages = stored;
                            }
                        }
                        messages = report.messages;
                    }
                }
                let req = ChatRequest::new(messages).with_tools(specs);
                let resp = if ctx.is_streaming() {
                    traced_chat_stream(
                        model.as_ref(),
                        req,
                        ctx.tracer.as_ref(),
                        ctx.run.as_ref(),
                        |ev| {
                            if let StreamEvent::TextDelta { text } = ev {
                                ctx.emit_token(text.clone());
                            }
                        },
                    )
                    .await?
                } else {
                    traced_chat(model.as_ref(), req, ctx.tracer.as_ref(), ctx.run.as_ref()).await?
                };
                s.messages.push(resp.message);
                Ok(s)
            }
        });

        let tools = self.tools.clone();
        let handoffs = self.handoffs.clone();
        let guard = self.loop_guard.clone();
        let unknown = self.unknown_outcome;
        g.add_command_node("tools", ["agent", END], move |mut s: AgentState, ctx| {
            let tools = tools.clone();
            let handoffs = handoffs.clone();
            let guard = guard.clone();
            async move {
                let calls = s.last().map(|m| m.tool_calls.clone()).unwrap_or_default();
                // Where these calls were requested, for durable ledger keys.
                let position = s.messages.len().saturating_sub(1).to_string();
                let mut nudged: Vec<RepeatedCall> = Vec::new();
                if let Some(g) = &guard {
                    let check = g.check(&s.messages);
                    if check.is_loop() {
                        ctx.emit(json!({ "loop_detected": &check }));
                        let action = match g.action {
                            LoopAction::Nudge if check.stuck => LoopAction::Stop,
                            a => a,
                        };
                        let stop = match action {
                            LoopAction::Nudge => {
                                nudged = check.repeated.clone();
                                false
                            }
                            LoopAction::Stop => true,
                            LoopAction::Interrupt => {
                                let proceed: bool =
                                    ctx.interrupt(json!({ "loop_detected": &check }))?;
                                !proceed
                            }
                        };
                        if stop {
                            for call in &calls {
                                s.messages.push(Message::tool_result(
                                    call.id.clone(),
                                    "skipped: the run was stopped by the loop guard",
                                ));
                            }
                            s.messages.push(Message::assistant(LOOP_STOP_REPLY));
                            return Ok(Command::goto(END).with_update(s));
                        }
                    }
                }
                let mut target = None;
                for call in &calls {
                    if let Some(r) = nudged.iter().find(|r| r.id == call.id) {
                        s.messages.push(Message::tool_result(
                            call.id.clone(),
                            LoopGuard::nudge_message(r),
                        ));
                        continue;
                    }
                    if let Some(h) = handoffs.iter().find(|h| h.tool_name == call.name) {
                        let reply = if target.is_none() {
                            target = Some(h.agent.clone());
                            format!("Successfully transferred to {}", h.agent)
                        } else {
                            "Ignored: already transferring to another agent".to_string()
                        };
                        s.messages
                            .push(Message::tool_result(call.id.clone(), reply));
                        continue;
                    }
                    let result =
                        crate::durable::durable_tool_call(&tools, call, &ctx, &position, unknown)
                            .await?;
                    s.messages.push(result);
                }
                Ok(match target {
                    // Leave this agent: the parent graph continues at the
                    // target with the conversation so far.
                    Some(agent) => Command::goto(agent.clone())
                        .with_json_update(json!({
                            "messages": s.messages,
                            "active_agent": agent,
                        }))
                        .to_parent(),
                    None => Command::goto("agent").with_update(s),
                })
            }
        });

        let structured = self.response_format.is_some();
        let route = move |s: &AgentState| {
            if s.last().is_some_and(Message::has_tool_calls) {
                "tools".to_string()
            } else if structured {
                "structured_response".to_string()
            } else {
                END.to_string()
            }
        };
        let targets = [
            "tools",
            if structured {
                "structured_response"
            } else {
                END
            },
        ];
        match self.post_model_hook {
            Some(hook) => {
                g.add_node_with_context("post_model_hook", move |s, ctx| hook(s, ctx));
                g.add_edge("agent", "post_model_hook");
                g.add_conditional_edges("post_model_hook", route, targets);
            }
            None => {
                g.add_conditional_edges("agent", route, targets);
            }
        }

        if let Some(format) = self.response_format {
            let model = self.model.clone();
            g.add_node("structured_response", move |mut s: AgentState| {
                let (model, format) = (model.clone(), format.clone());
                async move {
                    let v: Value = format
                        .generate(model.as_ref(), ChatRequest::new(s.messages.clone()))
                        .await?;
                    s.structured_response = Some(v);
                    Ok(s)
                }
            });
            g.add_edge("structured_response", END);
        }
        g.set_entry_point("agent");
        g
    }

    pub fn compile(self) -> Result<CompiledGraph<AgentState>, GraphError> {
        self.graph().compile()
    }
}

/// A tool spec with no parameters (used by handoff tools).
pub(crate) fn empty_tool(name: String, description: String) -> ToolSpec {
    ToolSpec {
        name,
        description,
        parameters: json!({ "type": "object", "properties": {} }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use sinew_core::{mock::ScriptedModel, FnTool, Role, ToolCall};

    #[tokio::test]
    async fn loops_through_tools_and_answers() {
        let model = Arc::new(ScriptedModel::new([
            Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "c1".into(),
                    name: "add".into(),
                    arguments: json!({"a": 2, "b": 40}),
                }],
            ),
            Message::assistant("The answer is 42."),
        ]));
        let tools = ToolBox::new().with(FnTool::new(
            "add",
            "add numbers",
            json!({"type": "object"}),
            |a: Value| {
                Ok(json!(
                    a["a"].as_i64().unwrap_or(0) + a["b"].as_i64().unwrap_or(0)
                ))
            },
        ));
        let graph = react_agent(model.clone(), tools).unwrap();
        let out = graph
            .invoke(AgentState::new(vec![Message::user("2 + 40?")]))
            .await
            .unwrap();

        let roles: Vec<Role> = out.messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![Role::User, Role::Assistant, Role::Tool, Role::Assistant]
        );
        assert_eq!(out.messages[2].content, "42");
        assert_eq!(out.last().unwrap().content, "The answer is 42.");
        // The second model call saw the tool result.
        assert_eq!(model.requests()[1].messages.len(), 3);
        assert_eq!(model.requests()[0].tools[0].name, "add");
    }

    #[tokio::test]
    async fn streams_model_tokens_through_the_graph() {
        use crate::GraphEvent;
        use futures_util::StreamExt;

        let model = Arc::new(ScriptedModel::new([
            Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "c1".into(),
                    name: "add".into(),
                    arguments: json!({"a": 2, "b": 40}),
                }],
            ),
            Message::assistant("The answer is 42."),
        ]));
        let tools = ToolBox::new().with(FnTool::new(
            "add",
            "add numbers",
            json!({"type": "object"}),
            |a: Value| Ok(json!(a["a"].as_i64().unwrap() + a["b"].as_i64().unwrap())),
        ));
        let graph = react_agent(model, tools).unwrap();
        let events: Vec<_> = graph
            .stream("t", Some(AgentState::new(vec![Message::user("2 + 40?")])))
            .collect()
            .await;
        let tokens: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                GraphEvent::Token { delta, node, .. } if node == "agent" => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(tokens, vec!["The", " answer", " is", " 42."]);
        let starts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                GraphEvent::NodeStart { node, .. } => Some(node.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec!["agent", "tools", "agent"]);
        let Some(GraphEvent::Completed { state }) = events.last() else {
            panic!("expected Completed, got {:?}", events.last())
        };
        // The streamed reply was accumulated back into a proper message,
        // including the tool call that arrived as deltas.
        assert_eq!(
            state.messages[1].tool_calls[0].arguments,
            json!({"a": 2, "b": 40})
        );
        assert_eq!(state.last().unwrap().content, "The answer is 42.");
    }

    fn long_history(turns: usize) -> Vec<Message> {
        let mut msgs = Vec::new();
        for i in 0..turns {
            msgs.push(Message::user(format!(
                "question {i}: {}",
                "padding ".repeat(60)
            )));
            msgs.push(Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: format!("old{i}"),
                    name: "add".into(),
                    arguments: json!({}),
                }],
            ));
            msgs.push(Message::tool_result(
                format!("old{i}"),
                "result ".repeat(200),
            ));
            msgs.push(Message::assistant(format!("answer {i}")));
        }
        msgs.push(Message::user("latest question"));
        msgs
    }

    #[tokio::test]
    async fn context_policy_shrinks_the_request_not_the_state() {
        use sinew_core::context::{
            check_tool_pairing, ContextPolicy, EstimateCounter, TokenCounter,
        };
        let model = Arc::new(ScriptedModel::new([Message::assistant("done")]));
        let graph = ReactAgent::new(model.clone(), ToolBox::new())
            .system_prompt("Be brief.")
            .context_policy(ContextPolicy::trimming(1_500))
            .compile()
            .unwrap();
        let input = long_history(40);
        let out = graph.invoke(AgentState::new(input.clone())).await.unwrap();
        assert_eq!(out.messages.len(), input.len() + 1);
        assert_eq!(out.messages[..input.len()], input[..]);
        let sent = &model.requests()[0].messages;
        assert!(EstimateCounter.count_messages(sent) <= 1_500);
        check_tool_pairing(sent).unwrap();
        assert_eq!(sent[0].content, "Be brief.");
        assert_eq!(sent.last().unwrap().content, "latest question");
    }

    #[tokio::test]
    async fn persisted_context_keeps_a_running_summary() {
        use sinew_core::context::{ContextPolicy, KeepRecent, Summarize, DEFAULT_SUMMARY_HEADER};
        let summarizer = Arc::new(ScriptedModel::new([
            Message::assistant("summary one"),
            Message::assistant("summary two"),
        ]));
        let policy = ContextPolicy::new(2_000).then(
            Summarize::new(summarizer.clone())
                .keep_recent(KeepRecent::Messages(4))
                .max_chunk_tokens(100_000),
        );
        let model = Arc::new(ScriptedModel::new([
            Message::assistant("first"),
            Message::assistant("second"),
        ]));
        let graph = ReactAgent::new(model.clone(), ToolBox::new())
            .system_prompt("Be brief.")
            .context_policy(policy)
            .persist_context(true)
            .compile()
            .unwrap();
        let out = graph
            .invoke(AgentState::new(long_history(20)))
            .await
            .unwrap();
        assert!(out.messages[0].content.starts_with(DEFAULT_SUMMARY_HEADER));
        assert!(out.messages[0].content.ends_with("summary one"));
        // The system prompt is sent but not stored.
        assert!(out.messages.iter().all(|m| m.content != "Be brief."));
        assert_eq!(model.requests()[0].messages[0].content, "Be brief.");

        let mut next = out.messages;
        next.extend(long_history(20));
        let out = graph.invoke(AgentState::new(next)).await.unwrap();
        let summaries: Vec<_> = out
            .messages
            .iter()
            .filter(|m| m.content.starts_with(DEFAULT_SUMMARY_HEADER))
            .collect();
        assert_eq!(summaries.len(), 1);
        assert!(summaries[0].content.ends_with("summary two"));
        // The second summarizer call started from the first summary.
        let input = &summarizer.requests()[1].messages[1].content;
        assert!(input.contains("Current summary:\nsummary one"));
        assert_eq!(model.requests()[1].messages[0].content, "Be brief.");
    }
}
