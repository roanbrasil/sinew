//! Ready-made executors: any [`ChatModel`] ([`ChatModelExecutor`]) and any
//! compiled graph, including the prebuilt `react_agent` ([`GraphExecutor`]).
//!
//! Both stream model tokens straight into an artifact (`artifact-update`
//! events with `append: true`) as the model produces them.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::json;
use sinew_core::{ChatModel, ChatRequest, Role as ChatRole, StreamEvent};
use sinew_graph::{prebuilt::AgentState, CompiledGraph, GraphEvent, State};

use crate::{
    executor::{AgentExecutor, BoxError, EventQueue, RequestContext},
    types::{Message, Part, Role, Task},
};

const DEFAULT_ARTIFACT: &str = "response";

/// Converts an A2A message to a chat message (`user` → user, `agent` →
/// assistant) using [`parts_text`](crate::types::parts_text) for content.
pub fn to_chat_message(m: &Message) -> sinew_core::Message {
    match m.role {
        Role::User => sinew_core::Message::user(m.text()),
        Role::Agent => sinew_core::Message::assistant(m.text()),
    }
}

/// The conversation of a request: the task's history (if the message
/// continues a task), the question the agent asked last (`status.message`),
/// then the new message.
pub fn conversation(task: Option<&Task>, message: &Message) -> Vec<sinew_core::Message> {
    let mut out = Vec::new();
    if let Some(t) = task {
        out.extend(t.history.iter().map(to_chat_message));
        if let Some(m) = &t.status.message {
            if !t.history.iter().any(|h| h.message_id == m.message_id) {
                out.push(to_chat_message(m));
            }
        }
    }
    out.push(to_chat_message(message));
    out
}

// ---------------------------------------------------------------------------
// ChatModel
// ---------------------------------------------------------------------------

/// Serves a [`ChatModel`]: each message becomes a streamed chat call whose
/// text deltas go out as artifact chunks; the task completes when the model
/// is done.
pub struct ChatModelExecutor {
    model: Arc<dyn ChatModel>,
    system: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    artifact_name: String,
}

impl ChatModelExecutor {
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            model,
            system: None,
            temperature: None,
            max_tokens: None,
            artifact_name: DEFAULT_ARTIFACT.into(),
        }
    }

    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system = Some(prompt.into());
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Name of the output artifact (default `response`).
    pub fn with_artifact_name(mut self, name: impl Into<String>) -> Self {
        self.artifact_name = name.into();
        self
    }
}

#[async_trait]
impl AgentExecutor for ChatModelExecutor {
    async fn execute(&self, ctx: RequestContext, events: EventQueue) -> Result<(), BoxError> {
        events.working();
        let mut messages = Vec::new();
        if let Some(s) = &self.system {
            messages.push(sinew_core::Message::system(s.clone()));
        }
        messages.extend(conversation(ctx.task.as_ref(), &ctx.message));
        let mut req = ChatRequest::new(messages);
        req.temperature = self.temperature;
        req.max_tokens = self.max_tokens;

        let mut stream = self.model.chat_stream(req).await?;
        let mut out = events.text_artifact(self.artifact_name.clone());
        while let Some(ev) = stream.next().await {
            if let StreamEvent::TextDelta { text } = ev? {
                out.push(text);
            }
        }
        out.finish();
        events.complete(None);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

type InputFn<S> = dyn Fn(&RequestContext, Option<S>) -> S + Send + Sync;
type OutputFn<S> = dyn Fn(&S) -> String + Send + Sync;
type ResumeFn<S> = dyn Fn(S, &RequestContext) -> Result<Option<S>, String> + Send + Sync;
type InterruptFn<S> = dyn Fn(&S, &[String]) -> String + Send + Sync;

/// Serves a [`CompiledGraph`] as an A2A agent.
///
/// - The graph thread id is the A2A `contextId`. With a checkpointer, a new
///   message in the same context continues the previous conversation (the
///   `input` function receives the last state).
/// - [`GraphEvent::Token`]s (e.g. from `react_agent`'s model node) stream
///   out as artifact chunks; if the graph emitted no tokens, `output(state)`
///   becomes the artifact when the run completes.
/// - An interrupt (`interrupt_before`) becomes `input-required`; the
///   client's reply (a message with the same `taskId`) goes through
///   `resume`, which may edit the paused state (`Ok(Some(state))`), resume it
///   unchanged (`Ok(None)`) or refuse (`Err(reason)`, the task is canceled).
///   Interrupts need a checkpointer to resume.
///
/// Messages for the same context should not run concurrently when the graph
/// has a checkpointer, since they share the thread.
pub struct GraphExecutor<S> {
    graph: CompiledGraph<S>,
    input: Arc<InputFn<S>>,
    output: Arc<OutputFn<S>>,
    resume: Arc<ResumeFn<S>>,
    interrupt: Arc<InterruptFn<S>>,
    artifact_name: String,
}

impl<S: State> GraphExecutor<S> {
    /// `input` builds the run's input state from the request and the
    /// thread's previous state (if any); `output` renders the final answer.
    pub fn new(
        graph: CompiledGraph<S>,
        input: impl Fn(&RequestContext, Option<S>) -> S + Send + Sync + 'static,
        output: impl Fn(&S) -> String + Send + Sync + 'static,
    ) -> Self {
        Self {
            graph,
            input: Arc::new(input),
            output: Arc::new(output),
            resume: Arc::new(|_, _| Ok(None)),
            interrupt: Arc::new(|_, next| {
                format!("Paused before `{}`. Reply to continue.", next.join("`, `"))
            }),
            artifact_name: DEFAULT_ARTIFACT.into(),
        }
    }

    /// How a reply to `input-required` resumes the paused state.
    pub fn with_resume(
        mut self,
        f: impl Fn(S, &RequestContext) -> Result<Option<S>, String> + Send + Sync + 'static,
    ) -> Self {
        self.resume = Arc::new(f);
        self
    }

    /// Text of the `input-required` message for an interrupt before `next`.
    pub fn with_interrupt_message(
        mut self,
        f: impl Fn(&S, &[String]) -> String + Send + Sync + 'static,
    ) -> Self {
        self.interrupt = Arc::new(f);
        self
    }

    pub fn with_artifact_name(mut self, name: impl Into<String>) -> Self {
        self.artifact_name = name.into();
        self
    }
}

impl GraphExecutor<AgentState> {
    /// For [`react_agent`](sinew_graph::prebuilt::react_agent) and other
    /// `AgentState` graphs: the user message is appended to the thread's
    /// previous messages, the answer is the last assistant message, an
    /// interrupt before `tools` asks for approval of the pending tool calls,
    /// and a reply of "no"/"reject"/"cancel"/"deny"/"stop" refuses them.
    pub fn agent(graph: CompiledGraph<AgentState>) -> Self {
        Self::agent_with_system(graph, None::<String>)
    }

    /// Like [`agent`](Self::agent) with a system prompt at the start of
    /// every new conversation.
    pub fn agent_with_system(
        graph: CompiledGraph<AgentState>,
        system: Option<impl Into<String>>,
    ) -> Self {
        let system = system.map(Into::into);
        Self::new(
            graph,
            move |ctx, prev| {
                let mut messages = match prev {
                    Some(s) if !s.messages.is_empty() => s.messages,
                    _ => system
                        .iter()
                        .map(|s| sinew_core::Message::system(s.clone()))
                        .collect(),
                };
                messages.push(to_chat_message(&ctx.message));
                AgentState::new(messages)
            },
            |s| {
                s.messages
                    .iter()
                    .rev()
                    .find(|m| m.role == ChatRole::Assistant && !m.content.is_empty())
                    .map(|m| m.content.clone())
                    .unwrap_or_default()
            },
        )
        .with_interrupt_message(|s, next| {
            let calls: Vec<String> = s
                .last()
                .map(|m| {
                    m.tool_calls
                        .iter()
                        .map(|c| format!("{}({})", c.name, c.arguments))
                        .collect()
                })
                .unwrap_or_default();
            if calls.is_empty() {
                format!(
                    "Paused before `{}`. Reply \"yes\" to continue or \"no\" to stop.",
                    next.join("`, `")
                )
            } else {
                format!(
                    "Approval needed to run: {}. Reply \"yes\" to continue or \"no\" to stop.",
                    calls.join(", ")
                )
            }
        })
        .with_resume(|_, ctx| {
            let reply = ctx.text().trim().to_lowercase();
            let refused = ["no", "n", "reject", "deny", "cancel", "stop"]
                .iter()
                .any(|w| reply == *w || reply.starts_with(&format!("{w} ")));
            if refused {
                Err(format!("Stopped by the user: {}", ctx.text().trim()))
            } else {
                Ok(None)
            }
        })
    }
}

#[async_trait]
impl<S: State> AgentExecutor for GraphExecutor<S> {
    async fn execute(&self, ctx: RequestContext, events: EventQueue) -> Result<(), BoxError> {
        events.working();
        let thread = ctx.context_id.clone();
        let snapshot = self.graph.get_state_snapshot(&thread).await.ok();

        let input = match snapshot {
            Some(snap) if ctx.is_resume() && !snap.next.is_empty() => {
                match (self.resume)(snap.state, &ctx) {
                    Ok(Some(edited)) => self.graph.update_state(&thread, edited).await?,
                    Ok(None) => {}
                    Err(reason) => {
                        let m = events.agent_message(vec![Part::text(reason)]);
                        events.status(crate::types::TaskState::Canceled, Some(m));
                        return Ok(());
                    }
                }
                None
            }
            snap => Some((self.input)(&ctx, snap.map(|s| s.state))),
        };

        let mut stream = self.graph.stream(thread, input);
        let mut out = events.text_artifact(self.artifact_name.clone());
        // Separate the text of successive nodes (e.g. agent turns).
        let mut node_had_tokens = false;
        let mut need_sep = false;
        while let Some(ev) = stream.next().await {
            match ev {
                GraphEvent::Token { delta, .. } => {
                    if need_sep {
                        out.push("\n\n");
                        need_sep = false;
                    }
                    node_had_tokens |= !delta.is_empty();
                    out.push(delta);
                }
                GraphEvent::NodeStart { .. } => {
                    need_sep = need_sep || (node_had_tokens && out.has_content());
                    node_had_tokens = false;
                }
                GraphEvent::Completed { state } => {
                    if !out.has_content() {
                        out.push((self.output)(&state));
                    }
                    out.finish();
                    events.complete(None);
                }
                GraphEvent::Interrupted { next, state, .. } => {
                    out.finish();
                    let text = (self.interrupt)(&state, &next);
                    let m = events.agent_message(vec![
                        Part::text(text),
                        Part::data(json!({ "interrupt": { "next": next } })),
                    ]);
                    events.input_required(m);
                }
                GraphEvent::Error(e) => return Err(e.into()),
                _ => {}
            }
        }
        Ok(())
    }
}
