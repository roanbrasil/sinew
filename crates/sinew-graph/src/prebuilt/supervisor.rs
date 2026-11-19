use std::sync::Arc;

use serde_json::json;
use sinew_core::{ChatModel, ChatRequest, Message, Role, ToolCall};
use sinew_trace::traced_chat;

use super::{AgentState, Handoff};
use crate::{Command, CompiledGraph, GraphError, StateGraph, END};

/// An agent managed by a [`Supervisor`]: its node name, what it is good at
/// (shown to the supervisor model), and its graph.
#[derive(Clone)]
pub struct NamedAgent {
    pub name: String,
    pub description: String,
    pub graph: CompiledGraph<AgentState>,
}

impl NamedAgent {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        graph: CompiledGraph<AgentState>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            graph,
        }
    }
}

/// What a worker agent adds to the shared conversation when it finishes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputMode {
    /// Every message the agent produced (tool calls and results included).
    #[default]
    FullHistory,
    /// Only the agent's final message.
    LastMessage,
}

/// A supervisor that routes a conversation among worker agents
/// (LangGraph's `create_supervisor`).
///
/// The `supervisor` node calls the model with one handoff tool per agent
/// (`transfer_to_<name>`). Calling one runs that agent (a subgraph on the
/// shared [`AgentState`]); when the agent finishes, control returns to the
/// supervisor with a `transfer_back_to_supervisor` exchange appended. A
/// reply without tool calls ends the run. Bound the number of rounds with
/// the compiled graph's recursion limit.
pub struct Supervisor {
    model: Arc<dyn ChatModel>,
    agents: Vec<NamedAgent>,
    prompt: Option<String>,
    output_mode: OutputMode,
    #[cfg(feature = "identity")]
    auth: Option<super::MessageAuth>,
}

impl Supervisor {
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            model,
            agents: Vec::new(),
            prompt: None,
            output_mode: OutputMode::default(),
            #[cfg(feature = "identity")]
            auth: None,
        }
    }

    /// Authenticates what each worker hands back (feature `identity`): the
    /// worker's output is signed with its credentials in `auth` (keyed by
    /// the worker's node name) and verified as a message to `supervisor`;
    /// the record goes to `auth`'s ledger. The `transfer_back_to_supervisor`
    /// result tells the supervisor model who verifiably sent the output and
    /// that identity claims or approvals inside it are not authenticated. A
    /// worker whose output fails verification (no credentials when
    /// signatures are required, a revoked identity or key) has its output
    /// replaced by a rejection notice.
    ///
    /// ```
    /// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
    /// use std::sync::Arc;
    /// use sinew_core::{identity::*, mock::ScriptedModel, Message, ToolBox, ToolCall};
    /// use sinew_graph::prebuilt::{react_agent, AgentState, MessageAuth, Supervisor};
    ///
    /// let issuer = AgentKey::from_bytes(&[1; 32]);
    /// let key = AgentKey::from_bytes(&[2; 32]);
    /// let billing = AgentCredentials::new(
    ///     key.clone(),
    ///     AgentIdentity::new("billing", key.public_key()).sign(&issuer).unwrap(),
    /// ).unwrap();
    /// let auth = MessageAuth::new(MessageVerifier::new(TrustStore::new().trust(issuer.public_key())))
    ///     .agent(billing);
    ///
    /// let worker = react_agent(Arc::new(ScriptedModel::new([Message::assistant("paid")])), ToolBox::new()).unwrap();
    /// let boss = Arc::new(ScriptedModel::new([
    ///     Message::assistant_with_tools("", vec![ToolCall { id: "c".into(), name: "transfer_to_billing".into(), arguments: serde_json::json!({}) }]),
    ///     Message::assistant("done"),
    /// ]));
    /// let graph = Supervisor::new(boss).agent("billing", "Invoices", worker).message_auth(auth.clone()).compile().unwrap();
    /// let out = graph.invoke(AgentState::new(vec![Message::user("is invoice 7 paid?")])).await.unwrap();
    /// let back = &out.messages[out.messages.len() - 2];
    /// assert!(back.content.contains("verified as agent `billing`"));
    /// assert!(auth.ledger().records()[0].is_verified());
    /// # });
    /// ```
    #[cfg(feature = "identity")]
    pub fn message_auth(mut self, auth: super::MessageAuth) -> Self {
        self.auth = Some(auth);
        self
    }

    pub fn agent(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        graph: CompiledGraph<AgentState>,
    ) -> Self {
        self.agents.push(NamedAgent::new(name, description, graph));
        self
    }

    /// Replaces the default system prompt of the supervisor.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn output_mode(mut self, mode: OutputMode) -> Self {
        self.output_mode = mode;
        self
    }

    /// The supervisor graph, not yet compiled.
    pub fn graph(self) -> StateGraph<AgentState> {
        let handoffs: Vec<Handoff> = self
            .agents
            .iter()
            .map(|a| {
                Handoff::to(
                    a.name.clone(),
                    format!("Assign the task to {}: {}", a.name, a.description),
                )
            })
            .collect();
        let prompt = self.prompt.unwrap_or_else(|| {
            let roster: Vec<String> = self
                .agents
                .iter()
                .map(|a| format!("- {}: {}", a.name, a.description))
                .collect();
            format!(
                "You are a supervisor managing these agents:\n{}\nAssign work to one agent at a \
                 time by calling its transfer tool; do not do the work yourself. When the task \
                 is done, answer the user directly with the final result.",
                roster.join("\n")
            )
        });
        let specs: Vec<_> = handoffs.iter().map(Handoff::spec).collect();
        let mut destinations: Vec<String> = self.agents.iter().map(|a| a.name.clone()).collect();
        destinations.push("supervisor".into());
        destinations.push(END.into());

        let mut g = StateGraph::<AgentState>::new();
        let model = self.model;
        g.add_command_node("supervisor", destinations, move |mut s: AgentState, ctx| {
            let model = model.clone();
            let handoffs = handoffs.clone();
            let mut messages = s.messages.clone();
            if messages.first().is_none_or(|m| m.role != Role::System) {
                messages.insert(0, Message::system(prompt.clone()));
            }
            let req = ChatRequest::new(messages).with_tools(specs.clone());
            async move {
                let resp =
                    traced_chat(model.as_ref(), req, ctx.tracer.as_ref(), ctx.run.as_ref()).await?;
                let calls = resp.message.tool_calls.clone();
                s.messages.push(resp.message);
                if calls.is_empty() {
                    return Ok(Command::end().with_update(s));
                }
                let mut target = None;
                for call in &calls {
                    let reply = match handoffs.iter().find(|h| h.tool_name == call.name) {
                        Some(h) if target.is_none() => {
                            target = Some(h.agent.clone());
                            format!("Successfully transferred to {}", h.agent)
                        }
                        Some(_) => "Ignored: only one transfer at a time".to_string(),
                        None => format!("Unknown tool `{}`; call a transfer tool", call.name),
                    };
                    s.messages
                        .push(Message::tool_result(call.id.clone(), reply));
                }
                let next = target.unwrap_or_else(|| "supervisor".to_string());
                Ok(Command::goto(next).with_update(s))
            }
        });

        #[cfg(feature = "identity")]
        let auth = self.auth;
        for agent in self.agents {
            let mode = self.output_mode;
            let name = agent.name.clone();
            #[cfg(feature = "identity")]
            let auth = auth.clone();
            g.add_subgraph_mapped(
                agent.name.clone(),
                agent.graph,
                AgentState::clone,
                move |mut parent: AgentState, child: AgentState| {
                    #[cfg_attr(not(feature = "identity"), allow(unused_mut))]
                    let mut back = "Successfully transferred back to supervisor".to_string();
                    let start = parent.messages.len();
                    match mode {
                        OutputMode::FullHistory => parent.messages = child.messages,
                        OutputMode::LastMessage => {
                            if let Some(last) = child.messages.last() {
                                parent.messages.push(last.clone());
                            }
                        }
                    }
                    #[cfg(feature = "identity")]
                    if let Some(auth) = &auth {
                        let start = start.min(parent.messages.len());
                        let content =
                            serde_json::to_string(&parent.messages[start..]).unwrap_or_default();
                        let record = auth.exchange(&name, "supervisor", None, &content);
                        if record.is_rejected() {
                            parent.messages.truncate(start);
                            parent.messages.push(Message::assistant(format!(
                                "[the output of `{name}` was dropped: it failed authentication]"
                            )));
                        }
                        back = format!(
                            "{back}. About the messages of `{name}` since the transfer: {}",
                            sinew_core::identity::peer_note(&record)
                        );
                    }
                    let _ = start;
                    let id = format!("back_{}_{}", name, parent.messages.len());
                    parent.messages.push(Message::assistant_with_tools(
                        "",
                        vec![ToolCall {
                            id: id.clone(),
                            name: "transfer_back_to_supervisor".into(),
                            arguments: json!({}),
                        }],
                    ));
                    parent.messages.push(Message::tool_result(id, back));
                    parent
                },
            );
            g.add_edge(agent.name, "supervisor");
        }
        g.set_entry_point("supervisor");
        g
    }

    pub fn compile(self) -> Result<CompiledGraph<AgentState>, GraphError> {
        self.graph().compile()
    }
}

/// Builds a [`Supervisor`] over `agents` with default options.
pub fn create_supervisor(
    model: Arc<dyn ChatModel>,
    agents: Vec<NamedAgent>,
) -> Result<CompiledGraph<AgentState>, GraphError> {
    let mut s = Supervisor::new(model);
    s.agents = agents;
    s.compile()
}
