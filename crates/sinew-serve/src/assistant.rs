//! Registering graphs: [`Assistant`] and the typed [`GraphAssistant`] builder.

use std::sync::Arc;

use bytes::Bytes;
use futures_util::{future::ready, StreamExt};
use serde::Serialize;
use serde_json::Value;
use sinew_core::{BoxStream, Message};
use sinew_graph::{prebuilt::AgentState, Checkpointer, CompiledGraph, GraphEvent, State};
use sinew_trace::Tracer;

use crate::{
    api::{diff, merge, AssistantInfo, StreamMode},
    sse::FrameWriter,
};

type InputFn<S> = Arc<dyn Fn(Option<S>, Value) -> Result<S, String> + Send + Sync>;
type ToStateFn<S> = Arc<dyn Fn(Vec<Message>) -> S + Send + Sync>;
type ReplyFn<S> = Arc<dyn Fn(&S) -> Option<Message> + Send + Sync>;

/// Which SSE events a run produces.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Modes {
    pub values: bool,
    pub updates: bool,
    pub tokens: bool,
    pub events: bool,
    pub custom: bool,
}

impl Modes {
    pub(crate) fn from_list(list: &[StreamMode]) -> Self {
        let mut m = Modes::default();
        if list.is_empty() {
            m.values = true;
        }
        for mode in list {
            match mode {
                StreamMode::Values => m.values = true,
                StreamMode::Updates => m.updates = true,
                StreamMode::Tokens => m.tokens = true,
                StreamMode::Events => m.events = true,
                StreamMode::Custom => m.custom = true,
            }
        }
        m
    }
}

/// Everything needed to start one execution of a graph.
pub(crate) struct StartSpec {
    /// `None` runs the stateless copy of the graph (no checkpoints).
    pub thread_id: Option<String>,
    pub input: Option<Value>,
    /// The thread's current state, for input mapping and update diffs.
    pub prev: Option<Value>,
    pub modes: Modes,
    pub recursion_limit: Option<usize>,
    pub interrupt_before: Option<Vec<String>>,
}

pub(crate) enum Outcome {
    Completed(Value),
    Interrupted { next: Vec<String>, values: Value },
    Error(String),
}

pub(crate) enum Step {
    /// Encoded SSE frames (possibly several).
    Frame(Bytes),
    Done(Outcome),
}

pub(crate) enum ChatPiece {
    Delta(String),
    Done(Result<Message, String>),
}

/// A registered graph with its state type erased.
pub(crate) trait ErasedGraph: Send + Sync {
    fn start(&self, spec: StartSpec) -> Result<BoxStream<'static, Step>, String>;
    /// Checks that `values` decodes into the graph's state type.
    fn validate(&self, values: &Value) -> Result<(), String>;
    /// Runs the graph on a chat transcript (OpenAI façade), if it supports it.
    fn chat(&self, messages: Vec<Message>, stream: bool) -> Option<BoxStream<'static, ChatPiece>>;
}

trait Binder: Send {
    fn bind(
        self: Box<Self>,
        cp: Arc<dyn Checkpointer>,
        tracer: Option<Tracer>,
    ) -> Arc<dyn ErasedGraph>;
}

/// A graph ready to be registered with [`ServerBuilder::assistant`](crate::ServerBuilder::assistant).
/// Build one with [`Assistant::graph`] or [`Assistant::agent`].
pub struct Assistant {
    pub(crate) info: AssistantInfo,
    binder: Box<dyn Binder>,
}

impl Assistant {
    /// Wraps any compiled graph. Register it without a checkpointer: the
    /// server attaches its own store.
    ///
    /// By default a run's `input` is shallow-merged over the thread's
    /// current state (top-level keys replace) and decoded into `S`; change
    /// that with [`GraphAssistant::with_input`].
    pub fn graph<S: State>(id: impl Into<String>, graph: CompiledGraph<S>) -> GraphAssistant<S> {
        let id = id.into();
        GraphAssistant {
            info: AssistantInfo {
                assistant_id: id.clone(),
                graph_id: id.clone(),
                name: id,
                description: String::new(),
                config: Value::Object(Default::default()),
                metadata: Value::Object(Default::default()),
                chat: false,
            },
            graph,
            input: Arc::new(|prev: Option<S>, input: Value| {
                let v = match prev {
                    Some(p) => merge(serde_json::to_value(p).map_err(|e| e.to_string())?, input),
                    None => input,
                };
                S::deserialize(v).map_err(|e| format!("invalid input: {e}"))
            }),
            chat: None,
        }
    }

    /// Wraps a chat agent such as [`react_agent`](sinew_graph::prebuilt::react_agent).
    ///
    /// Input `{"messages": [...]}` (or a bare array of messages, or a string
    /// for one user message) is appended to the thread's conversation, and
    /// the agent is exposed through `/v1/chat/completions`.
    pub fn agent(
        id: impl Into<String>,
        graph: CompiledGraph<AgentState>,
    ) -> GraphAssistant<AgentState> {
        Self::graph(id, graph)
            .with_input(|prev: Option<AgentState>, input: Value| {
                let mut state = prev.unwrap_or_default();
                state.messages.extend(messages_from_input(input)?);
                Ok(state)
            })
            .with_chat(AgentState::new, |s: &AgentState| {
                s.messages
                    .iter()
                    .rev()
                    .find(|m| m.role == sinew_core::Role::Assistant)
                    .cloned()
            })
    }

    pub fn id(&self) -> &str {
        &self.info.assistant_id
    }

    pub(crate) fn bind(
        self,
        cp: Arc<dyn Checkpointer>,
        tracer: Option<Tracer>,
    ) -> (AssistantInfo, Arc<dyn ErasedGraph>) {
        (self.info, self.binder.bind(cp, tracer))
    }
}

fn messages_from_input(input: Value) -> Result<Vec<Message>, String> {
    let list = match input {
        Value::String(s) => return Ok(vec![Message::user(s)]),
        Value::Object(mut o) => o.remove("messages").unwrap_or(Value::Array(Vec::new())),
        other => other,
    };
    serde_json::from_value(list).map_err(|e| format!("invalid messages: {e}"))
}

/// Typed builder returned by [`Assistant::graph`] and [`Assistant::agent`].
pub struct GraphAssistant<S> {
    info: AssistantInfo,
    graph: CompiledGraph<S>,
    input: InputFn<S>,
    chat: Option<(ToStateFn<S>, ReplyFn<S>)>,
}

impl<S: State> GraphAssistant<S> {
    /// Display name (defaults to the id).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.info.name = name.into();
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.info.description = description.into();
        self
    }

    /// Free-form configuration shown by `GET /assistants/{id}`.
    pub fn config(mut self, config: Value) -> Self {
        self.info.config = config;
        self
    }

    pub fn metadata(mut self, metadata: Value) -> Self {
        self.info.metadata = metadata;
        self
    }

    /// How a run's `input` becomes the state: receives the thread's current
    /// state (if any) and the raw input.
    pub fn with_input<F>(mut self, f: F) -> Self
    where
        F: Fn(Option<S>, Value) -> Result<S, String> + Send + Sync + 'static,
    {
        self.input = Arc::new(f);
        self
    }

    /// Exposes the graph through the OpenAI façade: `to_state` builds the
    /// initial state from the request's messages, `reply` extracts the
    /// assistant answer from the final state.
    pub fn with_chat<T, R>(mut self, to_state: T, reply: R) -> Self
    where
        T: Fn(Vec<Message>) -> S + Send + Sync + 'static,
        R: Fn(&S) -> Option<Message> + Send + Sync + 'static,
    {
        self.chat = Some((Arc::new(to_state), Arc::new(reply)));
        self.info.chat = true;
        self
    }
}

impl<S: State> From<GraphAssistant<S>> for Assistant {
    fn from(g: GraphAssistant<S>) -> Self {
        Assistant {
            info: g.info.clone(),
            binder: Box::new(g),
        }
    }
}

impl<S: State> Binder for GraphAssistant<S> {
    fn bind(
        self: Box<Self>,
        cp: Arc<dyn Checkpointer>,
        tracer: Option<Tracer>,
    ) -> Arc<dyn ErasedGraph> {
        let mut stateless = self.graph;
        if let Some(t) = tracer {
            stateless = stateless.with_tracer(t);
        }
        let stateful = stateless.clone().with_shared_checkpointer(cp);
        Arc::new(Bound {
            stateful,
            stateless,
            input: self.input,
            chat: self.chat,
        })
    }
}

struct Bound<S> {
    stateful: CompiledGraph<S>,
    stateless: CompiledGraph<S>,
    input: InputFn<S>,
    chat: Option<(ToStateFn<S>, ReplyFn<S>)>,
}

#[derive(Serialize)]
struct TokenOut<'a> {
    node: &'a str,
    delta: &'a str,
    step: usize,
    task: usize,
}

#[derive(Serialize)]
struct LifecycleOut<'a> {
    event: &'static str,
    node: &'a str,
    step: usize,
    task: usize,
}

#[derive(Serialize)]
struct CustomOut<'a> {
    node: &'a str,
    value: &'a Value,
    step: usize,
    task: usize,
}

/// Turns graph events into SSE frames, serializing state only for the
/// modes that need it.
struct Mapper {
    w: FrameWriter,
    modes: Modes,
    last: Option<Value>,
}

impl Mapper {
    fn map<S: State>(&mut self, ev: GraphEvent<S>) -> Option<Step> {
        match ev {
            GraphEvent::Token {
                node,
                step,
                task,
                delta,
            } if self.modes.tokens => Some(Step::Frame(self.w.event(
                "tokens",
                &TokenOut {
                    node: &node,
                    delta: &delta,
                    step,
                    task,
                },
            ))),
            GraphEvent::NodeStart { node, step, task } if self.modes.events => {
                Some(Step::Frame(self.w.event(
                    "events",
                    &LifecycleOut {
                        event: "node_start",
                        node: &node,
                        step,
                        task,
                    },
                )))
            }
            GraphEvent::NodeEnd {
                node,
                step,
                task,
                state,
            } => {
                if self.modes.events {
                    self.w.push_event(
                        "events",
                        &LifecycleOut {
                            event: "node_end",
                            node: &node,
                            step,
                            task,
                        },
                    );
                }
                if self.modes.updates {
                    match serde_json::to_value(&state) {
                        Ok(v) => {
                            let mut upd = serde_json::Map::with_capacity(1);
                            upd.insert(node, diff(self.last.as_ref(), &v));
                            self.w.push_event("updates", &upd);
                            if self.modes.values {
                                self.w.push_event("values", &v);
                            }
                            self.last = Some(v);
                        }
                        Err(e) => self
                            .w
                            .push_event("error", &serde_json::json!({"error": e.to_string()})),
                    }
                } else if self.modes.values {
                    self.w.push_event("values", &state);
                }
                (!self.w.is_empty()).then(|| Step::Frame(self.w.take()))
            }
            GraphEvent::Custom {
                node,
                step,
                task,
                value,
            } if self.modes.custom => Some(Step::Frame(self.w.event(
                "custom",
                &CustomOut {
                    node: &node,
                    value: &value,
                    step,
                    task,
                },
            ))),
            GraphEvent::Interrupted { next, state } => {
                Some(Step::Done(match serde_json::to_value(&state) {
                    Ok(values) => Outcome::Interrupted { next, values },
                    Err(e) => Outcome::Error(e.to_string()),
                }))
            }
            GraphEvent::Completed { state } => {
                Some(Step::Done(match serde_json::to_value(&state) {
                    Ok(v) => Outcome::Completed(v),
                    Err(e) => Outcome::Error(e.to_string()),
                }))
            }
            GraphEvent::Error(e) => Some(Step::Done(Outcome::Error(e.to_string()))),
            _ => None,
        }
    }
}

impl<S: State> ErasedGraph for Bound<S> {
    fn start(&self, spec: StartSpec) -> Result<BoxStream<'static, Step>, String> {
        let mut graph = if spec.thread_id.is_some() {
            self.stateful.clone()
        } else {
            self.stateless.clone()
        };
        if let Some(n) = spec.recursion_limit {
            graph = graph.with_recursion_limit(n);
        }
        if let Some(nodes) = spec.interrupt_before {
            graph = graph.interrupt_before(nodes);
        }
        let mut last = spec.prev;
        let input = match spec.input {
            Some(raw) => {
                let prev = last
                    .take()
                    .map(S::deserialize)
                    .transpose()
                    .map_err(|e| format!("stored state does not match the graph: {e}"))?;
                let s = (self.input)(prev, raw)?;
                if spec.modes.updates {
                    last = serde_json::to_value(&s).ok();
                }
                Some(s)
            }
            None if spec.thread_id.is_none() => {
                return Err("a stateless run needs `input`".into());
            }
            None => None,
        };
        let thread = spec
            .thread_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let mut mapper = Mapper {
            w: FrameWriter::default(),
            modes: spec.modes,
            last,
        };
        Ok(Box::pin(
            graph
                .stream(thread, input)
                .filter_map(move |ev| ready(mapper.map(ev))),
        ))
    }

    fn validate(&self, values: &Value) -> Result<(), String> {
        S::deserialize(values)
            .map(|_| ())
            .map_err(|e| format!("state does not match the graph: {e}"))
    }

    fn chat(&self, messages: Vec<Message>, stream: bool) -> Option<BoxStream<'static, ChatPiece>> {
        let (to_state, reply) = self.chat.clone()?;
        let state = to_state(messages);
        let graph = self.stateless.clone();
        let reply_of = move |s: &S| Ok(reply(s).unwrap_or_else(|| Message::assistant("")));
        if !stream {
            let fut = async move {
                ChatPiece::Done(match graph.invoke(state).await {
                    Ok(s) => reply_of(&s),
                    Err(e) => Err(e.to_string()),
                })
            };
            return Some(Box::pin(futures_util::stream::once(fut)));
        }
        let thread = uuid::Uuid::new_v4().to_string();
        Some(Box::pin(graph.stream(thread, Some(state)).filter_map(
            move |ev| {
                ready(match ev {
                    GraphEvent::Token { delta, .. } => Some(ChatPiece::Delta(delta)),
                    GraphEvent::Completed { state } | GraphEvent::Interrupted { state, .. } => {
                        Some(ChatPiece::Done(reply_of(&state)))
                    }
                    GraphEvent::Error(e) => Some(ChatPiece::Done(Err(e.to_string()))),
                    _ => None,
                })
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn agent_input_forms() {
        assert_eq!(messages_from_input(json!("hi")).unwrap()[0].content, "hi");
        let m =
            messages_from_input(json!({"messages": [{"role": "user", "content": "a"}]})).unwrap();
        assert_eq!(m.len(), 1);
        let m = messages_from_input(json!([{"role": "system", "content": "s"}])).unwrap();
        assert_eq!(m[0].role, sinew_core::Role::System);
        assert!(messages_from_input(json!({"messages": 3})).is_err());
    }

    #[test]
    fn default_modes_are_values() {
        let m = Modes::from_list(&[]);
        assert!(m.values && !m.tokens);
        let m = Modes::from_list(&[StreamMode::Tokens]);
        assert!(!m.values && m.tokens);
    }
}
