//! [`Command`]: what a node (or a caller) asks the graph to do next, and the
//! dynamic-interrupt types ([`Interrupt`], [`GraphInterrupt`]).

use std::{collections::HashMap, fmt};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{SendTo, END};

/// Where a [`Command`] applies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandGraph {
    /// The graph the node belongs to (default).
    #[default]
    Current,
    /// The closest parent graph: the node runs inside a subgraph and wants
    /// to update the parent's state and/or route the parent to other nodes
    /// (LangGraph's `Command.PARENT`). The subgraph run stops immediately.
    Parent,
}

/// A navigation target of a [`Command`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "S: Serialize", deserialize = "S: Deserialize<'de>"))]
pub enum Goto<S> {
    /// Run this node next (`END` ends the branch).
    Node(String),
    /// Run a node on its own input next (map-style branch).
    Send(SendTo<S>),
}

impl<S> From<&str> for Goto<S> {
    fn from(n: &str) -> Self {
        Goto::Node(n.to_string())
    }
}

impl<S> From<String> for Goto<S> {
    fn from(n: String) -> Self {
        Goto::Node(n)
    }
}

impl<S> From<SendTo<S>> for Goto<S> {
    fn from(s: SendTo<S>) -> Self {
        Goto::Send(s)
    }
}

/// Values that resume pending [`interrupt`](crate::NodeContext::interrupt)s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Resume {
    /// Resumes the first pending interrupt (in task order).
    Value(Value),
    /// Resumes interrupts by [`Interrupt::id`].
    ById(HashMap<String, Value>),
}

/// Combined state update and control flow, returned by nodes registered
/// with [`StateGraph::add_command_node`](crate::StateGraph::add_command_node)
/// and accepted as run input by
/// [`CompiledGraph::run_with`](crate::CompiledGraph::run_with).
///
/// As a **node output**:
/// - `update` replaces the state (like a plain node's return value);
///   `update_json` overlays top-level fields onto it (a partial update).
///   Without either, the node writes nothing.
/// - `goto` schedules nodes for the next superstep, in addition to the
///   node's outgoing edge (if it has one).
/// - `graph: Parent` sends the update and the `goto` to the parent graph
///   when the node runs inside a subgraph.
///
/// As **run input** (resuming a thread):
/// - `resume` answers pending dynamic interrupts;
/// - `update` / `update_json` edit the checkpointed state first;
/// - `goto` replaces the pending tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "S: Serialize", deserialize = "S: Deserialize<'de>"))]
pub struct Command<S> {
    #[serde(default)]
    pub update: Option<S>,
    #[serde(default)]
    pub update_json: Option<Value>,
    #[serde(default)]
    pub goto: Vec<Goto<S>>,
    #[serde(default)]
    pub graph: CommandGraph,
    #[serde(default)]
    pub resume: Option<Resume>,
}

impl<S> Default for Command<S> {
    fn default() -> Self {
        Self {
            update: None,
            update_json: None,
            goto: Vec::new(),
            graph: CommandGraph::Current,
            resume: None,
        }
    }
}

impl<S> Command<S> {
    /// Shorthand for [`CommandGraph::Parent`].
    pub const PARENT: CommandGraph = CommandGraph::Parent;

    /// An empty command (no update, no navigation).
    pub fn new() -> Self {
        Self::default()
    }

    /// Navigate to `node` (or `END`).
    pub fn goto(node: impl Into<Goto<S>>) -> Self {
        Self::new().with_goto(node)
    }

    /// Navigate to several nodes; they run in parallel next superstep.
    pub fn goto_many<I, T>(nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Goto<S>>,
    {
        let mut c = Self::new();
        c.goto = nodes.into_iter().map(Into::into).collect();
        c
    }

    /// End this branch: equivalent to `Command::goto(END)`.
    pub fn end() -> Self {
        Self::goto(END)
    }

    /// Replace the state.
    pub fn update(state: S) -> Self {
        Self::new().with_update(state)
    }

    /// Resume the first pending interrupt with `value`.
    pub fn resume(value: impl Serialize) -> Self {
        let mut c = Self::new();
        c.resume = Some(Resume::Value(
            serde_json::to_value(value).unwrap_or(Value::Null),
        ));
        c
    }

    /// Resume interrupts by id.
    pub fn resume_map<I, K>(values: I) -> Self
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let mut c = Self::new();
        c.resume = Some(Resume::ById(
            values.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        ));
        c
    }

    pub fn with_update(mut self, state: S) -> Self {
        self.update = Some(state);
        self
    }

    /// Partial update: top-level fields of `patch` (a JSON object) overwrite
    /// the same fields of the state. Applied after `update`.
    pub fn with_json_update(mut self, patch: Value) -> Self {
        self.update_json = Some(patch);
        self
    }

    pub fn with_goto(mut self, node: impl Into<Goto<S>>) -> Self {
        self.goto.push(node.into());
        self
    }

    pub fn with_send(mut self, send: SendTo<S>) -> Self {
        self.goto.push(Goto::Send(send));
        self
    }

    pub fn with_graph(mut self, graph: CommandGraph) -> Self {
        self.graph = graph;
        self
    }

    /// Apply this command to the parent graph (see [`CommandGraph::Parent`]).
    pub fn to_parent(self) -> Self {
        self.with_graph(CommandGraph::Parent)
    }

    pub(crate) fn has_write(&self) -> bool {
        self.update.is_some() || self.update_json.is_some()
    }
}

/// A pause requested by [`NodeContext::interrupt`](crate::NodeContext::interrupt).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interrupt {
    /// Stable id; resume a specific interrupt with [`Command::resume_map`].
    pub id: String,
    /// The payload passed to `interrupt` (e.g. a question for a human).
    pub value: Value,
    /// Node that raised it.
    pub node: String,
    /// Subgraph path of that node (empty for the root graph).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub namespace: Vec<String>,
}

/// Error value a node propagates (with `?`) to pause the run. Created by
/// [`NodeContext::interrupt`](crate::NodeContext::interrupt); you normally
/// never build one yourself. Do not swallow it.
#[derive(Debug, Clone)]
pub struct GraphInterrupt {
    pub interrupts: Vec<Interrupt>,
}

impl fmt::Display for GraphInterrupt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "graph interrupted")?;
        if let Some(i) = self.interrupts.first() {
            write!(f, " in `{}`: {}", i.node, i.value)?;
        }
        Ok(())
    }
}

impl std::error::Error for GraphInterrupt {}

/// Error returned by a node attempt that exceeded its timeout
/// ([`StateGraph::set_node_timeout`](crate::StateGraph::set_node_timeout)).
#[derive(Debug, Clone)]
pub struct NodeTimeout {
    pub after: std::time::Duration,
}

impl fmt::Display for NodeTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "timed out after {:?}", self.after)
    }
}

impl std::error::Error for NodeTimeout {}

/// A [`Command`] that a subgraph node sent to its parent graph.
#[derive(Debug, Clone)]
pub struct ParentCommand {
    /// Overlay for the parent's state (top-level JSON fields).
    pub update: Option<Value>,
    pub goto: Vec<Goto<Value>>,
}

/// What a run starts from.
#[derive(Debug, Clone)]
pub enum RunInput<S> {
    /// Start the thread over from the entry point with this state.
    Start(S),
    /// Continue the thread from its checkpoint.
    Continue,
    /// Continue the thread applying a [`Command`] (resume, update, goto).
    Command(Command<S>),
}

impl<S> From<S> for RunInput<S> {
    fn from(s: S) -> Self {
        RunInput::Start(s)
    }
}

impl<S> From<Option<S>> for RunInput<S> {
    fn from(s: Option<S>) -> Self {
        s.map_or(RunInput::Continue, RunInput::Start)
    }
}

impl<S> From<Command<S>> for RunInput<S> {
    fn from(c: Command<S>) -> Self {
        RunInput::Command(c)
    }
}

/// Per-run settings: which thread, optionally which checkpoint to start
/// from (replay / fork), and a recursion limit override.
#[derive(Debug, Clone, Default)]
pub struct RunConfig {
    pub thread_id: String,
    /// Continue from this checkpoint instead of the latest one. New
    /// checkpoints are chained to it, forming a branch.
    pub checkpoint_id: Option<String>,
    pub recursion_limit: Option<usize>,
}

impl RunConfig {
    pub fn new(thread_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            ..Self::default()
        }
    }

    pub fn at_checkpoint(mut self, checkpoint_id: impl Into<String>) -> Self {
        self.checkpoint_id = Some(checkpoint_id.into());
        self
    }

    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = Some(limit);
        self
    }
}

impl From<&str> for RunConfig {
    fn from(t: &str) -> Self {
        Self::new(t)
    }
}

impl From<String> for RunConfig {
    fn from(t: String) -> Self {
        Self::new(t)
    }
}

impl From<&String> for RunConfig {
    fn from(t: &String) -> Self {
        Self::new(t.clone())
    }
}

/// Overlays the top-level fields of `patch` onto `base` (both JSON objects).
pub(crate) fn overlay(base: &mut Value, patch: Value) {
    match (base, patch) {
        (Value::Object(b), Value::Object(p)) => {
            for (k, v) in p {
                b.insert(k, v);
            }
        }
        (b, p) => *b = p,
    }
}
