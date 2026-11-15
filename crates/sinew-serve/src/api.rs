//! Wire types shared by the server and [`ServeClient`](crate::ServeClient).
//!
//! Timestamps are Unix epoch milliseconds.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

/// A registered graph, as listed by `GET /assistants`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantInfo {
    pub assistant_id: String,
    /// Same as `assistant_id`: every assistant wraps one graph.
    pub graph_id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Free-form configuration published with the assistant.
    #[serde(default)]
    pub config: Value,
    #[serde(default)]
    pub metadata: Value,
    /// Whether the assistant is exposed through `/v1/chat/completions`.
    #[serde(default)]
    pub chat: bool,
}

/// Lifecycle of a thread.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThreadStatus {
    /// No run in flight and nothing pending.
    #[default]
    Idle,
    /// A run is executing or queued.
    Busy,
    /// The last run paused before a node; resume it with a new run.
    Interrupted,
    /// The last run failed.
    Error,
}

impl ThreadStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ThreadStatus::Idle => "idle",
            ThreadStatus::Busy => "busy",
            ThreadStatus::Interrupted => "interrupted",
            ThreadStatus::Error => "error",
        }
    }

    pub(crate) fn parse(s: &str) -> Self {
        match s {
            "busy" => ThreadStatus::Busy,
            "interrupted" => ThreadStatus::Interrupted,
            "error" => ThreadStatus::Error,
            _ => ThreadStatus::Idle,
        }
    }
}

/// A conversation thread: a durable sequence of checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Thread {
    pub thread_id: String,
    /// Assistant the thread is bound to (set by its first run).
    #[serde(default)]
    pub assistant_id: Option<String>,
    #[serde(default)]
    pub status: ThreadStatus,
    #[serde(default)]
    pub metadata: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Body of `POST /threads`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreadCreate {
    /// Choose the id yourself; a UUID is generated otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assistant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// `"raise"` (default, 409 when the id exists) or `"do_nothing"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub if_exists: Option<String>,
}

/// Query of `GET /threads` / body of `POST /threads/search`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreadQuery {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assistant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ThreadStatus>,
}

/// A thread's state at one checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreadState {
    pub thread_id: String,
    /// The graph state (`S` serialized).
    pub values: Value,
    /// Nodes of the next superstep; empty when the run finished.
    pub next: Vec<String>,
    /// Supersteps completed when the checkpoint was written.
    pub step: usize,
}

impl ThreadState {
    /// Decodes `values` into the graph's state type.
    pub fn values_as<S: serde::de::DeserializeOwned>(&self) -> Result<S, serde_json::Error> {
        S::deserialize(&self.values)
    }
}

/// Body of `POST /threads/{id}/state`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateUpdate {
    /// Top-level keys to overwrite in the current state (a shallow merge).
    /// A non-object value replaces the whole state.
    pub values: Value,
}

/// What a run streams over SSE. Several can be combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamMode {
    /// The state after every node (`event: values`).
    Values,
    /// Top-level keys each node changed, as `{node: {key: value}}` (`event: updates`).
    Updates,
    /// Model tokens emitted by nodes (`event: tokens`). `messages` is accepted as an alias.
    #[serde(alias = "messages")]
    Tokens,
    /// Node start/end lifecycle events (`event: events`).
    Events,
    /// Values emitted with `NodeContext::emit` (`event: custom`).
    Custom,
}

/// What to do when a thread already has a run in flight.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MultitaskStrategy {
    /// Refuse the new run with `409 Conflict`.
    #[default]
    Reject,
    /// Queue it; runs on a thread execute one at a time, in order.
    Enqueue,
    /// Cancel what is running or queued (checkpoints stay) and start this one.
    Interrupt,
}

/// What happens to a streamed run when the client disconnects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnDisconnect {
    #[default]
    Cancel,
    Continue,
}

/// Edits applied to a paused thread before it resumes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Command {
    /// Shallow-merged into the thread state before resuming.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update: Option<Value>,
    /// Accepted for LangGraph compatibility: an object is merged like `update`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume: Option<Value>,
}

/// Body of every run-creating endpoint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunCreate {
    pub assistant_id: String,
    /// New input. Without it (and with a checkpoint) the thread resumes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<Command>,
    /// Defaults to `["values"]`. A single string is accepted.
    #[serde(
        default,
        deserialize_with = "one_or_many",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub stream_mode: Vec<StreamMode>,
    #[serde(default)]
    pub multitask_strategy: MultitaskStrategy,
    #[serde(default)]
    pub on_disconnect: OnDisconnect,
    /// POSTed the final [`Run`] when the run ends (if the server allows it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webhook: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Overrides the graph's recursion limit for this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recursion_limit: Option<usize>,
    /// Overrides the graph's `interrupt_before` nodes for this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interrupt_before: Option<Vec<String>>,
    /// `"create"` (default) creates a missing thread; `"reject"` answers 404.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub if_not_exists: Option<String>,
}

impl RunCreate {
    /// A run of `assistant_id` with `input`.
    pub fn new(assistant_id: impl Into<String>, input: Value) -> Self {
        Self {
            assistant_id: assistant_id.into(),
            input: Some(input),
            ..Default::default()
        }
    }

    /// A run that resumes the thread's latest checkpoint.
    pub fn resume(assistant_id: impl Into<String>) -> Self {
        Self {
            assistant_id: assistant_id.into(),
            ..Default::default()
        }
    }

    pub fn with_stream_mode(mut self, modes: impl IntoIterator<Item = StreamMode>) -> Self {
        self.stream_mode = modes.into_iter().collect();
        self
    }

    pub fn with_update(mut self, update: Value) -> Self {
        self.command.get_or_insert_with(Command::default).update = Some(update);
        self
    }

    pub fn with_multitask(mut self, strategy: MultitaskStrategy) -> Self {
        self.multitask_strategy = strategy;
        self
    }

    pub fn with_webhook(mut self, url: impl Into<String>) -> Self {
        self.webhook = Some(url.into());
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_interrupt_before<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_before = Some(nodes.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_on_disconnect(mut self, on: OnDisconnect) -> Self {
        self.on_disconnect = on;
        self
    }
}

fn one_or_many<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<StreamMode>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(StreamMode),
        Many(Vec<StreamMode>),
        Null(()),
    }
    Ok(match OneOrMany::deserialize(d)? {
        OneOrMany::One(m) => vec![m],
        OneOrMany::Many(v) => v,
        OneOrMany::Null(()) => Vec::new(),
    })
}

/// Lifecycle of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    /// Waiting for an earlier run on the same thread (`enqueue`).
    Pending,
    Running,
    Success,
    Error,
    /// Paused before a node; see [`Run::interrupted_before`].
    Interrupted,
    Cancelled,
}

impl RunStatus {
    /// Whether the run is over.
    pub fn is_terminal(self) -> bool {
        !matches!(self, RunStatus::Pending | RunStatus::Running)
    }
}

/// A run, as returned by the run endpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub run_id: String,
    /// `None` for stateless runs.
    #[serde(default)]
    pub thread_id: Option<String>,
    pub assistant_id: String,
    pub status: RunStatus,
    pub created_at: i64,
    pub updated_at: i64,
    pub multitask_strategy: MultitaskStrategy,
    #[serde(default)]
    pub metadata: Value,
    /// Final (or paused) state once the run is over.
    #[serde(default)]
    pub output: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
    /// Nodes the run paused before, when `status` is `interrupted`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interrupted_before: Vec<String>,
}

impl Run {
    /// Decodes `output` into the graph's state type.
    pub fn output_as<S: serde::de::DeserializeOwned>(
        &self,
    ) -> Option<Result<S, serde_json::Error>> {
        self.output.as_ref().map(S::deserialize)
    }
}

/// Shallow merge: top-level keys of `patch` overwrite `base`. A non-object
/// patch (or base) replaces the whole value.
pub(crate) fn merge(base: Value, patch: Value) -> Value {
    match (base, patch) {
        (Value::Object(mut b), Value::Object(p)) => {
            for (k, v) in p {
                b.insert(k, v);
            }
            Value::Object(b)
        }
        (_, p) => p,
    }
}

/// Top-level keys whose value differs between `prev` and `next`.
pub(crate) fn diff(prev: Option<&Value>, next: &Value) -> Value {
    match (prev, next) {
        (Some(Value::Object(p)), Value::Object(n)) => {
            let mut out = Map::new();
            for (k, v) in n {
                if p.get(k) != Some(v) {
                    out.insert(k.clone(), v.clone());
                }
            }
            Value::Object(out)
        }
        _ => next.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stream_mode_accepts_string_array_and_alias() {
        let r: RunCreate =
            serde_json::from_value(json!({"assistant_id": "a", "stream_mode": "messages"}))
                .unwrap();
        assert_eq!(r.stream_mode, vec![StreamMode::Tokens]);
        let r: RunCreate = serde_json::from_value(
            json!({"assistant_id": "a", "stream_mode": ["values", "updates"]}),
        )
        .unwrap();
        assert_eq!(r.stream_mode, vec![StreamMode::Values, StreamMode::Updates]);
        let r: RunCreate = serde_json::from_value(json!({"assistant_id": "a"})).unwrap();
        assert!(r.stream_mode.is_empty());
        assert_eq!(r.multitask_strategy, MultitaskStrategy::Reject);
    }

    #[test]
    fn merge_and_diff_are_shallow() {
        let base = json!({"a": 1, "b": [1]});
        let merged = merge(base.clone(), json!({"b": [2], "c": true}));
        assert_eq!(merged, json!({"a": 1, "b": [2], "c": true}));
        assert_eq!(diff(Some(&base), &merged), json!({"b": [2], "c": true}));
        assert_eq!(diff(None, &merged), merged);
        assert_eq!(merge(json!(1), json!({"x": 1})), json!({"x": 1}));
    }
}
