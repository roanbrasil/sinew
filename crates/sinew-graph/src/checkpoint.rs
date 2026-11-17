use std::{collections::HashMap, sync::Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Interrupt;

/// State snapshot taken after every superstep (and when a run pauses).
///
/// `tasks` is the full set of node executions that run in the next superstep;
/// it has more than one entry after a fan-out. `next` is kept for
/// compatibility: it is the first task's node, or `END` when the run
/// finished. Checkpoints written before `tasks` existed deserialize with an
/// empty `tasks`, and resuming then falls back to `next`.
///
/// Every checkpoint written by the graph has a unique `checkpoint_id` and the
/// id of the checkpoint it follows (`parent_id`), so a thread's checkpoints
/// form a tree: replaying or forking from an older checkpoint starts a new
/// branch. Checkpoints written by subgraphs carry their `checkpoint_ns`
/// (empty for the root graph). Older stored checkpoints, without these
/// fields, still deserialize.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub thread_id: String,
    /// Supersteps completed so far in this thread.
    pub step: usize,
    pub next: String,
    pub state: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tasks: Vec<PendingTask>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub checkpoint_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Subgraph namespace (`"node:task"` segments joined by `|`), empty for
    /// the root graph.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub checkpoint_ns: String,
    /// Free-form metadata. The graph sets `source` (`input`, `loop`,
    /// `update`, `fork`, `interrupt`) and `writes` (nodes that ran).
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub metadata: Map<String, Value>,
    /// Unix time in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,
}

impl Checkpoint {
    /// A checkpoint whose next superstep runs `tasks` (empty = finished),
    /// with a fresh id.
    pub fn new(
        thread_id: impl Into<String>,
        step: usize,
        tasks: Vec<PendingTask>,
        state: Value,
    ) -> Self {
        Self {
            thread_id: thread_id.into(),
            step,
            next: tasks
                .first()
                .map_or_else(|| crate::END.to_string(), |t| t.node.clone()),
            state,
            tasks,
            checkpoint_id: new_checkpoint_id(),
            created_at: Some(now_millis()),
            ..Self::default()
        }
    }

    pub fn with_parent(mut self, parent_id: Option<String>) -> Self {
        self.parent_id = parent_id;
        self
    }

    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.checkpoint_ns = ns.into();
        self
    }

    pub fn with_metadata(mut self, key: &str, value: Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Nodes of the next superstep (empty when the run finished).
    pub fn next_nodes(&self) -> Vec<String> {
        if !self.tasks.is_empty() {
            self.tasks.iter().map(|t| t.node.clone()).collect()
        } else if self.next == crate::END {
            Vec::new()
        } else {
            vec![self.next.clone()]
        }
    }

    /// Pending dynamic interrupts, in task order.
    pub fn interrupts(&self) -> Vec<Interrupt> {
        self.tasks
            .iter()
            .flat_map(|t| t.interrupts.iter().cloned())
            .collect()
    }
}

/// A node execution scheduled for the next superstep. `input` is set for
/// tasks created with [`SendTo`](crate::SendTo): that branch runs on its own
/// input instead of the shared state.
///
/// When a superstep is interrupted, the checkpoint also records, per task,
/// the interrupts it raised, the resume values it already received, and the
/// result of tasks that completed (so they do not run again on resume).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PendingTask {
    pub node: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resume: Vec<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interrupts: Vec<Interrupt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
}

impl PendingTask {
    pub fn new(node: impl Into<String>, input: Option<Value>) -> Self {
        Self {
            node: node.into(),
            input,
            ..Self::default()
        }
    }
}

/// A fresh, time-ordered checkpoint id in UUID v7 layout: 48-bit Unix
/// milliseconds, then a per-process random seed mixed with a counter. Ids
/// sort by creation time across processes and never repeat within one.
/// Generated without a syscall for randomness on the hot path.
pub fn new_checkpoint_id() -> String {
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    };
    static SEED: OnceLock<u64> = OnceLock::new();
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seed = *SEED.get_or_init(|| {
        let b = uuid::Uuid::new_v4();
        let (hi, lo) = b.as_u64_pair();
        hi ^ lo
    });
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ms = now_millis() & 0xFFFF_FFFF_FFFF;
    let rand_a = (n & 0x0FFF) as u16;
    let rand_b = seed.wrapping_add(n.wrapping_mul(0x9E37_79B9_7F4A_7C15)) & 0x3FFF_FFFF_FFFF_FFFF;
    format!(
        "{:08x}-{:04x}-7{:03x}-{:04x}-{:012x}",
        ms >> 16,
        ms & 0xFFFF,
        rand_a,
        0x8000 | ((rand_b >> 48) as u16 & 0x3FFF),
        rand_b & 0xFFFF_FFFF_FFFF
    )
}

pub(crate) fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The durable record of one tool call (see [`Checkpointer::get_tool_entry`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ToolEntry {
    /// The call started; its outcome is unknown until `Done` is written.
    Pending {
        name: String,
        arguments: Value,
        started_ms: u64,
    },
    /// The call finished; `content` is the tool message sent to the model
    /// (errors included, as `"error: ..."`).
    Done { content: String },
}

/// Storage for checkpoints.
///
/// Only `put`, `latest` and `history` are required. The namespace- and
/// id-aware methods have default implementations on top of `history`;
/// database backends override them with indexed queries.
#[async_trait]
pub trait Checkpointer: Send + Sync {
    async fn put(&self, checkpoint: Checkpoint) -> Result<(), String>;
    /// Most recently written checkpoint of the thread's root graph.
    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>, String>;
    /// Every checkpoint of the thread (all namespaces), oldest first.
    async fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>, String>;

    /// True when this checkpointer keeps a tool-call ledger
    /// ([`get_tool_entry`](Self::get_tool_entry) /
    /// [`put_tool_entry`](Self::put_tool_entry)), which makes tool calls in
    /// prebuilt agents durable: a call that finished before a crash is not
    /// run again on resume. The default is `false` (no ledger).
    fn supports_tool_ledger(&self) -> bool {
        false
    }

    /// The ledger entry stored under `key` for `thread_id`.
    async fn get_tool_entry(
        &self,
        thread_id: &str,
        key: &str,
    ) -> Result<Option<ToolEntry>, String> {
        let _ = (thread_id, key);
        Ok(None)
    }

    /// Writes (or replaces) the ledger entry under `key` for `thread_id`.
    async fn put_tool_entry(
        &self,
        thread_id: &str,
        key: &str,
        entry: &ToolEntry,
    ) -> Result<(), String> {
        let _ = (thread_id, key, entry);
        Ok(())
    }

    /// Most recently written checkpoint of a namespace (`""` = root graph).
    async fn latest_in(&self, thread_id: &str, ns: &str) -> Result<Option<Checkpoint>, String> {
        if ns.is_empty() {
            if let Some(cp) = self.latest(thread_id).await? {
                if cp.checkpoint_ns.is_empty() {
                    return Ok(Some(cp));
                }
            }
        }
        Ok(self
            .history(thread_id)
            .await?
            .into_iter()
            .rev()
            .find(|c| c.checkpoint_ns == ns))
    }

    /// Checkpoints of one namespace, oldest first (in write order).
    async fn history_in(&self, thread_id: &str, ns: &str) -> Result<Vec<Checkpoint>, String> {
        Ok(self
            .history(thread_id)
            .await?
            .into_iter()
            .filter(|c| c.checkpoint_ns == ns)
            .collect())
    }

    /// A checkpoint by id.
    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<Option<Checkpoint>, String> {
        Ok(self
            .history(thread_id)
            .await?
            .into_iter()
            .find(|c| c.checkpoint_id == checkpoint_id))
    }
}

/// Process-local checkpointer. Swap for a database-backed one in production.
#[derive(Default)]
pub struct MemoryCheckpointer {
    threads: Mutex<HashMap<String, Vec<Checkpoint>>>,
    ledger: Mutex<HashMap<(String, String), ToolEntry>>,
}

impl MemoryCheckpointer {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Checkpointer for MemoryCheckpointer {
    fn supports_tool_ledger(&self) -> bool {
        true
    }

    async fn get_tool_entry(
        &self,
        thread_id: &str,
        key: &str,
    ) -> Result<Option<ToolEntry>, String> {
        Ok(self
            .ledger
            .lock()
            .unwrap()
            .get(&(thread_id.to_string(), key.to_string()))
            .cloned())
    }

    async fn put_tool_entry(
        &self,
        thread_id: &str,
        key: &str,
        entry: &ToolEntry,
    ) -> Result<(), String> {
        self.ledger
            .lock()
            .unwrap()
            .insert((thread_id.to_string(), key.to_string()), entry.clone());
        Ok(())
    }

    async fn put(&self, checkpoint: Checkpoint) -> Result<(), String> {
        self.threads
            .lock()
            .unwrap()
            .entry(checkpoint.thread_id.clone())
            .or_default()
            .push(checkpoint);
        Ok(())
    }

    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>, String> {
        self.latest_in(thread_id, "").await
    }

    async fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>, String> {
        Ok(self
            .threads
            .lock()
            .unwrap()
            .get(thread_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn latest_in(&self, thread_id: &str, ns: &str) -> Result<Option<Checkpoint>, String> {
        Ok(self
            .threads
            .lock()
            .unwrap()
            .get(thread_id)
            .and_then(|v| v.iter().rev().find(|c| c.checkpoint_ns == ns).cloned()))
    }

    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<Option<Checkpoint>, String> {
        Ok(self.threads.lock().unwrap().get(thread_id).and_then(|v| {
            v.iter()
                .rev()
                .find(|c| c.checkpoint_id == checkpoint_id)
                .cloned()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn old_checkpoints_still_deserialize() {
        let old = json!({"thread_id": "t", "step": 3, "next": "report", "state": {"n": 1}});
        let cp: Checkpoint = serde_json::from_value(old).unwrap();
        assert!(cp.tasks.is_empty());
        assert!(cp.checkpoint_id.is_empty());
        assert!(cp.parent_id.is_none());
        assert_eq!(cp.next_nodes(), vec!["report".to_string()]);

        let done: Checkpoint = serde_json::from_value(
            json!({"thread_id": "t", "step": 4, "next": crate::END, "state": null}),
        )
        .unwrap();
        assert!(done.next_nodes().is_empty());
    }

    #[test]
    fn new_checkpoints_carry_every_task() {
        let cp = Checkpoint::new(
            "t",
            1,
            vec![
                PendingTask::new("a", None),
                PendingTask::new("b", Some(json!({"doc": 1}))),
            ],
            json!({}),
        )
        .with_parent(Some("p".into()));
        assert_eq!(cp.next, "a");
        assert!(!cp.checkpoint_id.is_empty());
        let back: Checkpoint = serde_json::from_value(serde_json::to_value(&cp).unwrap()).unwrap();
        assert_eq!(back, cp);
        assert_eq!(back.next_nodes(), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn checkpoint_ids_are_unique_and_uuid_shaped() {
        let ids: std::collections::HashSet<String> =
            (0..10_000).map(|_| new_checkpoint_id()).collect();
        assert_eq!(ids.len(), 10_000);
        let one = ids.iter().next().unwrap();
        let parsed = uuid::Uuid::parse_str(one).unwrap();
        assert_eq!(parsed.get_version_num(), 7);
    }

    #[tokio::test]
    async fn memory_checkpointer_separates_namespaces() {
        let m = MemoryCheckpointer::new();
        let root = Checkpoint::new("t", 0, vec![], json!(1));
        let sub = Checkpoint::new("t", 0, vec![], json!(2)).with_namespace("sub:0");
        m.put(root.clone()).await.unwrap();
        m.put(sub.clone()).await.unwrap();
        assert_eq!(m.latest("t").await.unwrap(), Some(root.clone()));
        assert_eq!(m.latest_in("t", "sub:0").await.unwrap(), Some(sub.clone()));
        assert_eq!(
            m.get("t", &sub.checkpoint_id).await.unwrap(),
            Some(sub.clone())
        );
        assert_eq!(m.history_in("t", "").await.unwrap(), vec![root]);
        assert_eq!(m.history("t").await.unwrap().len(), 2);
    }
}
