use std::{collections::HashMap, sync::Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// State snapshot taken after every superstep (and when a run pauses).
///
/// `tasks` is the full set of node executions that run in the next superstep;
/// it has more than one entry after a fan-out. `next` is kept for
/// compatibility: it is the first task's node, or `END` when the run
/// finished. Checkpoints written before `tasks` existed deserialize with an
/// empty `tasks`, and resuming then falls back to `next`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub thread_id: String,
    /// Supersteps completed so far in this thread.
    pub step: usize,
    pub next: String,
    pub state: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tasks: Vec<PendingTask>,
}

impl Checkpoint {
    /// A checkpoint whose next superstep runs `tasks` (empty = finished).
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
        }
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
}

/// A node execution scheduled for the next superstep. `input` is set for
/// tasks created with [`SendTo`](crate::SendTo): that branch runs on its own
/// input instead of the shared state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingTask {
    pub node: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
}

#[async_trait]
pub trait Checkpointer: Send + Sync {
    async fn put(&self, checkpoint: Checkpoint) -> Result<(), String>;
    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>, String>;
    async fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>, String>;
}

/// Process-local checkpointer. Swap for a database-backed one in production.
#[derive(Default)]
pub struct MemoryCheckpointer {
    threads: Mutex<HashMap<String, Vec<Checkpoint>>>,
}

impl MemoryCheckpointer {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Checkpointer for MemoryCheckpointer {
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
        Ok(self
            .threads
            .lock()
            .unwrap()
            .get(thread_id)
            .and_then(|v| v.last().cloned()))
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
                PendingTask {
                    node: "a".into(),
                    input: None,
                },
                PendingTask {
                    node: "b".into(),
                    input: Some(json!({"doc": 1})),
                },
            ],
            json!({}),
        );
        assert_eq!(cp.next, "a");
        let back: Checkpoint = serde_json::from_value(serde_json::to_value(&cp).unwrap()).unwrap();
        assert_eq!(back, cp);
        assert_eq!(back.next_nodes(), vec!["a".to_string(), "b".to_string()]);
    }
}
