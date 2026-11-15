use std::{collections::HashMap, sync::Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// State snapshot taken after a step. `next` is the node that will run on
/// resume (`END` when the run finished).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub thread_id: String,
    pub step: usize,
    pub next: String,
    pub state: Value,
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
