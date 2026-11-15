//! Task persistence. The server keeps running tasks in memory and writes
//! snapshots to a [`TaskStore`] when a task is created, when its status
//! changes and when it finishes (not on every streamed token).

use std::{collections::HashMap, sync::RwLock};

use async_trait::async_trait;

use crate::types::Task;

/// Where finished and paused tasks live. Implement it over a database to
/// keep tasks across restarts; [`InMemoryTaskStore`] is the default.
#[async_trait]
pub trait TaskStore: Send + Sync {
    async fn get(&self, task_id: &str) -> Result<Option<Task>, String>;
    async fn save(&self, task: &Task) -> Result<(), String>;
    async fn delete(&self, task_id: &str) -> Result<(), String>;
}

/// Process-local task store.
#[derive(Default)]
pub struct InMemoryTaskStore {
    tasks: RwLock<HashMap<String, Task>>,
}

impl InMemoryTaskStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored tasks.
    pub fn len(&self) -> usize {
        self.tasks.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl TaskStore for InMemoryTaskStore {
    async fn get(&self, task_id: &str) -> Result<Option<Task>, String> {
        Ok(self.tasks.read().unwrap().get(task_id).cloned())
    }

    async fn save(&self, task: &Task) -> Result<(), String> {
        self.tasks
            .write()
            .unwrap()
            .insert(task.id.clone(), task.clone());
        Ok(())
    }

    async fn delete(&self, task_id: &str) -> Result<(), String> {
        self.tasks.write().unwrap().remove(task_id);
        Ok(())
    }
}
