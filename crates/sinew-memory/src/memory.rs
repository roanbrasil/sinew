//! Memory records, errors and reports.

use std::fmt;

use serde::{Deserialize, Serialize};
use sinew_graph::store::StoreError;

/// What a memory is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    /// Something true about the user or their world ("works as a nurse").
    #[default]
    Fact,
    /// A stable like, dislike or way of working ("prefers answers in Portuguese").
    Preference,
    /// Something that happened and may matter later ("moved to Montreal in May").
    Episode,
}

impl MemoryKind {
    /// The lowercase name used in prompts and JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryKind::Fact => "fact",
            MemoryKind::Preference => "preference",
            MemoryKind::Episode => "episode",
        }
    }
}

impl fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One long-term memory. Stored as the value of a [`Store`](sinew_graph::Store)
/// item whose key is [`id`](Self::id), under the user's namespace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    /// One self-contained sentence.
    pub content: String,
    #[serde(default)]
    pub kind: MemoryKind,
    /// Unix time in milliseconds.
    pub created_at: u64,
    /// Unix time in milliseconds; equals `created_at` until the memory is updated.
    pub updated_at: u64,
    /// Thread whose conversation produced (or last updated) this memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_thread: Option<String>,
    /// From 0 (trivial) to 1 (essential), as judged by the extraction model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f32>,
}

impl Memory {
    /// A new memory with a random id and the current time.
    pub fn new(content: impl Into<String>, kind: MemoryKind) -> Self {
        let now = now_millis();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            kind,
            created_at: now,
            updated_at: now,
            source_thread: None,
            importance: None,
        }
    }

    pub fn with_source_thread(mut self, thread_id: impl Into<String>) -> Self {
        self.source_thread = Some(thread_id.into());
        self
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = Some(importance.clamp(0.0, 1.0));
        self
    }
}

/// A recalled memory and its relevance to the query: cosine similarity with
/// embeddings, keyword overlap (0 to 1) without them, `0` for an empty query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub score: f32,
}

/// What [`MemoryManager::remember`](crate::MemoryManager::remember) changed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RememberReport {
    /// Memories that did not exist before.
    pub added: Vec<Memory>,
    /// Existing memories with new content (same id, new `updated_at`).
    pub updated: Vec<Memory>,
    /// Memories removed because the conversation contradicted or retracted them.
    pub deleted: Vec<Memory>,
}

impl RememberReport {
    /// Whether nothing changed.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.updated.is_empty() && self.deleted.is_empty()
    }
}

/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// Reading or writing the [`Store`](sinew_graph::Store) failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// The extraction or update model call failed (including invalid
    /// structured output after retries).
    #[error(transparent)]
    Model(#[from] sinew_core::Error),
    /// The [`Embeddings`](sinew_rag::Embeddings) call failed.
    #[error("embedding failed: {0}")]
    Embeddings(#[from] sinew_rag::Error),
    /// A stored value is not a [`Memory`].
    #[error("stored memory `{key}` is invalid: {source}")]
    InvalidRecord {
        key: String,
        source: serde_json::Error,
    },
    /// The agent graph failed.
    #[error(transparent)]
    Graph(#[from] sinew_graph::GraphError),
    /// A background save task panicked or was cancelled.
    #[error("background memory task failed: {0}")]
    Task(String),
}

pub type Result<T> = std::result::Result<T, MemoryError>;

pub(crate) fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}
