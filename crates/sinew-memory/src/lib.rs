//! Long-term memory for Sinew agents: facts, preferences and episodes that
//! survive across threads, on top of any [`Store`](sinew_graph::Store)
//! (LangMem / Mem0 style).
//!
//! - [`MemoryManager`] keeps one namespace per user (`["memories", user]` by
//!   default) and offers [`remember`](MemoryManager::remember) (extract
//!   memories from a conversation with any `ChatModel`, then add, update or
//!   delete against the most similar existing ones),
//!   [`recall`](MemoryManager::recall) (semantic search with
//!   [`Embeddings`](sinew_rag::Embeddings), keyword overlap without),
//!   [`forget`](MemoryManager::forget), [`add`](MemoryManager::add),
//!   [`list`](MemoryManager::list) and [`context`](MemoryManager::context)
//!   (recalled memories formatted for a system prompt).
//! - [`MemoryAgent`] wraps a prebuilt
//!   [`ReactAgent`](sinew_graph::prebuilt::ReactAgent): it recalls before
//!   each turn and remembers after it, in the background by default.
//! - `MemoryHook` (feature `harness`) does the same for the
//!   `sinew-harness` agent.
//!
//! Each [`Memory`] is stored as a JSON item keyed by its id, so any backend
//! works: `InMemoryStore`, `SqliteStore` (`sinew-checkpoint`), or your own.
//!
//! ```
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! use std::sync::Arc;
//! use serde_json::json;
//! use sinew_core::{mock::ScriptedModel, Message, ToolCall};
//! use sinew_graph::InMemoryStore;
//! use sinew_memory::{MemoryKind, MemoryManager};
//! use sinew_rag::FakeEmbeddings;
//!
//! // The extraction call answers through the `save_memories` tool.
//! let model = Arc::new(ScriptedModel::new([Message::assistant_with_tools("", vec![ToolCall {
//!     id: "1".into(),
//!     name: "save_memories".into(),
//!     arguments: json!({"memories": [
//!         {"content": "The user is vegetarian.", "kind": "preference", "importance": 0.9}
//!     ]}),
//! }])]));
//! let memory = MemoryManager::new(Arc::new(InMemoryStore::new()), model)
//!     .with_embeddings(FakeEmbeddings::new(64));
//!
//! let conversation = [Message::user("I'm vegetarian."), Message::assistant("Noted!")];
//! let report = memory.remember("ana", &conversation, Some("thread-1")).await?;
//! assert_eq!(report.added[0].kind, MemoryKind::Preference);
//!
//! let hits = memory.recall("ana", "is the user vegetarian?", 3).await?;
//! assert_eq!(hits[0].memory.content, "The user is vegetarian.");
//! # Ok::<_, sinew_memory::MemoryError>(()) }).unwrap();
//! ```
//!
//! See the [memory guide](https://github.com/roanbrasil/sinew/blob/main/docs/memory.md).
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

mod agent;
#[cfg(feature = "harness")]
mod hook;
mod manager;
mod memory;

pub use agent::{MemoryAgent, MemoryOptions, RememberScope};
#[cfg(feature = "harness")]
pub use hook::MemoryHook;
pub use manager::{
    format_memories, last_turn, last_user_text, transcript, MemoryManager, DEFAULT_CONTEXT_HEADER,
    DEFAULT_EXTRACTION_PROMPT, DEFAULT_NAMESPACE, DEFAULT_UPDATE_PROMPT,
};
pub use memory::{Memory, MemoryError, MemoryKind, RememberReport, Result, ScoredMemory};
