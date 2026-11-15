//! Persistent [`Checkpointer`] implementations for `sinew-graph`.
//!
//! - [`SqliteCheckpointer`] (feature `sqlite`, on by default): a single file,
//!   or an in-memory database for tests.
//! - [`PostgresCheckpointer`] (feature `postgres`): shared storage for several
//!   processes or hosts.
//!
//! Both store every [`Checkpoint`] as an opaque JSON blob, next to indexed
//! columns for the thread id, the step and the creation time. New fields added
//! to `Checkpoint` are persisted without a schema change.
//!
//! Besides the [`Checkpointer`] trait, both implement [`CheckpointStore`] for
//! housekeeping: listing and deleting threads, and pruning old checkpoints.
//!
//! ```no_run
//! # #[cfg(feature = "sqlite")]
//! # async fn demo() -> Result<(), sinew_checkpoint::CheckpointError> {
//! use sinew_checkpoint::{CheckpointStore, SqliteCheckpointer};
//!
//! // Keep at most 50 checkpoints per thread.
//! let store = SqliteCheckpointer::connect("sqlite://agent.db")
//!     .await?
//!     .with_retention(50);
//! println!("threads: {:?}", store.list_threads().await?);
//! // graph.with_checkpointer(store) ...
//! # Ok(()) }
//! ```
//!
//! [`Checkpoint`]: sinew_graph::Checkpoint

use async_trait::async_trait;
pub use sinew_graph::{Checkpoint, Checkpointer};
/// Re-exported so callers can build pools for `from_pool` without a version mismatch.
pub use sqlx;

#[cfg(feature = "postgres")]
mod postgres;
#[cfg(feature = "sqlite")]
mod sqlite;

#[cfg(feature = "postgres")]
pub use postgres::PostgresCheckpointer;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteCheckpointer;

/// Name of the table both backends create.
pub const TABLE: &str = "sinew_checkpoints";

/// Errors returned by the persistent checkpointers.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("checkpoint (de)serialization failed: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("step {0} does not fit in a signed 64-bit column")]
    StepOverflow(usize),
    #[error("stored step {0} is negative")]
    NegativeStep(i64),
}

/// Housekeeping operations shared by the persistent checkpointers.
#[async_trait]
pub trait CheckpointStore: Checkpointer {
    /// All thread ids, most recently written first.
    async fn list_threads(&self) -> Result<Vec<String>, CheckpointError>;

    /// Deletes every checkpoint of a thread. Returns the number of rows removed.
    async fn delete_thread(&self, thread_id: &str) -> Result<u64, CheckpointError>;

    /// Keeps only the `keep` most recently written checkpoints of a thread
    /// (`keep = 0` empties it). Returns the number of rows removed.
    async fn prune(&self, thread_id: &str, keep: usize) -> Result<u64, CheckpointError>;

    /// Applies [`prune`](CheckpointStore::prune) to every thread at once.
    async fn prune_all(&self, keep: usize) -> Result<u64, CheckpointError>;
}

/// Internal helpers shared by the backends.
#[cfg(any(feature = "sqlite", feature = "postgres"))]
mod util {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{Checkpoint, CheckpointError};

    pub(crate) fn step_to_db(step: usize) -> Result<i64, CheckpointError> {
        i64::try_from(step).map_err(|_| CheckpointError::StepOverflow(step))
    }

    pub(crate) fn keep_to_db(keep: usize) -> i64 {
        i64::try_from(keep).unwrap_or(i64::MAX)
    }

    #[cfg_attr(not(feature = "sqlite"), allow(dead_code))]
    pub(crate) fn now_millis() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0)
    }

    pub(crate) fn encode(checkpoint: &Checkpoint) -> Result<String, CheckpointError> {
        Ok(serde_json::to_string(checkpoint)?)
    }

    pub(crate) fn decode(blob: &str) -> Result<Checkpoint, CheckpointError> {
        Ok(serde_json::from_str(blob)?)
    }
}
