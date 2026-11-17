//! Thread storage: thread records plus the checkpointer every graph writes to.
//!
//! - [`MemoryStore`]: process-local, for tests and demos.
//! - [`SqliteStore`] (feature `sqlite`, default): a single file, built on
//!   [`sinew_checkpoint::SqliteCheckpointer`].
//! - [`PostgresStore`] (feature `postgres`): shared by several replicas.
//!
//! The SQL stores add one table, `sinew_serve_threads`, next to the
//! checkpoint table, in the same database.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use serde_json::Value;
use sinew_graph::{Checkpoint, Checkpointer};

use crate::api::{Thread, ThreadQuery, ThreadStatus};

/// Name of the table the SQL stores create for thread records.
pub const THREADS_TABLE: &str = "sinew_serve_threads";

/// Storage failure.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("storage error: {0}")]
    Backend(String),
}

impl From<sinew_checkpoint::CheckpointError> for StoreError {
    fn from(e: sinew_checkpoint::CheckpointError) -> Self {
        StoreError::Backend(e.to_string())
    }
}

impl From<sqlx::Error> for StoreError {
    fn from(e: sqlx::Error) -> Self {
        StoreError::Backend(e.to_string())
    }
}

/// Where threads and their checkpoints live.
#[async_trait]
pub trait ThreadStore: Send + Sync + 'static {
    /// The checkpointer the server attaches to every registered graph.
    fn checkpointer(&self) -> Arc<dyn Checkpointer>;
    /// Inserts a thread. Returns `false` (and changes nothing) if the id exists.
    async fn create_thread(&self, thread: &Thread) -> Result<bool, StoreError>;
    async fn get_thread(&self, thread_id: &str) -> Result<Option<Thread>, StoreError>;
    /// Overwrites an existing thread record.
    async fn update_thread(&self, thread: &Thread) -> Result<(), StoreError>;
    /// Threads, most recently updated first.
    async fn list_threads(&self, query: &ThreadQuery) -> Result<Vec<Thread>, StoreError>;
    /// Deletes the record and every checkpoint. Returns whether it existed.
    async fn delete_thread(&self, thread_id: &str) -> Result<bool, StoreError>;
    /// Cheap connectivity check for readiness probes.
    async fn ping(&self) -> Result<(), StoreError>;
}

const DEFAULT_LIMIT: usize = 100;
/// Largest page a `ThreadQuery` can ask for.
pub(crate) const MAX_LIMIT: usize = 10_000;

pub(crate) fn page(q: &ThreadQuery) -> (usize, usize) {
    (
        q.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT),
        q.offset.unwrap_or(0),
    )
}

// ---------------------------------------------------------------- memory

/// Checkpointer with deletion, used by [`MemoryStore`].
#[derive(Default)]
struct MemoryCheckpoints {
    threads: Mutex<HashMap<String, Vec<Checkpoint>>>,
}

#[async_trait]
impl Checkpointer for MemoryCheckpoints {
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

/// Process-local store. Everything is lost when the process exits.
#[derive(Clone, Default)]
pub struct MemoryStore {
    checkpoints: Arc<MemoryCheckpoints>,
    threads: Arc<Mutex<HashMap<String, Thread>>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ThreadStore for MemoryStore {
    fn checkpointer(&self) -> Arc<dyn Checkpointer> {
        self.checkpoints.clone()
    }

    async fn create_thread(&self, thread: &Thread) -> Result<bool, StoreError> {
        let mut map = self.threads.lock().unwrap();
        if map.contains_key(&thread.thread_id) {
            return Ok(false);
        }
        map.insert(thread.thread_id.clone(), thread.clone());
        Ok(true)
    }

    async fn get_thread(&self, thread_id: &str) -> Result<Option<Thread>, StoreError> {
        Ok(self.threads.lock().unwrap().get(thread_id).cloned())
    }

    async fn update_thread(&self, thread: &Thread) -> Result<(), StoreError> {
        self.threads
            .lock()
            .unwrap()
            .insert(thread.thread_id.clone(), thread.clone());
        Ok(())
    }

    async fn list_threads(&self, q: &ThreadQuery) -> Result<Vec<Thread>, StoreError> {
        let (limit, offset) = page(q);
        let mut all: Vec<Thread> = self
            .threads
            .lock()
            .unwrap()
            .values()
            .filter(|t| q.assistant_id.is_none() || t.assistant_id == q.assistant_id)
            .filter(|t| q.status.is_none_or(|s| t.status == s))
            .cloned()
            .collect();
        all.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then_with(|| b.thread_id.cmp(&a.thread_id))
        });
        Ok(all.into_iter().skip(offset).take(limit).collect())
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<bool, StoreError> {
        let had_cp = self
            .checkpoints
            .threads
            .lock()
            .unwrap()
            .remove(thread_id)
            .is_some();
        Ok(self.threads.lock().unwrap().remove(thread_id).is_some() || had_cp)
    }

    async fn ping(&self) -> Result<(), StoreError> {
        Ok(())
    }
}

fn decode_metadata(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or(Value::Null)
}

// ---------------------------------------------------------------- sqlite

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteStore;

#[cfg(feature = "sqlite")]
mod sqlite {
    use super::*;
    use sinew_checkpoint::{CheckpointStore, SqliteCheckpointer};
    use sqlx::Row;

    /// SQLite-backed store (thread table + [`SqliteCheckpointer`]).
    #[derive(Clone)]
    pub struct SqliteStore {
        checkpoints: SqliteCheckpointer,
    }

    impl SqliteStore {
        /// Opens (creating if needed) `url`, e.g. `sqlite://serve.db` or `sqlite::memory:`.
        pub async fn connect(url: &str) -> Result<Self, StoreError> {
            Self::new(SqliteCheckpointer::connect(url).await?).await
        }

        /// Uses an existing checkpointer (and its pool).
        pub async fn new(checkpoints: SqliteCheckpointer) -> Result<Self, StoreError> {
            sqlx::query(&format!(
                "CREATE TABLE IF NOT EXISTS {THREADS_TABLE} (
                    thread_id TEXT PRIMARY KEY,
                    assistant_id TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL)"
            ))
            .execute(checkpoints.pool())
            .await?;
            sqlx::query(&format!(
                "CREATE INDEX IF NOT EXISTS {THREADS_TABLE}_updated ON {THREADS_TABLE} (updated_at)"
            ))
            .execute(checkpoints.pool())
            .await?;
            Ok(Self { checkpoints })
        }

        fn row(r: &sqlx::sqlite::SqliteRow) -> Result<Thread, sqlx::Error> {
            Ok(Thread {
                thread_id: r.try_get("thread_id")?,
                assistant_id: r.try_get("assistant_id")?,
                status: ThreadStatus::parse(r.try_get::<&str, _>("status")?),
                metadata: decode_metadata(r.try_get::<&str, _>("metadata")?),
                created_at: r.try_get("created_at")?,
                updated_at: r.try_get("updated_at")?,
            })
        }
    }

    #[async_trait]
    impl ThreadStore for SqliteStore {
        fn checkpointer(&self) -> Arc<dyn Checkpointer> {
            Arc::new(self.checkpoints.clone())
        }

        async fn create_thread(&self, t: &Thread) -> Result<bool, StoreError> {
            let res = sqlx::query(&format!(
                "INSERT INTO {THREADS_TABLE} (thread_id, assistant_id, status, metadata, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(thread_id) DO NOTHING"
            ))
            .bind(&t.thread_id)
            .bind(&t.assistant_id)
            .bind(t.status.as_str())
            .bind(t.metadata.to_string())
            .bind(t.created_at)
            .bind(t.updated_at)
            .execute(self.checkpoints.pool())
            .await?;
            Ok(res.rows_affected() == 1)
        }

        async fn get_thread(&self, thread_id: &str) -> Result<Option<Thread>, StoreError> {
            let row = sqlx::query(&format!(
                "SELECT * FROM {THREADS_TABLE} WHERE thread_id = ?"
            ))
            .bind(thread_id)
            .fetch_optional(self.checkpoints.pool())
            .await?;
            Ok(row.as_ref().map(Self::row).transpose()?)
        }

        async fn update_thread(&self, t: &Thread) -> Result<(), StoreError> {
            sqlx::query(&format!(
                "UPDATE {THREADS_TABLE} SET assistant_id = ?, status = ?, metadata = ?, updated_at = ?
                 WHERE thread_id = ?"
            ))
            .bind(&t.assistant_id)
            .bind(t.status.as_str())
            .bind(t.metadata.to_string())
            .bind(t.updated_at)
            .bind(&t.thread_id)
            .execute(self.checkpoints.pool())
            .await?;
            Ok(())
        }

        async fn list_threads(&self, q: &ThreadQuery) -> Result<Vec<Thread>, StoreError> {
            let (limit, offset) = page(q);
            let rows = sqlx::query(&format!(
                "SELECT * FROM {THREADS_TABLE}
                 WHERE (? IS NULL OR assistant_id = ?) AND (? IS NULL OR status = ?)
                 ORDER BY updated_at DESC, thread_id DESC LIMIT ? OFFSET ?"
            ))
            .bind(&q.assistant_id)
            .bind(&q.assistant_id)
            .bind(q.status.map(ThreadStatus::as_str))
            .bind(q.status.map(ThreadStatus::as_str))
            .bind(limit as i64)
            .bind(offset as i64)
            .fetch_all(self.checkpoints.pool())
            .await?;
            Ok(rows.iter().map(Self::row).collect::<Result<_, _>>()?)
        }

        async fn delete_thread(&self, thread_id: &str) -> Result<bool, StoreError> {
            let cps = self.checkpoints.delete_thread(thread_id).await?;
            let res = sqlx::query(&format!("DELETE FROM {THREADS_TABLE} WHERE thread_id = ?"))
                .bind(thread_id)
                .execute(self.checkpoints.pool())
                .await?;
            Ok(res.rows_affected() > 0 || cps > 0)
        }

        async fn ping(&self) -> Result<(), StoreError> {
            sqlx::query("SELECT 1")
                .execute(self.checkpoints.pool())
                .await?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------- postgres

#[cfg(feature = "postgres")]
pub use postgres::PostgresStore;

#[cfg(feature = "postgres")]
mod postgres {
    use super::*;
    use sinew_checkpoint::{CheckpointStore, PostgresCheckpointer};
    use sqlx::Row;

    /// Postgres-backed store (thread table + [`PostgresCheckpointer`]).
    #[derive(Clone)]
    pub struct PostgresStore {
        checkpoints: PostgresCheckpointer,
    }

    impl PostgresStore {
        /// Connects to `url`, e.g. `postgres://user:pass@host/db`, and creates the schema.
        pub async fn connect(url: &str) -> Result<Self, StoreError> {
            Self::new(PostgresCheckpointer::connect(url).await?).await
        }

        /// Uses an existing checkpointer (and its pool).
        pub async fn new(checkpoints: PostgresCheckpointer) -> Result<Self, StoreError> {
            sqlx::query(&format!(
                "CREATE TABLE IF NOT EXISTS {THREADS_TABLE} (
                    thread_id TEXT PRIMARY KEY,
                    assistant_id TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at BIGINT NOT NULL,
                    updated_at BIGINT NOT NULL)"
            ))
            .execute(checkpoints.pool())
            .await?;
            sqlx::query(&format!(
                "CREATE INDEX IF NOT EXISTS {THREADS_TABLE}_updated ON {THREADS_TABLE} (updated_at)"
            ))
            .execute(checkpoints.pool())
            .await?;
            Ok(Self { checkpoints })
        }

        fn row(r: &sqlx::postgres::PgRow) -> Result<Thread, sqlx::Error> {
            Ok(Thread {
                thread_id: r.try_get("thread_id")?,
                assistant_id: r.try_get("assistant_id")?,
                status: ThreadStatus::parse(r.try_get::<&str, _>("status")?),
                metadata: decode_metadata(r.try_get::<&str, _>("metadata")?),
                created_at: r.try_get("created_at")?,
                updated_at: r.try_get("updated_at")?,
            })
        }
    }

    #[async_trait]
    impl ThreadStore for PostgresStore {
        fn checkpointer(&self) -> Arc<dyn Checkpointer> {
            Arc::new(self.checkpoints.clone())
        }

        async fn create_thread(&self, t: &Thread) -> Result<bool, StoreError> {
            let res = sqlx::query(&format!(
                "INSERT INTO {THREADS_TABLE} (thread_id, assistant_id, status, metadata, created_at, updated_at)
                 VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (thread_id) DO NOTHING"
            ))
            .bind(&t.thread_id)
            .bind(&t.assistant_id)
            .bind(t.status.as_str())
            .bind(t.metadata.to_string())
            .bind(t.created_at)
            .bind(t.updated_at)
            .execute(self.checkpoints.pool())
            .await?;
            Ok(res.rows_affected() == 1)
        }

        async fn get_thread(&self, thread_id: &str) -> Result<Option<Thread>, StoreError> {
            let row = sqlx::query(&format!(
                "SELECT * FROM {THREADS_TABLE} WHERE thread_id = $1"
            ))
            .bind(thread_id)
            .fetch_optional(self.checkpoints.pool())
            .await?;
            Ok(row.as_ref().map(Self::row).transpose()?)
        }

        async fn update_thread(&self, t: &Thread) -> Result<(), StoreError> {
            sqlx::query(&format!(
                "UPDATE {THREADS_TABLE} SET assistant_id = $1, status = $2, metadata = $3, updated_at = $4
                 WHERE thread_id = $5"
            ))
            .bind(&t.assistant_id)
            .bind(t.status.as_str())
            .bind(t.metadata.to_string())
            .bind(t.updated_at)
            .bind(&t.thread_id)
            .execute(self.checkpoints.pool())
            .await?;
            Ok(())
        }

        async fn list_threads(&self, q: &ThreadQuery) -> Result<Vec<Thread>, StoreError> {
            let (limit, offset) = page(q);
            let rows = sqlx::query(&format!(
                "SELECT * FROM {THREADS_TABLE}
                 WHERE ($1::TEXT IS NULL OR assistant_id = $1) AND ($2::TEXT IS NULL OR status = $2)
                 ORDER BY updated_at DESC, thread_id DESC LIMIT $3 OFFSET $4"
            ))
            .bind(&q.assistant_id)
            .bind(q.status.map(ThreadStatus::as_str))
            .bind(limit as i64)
            .bind(offset as i64)
            .fetch_all(self.checkpoints.pool())
            .await?;
            Ok(rows.iter().map(Self::row).collect::<Result<_, _>>()?)
        }

        async fn delete_thread(&self, thread_id: &str) -> Result<bool, StoreError> {
            let cps = self.checkpoints.delete_thread(thread_id).await?;
            let res = sqlx::query(&format!("DELETE FROM {THREADS_TABLE} WHERE thread_id = $1"))
                .bind(thread_id)
                .execute(self.checkpoints.pool())
                .await?;
            Ok(res.rows_affected() > 0 || cps > 0)
        }

        async fn ping(&self) -> Result<(), StoreError> {
            sqlx::query("SELECT 1")
                .execute(self.checkpoints.pool())
                .await?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn thread(id: &str, updated: i64) -> Thread {
        Thread {
            thread_id: id.into(),
            assistant_id: Some("a".into()),
            status: ThreadStatus::Idle,
            metadata: json!({"k": id}),
            created_at: 1,
            updated_at: updated,
        }
    }

    async fn exercise(store: &dyn ThreadStore) {
        assert!(store.create_thread(&thread("t1", 10)).await.unwrap());
        assert!(!store.create_thread(&thread("t1", 99)).await.unwrap());
        assert!(store.create_thread(&thread("t2", 20)).await.unwrap());
        let got = store.get_thread("t1").await.unwrap().unwrap();
        assert_eq!(got.metadata, json!({"k": "t1"}));
        assert_eq!(got.updated_at, 10);

        let mut t = got.clone();
        t.status = ThreadStatus::Interrupted;
        t.updated_at = 30;
        store.update_thread(&t).await.unwrap();
        let list = store.list_threads(&ThreadQuery::default()).await.unwrap();
        let ids: Vec<_> = list.iter().map(|t| t.thread_id.as_str()).collect();
        assert_eq!(ids, ["t1", "t2"]);
        let only = store
            .list_threads(&ThreadQuery {
                status: Some(ThreadStatus::Interrupted),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(only.len(), 1);
        let paged = store
            .list_threads(&ThreadQuery {
                limit: Some(1),
                offset: Some(1),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(paged[0].thread_id, "t2");

        let cp = store.checkpointer();
        cp.put(Checkpoint::new("t1", 1, vec![], json!({"n": 1})))
            .await
            .unwrap();
        assert!(cp.latest("t1").await.unwrap().is_some());
        assert!(store.delete_thread("t1").await.unwrap());
        assert!(cp.latest("t1").await.unwrap().is_none());
        assert!(store.get_thread("t1").await.unwrap().is_none());
        assert!(!store.delete_thread("t1").await.unwrap());
        store.ping().await.unwrap();
    }

    #[tokio::test]
    async fn memory_store_round_trips() {
        exercise(&MemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_store_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let url = format!("sqlite://{}", dir.path().join("serve.db").display());
        exercise(&SqliteStore::connect(&url).await.unwrap()).await;
        exercise(&SqliteStore::connect("sqlite::memory:").await.unwrap()).await;
    }

    /// Needs `SINEW_SERVE_PG_URL`; skipped otherwise.
    #[cfg(feature = "postgres")]
    #[tokio::test]
    async fn postgres_store_round_trips() {
        let Ok(url) = std::env::var("SINEW_SERVE_PG_URL") else {
            eprintln!("SINEW_SERVE_PG_URL not set; skipping");
            return;
        };
        let store = PostgresStore::connect(&url).await.unwrap();
        for id in ["t1", "t2"] {
            store.delete_thread(id).await.unwrap();
        }
        exercise(&store).await;
        store.delete_thread("t2").await.unwrap();
    }
}
