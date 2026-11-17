use std::{str::FromStr, time::Duration};

use async_trait::async_trait;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous},
    Row, SqlitePool,
};

use crate::{
    util::{decode_row, encode, id_to_db, keep_to_db, legacy_row_id, now_millis, step_to_db},
    Checkpoint, CheckpointError, CheckpointStore, Checkpointer,
};

const SCHEMA: &[&str] = &[
    // Durable tool calls (see `Checkpointer::get_tool_entry`).
    "CREATE TABLE IF NOT EXISTS sinew_tool_ledger (
        thread_id  TEXT    NOT NULL,
        key        TEXT    NOT NULL,
        entry      TEXT    NOT NULL,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY (thread_id, key)
    )",
    "CREATE TABLE IF NOT EXISTS sinew_checkpoints (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id  TEXT    NOT NULL,
        step       INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        checkpoint TEXT    NOT NULL,
        checkpoint_id TEXT,
        checkpoint_ns TEXT NOT NULL DEFAULT ''
    )",
    "CREATE INDEX IF NOT EXISTS sinew_checkpoints_thread_id
        ON sinew_checkpoints (thread_id, id)",
    "CREATE INDEX IF NOT EXISTS sinew_checkpoints_thread_step
        ON sinew_checkpoints (thread_id, step, id)",
];

/// Columns added after the first release. Tables created by older versions
/// get them through `ALTER TABLE` (existing rows: no id, root namespace).
const MIGRATIONS: &[(&str, &str)] = &[
    (
        "checkpoint_id",
        "ALTER TABLE sinew_checkpoints ADD COLUMN checkpoint_id TEXT",
    ),
    (
        "checkpoint_ns",
        "ALTER TABLE sinew_checkpoints ADD COLUMN checkpoint_ns TEXT NOT NULL DEFAULT ''",
    ),
];

/// Indexes over the migrated columns (created after the migrations ran).
const LATE_INDEXES: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS sinew_checkpoints_thread_ns
        ON sinew_checkpoints (thread_id, checkpoint_ns, id)",
    "CREATE INDEX IF NOT EXISTS sinew_checkpoints_checkpoint_id
        ON sinew_checkpoints (thread_id, checkpoint_id)",
];

/// [`Checkpointer`] backed by SQLite (a file, or memory for tests).
///
/// File databases are opened in WAL mode with a busy timeout, so several
/// tasks (or processes) can write concurrently. Each `put` (plus its optional
/// retention pruning) runs in one transaction.
///
/// `latest` returns the most recently written root checkpoint of a thread;
/// `history` returns all of them (every namespace) ordered by step, then
/// insertion order; `history_in` returns one namespace in write order.
#[derive(Clone, Debug)]
pub struct SqliteCheckpointer {
    pool: SqlitePool,
    retention: Option<usize>,
}

impl SqliteCheckpointer {
    /// Opens (creating it if needed) the database at `url`, e.g.
    /// `sqlite://agent.db` or `sqlite::memory:`, and creates the schema.
    ///
    /// In-memory URLs use a single pooled connection that is never recycled,
    /// so the database lives as long as this checkpointer (and its clones).
    pub async fn connect(url: &str) -> Result<Self, CheckpointError> {
        Self::from_pool(open_pool(url).await?).await
    }

    /// A fresh private in-memory database (handy for tests).
    pub async fn in_memory() -> Result<Self, CheckpointError> {
        Self::connect("sqlite::memory:").await
    }

    /// Wraps an existing pool and creates the schema if it is missing.
    pub async fn from_pool(pool: SqlitePool) -> Result<Self, CheckpointError> {
        let this = Self {
            pool,
            retention: None,
        };
        this.ensure_schema().await?;
        Ok(this)
    }

    /// Keeps at most `keep` checkpoints per thread, pruning the oldest ones in
    /// the same transaction as each `put`. Values below 1 are treated as 1, so
    /// the latest checkpoint is always kept.
    pub fn with_retention(mut self, keep: usize) -> Self {
        self.retention = Some(keep.max(1));
        self
    }

    /// The underlying connection pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Creates the table and indexes if they do not exist, and adds the
    /// columns newer versions need (`checkpoint_id`, `checkpoint_ns`) to
    /// tables created by older ones. Idempotent.
    pub async fn ensure_schema(&self) -> Result<(), CheckpointError> {
        let mut tx = self.pool.begin().await?;
        for ddl in SCHEMA {
            sqlx::query(ddl).execute(&mut *tx).await?;
        }
        tx.commit().await?;
        let columns: Vec<String> =
            sqlx::query("SELECT name FROM pragma_table_info('sinew_checkpoints')")
                .fetch_all(&self.pool)
                .await?
                .iter()
                .map(|r| r.try_get::<String, _>(0))
                .collect::<Result<_, _>>()?;
        for (column, ddl) in MIGRATIONS {
            if columns.iter().any(|c| c == column) {
                continue;
            }
            match sqlx::query(ddl).execute(&self.pool).await {
                Ok(_) => {}
                // Another process migrated first.
                Err(e) if e.to_string().contains("duplicate column") => {}
                Err(e) => return Err(e.into()),
            }
        }
        for ddl in LATE_INDEXES {
            sqlx::query(ddl).execute(&self.pool).await?;
        }
        Ok(())
    }

    async fn put_checkpoint(&self, checkpoint: &Checkpoint) -> Result<(), CheckpointError> {
        let step = step_to_db(checkpoint.step)?;
        let blob = encode(checkpoint)?;
        // The transaction starts with a write, so it takes the write lock
        // right away and never has to upgrade a read lock (which could fail
        // with SQLITE_BUSY under concurrent writers).
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            "INSERT INTO sinew_checkpoints
                 (thread_id, step, created_at, checkpoint, checkpoint_id, checkpoint_ns)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&checkpoint.thread_id)
        .bind(step)
        .bind(now_millis())
        .bind(blob)
        .bind(id_to_db(checkpoint))
        .bind(&checkpoint.checkpoint_ns)
        .execute(&mut *tx)
        .await?;
        if let Some(keep) = self.retention {
            prune_thread(&mut tx, &checkpoint.thread_id, keep).await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn latest_checkpoint(
        &self,
        thread_id: &str,
        ns: &str,
    ) -> Result<Option<Checkpoint>, CheckpointError> {
        let row = sqlx::query(
            "SELECT id, checkpoint FROM sinew_checkpoints
             WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY id DESC LIMIT 1",
        )
        .bind(thread_id)
        .bind(ns)
        .fetch_optional(&self.pool)
        .await?;
        row.map(|r| decode_row(r.try_get(0)?, r.try_get::<&str, _>(1)?))
            .transpose()
    }

    async fn thread_history(&self, thread_id: &str) -> Result<Vec<Checkpoint>, CheckpointError> {
        let rows = sqlx::query(
            "SELECT id, checkpoint FROM sinew_checkpoints
             WHERE thread_id = ? ORDER BY step, id",
        )
        .bind(thread_id)
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|r| decode_row(r.try_get(0)?, r.try_get::<&str, _>(1)?))
            .collect()
    }

    async fn ns_history(
        &self,
        thread_id: &str,
        ns: &str,
    ) -> Result<Vec<Checkpoint>, CheckpointError> {
        let rows = sqlx::query(
            "SELECT id, checkpoint FROM sinew_checkpoints
             WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY id",
        )
        .bind(thread_id)
        .bind(ns)
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|r| decode_row(r.try_get(0)?, r.try_get::<&str, _>(1)?))
            .collect()
    }

    async fn by_id(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<Option<Checkpoint>, CheckpointError> {
        let row = match legacy_row_id(checkpoint_id) {
            Some(row_id) => {
                sqlx::query(
                    "SELECT id, checkpoint FROM sinew_checkpoints
                     WHERE thread_id = ? AND id = ?",
                )
                .bind(thread_id)
                .bind(row_id)
                .fetch_optional(&self.pool)
                .await?
            }
            None => {
                sqlx::query(
                    "SELECT id, checkpoint FROM sinew_checkpoints
                     WHERE thread_id = ? AND checkpoint_id = ? ORDER BY id DESC LIMIT 1",
                )
                .bind(thread_id)
                .bind(checkpoint_id)
                .fetch_optional(&self.pool)
                .await?
            }
        };
        row.map(|r| decode_row(r.try_get(0)?, r.try_get::<&str, _>(1)?))
            .transpose()
    }
}

/// Opens a pool for `url`: WAL + busy timeout for files, one long-lived
/// connection for in-memory databases.
pub(crate) async fn open_pool(url: &str) -> Result<SqlitePool, CheckpointError> {
    let in_memory = url.contains(":memory:") || url.contains("mode=memory");
    let options = SqliteConnectOptions::from_str(url)?
        .create_if_missing(true)
        .busy_timeout(Duration::from_secs(10));
    let (options, pool) = if in_memory {
        (
            options,
            SqlitePoolOptions::new()
                .max_connections(1)
                .min_connections(1)
                .idle_timeout(None)
                .max_lifetime(None),
        )
    } else {
        (
            options
                .journal_mode(SqliteJournalMode::Wal)
                .synchronous(SqliteSynchronous::Normal),
            SqlitePoolOptions::new().max_connections(8),
        )
    };
    Ok(pool.connect_with(options).await?)
}

async fn prune_thread(
    conn: &mut sqlx::SqliteConnection,
    thread_id: &str,
    keep: usize,
) -> Result<u64, CheckpointError> {
    let res = sqlx::query(
        "DELETE FROM sinew_checkpoints
         WHERE thread_id = ?1 AND id NOT IN (
             SELECT id FROM sinew_checkpoints
             WHERE thread_id = ?1 ORDER BY id DESC LIMIT ?2
         )",
    )
    .bind(thread_id)
    .bind(keep_to_db(keep))
    .execute(conn)
    .await?;
    Ok(res.rows_affected())
}

#[async_trait]
impl Checkpointer for SqliteCheckpointer {
    fn supports_tool_ledger(&self) -> bool {
        true
    }

    async fn get_tool_entry(
        &self,
        thread_id: &str,
        key: &str,
    ) -> Result<Option<sinew_graph::ToolEntry>, String> {
        let row =
            sqlx::query("SELECT entry FROM sinew_tool_ledger WHERE thread_id = ? AND key = ?")
                .bind(thread_id)
                .bind(key)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| e.to_string())?;
        row.map(|r| {
            let text: String = r.try_get(0).map_err(|e| e.to_string())?;
            serde_json::from_str(&text).map_err(|e| e.to_string())
        })
        .transpose()
    }

    async fn put_tool_entry(
        &self,
        thread_id: &str,
        key: &str,
        entry: &sinew_graph::ToolEntry,
    ) -> Result<(), String> {
        let text = serde_json::to_string(entry).map_err(|e| e.to_string())?;
        sqlx::query(
            "INSERT INTO sinew_tool_ledger (thread_id, key, entry, updated_at) VALUES (?, ?, ?, ?)
             ON CONFLICT (thread_id, key) DO UPDATE SET entry = excluded.entry, updated_at = excluded.updated_at",
        )
        .bind(thread_id)
        .bind(key)
        .bind(text)
        .bind(now_millis())
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn put(&self, checkpoint: Checkpoint) -> Result<(), String> {
        self.put_checkpoint(&checkpoint)
            .await
            .map_err(|e| e.to_string())
    }

    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>, String> {
        self.latest_checkpoint(thread_id, "")
            .await
            .map_err(|e| e.to_string())
    }

    async fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>, String> {
        self.thread_history(thread_id)
            .await
            .map_err(|e| e.to_string())
    }

    async fn latest_in(&self, thread_id: &str, ns: &str) -> Result<Option<Checkpoint>, String> {
        self.latest_checkpoint(thread_id, ns)
            .await
            .map_err(|e| e.to_string())
    }

    async fn history_in(&self, thread_id: &str, ns: &str) -> Result<Vec<Checkpoint>, String> {
        self.ns_history(thread_id, ns)
            .await
            .map_err(|e| e.to_string())
    }

    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<Option<Checkpoint>, String> {
        self.by_id(thread_id, checkpoint_id)
            .await
            .map_err(|e| e.to_string())
    }
}

#[async_trait]
impl CheckpointStore for SqliteCheckpointer {
    async fn list_threads(&self) -> Result<Vec<String>, CheckpointError> {
        let rows = sqlx::query(
            "SELECT thread_id FROM sinew_checkpoints
             GROUP BY thread_id ORDER BY MAX(id) DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|r| Ok(r.try_get::<String, _>(0)?))
            .collect()
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<u64, CheckpointError> {
        let res = sqlx::query("DELETE FROM sinew_checkpoints WHERE thread_id = ?")
            .bind(thread_id)
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM sinew_tool_ledger WHERE thread_id = ?")
            .bind(thread_id)
            .execute(&self.pool)
            .await?;
        Ok(res.rows_affected())
    }

    async fn prune(&self, thread_id: &str, keep: usize) -> Result<u64, CheckpointError> {
        let mut conn = self.pool.acquire().await?;
        prune_thread(&mut conn, thread_id, keep).await
    }

    async fn prune_all(&self, keep: usize) -> Result<u64, CheckpointError> {
        let res = sqlx::query(
            "DELETE FROM sinew_checkpoints WHERE id IN (
                 SELECT id FROM (
                     SELECT id, ROW_NUMBER() OVER (
                         PARTITION BY thread_id ORDER BY id DESC
                     ) AS rn
                     FROM sinew_checkpoints
                 ) WHERE rn > ?
             )",
        )
        .bind(keep_to_db(keep))
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected())
    }
}
