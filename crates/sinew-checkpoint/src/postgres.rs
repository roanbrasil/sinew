use async_trait::async_trait;
use sqlx::{postgres::PgPoolOptions, PgPool, Row};

use crate::{
    util::{decode, encode, keep_to_db, step_to_db},
    Checkpoint, CheckpointError, CheckpointStore, Checkpointer,
};

/// Advisory lock key that serializes concurrent `ensure_schema` calls
/// (`CREATE TABLE IF NOT EXISTS` is not race-free in Postgres).
const SCHEMA_LOCK: i64 = 0x5_1E7E_C4EC;

const SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS sinew_checkpoints (
        id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        thread_id  TEXT        NOT NULL,
        step       BIGINT      NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        checkpoint JSONB       NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS sinew_checkpoints_thread_id
        ON sinew_checkpoints (thread_id, id)",
    "CREATE INDEX IF NOT EXISTS sinew_checkpoints_thread_step
        ON sinew_checkpoints (thread_id, step, id)",
];

/// [`Checkpointer`] backed by Postgres, for several processes or hosts sharing
/// the same threads.
///
/// Each `put` (plus its optional retention pruning) runs in one transaction
/// holding a per-thread advisory lock, so concurrent writers to the same
/// thread commit in id order and pruning never races an insert. Checkpoints
/// are stored as `JSONB`, so they can be queried with SQL too.
///
/// The table is created unqualified, so it lands in the first schema of the
/// connection's `search_path`.
#[derive(Clone, Debug)]
pub struct PostgresCheckpointer {
    pool: PgPool,
    retention: Option<usize>,
}

impl PostgresCheckpointer {
    /// Connects to `url` (e.g. `postgres://user:pass@host/db`) and creates
    /// the schema if it is missing.
    pub async fn connect(url: &str) -> Result<Self, CheckpointError> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(url)
            .await?;
        Self::from_pool(pool).await
    }

    /// Wraps an existing pool and creates the schema if it is missing.
    pub async fn from_pool(pool: PgPool) -> Result<Self, CheckpointError> {
        let this = Self {
            pool,
            retention: None,
        };
        this.ensure_schema().await?;
        Ok(this)
    }

    /// Keeps at most `keep` checkpoints per thread, pruning the oldest ones in
    /// the same transaction as each `put`. Values below 1 are treated as 1.
    pub fn with_retention(mut self, keep: usize) -> Self {
        self.retention = Some(keep.max(1));
        self
    }

    /// The underlying connection pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Creates the table and indexes if they do not exist. Idempotent and safe
    /// to call from several processes at once.
    pub async fn ensure_schema(&self) -> Result<(), CheckpointError> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)")
            .bind(SCHEMA_LOCK)
            .execute(&mut *tx)
            .await?;
        for ddl in SCHEMA {
            sqlx::query(ddl).execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn put_checkpoint(&self, checkpoint: &Checkpoint) -> Result<(), CheckpointError> {
        let step = step_to_db(checkpoint.step)?;
        let blob = encode(checkpoint)?;
        let mut tx = self.pool.begin().await?;
        lock_thread(&mut tx, &checkpoint.thread_id).await?;
        sqlx::query(
            "INSERT INTO sinew_checkpoints (thread_id, step, checkpoint)
             VALUES ($1, $2, $3::jsonb)",
        )
        .bind(&checkpoint.thread_id)
        .bind(step)
        .bind(blob)
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
    ) -> Result<Option<Checkpoint>, CheckpointError> {
        let row = sqlx::query(
            "SELECT checkpoint::text FROM sinew_checkpoints
             WHERE thread_id = $1 ORDER BY id DESC LIMIT 1",
        )
        .bind(thread_id)
        .fetch_optional(&self.pool)
        .await?;
        row.map(|r| decode(r.try_get::<&str, _>(0)?)).transpose()
    }

    async fn thread_history(&self, thread_id: &str) -> Result<Vec<Checkpoint>, CheckpointError> {
        let rows = sqlx::query(
            "SELECT checkpoint::text FROM sinew_checkpoints
             WHERE thread_id = $1 ORDER BY step, id",
        )
        .bind(thread_id)
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|r| decode(r.try_get::<&str, _>(0)?))
            .collect()
    }
}

/// Serializes writers of one thread until the transaction ends.
async fn lock_thread(conn: &mut sqlx::PgConnection, thread_id: &str) -> Result<(), sqlx::Error> {
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
        .bind(thread_id)
        .execute(conn)
        .await?;
    Ok(())
}

async fn prune_thread(
    conn: &mut sqlx::PgConnection,
    thread_id: &str,
    keep: usize,
) -> Result<u64, CheckpointError> {
    let res = sqlx::query(
        "DELETE FROM sinew_checkpoints
         WHERE thread_id = $1 AND id NOT IN (
             SELECT id FROM sinew_checkpoints
             WHERE thread_id = $1 ORDER BY id DESC LIMIT $2
         )",
    )
    .bind(thread_id)
    .bind(keep_to_db(keep))
    .execute(conn)
    .await?;
    Ok(res.rows_affected())
}

#[async_trait]
impl Checkpointer for PostgresCheckpointer {
    async fn put(&self, checkpoint: Checkpoint) -> Result<(), String> {
        self.put_checkpoint(&checkpoint)
            .await
            .map_err(|e| e.to_string())
    }

    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>, String> {
        self.latest_checkpoint(thread_id)
            .await
            .map_err(|e| e.to_string())
    }

    async fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>, String> {
        self.thread_history(thread_id)
            .await
            .map_err(|e| e.to_string())
    }
}

#[async_trait]
impl CheckpointStore for PostgresCheckpointer {
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
        let mut tx = self.pool.begin().await?;
        lock_thread(&mut tx, thread_id).await?;
        let res = sqlx::query("DELETE FROM sinew_checkpoints WHERE thread_id = $1")
            .bind(thread_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(res.rows_affected())
    }

    async fn prune(&self, thread_id: &str, keep: usize) -> Result<u64, CheckpointError> {
        let mut tx = self.pool.begin().await?;
        lock_thread(&mut tx, thread_id).await?;
        let n = prune_thread(&mut tx, thread_id, keep).await?;
        tx.commit().await?;
        Ok(n)
    }

    async fn prune_all(&self, keep: usize) -> Result<u64, CheckpointError> {
        let res = sqlx::query(
            "DELETE FROM sinew_checkpoints WHERE id IN (
                 SELECT id FROM (
                     SELECT id, ROW_NUMBER() OVER (
                         PARTITION BY thread_id ORDER BY id DESC
                     ) AS rn
                     FROM sinew_checkpoints
                 ) ranked WHERE rn > $1
             )",
        )
        .bind(keep_to_db(keep))
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected())
    }
}
