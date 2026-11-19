//! [`PostgresAuditStore`]: the audit log in a Postgres table.

use async_trait::async_trait;
use sqlx::{postgres::PgPoolOptions, PgPool, Row};

use super::{db, AuditEntry, AuditError, AuditStore};

/// Serializes concurrent `ensure_schema` calls.
const SCHEMA_LOCK: i64 = 0x5_1E7E_A0D1;

const SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS sinew_audit_log (
        seq       BIGINT PRIMARY KEY,
        time_ms   BIGINT NOT NULL,
        kind      TEXT   NOT NULL,
        payload   TEXT   NOT NULL,
        prev_hash TEXT   NOT NULL,
        hash      TEXT   NOT NULL
    )",
    "CREATE OR REPLACE FUNCTION sinew_audit_log_append_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'sinew_audit_log is append-only'; END $$",
    "DROP TRIGGER IF EXISTS sinew_audit_log_append_only ON sinew_audit_log",
    "CREATE TRIGGER sinew_audit_log_append_only
        BEFORE UPDATE OR DELETE OR TRUNCATE ON sinew_audit_log
        FOR EACH STATEMENT EXECUTE FUNCTION sinew_audit_log_append_only()",
];

/// [`AuditStore`] in a Postgres table, `sinew_audit_log`, for several
/// processes writing one log (a second writer appending the same `seq`
/// gets a conflict, and [`AuditLog`](super::AuditLog) retries after the new
/// head).
///
/// A statement trigger refuses `UPDATE`, `DELETE` and `TRUNCATE`; grant the
/// application role only `INSERT` and `SELECT` on the table as well. The
/// payload is stored as `TEXT` (not `JSONB`) so it reads back exactly as
/// written. The table is created unqualified, in the first schema of the
/// connection's `search_path`.
#[derive(Clone, Debug)]
pub struct PostgresAuditStore {
    pool: PgPool,
}

impl PostgresAuditStore {
    /// Connects to `url` (`postgres://user:pass@host/db`) and creates the
    /// table if it is missing.
    pub async fn connect(url: &str) -> Result<Self, AuditError> {
        let pool = PgPoolOptions::new().max_connections(5).connect(url).await?;
        Self::from_pool(pool).await
    }

    /// Wraps an existing pool and creates the table if it is missing.
    pub async fn from_pool(pool: PgPool) -> Result<Self, AuditError> {
        let mut tx = pool.begin().await?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)")
            .bind(SCHEMA_LOCK)
            .execute(&mut *tx)
            .await?;
        for ddl in SCHEMA {
            sqlx::query(ddl).execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(Self { pool })
    }

    /// The underlying connection pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

fn row_to_entry(row: &sqlx::postgres::PgRow) -> Result<AuditEntry, AuditError> {
    db::entry(
        row.try_get("seq")?,
        row.try_get("time_ms")?,
        row.try_get("kind")?,
        row.try_get("payload")?,
        row.try_get("prev_hash")?,
        row.try_get("hash")?,
    )
}

const SELECT: &str = "SELECT seq, time_ms, kind, payload, prev_hash, hash FROM sinew_audit_log";

#[async_trait]
impl AuditStore for PostgresAuditStore {
    async fn append(&self, entry: &AuditEntry) -> Result<(), AuditError> {
        let res = sqlx::query(
            "INSERT INTO sinew_audit_log (seq, time_ms, kind, payload, prev_hash, hash)
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(db::int(entry.seq)?)
        .bind(db::int(entry.time_ms)?)
        .bind(entry.kind.as_str())
        .bind(entry.payload.to_string())
        .bind(&entry.prev_hash)
        .bind(&entry.hash)
        .execute(&self.pool)
        .await;
        db::insert_result(res, entry.seq)
    }

    async fn last(&self) -> Result<Option<AuditEntry>, AuditError> {
        let row = sqlx::query(&format!("{SELECT} ORDER BY seq DESC LIMIT 1"))
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(row_to_entry).transpose()
    }

    async fn entries(&self, from: u64, limit: usize) -> Result<Vec<AuditEntry>, AuditError> {
        let rows = sqlx::query(&format!("{SELECT} WHERE seq >= $1 ORDER BY seq LIMIT $2"))
            .bind(db::int(from)?)
            .bind(i64::try_from(limit).unwrap_or(i64::MAX))
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(row_to_entry).collect()
    }
}
