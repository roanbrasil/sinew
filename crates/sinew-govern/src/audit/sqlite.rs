//! [`SqliteAuditStore`]: the audit log in a SQLite table.

use std::{str::FromStr, time::Duration};

use async_trait::async_trait;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous},
    Row, SqlitePool,
};

use super::{db, AuditEntry, AuditError, AuditStore};

const SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS sinew_audit_log (
        seq       INTEGER PRIMARY KEY,
        time_ms   INTEGER NOT NULL,
        kind      TEXT    NOT NULL,
        payload   TEXT    NOT NULL,
        prev_hash TEXT    NOT NULL,
        hash      TEXT    NOT NULL
    )",
    "CREATE TRIGGER IF NOT EXISTS sinew_audit_log_no_update
        BEFORE UPDATE ON sinew_audit_log
        BEGIN SELECT RAISE(ABORT, 'sinew_audit_log is append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS sinew_audit_log_no_delete
        BEFORE DELETE ON sinew_audit_log
        BEGIN SELECT RAISE(ABORT, 'sinew_audit_log is append-only'); END",
];

/// [`AuditStore`] in a SQLite table, `sinew_audit_log`.
///
/// Triggers refuse `UPDATE` and `DELETE` on the table, so a bug or a
/// careless query cannot change the log; someone who drops the triggers
/// can, and the hash chain and checkpoints then show it. The database is
/// opened in WAL mode with `synchronous = FULL`.
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
/// use std::sync::Arc;
/// use serde_json::json;
/// use sinew_govern::audit::{AuditKind, AuditLog, SqliteAuditStore, VerifyOptions};
///
/// let store = Arc::new(SqliteAuditStore::connect("sqlite::memory:").await.unwrap());
/// let audit = AuditLog::new(store.clone());
/// audit.append(AuditKind::Other, json!({"n": 1})).await.unwrap();
/// assert!(audit.verify(&VerifyOptions::new()).await.unwrap().is_ok());
///
/// // The table refuses edits.
/// let edit = sinew_govern::audit::sqlx::query("UPDATE sinew_audit_log SET payload = '{}'").execute(store.pool()).await;
/// assert!(edit.is_err());
/// # });
/// ```
#[derive(Clone, Debug)]
pub struct SqliteAuditStore {
    pool: SqlitePool,
}

impl SqliteAuditStore {
    /// Opens (or creates) the database at `url` (`sqlite://audit.db`,
    /// `sqlite::memory:`) and creates the table if it is missing.
    pub async fn connect(url: &str) -> Result<Self, AuditError> {
        let in_memory = url.contains(":memory:") || url.contains("mode=memory");
        let options = SqliteConnectOptions::from_str(url)?
            .create_if_missing(true)
            .busy_timeout(Duration::from_secs(10));
        let pool = if in_memory {
            SqlitePoolOptions::new()
                .max_connections(1)
                .min_connections(1)
                .idle_timeout(None)
                .max_lifetime(None)
                .connect_with(options)
                .await?
        } else {
            SqlitePoolOptions::new()
                .max_connections(4)
                .connect_with(
                    options
                        .journal_mode(SqliteJournalMode::Wal)
                        .synchronous(SqliteSynchronous::Full),
                )
                .await?
        };
        Self::from_pool(pool).await
    }

    /// Wraps an existing pool and creates the table if it is missing.
    pub async fn from_pool(pool: SqlitePool) -> Result<Self, AuditError> {
        let mut tx = pool.begin().await?;
        for ddl in SCHEMA {
            sqlx::query(ddl).execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(Self { pool })
    }

    /// The underlying connection pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

fn row_to_entry(row: &sqlx::sqlite::SqliteRow) -> Result<AuditEntry, AuditError> {
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
impl AuditStore for SqliteAuditStore {
    async fn append(&self, entry: &AuditEntry) -> Result<(), AuditError> {
        let res = sqlx::query(
            "INSERT INTO sinew_audit_log (seq, time_ms, kind, payload, prev_hash, hash)
             VALUES (?, ?, ?, ?, ?, ?)",
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
        let rows = sqlx::query(&format!("{SELECT} WHERE seq >= ? ORDER BY seq LIMIT ?"))
            .bind(db::int(from)?)
            .bind(i64::try_from(limit).unwrap_or(i64::MAX))
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(row_to_entry).collect()
    }
}
