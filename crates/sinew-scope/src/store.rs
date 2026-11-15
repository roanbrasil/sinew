//! SQLite storage for runs, trace summaries, datasets, experiments, prompts,
//! feedback, annotation queues, online-evaluation rules and alerts.
//!
//! Runs are upserted by id. After each ingest batch the summary row of every
//! touched trace (root run, latency, tokens, cost, error count) is
//! recomputed, so trace listing and filtering are single-table queries.
//!
//! The schema version lives in `PRAGMA user_version`; [`Store::open`]
//! migrates older databases forward (see [`SCHEMA_VERSION`]).

mod admin;
mod feedback;
mod monitor;
mod prompts;

pub use admin::{
    rule_matches, Alert, AlertInput, RetentionResult, Rule, RuleInput, RunFilter, Settings,
    ALERT_METRICS,
};
pub use feedback::{FeedbackInput, FeedbackRow, Queue, QueueItem, QueueSummary};
pub use monitor::{GroupStats, MetricsQuery, MetricsReport, RunPage, RunQuery, RunRow, TimeBucket};
pub use prompts::{DiffLine, PromptDetail, PromptDiff, PromptPushInput, PromptSummary};

use std::{
    collections::BTreeSet,
    str::FromStr,
    sync::{Arc, RwLock},
};

use crate::pricing::{run_cost, run_model, run_ttft, Pricing};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinew_eval::{Dataset, Example, ExperimentReport};
use sinew_trace::{Run, RunKind};
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteRow},
    QueryBuilder, Row, Sqlite, SqliteConnection, SqlitePool,
};

pub type Result<T> = std::result::Result<T, StoreError>;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),
    #[error("invalid data: {0}")]
    Invalid(String),
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    trace_id      TEXT NOT NULL,
    parent_id     TEXT,
    project       TEXT NOT NULL,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL,
    start_ms      INTEGER NOT NULL,
    end_ms        INTEGER,
    inputs        TEXT NOT NULL,
    outputs       TEXT,
    error         TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    metadata      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_trace ON runs(trace_id, start_ms);
CREATE INDEX IF NOT EXISTS runs_kind ON runs(kind, trace_id);

CREATE TABLE IF NOT EXISTS traces (
    trace_id      TEXT PRIMARY KEY,
    project       TEXT NOT NULL,
    root_id       TEXT NOT NULL,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL,
    start_ms      INTEGER NOT NULL,
    end_ms        INTEGER,
    latency_ms    INTEGER,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    run_count     INTEGER NOT NULL,
    error_count   INTEGER NOT NULL,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS traces_project_start ON traces(project, start_ms);
CREATE INDEX IF NOT EXISTS traces_start ON traces(start_ms);

CREATE TABLE IF NOT EXISTS datasets (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    created_ms  INTEGER NOT NULL,
    updated_ms  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS examples (
    dataset   TEXT NOT NULL,
    idx       INTEGER NOT NULL,
    id        TEXT NOT NULL,
    inputs    TEXT NOT NULL,
    reference TEXT,
    metadata  TEXT NOT NULL,
    PRIMARY KEY (dataset, idx)
);
CREATE TABLE IF NOT EXISTS experiments (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    dataset    TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    aggregates TEXT NOT NULL,
    report     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS experiments_dataset ON experiments(dataset, created_ms);
"#;

/// Current schema version (`PRAGMA user_version`).
///
/// - 1 (or 0): runs, traces, datasets, examples, experiments.
/// - 2: `runs.model`, `runs.cost_usd`, `runs.ttft_ms`, `traces.cost_usd`;
///   settings, prompts, feedback, annotation queues, rules, alerts, pairwise
///   tables; search and monitoring indexes. Costs are backfilled.
/// - 3: `runs_fts` full-text index (FTS5) over run name, inputs, outputs and
///   error, kept in sync by triggers. Skipped when SQLite lacks FTS5 (search
///   then falls back to `LIKE`).
pub const SCHEMA_VERSION: i64 = 3;

const SCHEMA_V2: &str = r#"
CREATE INDEX IF NOT EXISTS runs_start ON runs(start_ms);
CREATE INDEX IF NOT EXISTS runs_project_start ON runs(project, start_ms);
CREATE INDEX IF NOT EXISTS runs_model_start ON runs(model, start_ms);
CREATE INDEX IF NOT EXISTS runs_name_start ON runs(name, start_ms);
CREATE INDEX IF NOT EXISTS runs_parent ON runs(parent_id);
CREATE INDEX IF NOT EXISTS traces_name ON traces(name, start_ms);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompts (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    created_ms  INTEGER NOT NULL,
    updated_ms  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS prompt_versions (
    name            TEXT NOT NULL,
    commit_hash     TEXT NOT NULL,
    version         INTEGER NOT NULL,
    template        TEXT NOT NULL,
    messages        TEXT,
    input_variables TEXT NOT NULL,
    model_config    TEXT NOT NULL,
    metadata        TEXT NOT NULL,
    message         TEXT,
    parent_commit   TEXT,
    created_ms      INTEGER NOT NULL,
    PRIMARY KEY (name, commit_hash),
    UNIQUE (name, version)
);
CREATE TABLE IF NOT EXISTS prompt_tags (
    name        TEXT NOT NULL,
    tag         TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    updated_ms  INTEGER NOT NULL,
    PRIMARY KEY (name, tag)
);

CREATE TABLE IF NOT EXISTS feedback (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL,
    trace_id   TEXT,
    key        TEXT NOT NULL,
    score      REAL,
    value      TEXT,
    comment    TEXT,
    source     TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_run ON feedback(run_id, created_ms);
CREATE INDEX IF NOT EXISTS feedback_trace ON feedback(trace_id);
CREATE INDEX IF NOT EXISTS feedback_key ON feedback(key, created_ms);

CREATE TABLE IF NOT EXISTS annotation_queues (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL,
    keys        TEXT NOT NULL,
    created_ms  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS queue_items (
    queue_id TEXT NOT NULL,
    run_id   TEXT NOT NULL,
    status   TEXT NOT NULL,
    added_ms INTEGER NOT NULL,
    done_ms  INTEGER,
    PRIMARY KEY (queue_id, run_id)
);
CREATE INDEX IF NOT EXISTS queue_items_status ON queue_items(queue_id, status, added_ms);

CREATE TABLE IF NOT EXISTS rules (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    enabled        INTEGER NOT NULL,
    filter         TEXT NOT NULL,
    sample_rate    REAL NOT NULL,
    evaluators     TEXT NOT NULL,
    created_ms     INTEGER NOT NULL,
    updated_ms     INTEGER NOT NULL,
    runs_evaluated INTEGER NOT NULL DEFAULT 0,
    last_run_ms    INTEGER,
    last_error     TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    enabled         INTEGER NOT NULL,
    metric          TEXT NOT NULL,
    threshold       REAL NOT NULL,
    window_ms       INTEGER NOT NULL,
    filter          TEXT NOT NULL,
    webhook_url     TEXT NOT NULL,
    cooldown_ms     INTEGER NOT NULL,
    created_ms      INTEGER NOT NULL,
    last_checked_ms INTEGER,
    last_value      REAL,
    last_fired_ms   INTEGER,
    last_status     TEXT,
    last_error      TEXT
);

CREATE TABLE IF NOT EXISTS pairwise (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    dataset    TEXT NOT NULL,
    a_id       TEXT NOT NULL,
    b_id       TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    summary    TEXT NOT NULL,
    report     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS pairwise_dataset ON pairwise(dataset, created_ms);
"#;

const SCHEMA_FTS: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS runs_fts USING fts5(
    name, inputs, outputs, error, content='runs', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS runs_fts_ai AFTER INSERT ON runs BEGIN
    INSERT INTO runs_fts(rowid, name, inputs, outputs, error)
    VALUES (new.rowid, new.name, new.inputs, new.outputs, new.error);
END;
CREATE TRIGGER IF NOT EXISTS runs_fts_ad AFTER DELETE ON runs BEGIN
    INSERT INTO runs_fts(runs_fts, rowid, name, inputs, outputs, error)
    VALUES ('delete', old.rowid, old.name, old.inputs, old.outputs, old.error);
END;
CREATE TRIGGER IF NOT EXISTS runs_fts_au AFTER UPDATE ON runs BEGIN
    INSERT INTO runs_fts(runs_fts, rowid, name, inputs, outputs, error)
    VALUES ('delete', old.rowid, old.name, old.inputs, old.outputs, old.error);
    INSERT INTO runs_fts(rowid, name, inputs, outputs, error)
    VALUES (new.rowid, new.name, new.inputs, new.outputs, new.error);
END;
INSERT INTO runs_fts(runs_fts) VALUES ('rebuild');
"#;

/// A run plus the project it was filed under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredRun {
    pub project: String,
    #[serde(flatten)]
    pub run: Run,
    /// Model used (LLM runs), from metadata or OTLP attributes.
    #[serde(default)]
    pub model: Option<String>,
    /// Cost in USD from the pricing table (or `metadata.cost_usd`).
    #[serde(default)]
    pub cost_usd: Option<f64>,
    /// Time to first token in ms (streamed LLM runs).
    #[serde(default)]
    pub ttft_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceSummary {
    pub trace_id: String,
    pub project: String,
    pub root_id: String,
    pub name: String,
    pub kind: String,
    pub start_ms: i64,
    pub end_ms: Option<i64>,
    pub latency_ms: Option<i64>,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub run_count: i64,
    pub error_count: i64,
    /// Root error, or the first error in the trace.
    pub error: Option<String>,
    /// Sum of the run costs, USD.
    #[serde(default)]
    pub cost_usd: f64,
}

/// Filters and pagination for [`Store::list_traces`]. All fields optional.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TraceQuery {
    pub project: Option<String>,
    /// Substring of the root run name.
    pub name: Option<String>,
    /// Keeps traces that contain at least one run of this kind.
    pub kind: Option<String>,
    /// Only traces with at least one failed run.
    pub errors: bool,
    /// Trace start, inclusive, in Unix milliseconds.
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
    /// Full-text search over the inputs, outputs, names and errors of every
    /// run of the trace.
    pub q: Option<String>,
    /// The root run's `metadata.tags` contains this tag.
    pub tag: Option<String>,
    /// Root run metadata equality filters (`meta.<key>=<value>` in the API).
    #[serde(skip)]
    pub metadata: Vec<(String, String)>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TraceStats {
    pub trace_count: i64,
    pub error_count: i64,
    pub error_rate: f64,
    pub run_count: i64,
    pub avg_latency_ms: Option<f64>,
    pub p50_latency_ms: Option<i64>,
    pub p95_latency_ms: Option<i64>,
    pub max_latency_ms: Option<i64>,
    pub input_tokens: i64,
    pub output_tokens: i64,
    #[serde(default)]
    pub p99_latency_ms: Option<i64>,
    #[serde(default)]
    pub cost_usd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TracePage {
    pub traces: Vec<TraceSummary>,
    /// Number of traces matching the filters (ignores limit/offset).
    pub total: i64,
    pub limit: i64,
    pub offset: i64,
    /// Aggregates over every matching trace.
    pub stats: TraceStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceDetail {
    pub summary: TraceSummary,
    /// Every run of the trace, ordered by start time.
    pub runs: Vec<StoredRun>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectSummary {
    pub project: String,
    pub trace_count: i64,
    pub last_start_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub name: String,
    pub description: String,
    pub created_ms: i64,
    pub updated_ms: i64,
    pub example_count: i64,
    pub experiment_count: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub id: String,
    pub name: String,
    pub dataset: String,
    pub created_ms: i64,
    pub aggregates: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestResult {
    pub accepted: usize,
    pub traces: usize,
}

pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn kind_str(kind: RunKind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "chain".into())
}

fn parse_kind(s: &str) -> RunKind {
    serde_json::from_value(Value::String(s.to_string())).unwrap_or(RunKind::Chain)
}

pub(crate) fn to_json(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "null".into())
}

pub(crate) fn from_json(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or(Value::Null)
}

pub(crate) fn row_to_run(row: &SqliteRow) -> Result<StoredRun> {
    let opt_u32 = |col: &str| -> Result<Option<u32>> {
        Ok(row
            .try_get::<Option<i64>, _>(col)?
            .map(|v| v.clamp(0, u32::MAX as i64) as u32))
    };
    Ok(StoredRun {
        project: row.try_get("project")?,
        run: Run {
            id: row.try_get("id")?,
            trace_id: row.try_get("trace_id")?,
            parent_id: row.try_get("parent_id")?,
            name: row.try_get("name")?,
            kind: parse_kind(&row.try_get::<String, _>("kind")?),
            start_ms: row.try_get::<i64, _>("start_ms")?.max(0) as u64,
            end_ms: row
                .try_get::<Option<i64>, _>("end_ms")?
                .map(|v| v.max(0) as u64),
            inputs: from_json(&row.try_get::<String, _>("inputs")?),
            outputs: row
                .try_get::<Option<String>, _>("outputs")?
                .map(|s| from_json(&s)),
            error: row.try_get("error")?,
            input_tokens: opt_u32("input_tokens")?,
            output_tokens: opt_u32("output_tokens")?,
            metadata: from_json(&row.try_get::<String, _>("metadata")?),
        },
        model: row.try_get("model")?,
        cost_usd: row.try_get("cost_usd")?,
        ttft_ms: row.try_get("ttft_ms")?,
    })
}

/// Like [`row_to_run`] without the derived columns (works on any `runs` row).
fn row_to_run_basic(row: &SqliteRow, project: String) -> Result<StoredRun> {
    let opt_u32 = |col: &str| -> Result<Option<u32>> {
        Ok(row
            .try_get::<Option<i64>, _>(col)?
            .map(|v| v.clamp(0, u32::MAX as i64) as u32))
    };
    Ok(StoredRun {
        project,
        run: Run {
            id: row.try_get("id")?,
            trace_id: row.try_get("trace_id")?,
            parent_id: row.try_get("parent_id")?,
            name: row.try_get("name")?,
            kind: parse_kind(&row.try_get::<String, _>("kind")?),
            start_ms: row.try_get::<i64, _>("start_ms")?.max(0) as u64,
            end_ms: row
                .try_get::<Option<i64>, _>("end_ms")?
                .map(|v| v.max(0) as u64),
            inputs: from_json(&row.try_get::<String, _>("inputs")?),
            outputs: row
                .try_get::<Option<String>, _>("outputs")?
                .map(|s| from_json(&s)),
            error: row.try_get("error")?,
            input_tokens: opt_u32("input_tokens")?,
            output_tokens: opt_u32("output_tokens")?,
            metadata: from_json(&row.try_get::<String, _>("metadata")?),
        },
        model: None,
        cost_usd: None,
        ttft_ms: None,
    })
}

/// `ALTER TABLE ADD COLUMN` unless the column exists (idempotent migrations).
async fn add_column(
    tx: &mut SqliteConnection,
    table: &str,
    column: &str,
    decl: &str,
) -> Result<()> {
    let exists: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM pragma_table_info(?) WHERE name = ?")
            .bind(table)
            .bind(column)
            .fetch_one(&mut *tx)
            .await?;
    if exists == 0 {
        sqlx::raw_sql(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"))
            .execute(&mut *tx)
            .await?;
    }
    Ok(())
}

fn row_to_summary(row: &SqliteRow) -> Result<TraceSummary> {
    Ok(TraceSummary {
        trace_id: row.try_get("trace_id")?,
        project: row.try_get("project")?,
        root_id: row.try_get("root_id")?,
        name: row.try_get("name")?,
        kind: row.try_get("kind")?,
        start_ms: row.try_get("start_ms")?,
        end_ms: row.try_get("end_ms")?,
        latency_ms: row.try_get("latency_ms")?,
        input_tokens: row.try_get("input_tokens")?,
        output_tokens: row.try_get("output_tokens")?,
        run_count: row.try_get("run_count")?,
        error_count: row.try_get("error_count")?,
        error: row.try_get("error")?,
        cost_usd: row.try_get::<Option<f64>, _>("cost_usd")?.unwrap_or(0.0),
    })
}

/// Escapes `%`, `_` and `\` for a `LIKE ... ESCAPE '\'` substring pattern.
pub(crate) fn like_pattern(s: &str) -> String {
    format!(
        "%{}%",
        s.replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_")
    )
}

/// Turns free text into an FTS5 query: every word must appear (prefix match
/// on each), quotes are escaped so user input can never break the syntax.
pub(crate) fn fts_query(text: &str) -> Option<String> {
    let terms: Vec<String> = text
        .split_whitespace()
        .map(|t| format!("\"{}\"*", t.replace('"', "\"\"")))
        .collect();
    (!terms.is_empty()).then(|| terms.join(" "))
}

/// SQLite JSON path for a top-level metadata key (quoted, so dots are fine).
pub(crate) fn json_key_path(key: &str) -> String {
    format!("$.\"{}\"", key.replace('"', ""))
}

/// Normalizes a metadata filter value for comparison with
/// `CAST(json_extract(..) AS TEXT)` (JSON booleans come back as 1/0).
pub(crate) fn meta_value(v: &str) -> String {
    match v {
        "true" => "1".into(),
        "false" => "0".into(),
        _ => v.to_string(),
    }
}

/// Builds `<prefix> WHERE ...` with the trace filters bound as parameters.
fn filtered<'a>(prefix: &str, q: &TraceQuery, fts: bool) -> QueryBuilder<'a, Sqlite> {
    let mut qb = QueryBuilder::new(prefix);
    qb.push(" WHERE 1 = 1");
    if let Some(p) = q.project.as_ref().filter(|p| !p.is_empty()) {
        qb.push(" AND project = ").push_bind(p.clone());
    }
    if let Some(n) = q.name.as_ref().filter(|n| !n.is_empty()) {
        qb.push(" AND name LIKE ")
            .push_bind(like_pattern(n))
            .push(" ESCAPE '\\'");
    }
    if let Some(t) = q.tag.as_ref().filter(|t| !t.is_empty()) {
        qb.push(
            " AND EXISTS (SELECT 1 FROM runs rr, json_each(rr.metadata, '$.tags') jt \
             WHERE rr.id = traces.root_id AND jt.value = ",
        )
        .push_bind(t.clone())
        .push(")");
    }
    for (k, v) in &q.metadata {
        qb.push(" AND EXISTS (SELECT 1 FROM runs rr WHERE rr.id = traces.root_id AND CAST(json_extract(rr.metadata, ")
            .push_bind(json_key_path(k))
            .push(") AS TEXT) = ")
            .push_bind(meta_value(v))
            .push(")");
    }
    if let Some(text) = q.q.as_ref().filter(|t| !t.trim().is_empty()) {
        match fts_query(text).filter(|_| fts) {
            Some(m) => {
                qb.push(
                    " AND trace_id IN (SELECT r.trace_id FROM runs r WHERE r.rowid IN \
                     (SELECT rowid FROM runs_fts WHERE runs_fts MATCH ",
                )
                .push_bind(m)
                .push("))");
            }
            None => {
                let pat = like_pattern(text.trim());
                qb.push(" AND trace_id IN (SELECT r.trace_id FROM runs r WHERE r.inputs LIKE ")
                    .push_bind(pat.clone())
                    .push(" ESCAPE '\\' OR r.outputs LIKE ")
                    .push_bind(pat.clone())
                    .push(" ESCAPE '\\' OR r.name LIKE ")
                    .push_bind(pat.clone())
                    .push(" ESCAPE '\\' OR r.error LIKE ")
                    .push_bind(pat)
                    .push(" ESCAPE '\\')");
            }
        }
    }
    if let Some(k) = q.kind.as_ref().filter(|k| !k.is_empty()) {
        qb.push(
            " AND EXISTS (SELECT 1 FROM runs r WHERE r.trace_id = traces.trace_id AND r.kind = ",
        )
        .push_bind(k.to_lowercase())
        .push(")");
    }
    if q.errors {
        qb.push(" AND error_count > 0");
    }
    if let Some(f) = q.from_ms {
        qb.push(" AND start_ms >= ").push_bind(f);
    }
    if let Some(t) = q.to_ms {
        qb.push(" AND start_ms <= ").push_bind(t);
    }
    qb
}

/// Handle to the database. Cheap to clone.
#[derive(Clone)]
pub struct Store {
    pool: SqlitePool,
    pricing: Arc<RwLock<Pricing>>,
    fts: bool,
}

impl Store {
    /// Opens (or creates) a SQLite database file.
    pub async fn open(path: &str) -> Result<Self> {
        let opts = SqliteConnectOptions::from_str(path)
            .or_else(|_| SqliteConnectOptions::from_str(&format!("sqlite://{path}")))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(std::time::Duration::from_secs(5));
        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(opts)
            .await?;
        Self::init(pool).await
    }

    /// A private in-memory database (tests, examples).
    pub async fn memory() -> Result<Self> {
        let opts = SqliteConnectOptions::from_str("sqlite::memory:")?;
        // One connection: every in-memory connection is its own database.
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .idle_timeout(None)
            .max_lifetime(None)
            .connect_with(opts)
            .await?;
        Self::init(pool).await
    }

    async fn init(pool: SqlitePool) -> Result<Self> {
        sqlx::raw_sql(SCHEMA).execute(&pool).await?;
        let version: i64 = sqlx::query_scalar("PRAGMA user_version")
            .fetch_one(&pool)
            .await?;
        if version < 2 {
            let mut tx = pool.begin().await?;
            add_column(&mut tx, "runs", "model", "TEXT").await?;
            add_column(&mut tx, "runs", "cost_usd", "REAL").await?;
            add_column(&mut tx, "runs", "ttft_ms", "INTEGER").await?;
            add_column(&mut tx, "traces", "cost_usd", "REAL NOT NULL DEFAULT 0").await?;
            sqlx::raw_sql(SCHEMA_V2).execute(&mut *tx).await?;
            sqlx::raw_sql("PRAGMA user_version = 2")
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
        }
        let mut store = Self {
            pool,
            pricing: Arc::new(RwLock::new(Pricing::builtin())),
            fts: false,
        };
        if let Some(p) = store.stored_pricing().await? {
            store.set_pricing_in_memory(p);
        }
        if version < 2 {
            // Backfill model, cost and ttft of runs stored before v2.
            store.recompute_costs().await?;
        }
        if version < 3 {
            let mut tx = store.pool.begin().await?;
            if sqlx::raw_sql(SCHEMA_FTS).execute(&mut *tx).await.is_ok() {
                sqlx::raw_sql("PRAGMA user_version = 3")
                    .execute(&mut *tx)
                    .await?;
                tx.commit().await?;
            } else {
                tx.rollback().await?;
            }
        }
        store.fts = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'runs_fts'",
        )
        .fetch_one(&store.pool)
        .await?
            > 0;
        Ok(store)
    }

    /// Whether full-text search uses the FTS5 index (else `LIKE`).
    pub fn has_fts(&self) -> bool {
        self.fts
    }

    /// The schema version of the open database.
    pub async fn schema_version(&self) -> Result<i64> {
        Ok(sqlx::query_scalar("PRAGMA user_version")
            .fetch_one(&self.pool)
            .await?)
    }

    pub(crate) fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// The pricing table in use.
    pub fn pricing(&self) -> Pricing {
        self.pricing.read().map(|p| p.clone()).unwrap_or_default()
    }

    fn set_pricing_in_memory(&self, p: Pricing) {
        if let Ok(mut w) = self.pricing.write() {
            *w = p;
        }
    }

    /// Replaces the pricing table (persisted in `settings`) and recomputes
    /// the cost of every stored run. Returns how many runs were updated.
    pub async fn set_pricing(&self, p: Pricing) -> Result<u64> {
        p.validate().map_err(StoreError::Invalid)?;
        let text = serde_json::to_string(&p).map_err(|e| StoreError::Invalid(e.to_string()))?;
        sqlx::query(
            "INSERT INTO settings (key, value) VALUES ('pricing', ?) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        )
        .bind(text)
        .execute(&self.pool)
        .await?;
        self.set_pricing_in_memory(p);
        self.recompute_costs().await
    }

    /// Restores the built-in pricing table.
    pub async fn reset_pricing(&self) -> Result<u64> {
        sqlx::query("DELETE FROM settings WHERE key = 'pricing'")
            .execute(&self.pool)
            .await?;
        self.set_pricing_in_memory(Pricing::builtin());
        self.recompute_costs().await
    }

    async fn stored_pricing(&self) -> Result<Option<Pricing>> {
        let text: Option<String> =
            sqlx::query_scalar("SELECT value FROM settings WHERE key = 'pricing'")
                .fetch_optional(&self.pool)
                .await?;
        Ok(text.and_then(|t| serde_json::from_str(&t).ok()))
    }

    /// Recomputes model, cost and time to first token of every run, then the
    /// trace cost totals. Returns how many runs have a cost.
    pub async fn recompute_costs(&self) -> Result<u64> {
        let pricing = self.pricing();
        let rows = sqlx::query("SELECT * FROM runs")
            .fetch_all(&self.pool)
            .await?;
        let mut tx = self.pool.begin().await?;
        let mut priced = 0u64;
        for row in &rows {
            let project: String = row.try_get("project")?;
            let mut run = row_to_run_basic(row, project)?;
            let model = run_model(&run.run);
            let cost = run_cost(&pricing, &run.run, model.as_deref());
            priced += u64::from(cost.is_some());
            run.model = model;
            sqlx::query("UPDATE runs SET model = ?, cost_usd = ?, ttft_ms = ? WHERE id = ?")
                .bind(&run.model)
                .bind(cost)
                .bind(run_ttft(&run.run))
                .bind(&run.run.id)
                .execute(&mut *tx)
                .await?;
        }
        sqlx::query(
            "UPDATE traces SET cost_usd = COALESCE((SELECT SUM(cost_usd) FROM runs \
             WHERE runs.trace_id = traces.trace_id), 0)",
        )
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(priced)
    }

    /// Upserts runs (by id), each filed under its project, and refreshes the
    /// summaries of the traces they belong to.
    pub async fn ingest(&self, runs: &[(String, Run)]) -> Result<IngestResult> {
        let pricing = self.pricing();
        let mut tx = self.pool.begin().await?;
        let mut traces = BTreeSet::new();
        for (project, r) in runs {
            if r.id.is_empty() || r.trace_id.is_empty() {
                return Err(StoreError::Invalid(
                    "run id and trace_id are required".into(),
                ));
            }
            let model = run_model(r);
            let cost = run_cost(&pricing, r, model.as_deref());
            sqlx::query(
                "INSERT INTO runs (id, trace_id, parent_id, project, name, kind, start_ms, end_ms, \
                 inputs, outputs, error, input_tokens, output_tokens, metadata, model, cost_usd, \
                 ttft_ms) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
                 ON CONFLICT(id) DO UPDATE SET trace_id = excluded.trace_id, \
                 parent_id = excluded.parent_id, project = excluded.project, name = excluded.name, \
                 kind = excluded.kind, start_ms = excluded.start_ms, end_ms = excluded.end_ms, \
                 inputs = excluded.inputs, outputs = excluded.outputs, error = excluded.error, \
                 input_tokens = excluded.input_tokens, output_tokens = excluded.output_tokens, \
                 metadata = excluded.metadata, model = excluded.model, \
                 cost_usd = excluded.cost_usd, ttft_ms = excluded.ttft_ms",
            )
            .bind(&r.id)
            .bind(&r.trace_id)
            .bind(r.parent_id.as_deref().filter(|p| !p.is_empty()))
            .bind(project)
            .bind(&r.name)
            .bind(kind_str(r.kind))
            .bind(r.start_ms as i64)
            .bind(r.end_ms.map(|v| v as i64))
            .bind(to_json(&r.inputs))
            .bind(r.outputs.as_ref().map(to_json))
            .bind(&r.error)
            .bind(r.input_tokens.map(i64::from))
            .bind(r.output_tokens.map(i64::from))
            .bind(to_json(&r.metadata))
            .bind(&model)
            .bind(cost)
            .bind(run_ttft(r))
            .execute(&mut *tx)
            .await?;
            traces.insert(r.trace_id.clone());
        }
        for t in &traces {
            refresh_trace(&mut tx, t).await?;
        }
        tx.commit().await?;
        Ok(IngestResult {
            accepted: runs.len(),
            traces: traces.len(),
        })
    }

    pub async fn list_traces(&self, q: &TraceQuery) -> Result<TracePage> {
        let limit = q.limit.unwrap_or(50).clamp(1, 1000);
        let offset = q.offset.unwrap_or(0).max(0);

        let fts = self.fts;
        let mut qb = filtered("SELECT * FROM traces", q, fts);
        qb.push(" ORDER BY start_ms DESC, trace_id LIMIT ")
            .push_bind(limit)
            .push(" OFFSET ")
            .push_bind(offset);
        let traces = qb
            .build()
            .fetch_all(&self.pool)
            .await?
            .iter()
            .map(row_to_summary)
            .collect::<Result<Vec<_>>>()?;

        let row = filtered(
            "SELECT COUNT(*) AS n, \
             SUM(CASE WHEN error_count > 0 THEN 1 ELSE 0 END) AS errs, \
             SUM(run_count) AS runs, AVG(latency_ms) AS avg_lat, MAX(latency_ms) AS max_lat, \
             COUNT(latency_ms) AS n_lat, SUM(input_tokens) AS itok, SUM(output_tokens) AS otok, \
             SUM(cost_usd) AS cost \
             FROM traces",
            q,
            fts,
        )
        .build()
        .fetch_one(&self.pool)
        .await?;
        let total: i64 = row.try_get("n")?;
        let errs: i64 = row.try_get::<Option<i64>, _>("errs")?.unwrap_or(0);
        let n_lat: i64 = row.try_get("n_lat")?;

        let percentile = |p: f64| {
            let q = q.clone();
            let pool = self.pool.clone();
            async move {
                if n_lat == 0 {
                    return Ok::<_, StoreError>(None);
                }
                let k = ((n_lat - 1) as f64 * p).round() as i64;
                let mut qb = filtered("SELECT latency_ms FROM traces", &q, fts);
                qb.push(" AND latency_ms IS NOT NULL ORDER BY latency_ms LIMIT 1 OFFSET ")
                    .push_bind(k);
                let r = qb.build().fetch_optional(&pool).await?;
                Ok(match r {
                    Some(r) => r.try_get::<Option<i64>, _>(0)?,
                    None => None,
                })
            }
        };

        let stats = TraceStats {
            trace_count: total,
            error_count: errs,
            error_rate: if total > 0 {
                errs as f64 / total as f64
            } else {
                0.0
            },
            run_count: row.try_get::<Option<i64>, _>("runs")?.unwrap_or(0),
            avg_latency_ms: row.try_get("avg_lat")?,
            p50_latency_ms: percentile(0.5).await?,
            p95_latency_ms: percentile(0.95).await?,
            max_latency_ms: row.try_get("max_lat")?,
            input_tokens: row.try_get::<Option<i64>, _>("itok")?.unwrap_or(0),
            output_tokens: row.try_get::<Option<i64>, _>("otok")?.unwrap_or(0),
            p99_latency_ms: percentile(0.99).await?,
            cost_usd: row.try_get::<Option<f64>, _>("cost")?.unwrap_or(0.0),
        };
        Ok(TracePage {
            traces,
            total,
            limit,
            offset,
            stats,
        })
    }

    pub async fn trace(&self, trace_id: &str) -> Result<Option<TraceDetail>> {
        let Some(row) = sqlx::query("SELECT * FROM traces WHERE trace_id = ?")
            .bind(trace_id)
            .fetch_optional(&self.pool)
            .await?
        else {
            return Ok(None);
        };
        let summary = row_to_summary(&row)?;
        let runs = sqlx::query("SELECT * FROM runs WHERE trace_id = ? ORDER BY start_ms, rowid")
            .bind(trace_id)
            .fetch_all(&self.pool)
            .await?
            .iter()
            .map(row_to_run)
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(TraceDetail { summary, runs }))
    }

    pub async fn run(&self, id: &str) -> Result<Option<StoredRun>> {
        sqlx::query("SELECT * FROM runs WHERE id = ?")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?
            .as_ref()
            .map(row_to_run)
            .transpose()
    }

    pub async fn projects(&self) -> Result<Vec<ProjectSummary>> {
        let rows = sqlx::query(
            "SELECT project, COUNT(*) AS n, MAX(start_ms) AS last FROM traces \
             GROUP BY project ORDER BY last DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|r| {
                Ok(ProjectSummary {
                    project: r.try_get("project")?,
                    trace_count: r.try_get("n")?,
                    last_start_ms: r.try_get("last")?,
                })
            })
            .collect()
    }

    /// Creates or replaces a dataset and all of its examples.
    pub async fn put_dataset(&self, ds: &Dataset) -> Result<()> {
        if ds.name.trim().is_empty() {
            return Err(StoreError::Invalid("dataset name is required".into()));
        }
        let mut tx = self.pool.begin().await?;
        put_dataset_tx(&mut tx, ds).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn datasets(&self) -> Result<Vec<DatasetSummary>> {
        let rows = sqlx::query(
            "SELECT d.name, d.description, d.created_ms, d.updated_ms, \
             (SELECT COUNT(*) FROM examples e WHERE e.dataset = d.name) AS examples, \
             (SELECT COUNT(*) FROM experiments x WHERE x.dataset = d.name) AS experiments \
             FROM datasets d ORDER BY d.updated_ms DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|r| {
                Ok(DatasetSummary {
                    name: r.try_get("name")?,
                    description: r.try_get("description")?,
                    created_ms: r.try_get("created_ms")?,
                    updated_ms: r.try_get("updated_ms")?,
                    example_count: r.try_get("examples")?,
                    experiment_count: r.try_get("experiments")?,
                })
            })
            .collect()
    }

    pub async fn dataset(&self, name: &str) -> Result<Option<Dataset>> {
        let Some(row) = sqlx::query("SELECT description FROM datasets WHERE name = ?")
            .bind(name)
            .fetch_optional(&self.pool)
            .await?
        else {
            return Ok(None);
        };
        let examples = sqlx::query("SELECT * FROM examples WHERE dataset = ? ORDER BY idx")
            .bind(name)
            .fetch_all(&self.pool)
            .await?
            .iter()
            .map(|r| {
                Ok(Example {
                    id: r.try_get("id")?,
                    inputs: from_json(&r.try_get::<String, _>("inputs")?),
                    reference: r
                        .try_get::<Option<String>, _>("reference")?
                        .map(|s| from_json(&s)),
                    metadata: from_json(&r.try_get::<String, _>("metadata")?),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Dataset {
            name: name.to_string(),
            description: row.try_get("description")?,
            examples,
        }))
    }

    /// Stores an experiment report (upsert on its id). If its dataset is
    /// unknown, the dataset is created from the report's examples.
    pub async fn put_experiment(&self, r: &ExperimentReport) -> Result<()> {
        if r.id.is_empty() || r.dataset.is_empty() {
            return Err(StoreError::Invalid(
                "experiment id and dataset are required".into(),
            ));
        }
        let mut tx = self.pool.begin().await?;
        let known: Option<i64> = sqlx::query_scalar("SELECT 1 FROM datasets WHERE name = ?")
            .bind(&r.dataset)
            .fetch_optional(&mut *tx)
            .await?;
        if known.is_none() {
            let ds = Dataset {
                name: r.dataset.clone(),
                description: "created from an uploaded experiment".into(),
                examples: r
                    .results
                    .iter()
                    .map(|x| Example {
                        id: x.example_id.clone(),
                        inputs: x.inputs.clone(),
                        reference: x.reference.clone(),
                        metadata: Value::Null,
                    })
                    .collect(),
            };
            put_dataset_tx(&mut tx, &ds).await?;
        }
        let aggregates = serde_json::to_string(&r.aggregates).unwrap_or_else(|_| "{}".into());
        let report = serde_json::to_string(r).map_err(|e| StoreError::Invalid(e.to_string()))?;
        sqlx::query(
            "INSERT INTO experiments (id, name, dataset, created_ms, aggregates, report) \
             VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name, \
             dataset = excluded.dataset, created_ms = excluded.created_ms, \
             aggregates = excluded.aggregates, report = excluded.report",
        )
        .bind(&r.id)
        .bind(&r.name)
        .bind(&r.dataset)
        .bind(r.created_ms as i64)
        .bind(aggregates)
        .bind(report)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn experiments(&self, dataset: Option<&str>) -> Result<Vec<ExperimentSummary>> {
        let mut qb: QueryBuilder<'_, Sqlite> =
            QueryBuilder::new("SELECT id, name, dataset, created_ms, aggregates FROM experiments");
        if let Some(d) = dataset.filter(|d| !d.is_empty()) {
            qb.push(" WHERE dataset = ").push_bind(d.to_string());
        }
        qb.push(" ORDER BY created_ms DESC");
        let rows = qb.build().fetch_all(&self.pool).await?;
        rows.iter()
            .map(|r| {
                Ok(ExperimentSummary {
                    id: r.try_get("id")?,
                    name: r.try_get("name")?,
                    dataset: r.try_get("dataset")?,
                    created_ms: r.try_get("created_ms")?,
                    aggregates: from_json(&r.try_get::<String, _>("aggregates")?),
                })
            })
            .collect()
    }

    pub async fn experiment(&self, id: &str) -> Result<Option<ExperimentReport>> {
        let Some(text): Option<String> =
            sqlx::query_scalar("SELECT report FROM experiments WHERE id = ?")
                .bind(id)
                .fetch_optional(&self.pool)
                .await?
        else {
            return Ok(None);
        };
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| StoreError::Invalid(e.to_string()))
    }
}

pub(crate) async fn put_dataset_tx(tx: &mut SqliteConnection, ds: &Dataset) -> Result<()> {
    let now = now_ms();
    sqlx::query(
        "INSERT INTO datasets (name, description, created_ms, updated_ms) VALUES (?, ?, ?, ?) \
         ON CONFLICT(name) DO UPDATE SET description = excluded.description, \
         updated_ms = excluded.updated_ms",
    )
    .bind(&ds.name)
    .bind(&ds.description)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await?;
    sqlx::query("DELETE FROM examples WHERE dataset = ?")
        .bind(&ds.name)
        .execute(&mut *tx)
        .await?;
    for (i, ex) in ds.examples.iter().enumerate() {
        sqlx::query(
            "INSERT INTO examples (dataset, idx, id, inputs, reference, metadata) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&ds.name)
        .bind(i as i64)
        .bind(&ex.id)
        .bind(to_json(&ex.inputs))
        .bind(ex.reference.as_ref().map(to_json))
        .bind(to_json(&ex.metadata))
        .execute(&mut *tx)
        .await?;
    }
    Ok(())
}

/// Recomputes the summary row of one trace from its runs.
async fn refresh_trace(tx: &mut SqliteConnection, trace_id: &str) -> Result<()> {
    let agg = sqlx::query(
        "SELECT MIN(start_ms) AS s, MAX(end_ms) AS e, \
         SUM(COALESCE(input_tokens, 0)) AS itok, SUM(COALESCE(output_tokens, 0)) AS otok, \
         COUNT(*) AS n, SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errs, \
         COALESCE(SUM(cost_usd), 0.0) AS cost \
         FROM runs WHERE trace_id = ?",
    )
    .bind(trace_id)
    .fetch_one(&mut *tx)
    .await?;
    // Root: a run without parent, else one whose parent is not (yet) known,
    // else the earliest run.
    let Some(root) = sqlx::query(
        "SELECT id, project, name, kind, error FROM runs r WHERE trace_id = ? ORDER BY \
         CASE WHEN parent_id IS NULL THEN 0 \
              WHEN parent_id NOT IN (SELECT id FROM runs WHERE trace_id = r.trace_id) THEN 1 \
              ELSE 2 END, start_ms LIMIT 1",
    )
    .bind(trace_id)
    .fetch_optional(&mut *tx)
    .await?
    else {
        return Ok(());
    };
    let first_error: Option<String> = sqlx::query_scalar(
        "SELECT error FROM runs WHERE trace_id = ? AND error IS NOT NULL ORDER BY start_ms LIMIT 1",
    )
    .bind(trace_id)
    .fetch_optional(&mut *tx)
    .await?;
    let start: i64 = agg.try_get("s")?;
    let end: Option<i64> = agg.try_get("e")?;
    let root_error: Option<String> = root.try_get("error")?;
    sqlx::query(
        "INSERT INTO traces (trace_id, project, root_id, name, kind, start_ms, end_ms, latency_ms, \
         input_tokens, output_tokens, run_count, error_count, error, cost_usd) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
         ON CONFLICT(trace_id) DO UPDATE SET project = excluded.project, root_id = excluded.root_id, \
         name = excluded.name, kind = excluded.kind, start_ms = excluded.start_ms, \
         end_ms = excluded.end_ms, latency_ms = excluded.latency_ms, \
         input_tokens = excluded.input_tokens, output_tokens = excluded.output_tokens, \
         run_count = excluded.run_count, error_count = excluded.error_count, error = excluded.error, \
         cost_usd = excluded.cost_usd",
    )
    .bind(trace_id)
    .bind(root.try_get::<String, _>("project")?)
    .bind(root.try_get::<String, _>("id")?)
    .bind(root.try_get::<String, _>("name")?)
    .bind(root.try_get::<String, _>("kind")?)
    .bind(start)
    .bind(end)
    .bind(end.map(|e| (e - start).max(0)))
    .bind(agg.try_get::<i64, _>("itok")?)
    .bind(agg.try_get::<i64, _>("otok")?)
    .bind(agg.try_get::<i64, _>("n")?)
    .bind(agg.try_get::<i64, _>("errs")?)
    .bind(root_error.or(first_error))
    .bind(agg.try_get::<f64, _>("cost")?)
    .execute(&mut *tx)
    .await?;
    Ok(())
}
