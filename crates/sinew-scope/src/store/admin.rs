//! Settings (retention), online-evaluation rules, alerts and data retention.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{sqlite::SqliteRow, Row};

use super::{feedback::uuid_like, now_ms, Result, Store, StoreError};

/// Server settings stored in the database (`GET/PUT /api/settings`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Traces older than this many days are deleted (with their runs and
    /// feedback) by the retention job. `None` keeps everything.
    pub retention_days: Option<u32>,
}

/// Which runs a rule or alert looks at.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunFilter {
    pub project: Option<String>,
    /// Run kind (`llm`, `tool`, ...). Rules only.
    pub kind: Option<String>,
    /// Exact run name (rules) or substring of the trace name (alerts).
    pub name: Option<String>,
    pub tag: Option<String>,
    /// Rules: only failed runs. Alerts: ignored.
    pub errors_only: bool,
    /// Rules: only root runs.
    pub root_only: bool,
    /// Metadata equality (top-level keys).
    pub metadata: BTreeMap<String, Value>,
    /// Alerts: `trace` (default) or a run kind.
    pub level: Option<String>,
}

/// An online-evaluation rule: sampled incoming runs are scored by the
/// evaluators and the scores stored as feedback (source `rule:<name>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub filter: RunFilter,
    /// Fraction of matching runs evaluated, 0..1 (deterministic per run id).
    pub sample_rate: f64,
    /// Evaluator specs, e.g. `{"type": "contains", "needle": "Paris"}` or
    /// `{"type": "llm_judge", "model": "openai:gpt-4.1-mini", "rubric": "..."}`.
    pub evaluators: Vec<Value>,
    pub created_ms: i64,
    pub updated_ms: i64,
    pub runs_evaluated: i64,
    pub last_run_ms: Option<i64>,
    pub last_error: Option<String>,
}

/// Body of `POST /api/rules` and `PUT /api/rules/{id}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RuleInput {
    pub name: String,
    pub enabled: bool,
    pub filter: RunFilter,
    pub sample_rate: f64,
    pub evaluators: Vec<Value>,
}

impl Default for RuleInput {
    fn default() -> Self {
        Self {
            name: String::new(),
            enabled: true,
            filter: RunFilter::default(),
            sample_rate: 1.0,
            evaluators: Vec::new(),
        }
    }
}

/// A threshold alert evaluated periodically over a sliding window; when
/// the metric crosses the threshold, a JSON payload is POSTed to the webhook.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    /// `error_rate`, `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`,
    /// `avg_latency_ms`, `ttft_p95_ms`, `cost_usd` or `count`.
    pub metric: String,
    /// Fires when the metric is greater than this.
    pub threshold: f64,
    pub window_ms: i64,
    pub filter: RunFilter,
    pub webhook_url: String,
    /// Minimum time between two notifications.
    pub cooldown_ms: i64,
    pub created_ms: i64,
    pub last_checked_ms: Option<i64>,
    pub last_value: Option<f64>,
    pub last_fired_ms: Option<i64>,
    /// `ok`, `firing` or `no_data`.
    pub last_status: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AlertInput {
    pub name: String,
    pub enabled: bool,
    pub metric: String,
    pub threshold: f64,
    pub window_ms: i64,
    pub filter: RunFilter,
    pub webhook_url: String,
    pub cooldown_ms: i64,
}

impl Default for AlertInput {
    fn default() -> Self {
        Self {
            name: String::new(),
            enabled: true,
            metric: "error_rate".into(),
            threshold: 0.1,
            window_ms: 15 * 60 * 1000,
            filter: RunFilter::default(),
            webhook_url: String::new(),
            cooldown_ms: 60 * 60 * 1000,
        }
    }
}

pub const ALERT_METRICS: &[&str] = &[
    "error_rate",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "avg_latency_ms",
    "ttft_p95_ms",
    "cost_usd",
    "count",
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionResult {
    pub traces: u64,
    pub runs: u64,
    pub feedback: u64,
}

fn row_to_rule(r: &SqliteRow) -> Result<Rule> {
    Ok(Rule {
        id: r.try_get("id")?,
        name: r.try_get("name")?,
        enabled: r.try_get::<i64, _>("enabled")? != 0,
        filter: serde_json::from_str(&r.try_get::<String, _>("filter")?).unwrap_or_default(),
        sample_rate: r.try_get("sample_rate")?,
        evaluators: serde_json::from_str(&r.try_get::<String, _>("evaluators")?)
            .unwrap_or_default(),
        created_ms: r.try_get("created_ms")?,
        updated_ms: r.try_get("updated_ms")?,
        runs_evaluated: r.try_get("runs_evaluated")?,
        last_run_ms: r.try_get("last_run_ms")?,
        last_error: r.try_get("last_error")?,
    })
}

fn row_to_alert(r: &SqliteRow) -> Result<Alert> {
    Ok(Alert {
        id: r.try_get("id")?,
        name: r.try_get("name")?,
        enabled: r.try_get::<i64, _>("enabled")? != 0,
        metric: r.try_get("metric")?,
        threshold: r.try_get("threshold")?,
        window_ms: r.try_get("window_ms")?,
        filter: serde_json::from_str(&r.try_get::<String, _>("filter")?).unwrap_or_default(),
        webhook_url: r.try_get("webhook_url")?,
        cooldown_ms: r.try_get("cooldown_ms")?,
        created_ms: r.try_get("created_ms")?,
        last_checked_ms: r.try_get("last_checked_ms")?,
        last_value: r.try_get("last_value")?,
        last_fired_ms: r.try_get("last_fired_ms")?,
        last_status: r.try_get("last_status")?,
        last_error: r.try_get("last_error")?,
    })
}

fn validate_rule(r: &RuleInput) -> Result<()> {
    if r.name.trim().is_empty() {
        return Err(StoreError::Invalid("rule name is required".into()));
    }
    if !(0.0..=1.0).contains(&r.sample_rate) {
        return Err(StoreError::Invalid(
            "sample_rate must be between 0 and 1".into(),
        ));
    }
    if r.evaluators.is_empty() {
        return Err(StoreError::Invalid(
            "a rule needs at least one evaluator".into(),
        ));
    }
    for e in &r.evaluators {
        crate::engine::validate_evaluator(e).map_err(StoreError::Invalid)?;
    }
    Ok(())
}

fn validate_alert(a: &AlertInput) -> Result<()> {
    if a.name.trim().is_empty() {
        return Err(StoreError::Invalid("alert name is required".into()));
    }
    if !ALERT_METRICS.contains(&a.metric.as_str()) {
        return Err(StoreError::Invalid(format!(
            "unknown metric `{}` (one of {})",
            a.metric,
            ALERT_METRICS.join(", ")
        )));
    }
    if !a.threshold.is_finite() || a.window_ms <= 0 || a.cooldown_ms < 0 {
        return Err(StoreError::Invalid(
            "threshold must be finite, window_ms positive, cooldown_ms non-negative".into(),
        ));
    }
    if !(a.webhook_url.starts_with("http://") || a.webhook_url.starts_with("https://")) {
        return Err(StoreError::Invalid(
            "webhook_url must be an http(s) URL".into(),
        ));
    }
    Ok(())
}

impl Store {
    pub async fn settings(&self) -> Result<Settings> {
        let text: Option<String> =
            sqlx::query_scalar("SELECT value FROM settings WHERE key = 'general'")
                .fetch_optional(self.pool())
                .await?;
        Ok(text
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default())
    }

    pub async fn set_settings(&self, s: &Settings) -> Result<()> {
        if s.retention_days == Some(0) {
            return Err(StoreError::Invalid(
                "retention_days must be at least 1".into(),
            ));
        }
        sqlx::query(
            "INSERT INTO settings (key, value) VALUES ('general', ?) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        )
        .bind(serde_json::to_string(s).unwrap_or_else(|_| "{}".into()))
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Deletes traces that started before `cutoff_ms`, with their runs,
    /// feedback and queue items.
    pub async fn purge_before(&self, cutoff_ms: i64) -> Result<RetentionResult> {
        let mut tx = self.pool().begin().await?;
        sqlx::query("CREATE TEMP TABLE IF NOT EXISTS purge_traces (trace_id TEXT PRIMARY KEY)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM purge_traces")
            .execute(&mut *tx)
            .await?;
        sqlx::query("INSERT INTO purge_traces SELECT trace_id FROM traces WHERE start_ms < ?")
            .bind(cutoff_ms)
            .execute(&mut *tx)
            .await?;
        let feedback = sqlx::query(
            "DELETE FROM feedback WHERE trace_id IN (SELECT trace_id FROM purge_traces)",
        )
        .execute(&mut *tx)
        .await?
        .rows_affected();
        sqlx::query(
            "DELETE FROM queue_items WHERE run_id IN (SELECT id FROM runs WHERE trace_id IN \
             (SELECT trace_id FROM purge_traces))",
        )
        .execute(&mut *tx)
        .await?;
        let runs =
            sqlx::query("DELETE FROM runs WHERE trace_id IN (SELECT trace_id FROM purge_traces)")
                .execute(&mut *tx)
                .await?
                .rows_affected();
        let traces =
            sqlx::query("DELETE FROM traces WHERE trace_id IN (SELECT trace_id FROM purge_traces)")
                .execute(&mut *tx)
                .await?
                .rows_affected();
        sqlx::query("DELETE FROM purge_traces")
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(RetentionResult {
            traces,
            runs,
            feedback,
        })
    }

    /// Applies the configured retention now (no-op without `retention_days`).
    pub async fn apply_retention(&self) -> Result<RetentionResult> {
        match self.settings().await?.retention_days {
            Some(days) => {
                self.purge_before(now_ms() - i64::from(days) * 86_400_000)
                    .await
            }
            None => Ok(RetentionResult::default()),
        }
    }

    pub async fn rules(&self) -> Result<Vec<Rule>> {
        sqlx::query("SELECT * FROM rules ORDER BY created_ms")
            .fetch_all(self.pool())
            .await?
            .iter()
            .map(row_to_rule)
            .collect()
    }

    pub async fn rule(&self, id: &str) -> Result<Option<Rule>> {
        sqlx::query("SELECT * FROM rules WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await?
            .as_ref()
            .map(row_to_rule)
            .transpose()
    }

    /// Creates (`id = None`) or replaces a rule.
    pub async fn put_rule(&self, id: Option<&str>, r: &RuleInput) -> Result<Rule> {
        validate_rule(r)?;
        let now = now_ms();
        let id = id.map(str::to_string).unwrap_or_else(uuid_like);
        sqlx::query(
            "INSERT INTO rules (id, name, enabled, filter, sample_rate, evaluators, created_ms, updated_ms) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name, \
             enabled = excluded.enabled, filter = excluded.filter, sample_rate = excluded.sample_rate, \
             evaluators = excluded.evaluators, updated_ms = excluded.updated_ms",
        )
        .bind(&id)
        .bind(r.name.trim())
        .bind(i64::from(r.enabled))
        .bind(serde_json::to_string(&r.filter).unwrap_or_else(|_| "{}".into()))
        .bind(r.sample_rate)
        .bind(serde_json::to_string(&r.evaluators).unwrap_or_else(|_| "[]".into()))
        .bind(now)
        .bind(now)
        .execute(self.pool())
        .await?;
        self.rule(&id)
            .await?
            .ok_or_else(|| StoreError::Invalid("rule vanished".into()))
    }

    pub async fn delete_rule(&self, id: &str) -> Result<bool> {
        let r = sqlx::query("DELETE FROM rules WHERE id = ?")
            .bind(id)
            .execute(self.pool())
            .await?;
        Ok(r.rows_affected() > 0)
    }

    pub(crate) async fn record_rule_run(
        &self,
        id: &str,
        evaluated: i64,
        error: Option<&str>,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE rules SET runs_evaluated = runs_evaluated + ?, last_run_ms = ?, \
             last_error = COALESCE(?, last_error) WHERE id = ?",
        )
        .bind(evaluated)
        .bind(now_ms())
        .bind(error)
        .bind(id)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn alerts(&self) -> Result<Vec<Alert>> {
        sqlx::query("SELECT * FROM alerts ORDER BY created_ms")
            .fetch_all(self.pool())
            .await?
            .iter()
            .map(row_to_alert)
            .collect()
    }

    pub async fn alert(&self, id: &str) -> Result<Option<Alert>> {
        sqlx::query("SELECT * FROM alerts WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await?
            .as_ref()
            .map(row_to_alert)
            .transpose()
    }

    pub async fn put_alert(&self, id: Option<&str>, a: &AlertInput) -> Result<Alert> {
        validate_alert(a)?;
        let id = id.map(str::to_string).unwrap_or_else(uuid_like);
        sqlx::query(
            "INSERT INTO alerts (id, name, enabled, metric, threshold, window_ms, filter, webhook_url, \
             cooldown_ms, created_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET name = excluded.name, enabled = excluded.enabled, \
             metric = excluded.metric, threshold = excluded.threshold, window_ms = excluded.window_ms, \
             filter = excluded.filter, webhook_url = excluded.webhook_url, cooldown_ms = excluded.cooldown_ms",
        )
        .bind(&id)
        .bind(a.name.trim())
        .bind(i64::from(a.enabled))
        .bind(&a.metric)
        .bind(a.threshold)
        .bind(a.window_ms)
        .bind(serde_json::to_string(&a.filter).unwrap_or_else(|_| "{}".into()))
        .bind(&a.webhook_url)
        .bind(a.cooldown_ms)
        .bind(now_ms())
        .execute(self.pool())
        .await?;
        self.alert(&id)
            .await?
            .ok_or_else(|| StoreError::Invalid("alert vanished".into()))
    }

    pub async fn delete_alert(&self, id: &str) -> Result<bool> {
        let r = sqlx::query("DELETE FROM alerts WHERE id = ?")
            .bind(id)
            .execute(self.pool())
            .await?;
        Ok(r.rows_affected() > 0)
    }

    pub(crate) async fn record_alert_check(
        &self,
        id: &str,
        value: Option<f64>,
        status: &str,
        fired: bool,
        error: Option<&str>,
    ) -> Result<()> {
        let now = now_ms();
        sqlx::query(
            "UPDATE alerts SET last_checked_ms = ?, last_value = ?, last_status = ?, \
             last_fired_ms = CASE WHEN ? THEN ? ELSE last_fired_ms END, last_error = ? WHERE id = ?",
        )
        .bind(now)
        .bind(value)
        .bind(status)
        .bind(fired)
        .bind(now)
        .bind(error)
        .bind(id)
        .execute(self.pool())
        .await?;
        Ok(())
    }
}

/// Whether a run matches a rule filter (used by the online evaluator).
pub fn rule_matches(f: &RunFilter, project: &str, run: &sinew_trace::Run, is_root: bool) -> bool {
    if f.project
        .as_deref()
        .is_some_and(|p| !p.is_empty() && p != project)
    {
        return false;
    }
    if let Some(k) = f.kind.as_deref().filter(|k| !k.is_empty()) {
        if super::kind_str(run.kind) != k.to_lowercase() {
            return false;
        }
    }
    if f.name
        .as_deref()
        .is_some_and(|n| !n.is_empty() && n != run.name)
    {
        return false;
    }
    if let Some(t) = f.tag.as_deref().filter(|t| !t.is_empty()) {
        let has = run.metadata["tags"]
            .as_array()
            .is_some_and(|a| a.iter().any(|x| x == t));
        if !has {
            return false;
        }
    }
    if f.errors_only && run.error.is_none() {
        return false;
    }
    if f.root_only && !is_root {
        return false;
    }
    f.metadata
        .iter()
        .all(|(k, v)| run.metadata.get(k) == Some(v))
}
