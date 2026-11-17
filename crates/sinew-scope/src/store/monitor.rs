//! Run search (full text, metadata filters, cursor pagination) and
//! monitoring metrics (latency percentiles, time to first token, error rate,
//! tokens and cost over time and per group).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{QueryBuilder, Row, Sqlite};

use super::{from_json, fts_query, json_key_path, like_pattern, meta_value, Result, Store};

/// Filters and pagination for [`Store::search_runs`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RunQuery {
    pub project: Option<String>,
    /// Full text over name, inputs, outputs and error.
    pub q: Option<String>,
    pub kind: Option<String>,
    /// Substring of the run name.
    pub name: Option<String>,
    /// Model prefix (`gpt-4o` matches `gpt-4o-mini`).
    pub model: Option<String>,
    /// `metadata.tags` contains this tag.
    pub tag: Option<String>,
    pub trace_id: Option<String>,
    pub errors: bool,
    /// Only root runs (one per trace).
    pub root: bool,
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    pub min_latency_ms: Option<i64>,
    /// Runs that have feedback with this key.
    pub feedback_key: Option<String>,
    /// `meta.<key>=<value>` in the API.
    #[serde(skip)]
    pub metadata: Vec<(String, String)>,
    pub limit: Option<i64>,
    /// Opaque cursor from the previous page (`next_cursor`).
    pub cursor: Option<String>,
}

/// A run in search results (inputs/outputs shortened to previews).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRow {
    pub id: String,
    pub trace_id: String,
    pub parent_id: Option<String>,
    pub project: String,
    pub name: String,
    pub kind: String,
    pub start_ms: i64,
    pub end_ms: Option<i64>,
    pub latency_ms: Option<i64>,
    pub error: Option<String>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub model: Option<String>,
    pub cost_usd: Option<f64>,
    pub ttft_ms: Option<i64>,
    pub tags: Value,
    pub input_preview: String,
    pub output_preview: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunPage {
    pub runs: Vec<RunRow>,
    /// Pass as `cursor` to get the next (older) page; `None` at the end.
    pub next_cursor: Option<String>,
}

/// Parameters of [`Store::metrics`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsQuery {
    pub project: Option<String>,
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    /// Bucket width; default: the range split in about 60 buckets.
    pub bucket_ms: Option<i64>,
    /// `trace` (default: one point per trace, totals over its runs) or a run
    /// kind (`llm`, `tool`, `node`, `graph`, `chain`) or `run` (every run).
    pub level: Option<String>,
    /// `name` (default), `model`, `project` or `kind`.
    pub group_by: Option<String>,
    pub name: Option<String>,
    pub model: Option<String>,
    pub tag: Option<String>,
    #[serde(skip)]
    pub metadata: Vec<(String, String)>,
}

/// Aggregates over a set of runs or traces.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GroupStats {
    pub key: String,
    pub count: i64,
    pub errors: i64,
    pub error_rate: f64,
    pub avg_latency_ms: Option<f64>,
    pub p50_latency_ms: Option<i64>,
    pub p95_latency_ms: Option<i64>,
    pub p99_latency_ms: Option<i64>,
    pub ttft_p50_ms: Option<i64>,
    pub ttft_p95_ms: Option<i64>,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cost_usd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeBucket {
    pub start_ms: i64,
    #[serde(flatten)]
    pub stats: GroupStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricsReport {
    pub level: String,
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    pub bucket_ms: i64,
    pub totals: GroupStats,
    pub buckets: Vec<TimeBucket>,
    /// Sorted by count, descending.
    pub groups: Vec<GroupStats>,
    /// True when more than `METRICS_ROW_CAP` (200,000) points matched (the newest
    /// are used).
    pub truncated: bool,
}

/// Maximum points one metrics query aggregates.
pub const METRICS_ROW_CAP: i64 = 200_000;

struct Point {
    start: i64,
    latency: Option<i64>,
    error: bool,
    itok: i64,
    otok: i64,
    cost: f64,
    ttft: Option<i64>,
    group: String,
}

fn pct(sorted: &[i64], p: f64) -> Option<i64> {
    if sorted.is_empty() {
        None
    } else {
        Some(sorted[((sorted.len() - 1) as f64 * p).round() as usize])
    }
}

fn stats(key: String, points: &[&Point]) -> GroupStats {
    let mut lat: Vec<i64> = points.iter().filter_map(|p| p.latency).collect();
    lat.sort_unstable();
    let mut ttft: Vec<i64> = points.iter().filter_map(|p| p.ttft).collect();
    ttft.sort_unstable();
    let count = points.len() as i64;
    let errors = points.iter().filter(|p| p.error).count() as i64;
    GroupStats {
        key,
        count,
        errors,
        error_rate: if count > 0 {
            errors as f64 / count as f64
        } else {
            0.0
        },
        avg_latency_ms: (!lat.is_empty())
            .then(|| lat.iter().sum::<i64>() as f64 / lat.len() as f64),
        p50_latency_ms: pct(&lat, 0.5),
        p95_latency_ms: pct(&lat, 0.95),
        p99_latency_ms: pct(&lat, 0.99),
        ttft_p50_ms: pct(&ttft, 0.5),
        ttft_p95_ms: pct(&ttft, 0.95),
        input_tokens: points.iter().map(|p| p.itok).sum(),
        output_tokens: points.iter().map(|p| p.otok).sum(),
        cost_usd: points.iter().map(|p| p.cost).sum(),
    }
}

/// Cursor: `<start_ms>:<id>`.
fn parse_cursor(c: &str) -> Option<(i64, String)> {
    let (s, id) = c.split_once(':')?;
    Some((s.parse().ok()?, id.to_string()))
}

impl Store {
    fn push_run_filters(&self, qb: &mut QueryBuilder<Sqlite>, q: &RunQuery) {
        if let Some(p) = q.project.as_ref().filter(|p| !p.is_empty()) {
            qb.push(" AND runs.project = ").push_bind(p.clone());
        }
        if let Some(k) = q.kind.as_ref().filter(|k| !k.is_empty()) {
            qb.push(" AND runs.kind = ").push_bind(k.to_lowercase());
        }
        if let Some(n) = q.name.as_ref().filter(|n| !n.is_empty()) {
            qb.push(" AND runs.name LIKE ")
                .push_bind(like_pattern(n))
                .push(" ESCAPE '\\'");
        }
        if let Some(m) = q.model.as_ref().filter(|m| !m.is_empty()) {
            qb.push(" AND runs.model LIKE ")
                .push_bind(format!("{}%", m.replace('%', "")))
                .push(" ");
        }
        if let Some(t) = q.trace_id.as_ref().filter(|t| !t.is_empty()) {
            qb.push(" AND runs.trace_id = ").push_bind(t.clone());
        }
        if let Some(t) = q.tag.as_ref().filter(|t| !t.is_empty()) {
            qb.push(" AND EXISTS (SELECT 1 FROM json_each(runs.metadata, '$.tags') jt WHERE jt.value = ")
                .push_bind(t.clone())
                .push(")");
        }
        for (k, v) in &q.metadata {
            qb.push(" AND CAST(json_extract(runs.metadata, ")
                .push_bind(json_key_path(k))
                .push(") AS TEXT) = ")
                .push_bind(meta_value(v));
        }
        if q.errors {
            qb.push(" AND runs.error IS NOT NULL");
        }
        if q.root {
            qb.push(" AND runs.id IN (SELECT root_id FROM traces)");
        }
        if let Some(f) = q.from_ms {
            qb.push(" AND runs.start_ms >= ").push_bind(f);
        }
        if let Some(t) = q.to_ms {
            qb.push(" AND runs.start_ms <= ").push_bind(t);
        }
        if let Some(l) = q.min_latency_ms {
            qb.push(" AND runs.end_ms - runs.start_ms >= ").push_bind(l);
        }
        if let Some(k) = q.feedback_key.as_ref().filter(|k| !k.is_empty()) {
            qb.push(" AND runs.id IN (SELECT run_id FROM feedback WHERE key = ")
                .push_bind(k.clone())
                .push(")");
        }
        if let Some(text) = q.q.as_ref().filter(|t| !t.trim().is_empty()) {
            match fts_query(text).filter(|_| self.has_fts()) {
                Some(m) => {
                    qb.push(" AND runs.rowid IN (SELECT rowid FROM runs_fts WHERE runs_fts MATCH ")
                        .push_bind(m)
                        .push(")");
                }
                None => {
                    let pat = like_pattern(text.trim());
                    qb.push(" AND (runs.inputs LIKE ")
                        .push_bind(pat.clone())
                        .push(" ESCAPE '\\' OR runs.outputs LIKE ")
                        .push_bind(pat.clone())
                        .push(" ESCAPE '\\' OR runs.name LIKE ")
                        .push_bind(pat.clone())
                        .push(" ESCAPE '\\' OR runs.error LIKE ")
                        .push_bind(pat)
                        .push(" ESCAPE '\\')");
                }
            }
        }
    }

    /// Searches runs, newest first, with keyset (cursor) pagination so deep
    /// pages stay as fast as the first one.
    pub async fn search_runs(&self, q: &RunQuery) -> Result<RunPage> {
        let limit = q.limit.unwrap_or(50).clamp(1, 500);
        let mut qb: QueryBuilder<Sqlite> = QueryBuilder::new(
            "SELECT runs.id, runs.trace_id, runs.parent_id, runs.project, runs.name, runs.kind, \
             runs.start_ms, runs.end_ms, runs.error, runs.input_tokens, runs.output_tokens, \
             runs.model, runs.cost_usd, runs.ttft_ms, json_extract(runs.metadata, '$.tags') AS tags, \
             substr(runs.inputs, 1, 300) AS input_preview, substr(runs.outputs, 1, 300) AS output_preview \
             FROM runs WHERE 1 = 1",
        );
        self.push_run_filters(&mut qb, q);
        if let Some((s, id)) = q.cursor.as_deref().and_then(parse_cursor) {
            qb.push(" AND (runs.start_ms < ")
                .push_bind(s)
                .push(" OR (runs.start_ms = ")
                .push_bind(s)
                .push(" AND runs.id < ")
                .push_bind(id)
                .push("))");
        }
        qb.push(" ORDER BY runs.start_ms DESC, runs.id DESC LIMIT ")
            .push_bind(limit + 1);
        let rows = qb.build().fetch_all(self.pool()).await?;
        let mut runs = rows
            .iter()
            .map(|r| {
                let start: i64 = r.try_get("start_ms")?;
                let end: Option<i64> = r.try_get("end_ms")?;
                Ok(RunRow {
                    id: r.try_get("id")?,
                    trace_id: r.try_get("trace_id")?,
                    parent_id: r.try_get("parent_id")?,
                    project: r.try_get("project")?,
                    name: r.try_get("name")?,
                    kind: r.try_get("kind")?,
                    start_ms: start,
                    end_ms: end,
                    latency_ms: end.map(|e| (e - start).max(0)),
                    error: r.try_get("error")?,
                    input_tokens: r.try_get("input_tokens")?,
                    output_tokens: r.try_get("output_tokens")?,
                    model: r.try_get("model")?,
                    cost_usd: r.try_get("cost_usd")?,
                    ttft_ms: r.try_get("ttft_ms")?,
                    tags: r
                        .try_get::<Option<String>, _>("tags")?
                        .map(|t| from_json(&t))
                        .unwrap_or(Value::Null),
                    input_preview: r.try_get("input_preview")?,
                    output_preview: r.try_get("output_preview")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let next_cursor = if runs.len() as i64 > limit {
            runs.truncate(limit as usize);
            runs.last().map(|r| format!("{}:{}", r.start_ms, r.id))
        } else {
            None
        };
        Ok(RunPage { runs, next_cursor })
    }

    /// Monitoring aggregates over traces or runs.
    pub async fn metrics(&self, q: &MetricsQuery) -> Result<MetricsReport> {
        let level = q
            .level
            .clone()
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| "trace".into())
            .to_lowercase();
        let group_by = match q.group_by.as_deref().unwrap_or("name") {
            "model" => "model",
            "project" => "project",
            "kind" => "kind",
            _ => "name",
        };
        let mut qb: QueryBuilder<Sqlite> = if level == "trace" {
            let group = if group_by == "model" {
                "name"
            } else {
                group_by
            };
            let mut qb = QueryBuilder::new(format!(
                "SELECT start_ms, latency_ms AS lat, error_count > 0 AS err, input_tokens AS itok, \
                 output_tokens AS otok, cost_usd AS cost, NULL AS ttft, {group} AS grp FROM traces t WHERE 1 = 1"
            ));
            if let Some(p) = q.project.as_ref().filter(|p| !p.is_empty()) {
                qb.push(" AND t.project = ").push_bind(p.clone());
            }
            if let Some(n) = q.name.as_ref().filter(|n| !n.is_empty()) {
                qb.push(" AND t.name LIKE ")
                    .push_bind(like_pattern(n))
                    .push(" ESCAPE '\\'");
            }
            if let Some(t) = q.tag.as_ref().filter(|t| !t.is_empty()) {
                qb.push(" AND EXISTS (SELECT 1 FROM runs rr, json_each(rr.metadata, '$.tags') jt WHERE rr.id = t.root_id AND jt.value = ")
                    .push_bind(t.clone())
                    .push(")");
            }
            for (k, v) in &q.metadata {
                qb.push(" AND EXISTS (SELECT 1 FROM runs rr WHERE rr.id = t.root_id AND CAST(json_extract(rr.metadata, ")
                    .push_bind(json_key_path(k))
                    .push(") AS TEXT) = ")
                    .push_bind(meta_value(v))
                    .push(")");
            }
            if let Some(f) = q.from_ms {
                qb.push(" AND t.start_ms >= ").push_bind(f);
            }
            if let Some(t) = q.to_ms {
                qb.push(" AND t.start_ms <= ").push_bind(t);
            }
            qb
        } else {
            let mut qb = QueryBuilder::new(format!(
                "SELECT start_ms, end_ms - start_ms AS lat, error IS NOT NULL AS err, \
                 COALESCE(input_tokens, 0) AS itok, COALESCE(output_tokens, 0) AS otok, \
                 COALESCE(cost_usd, 0.0) AS cost, ttft_ms AS ttft, COALESCE({group_by}, '(none)') AS grp \
                 FROM runs WHERE 1 = 1"
            ));
            let rq = RunQuery {
                project: q.project.clone(),
                kind: (level != "run").then(|| level.clone()),
                name: q.name.clone(),
                model: q.model.clone(),
                tag: q.tag.clone(),
                metadata: q.metadata.clone(),
                from_ms: q.from_ms,
                to_ms: q.to_ms,
                ..Default::default()
            };
            self.push_run_filters(&mut qb, &rq);
            qb
        };
        qb.push(" ORDER BY start_ms DESC LIMIT ")
            .push_bind(METRICS_ROW_CAP + 1);
        let rows = qb.build().fetch_all(self.pool()).await?;
        let truncated = rows.len() as i64 > METRICS_ROW_CAP;
        let mut points = rows
            .iter()
            .take(METRICS_ROW_CAP as usize)
            .map(|r| {
                Ok(Point {
                    start: r.try_get("start_ms")?,
                    latency: r.try_get("lat")?,
                    error: r.try_get::<Option<bool>, _>("err")?.unwrap_or(false),
                    itok: r.try_get::<Option<i64>, _>("itok")?.unwrap_or(0),
                    otok: r.try_get::<Option<i64>, _>("otok")?.unwrap_or(0),
                    cost: r.try_get::<Option<f64>, _>("cost")?.unwrap_or(0.0),
                    ttft: r.try_get("ttft")?,
                    group: r
                        .try_get::<Option<String>, _>("grp")?
                        .unwrap_or_else(|| "(none)".into()),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        points.reverse();
        let first = points.first().map(|p| p.start);
        let last = points.last().map(|p| p.start);
        let from = q.from_ms.or(first).unwrap_or(0);
        let to = q.to_ms.or(last).unwrap_or(from);
        let bucket_ms = q
            .bucket_ms
            .filter(|b| *b > 0)
            .unwrap_or_else(|| ((to - from) / 60).max(1000))
            .max(1);
        let all: Vec<&Point> = points.iter().collect();
        let mut by_bucket: BTreeMap<i64, Vec<&Point>> = BTreeMap::new();
        let mut by_group: BTreeMap<String, Vec<&Point>> = BTreeMap::new();
        for p in &points {
            let b = from + ((p.start - from).max(0) / bucket_ms) * bucket_ms;
            by_bucket.entry(b).or_default().push(p);
            by_group.entry(p.group.clone()).or_default().push(p);
        }
        let mut groups: Vec<GroupStats> = by_group.into_iter().map(|(k, v)| stats(k, &v)).collect();
        groups.sort_by(|a, b| b.count.cmp(&a.count).then(a.key.cmp(&b.key)));
        Ok(MetricsReport {
            level,
            from_ms: q.from_ms.or(first),
            to_ms: q.to_ms.or(last),
            bucket_ms,
            totals: stats("all".into(), &all),
            buckets: by_bucket
                .into_iter()
                .map(|(start_ms, v)| TimeBucket {
                    start_ms,
                    stats: stats(String::new(), &v),
                })
                .collect(),
            groups,
            truncated,
        })
    }
}
