//! Background work: online evaluation rules, threshold alerts and data
//! retention.
//!
//! - **Rules**: every ingested batch is handed to a worker (bounded queue,
//!   never blocks ingest). Each enabled rule whose filter matches a finished
//!   run samples it (deterministically by run id, at `sample_rate`), runs the
//!   rule's `sinew-eval` evaluators on the run's inputs/outputs (reference:
//!   `metadata.reference`) and stores every score as feedback with source
//!   `rule:<rule name>`.
//! - **Alerts**: every `alert_interval` each enabled alert computes its metric
//!   over the last `window_ms`; above the threshold it POSTs a JSON payload to
//!   the webhook (at most once per `cooldown_ms`).
//! - **Retention**: every `retention_interval` traces older than
//!   `retention_days` are deleted.
//!
//! Evaluator specs (JSON) accepted by rules:
//!
//! | `type` | Fields | Evaluator |
//! |---|---|---|
//! | `exact_match` | `strict` | [`ExactMatch`] against `metadata.reference` |
//! | `contains` | `needle` (else the reference), `ignore_case` | [`Contains`] |
//! | `regex` | `pattern` | [`RegexMatch`] |
//! | `json_field` | `path`, `expected` | [`JsonFieldEquals`] |
//! | `numeric` | `tolerance`, `relative`, `path` | [`NumericTolerance`] |
//! | `llm_judge` | `model` (`provider:model`), `rubric` | [`LlmJudge`] |
//! | `not_empty` | | output is not empty |
//! | `is_json` | | output is (or parses as) a JSON object or array |
//!
//! Every spec also takes `key` to rename the score.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::ChatModel;
use sinew_eval::{
    as_text, Contains, EvalInput, Evaluator, ExactMatch, JsonFieldEquals, LlmJudge,
    NumericTolerance, RegexMatch, Score,
};
use sinew_trace::Run;
use tokio::sync::mpsc;

use crate::store::{rule_matches, Alert, FeedbackInput, MetricsQuery, Rule, RunQuery, Store};

/// Builds a chat model from a `provider:model` spec (LLM-judge rules).
pub type ModelFactory = Arc<dyn Fn(&str) -> Result<Arc<dyn ChatModel>, String> + Send + Sync>;

/// The default factory: `sinew_core::providers::from_spec`.
pub fn default_model_factory() -> ModelFactory {
    Arc::new(|spec: &str| sinew_core::providers::from_spec(spec).map_err(|e| e.to_string()))
}

/// Background engine settings.
#[derive(Clone)]
pub struct EngineConfig {
    pub model_factory: ModelFactory,
    /// How often alerts are checked (default 30 s).
    pub alert_interval: Duration,
    /// How often retention runs (default 1 h).
    pub retention_interval: Duration,
    /// Webhook request timeout (default 10 s).
    pub webhook_timeout: Duration,
    /// Ingest batches waiting for rule evaluation; extra batches are dropped.
    pub queue_capacity: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_factory: default_model_factory(),
            alert_interval: Duration::from_secs(30),
            retention_interval: Duration::from_secs(3600),
            webhook_timeout: Duration::from_secs(10),
            queue_capacity: 1000,
        }
    }
}

struct NotEmpty;

#[async_trait]
impl Evaluator for NotEmpty {
    fn name(&self) -> &str {
        "not_empty"
    }
    async fn evaluate(&self, i: EvalInput<'_>) -> Score {
        let empty = match i.output {
            Value::Null => true,
            Value::String(s) => s.trim().is_empty(),
            Value::Array(a) => a.is_empty(),
            Value::Object(o) => {
                o.is_empty()
                    || o.get("content")
                        .is_some_and(|c| c.as_str().is_some_and(|s| s.trim().is_empty()))
                        && !o.contains_key("tool_calls")
            }
            _ => false,
        };
        Score::pass("not_empty", !empty)
    }
}

struct IsJson;

#[async_trait]
impl Evaluator for IsJson {
    fn name(&self) -> &str {
        "is_json"
    }
    async fn evaluate(&self, i: EvalInput<'_>) -> Score {
        let text = match i.output {
            Value::Object(o) => o
                .get("content")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| as_text(i.output)),
            other => as_text(other),
        };
        let t = text
            .trim()
            .trim_start_matches("```json")
            .trim_matches('`')
            .trim();
        let ok = matches!(
            serde_json::from_str::<Value>(t),
            Ok(Value::Object(_) | Value::Array(_))
        );
        Score::pass("is_json", ok)
    }
}

/// Renames the score key of another evaluator.
struct Renamed {
    key: String,
    inner: Box<dyn Evaluator>,
}

#[async_trait]
impl Evaluator for Renamed {
    fn name(&self) -> &str {
        &self.key
    }
    async fn evaluate(&self, i: EvalInput<'_>) -> Score {
        let mut s = self.inner.evaluate(i).await;
        s.key = self.key.clone();
        s
    }
}

fn str_field<'a>(spec: &'a Value, k: &str) -> Option<&'a str> {
    spec.get(k)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// Checks an evaluator spec without building models.
pub fn validate_evaluator(spec: &Value) -> Result<(), String> {
    let t = str_field(spec, "type").ok_or("evaluator spec needs a `type`")?;
    match t {
        "llm_judge" => {
            let m = str_field(spec, "model").ok_or("llm_judge needs `model` (provider:model)")?;
            if !m.contains(':') && !m.contains('|') {
                return Err(format!(
                    "llm_judge model `{m}` must look like provider:model"
                ));
            }
            str_field(spec, "rubric").ok_or("llm_judge needs a `rubric`")?;
            Ok(())
        }
        _ => build_evaluator(spec, None).map(|_| ()),
    }
}

/// Builds an evaluator from its JSON spec. `factory` is required for
/// `llm_judge`.
pub fn build_evaluator(
    spec: &Value,
    factory: Option<&ModelFactory>,
) -> Result<Box<dyn Evaluator>, String> {
    let t = str_field(spec, "type").ok_or("evaluator spec needs a `type`")?;
    let e: Box<dyn Evaluator> = match t {
        "exact_match" => Box::new(if spec["strict"].as_bool().unwrap_or(false) {
            ExactMatch::strict()
        } else {
            ExactMatch::new()
        }),
        "contains" => {
            let c = match str_field(spec, "needle") {
                Some(n) => Contains::new(n),
                None => Contains::reference(),
            };
            Box::new(if spec["ignore_case"].as_bool().unwrap_or(false) {
                c.ignore_case()
            } else {
                c
            })
        }
        "regex" => Box::new(
            RegexMatch::new(str_field(spec, "pattern").ok_or("regex needs a `pattern`")?)
                .map_err(|e| format!("invalid regex: {e}"))?,
        ),
        "json_field" => {
            let path = str_field(spec, "path").ok_or("json_field needs a `path`")?;
            Box::new(match spec.get("expected") {
                Some(v) => JsonFieldEquals::expecting(path, v.clone()),
                None => JsonFieldEquals::new(path),
            })
        }
        "numeric" => {
            let tol = spec["tolerance"].as_f64().unwrap_or(1e-6);
            let n = if spec["relative"].as_bool().unwrap_or(false) {
                NumericTolerance::relative(tol)
            } else {
                NumericTolerance::absolute(tol)
            };
            Box::new(match str_field(spec, "path") {
                Some(p) => n.at(p),
                None => n,
            })
        }
        "llm_judge" => {
            let factory = factory.ok_or("llm_judge needs a model factory")?;
            let m = str_field(spec, "model").ok_or("llm_judge needs `model`")?;
            let rubric = str_field(spec, "rubric").ok_or("llm_judge needs a `rubric`")?;
            Box::new(LlmJudge::new(factory(m)?, rubric))
        }
        "not_empty" => Box::new(NotEmpty),
        "is_json" => Box::new(IsJson),
        other => return Err(format!("unknown evaluator type `{other}`")),
    };
    Ok(match str_field(spec, "key") {
        Some(k) => Box::new(Renamed {
            key: k.to_string(),
            inner: e,
        }),
        None => e,
    })
}

/// Deterministic sampling: the same run is always in or out for a rule.
pub fn sampled(rule_id: &str, run_id: &str, rate: f64) -> bool {
    if rate >= 1.0 {
        return true;
    }
    if rate <= 0.0 {
        return false;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in rule_id.bytes().chain([0u8]).chain(run_id.bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // splitmix64 finalizer: spreads FNV's weak high bits.
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    ((h >> 11) as f64 / (1u64 << 53) as f64) < rate
}

/// Handle to the background engine. Cheap to clone.
#[derive(Clone)]
pub struct Engine {
    store: Store,
    config: EngineConfig,
    tx: mpsc::Sender<Vec<(String, Run)>>,
    http: reqwest::Client,
}

impl Engine {
    /// Spawns the rule worker, the alert loop and the retention loop on the
    /// current Tokio runtime.
    pub fn start(store: Store, config: EngineConfig) -> Self {
        let (tx, mut rx) = mpsc::channel::<Vec<(String, Run)>>(config.queue_capacity.max(1));
        let http = reqwest::Client::builder()
            .timeout(config.webhook_timeout)
            .build()
            .unwrap_or_default();
        let engine = Self {
            store,
            config,
            tx,
            http,
        };
        let worker = engine.clone();
        tokio::spawn(async move {
            while let Some(batch) = rx.recv().await {
                let _ = worker.evaluate_runs(&batch).await;
            }
        });
        let alerts = engine.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(alerts.config.alert_interval);
            tick.tick().await;
            loop {
                tick.tick().await;
                let _ = alerts.check_alerts().await;
            }
        });
        let retention = engine.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(retention.config.retention_interval);
            loop {
                tick.tick().await;
                let _ = retention.store.apply_retention().await;
            }
        });
        engine
    }

    /// Queues an ingested batch for rule evaluation (dropped when the queue
    /// is full; ingest never waits).
    pub fn submit(&self, runs: Vec<(String, Run)>) {
        if !runs.is_empty() {
            let _ = self.tx.try_send(runs);
        }
    }

    /// Runs every enabled rule on `runs` now. Returns how many feedback
    /// entries were stored.
    pub async fn evaluate_runs(&self, runs: &[(String, Run)]) -> crate::store::Result<usize> {
        let rules: Vec<Rule> = self
            .store
            .rules()
            .await?
            .into_iter()
            .filter(|r| r.enabled)
            .collect();
        let mut stored = 0;
        for rule in &rules {
            let (n, err) = self.apply_rule(rule, runs).await;
            stored += n;
            if n > 0 || err.is_some() {
                self.store
                    .record_rule_run(&rule.id, n as i64, err.as_deref())
                    .await?;
            }
        }
        Ok(stored)
    }

    async fn apply_rule(&self, rule: &Rule, runs: &[(String, Run)]) -> (usize, Option<String>) {
        let targets: Vec<&(String, Run)> = runs
            .iter()
            .filter(|(p, r)| {
                r.end_ms.is_some()
                    && rule_matches(&rule.filter, p, r, r.parent_id.is_none())
                    && sampled(&rule.id, &r.id, rule.sample_rate)
            })
            .collect();
        if targets.is_empty() {
            return (0, None);
        }
        let mut evaluators = Vec::new();
        for spec in &rule.evaluators {
            match build_evaluator(spec, Some(&self.config.model_factory)) {
                Ok(e) => evaluators.push(e),
                Err(e) => return (0, Some(e)),
            }
        }
        let mut stored = 0;
        let mut last_err = None;
        for (_, run) in targets {
            let output = run.outputs.clone().unwrap_or(Value::Null);
            let reference = run.metadata.get("reference").cloned();
            for e in &evaluators {
                let score = e
                    .evaluate(EvalInput {
                        inputs: &run.inputs,
                        output: &output,
                        reference: reference.as_ref(),
                    })
                    .await;
                let f = FeedbackInput {
                    run_id: run.id.clone(),
                    key: score.key.clone(),
                    score: Some(score.score),
                    value: None,
                    comment: score.reason.clone(),
                    source: Some(format!("rule:{}", rule.name)),
                };
                match self.store.add_feedback(&f).await {
                    Ok(_) => stored += 1,
                    Err(e) => last_err = Some(e.to_string()),
                }
            }
        }
        (stored, last_err)
    }

    /// Runs one rule over the latest `limit` matching runs (sampling still
    /// applies). Returns how many feedback entries were stored.
    pub async fn backfill_rule(&self, rule: &Rule, limit: i64) -> crate::store::Result<usize> {
        let page = self
            .store
            .search_runs(&RunQuery {
                project: rule.filter.project.clone(),
                kind: rule.filter.kind.clone(),
                tag: rule.filter.tag.clone(),
                errors: rule.filter.errors_only,
                root: rule.filter.root_only,
                limit: Some(limit.clamp(1, 500)),
                ..Default::default()
            })
            .await?;
        let mut runs = Vec::new();
        for r in page.runs {
            if let Some(s) = self.store.run(&r.id).await? {
                runs.push((s.project, s.run));
            }
        }
        let (n, err) = self.apply_rule(rule, &runs).await;
        self.store
            .record_rule_run(&rule.id, n as i64, err.as_deref())
            .await?;
        Ok(n)
    }

    /// Checks every enabled alert now; returns one result per alert.
    pub async fn check_alerts(&self) -> crate::store::Result<Vec<Value>> {
        let mut out = Vec::new();
        for a in self.store.alerts().await? {
            if a.enabled {
                out.push(self.check_alert(&a).await?);
            }
        }
        Ok(out)
    }

    /// Checks one alert now (and notifies if it fires and the cooldown has
    /// passed). Returns `{id, value, status, fired, error}`.
    pub async fn check_alert(&self, a: &Alert) -> crate::store::Result<Value> {
        let now = crate::store::now_ms();
        let m = self
            .store
            .metrics(&MetricsQuery {
                project: a.filter.project.clone(),
                from_ms: Some(now - a.window_ms),
                to_ms: Some(now),
                bucket_ms: Some(a.window_ms.max(1)),
                level: a.filter.level.clone(),
                name: a.filter.name.clone(),
                tag: a.filter.tag.clone(),
                metadata: a
                    .filter
                    .metadata
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            v.as_str()
                                .map(str::to_string)
                                .unwrap_or_else(|| v.to_string()),
                        )
                    })
                    .collect(),
                ..Default::default()
            })
            .await?;
        let t = &m.totals;
        let value = if t.count == 0 {
            None
        } else {
            match a.metric.as_str() {
                "error_rate" => Some(t.error_rate),
                "p50_latency_ms" => t.p50_latency_ms.map(|v| v as f64),
                "p95_latency_ms" => t.p95_latency_ms.map(|v| v as f64),
                "p99_latency_ms" => t.p99_latency_ms.map(|v| v as f64),
                "avg_latency_ms" => t.avg_latency_ms,
                "ttft_p95_ms" => t.ttft_p95_ms.map(|v| v as f64),
                "cost_usd" => Some(t.cost_usd),
                "count" => Some(t.count as f64),
                _ => None,
            }
        };
        let firing = value.is_some_and(|v| v > a.threshold);
        let status = match (value, firing) {
            (None, _) => "no_data",
            (_, true) => "firing",
            _ => "ok",
        };
        let cooled = a.last_fired_ms.is_none_or(|f| now - f >= a.cooldown_ms);
        let mut fired = false;
        let mut error = None;
        if firing && cooled {
            let v = value.unwrap_or_default();
            let payload = json!({
                "text": format!(
                    "[sinew-scope] alert `{}`: {} = {:.4} > {} over the last {} min ({} points)",
                    a.name, a.metric, v, a.threshold, a.window_ms / 60_000, t.count
                ),
                "alert": { "id": a.id, "name": a.name, "metric": a.metric, "threshold": a.threshold, "window_ms": a.window_ms },
                "value": v,
                "status": "firing",
                "count": t.count,
                "filter": a.filter,
                "fired_ms": now,
            });
            match self.http.post(&a.webhook_url).json(&payload).send().await {
                Ok(r) if r.status().is_success() => fired = true,
                Ok(r) => error = Some(format!("webhook returned {}", r.status())),
                Err(e) => error = Some(format!("webhook failed: {e}")),
            }
        }
        self.store
            .record_alert_check(&a.id, value, status, fired, error.as_deref())
            .await?;
        Ok(json!({ "id": a.id, "value": value, "status": status, "fired": fired, "error": error }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_is_deterministic_and_roughly_uniform() {
        let n = (0..10_000)
            .filter(|i| sampled("rule", &format!("run-{i}"), 0.25))
            .count();
        assert!((2000..3000).contains(&n), "{n}");
        assert_eq!(sampled("r", "x", 0.5), sampled("r", "x", 0.5));
        assert!(sampled("r", "x", 1.0));
        assert!(!sampled("r", "x", 0.0));
    }

    #[tokio::test]
    async fn builds_specs() {
        for spec in [
            json!({"type": "contains", "needle": "a", "key": "has_a"}),
            json!({"type": "regex", "pattern": "^x"}),
            json!({"type": "json_field", "path": "a.b", "expected": 1}),
            json!({"type": "numeric", "tolerance": 0.1}),
            json!({"type": "exact_match"}),
            json!({"type": "not_empty"}),
            json!({"type": "is_json"}),
        ] {
            assert!(validate_evaluator(&spec).is_ok(), "{spec}");
        }
        assert!(validate_evaluator(
            &json!({"type": "llm_judge", "model": "openai:gpt-4.1-mini", "rubric": "ok?"})
        )
        .is_ok());
        assert!(validate_evaluator(&json!({"type": "llm_judge", "rubric": "x"})).is_err());
        assert!(validate_evaluator(&json!({"type": "nope"})).is_err());
        assert!(validate_evaluator(&json!({"type": "regex", "pattern": "("})).is_err());

        let e = build_evaluator(
            &json!({"type": "contains", "needle": "a", "key": "has_a"}),
            None,
        )
        .unwrap();
        let s = e
            .evaluate(EvalInput {
                inputs: &json!({}),
                output: &json!("cat"),
                reference: None,
            })
            .await;
        assert_eq!((s.key.as_str(), s.score), ("has_a", 1.0));
        let j = build_evaluator(&json!({"type": "is_json"}), None).unwrap();
        let s = j
            .evaluate(EvalInput {
                inputs: &json!({}),
                output: &json!({"role": "assistant", "content": "{\"a\": 1}"}),
                reference: None,
            })
            .await;
        assert_eq!(s.score, 1.0);
    }
}
