//! Datasets, evaluators and experiments: the LangSmith evaluation loop.
//!
//! 1. Build or load a [`Dataset`] of examples (inputs + optional reference).
//! 2. Pick [`Evaluator`]s: [`ExactMatch`], [`Contains`], [`RegexMatch`],
//!    [`JsonFieldEquals`], [`NumericTolerance`] or an [`LlmJudge`].
//! 3. Call [`evaluate`] with your async target function. You get an
//!    [`ExperimentReport`] with per-example scores, aggregates and latency.
//! 4. Optionally upload dataset and report to `sinew-scope serve` with
//!    [`ScopeClient`] to browse and compare experiments in the web UI.
//!
//! ```no_run
//! # async fn demo() -> Result<(), sinew_eval::EvalError> {
//! use serde_json::{json, Value};
//! use sinew_eval::*;
//!
//! let mut ds = Dataset::new("capitals");
//! ds.push(json!({"country": "France"}), Some(json!("Paris")));
//! let evaluators: Vec<Box<dyn Evaluator>> = vec![Box::new(Contains::reference())];
//! let report = evaluate(
//!     |inputs: Value| async move { Ok::<_, String>(json!(format!("{} → Paris", inputs["country"]))) },
//!     &ds,
//!     &evaluators,
//!     4,
//! )
//! .await;
//! println!("{}", report.summary());
//! ScopeClient::new("http://127.0.0.1:7878").upload_experiment(&ds, &report).await?;
//! # Ok(()) }
//! ```
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

mod client;
mod dataset;
mod evaluators;

use std::{
    collections::BTreeMap,
    fmt::Display,
    future::Future,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use client::ScopeClient;
pub use dataset::{Dataset, Example};
pub use evaluators::{
    as_text, extract_number, get_path, parse_judge_reply, Contains, EvalInput, Evaluator,
    ExactMatch, JsonFieldEquals, LlmJudge, NumericTolerance, RegexMatch, Score,
};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("line {line}: {message}")]
    Parse { line: usize, message: String },
    #[error("HTTP error: {0}")]
    Http(String),
}

/// Outcome of the target and every evaluator on one example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExampleResult {
    pub example_id: String,
    pub index: usize,
    pub inputs: Value,
    #[serde(default)]
    pub reference: Option<Value>,
    #[serde(default)]
    pub output: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
    pub latency_ms: u64,
    #[serde(default)]
    pub scores: Vec<Score>,
}

impl ExampleResult {
    pub fn score(&self, key: &str) -> Option<f64> {
        self.scores.iter().find(|s| s.key == key).map(|s| s.score)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScoreStats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Aggregates {
    pub count: usize,
    pub errors: usize,
    pub mean_latency_ms: f64,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub max_latency_ms: u64,
    /// Per evaluator key.
    pub scores: BTreeMap<String, ScoreStats>,
}

impl Aggregates {
    pub fn from_results(results: &[ExampleResult]) -> Self {
        let mut lat: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
        lat.sort_unstable();
        let pct = |p: f64| -> u64 {
            if lat.is_empty() {
                0
            } else {
                lat[((lat.len() - 1) as f64 * p).round() as usize]
            }
        };
        let mut scores: BTreeMap<String, ScoreStats> = BTreeMap::new();
        for s in results.iter().flat_map(|r| &r.scores) {
            let e = scores.entry(s.key.clone()).or_insert(ScoreStats {
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
                ..Default::default()
            });
            e.mean += s.score;
            e.count += 1;
            e.min = e.min.min(s.score);
            e.max = e.max.max(s.score);
        }
        for e in scores.values_mut() {
            e.mean /= e.count as f64;
        }
        Self {
            count: results.len(),
            errors: results.iter().filter(|r| r.error.is_some()).count(),
            mean_latency_ms: if lat.is_empty() {
                0.0
            } else {
                lat.iter().sum::<u64>() as f64 / lat.len() as f64
            },
            p50_latency_ms: pct(0.5),
            p95_latency_ms: pct(0.95),
            max_latency_ms: lat.last().copied().unwrap_or(0),
            scores,
        }
    }
}

/// A full experiment: one run of a target over a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentReport {
    pub id: String,
    pub name: String,
    pub dataset: String,
    pub created_ms: u64,
    #[serde(default)]
    pub metadata: Value,
    pub results: Vec<ExampleResult>,
    pub aggregates: Aggregates,
}

impl ExperimentReport {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Mean score of one evaluator key.
    pub fn mean(&self, key: &str) -> Option<f64> {
        self.aggregates.scores.get(key).map(|s| s.mean)
    }

    /// Short multi-line text summary.
    pub fn summary(&self) -> String {
        let a = &self.aggregates;
        let mut out = format!(
            "experiment {} on {}: {} examples, {} errors, latency mean {:.0}ms p50 {}ms p95 {}ms",
            self.name,
            self.dataset,
            a.count,
            a.errors,
            a.mean_latency_ms,
            a.p50_latency_ms,
            a.p95_latency_ms
        );
        for (k, s) in &a.scores {
            out.push_str(&format!(
                "\n  {k:<24} mean {:.3}  min {:.2}  max {:.2}",
                s.mean, s.min, s.max
            ));
        }
        out
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Runs `target` on every example (at most `concurrency` at once), scores each
/// output with every evaluator and aggregates. Target errors are recorded on
/// the example (evaluators are skipped for it) and never abort the run.
pub async fn evaluate<F, Fut, E>(
    target: F,
    dataset: &Dataset,
    evaluators: &[Box<dyn Evaluator>],
    concurrency: usize,
) -> ExperimentReport
where
    F: Fn(Value) -> Fut,
    Fut: Future<Output = Result<Value, E>>,
    E: Display,
{
    let created_ms = now_ms();
    let target = &target;
    let mut results: Vec<ExampleResult> =
        futures::stream::iter(dataset.examples.iter().enumerate())
            .map(|(index, ex)| async move {
                let started = Instant::now();
                let outcome = target(ex.inputs.clone()).await;
                let latency_ms = started.elapsed().as_millis() as u64;
                let (output, error, scores) = match outcome {
                    Ok(output) => {
                        let mut scores = Vec::with_capacity(evaluators.len());
                        for e in evaluators {
                            scores.push(
                                e.evaluate(EvalInput {
                                    inputs: &ex.inputs,
                                    output: &output,
                                    reference: ex.reference.as_ref(),
                                })
                                .await,
                            );
                        }
                        (Some(output), None, scores)
                    }
                    Err(e) => (None, Some(e.to_string()), Vec::new()),
                };
                ExampleResult {
                    example_id: ex.id.clone(),
                    index,
                    inputs: ex.inputs.clone(),
                    reference: ex.reference.clone(),
                    output,
                    error,
                    latency_ms,
                    scores,
                }
            })
            .buffer_unordered(concurrency.max(1))
            .collect()
            .await;
    results.sort_by_key(|r| r.index);
    let aggregates = Aggregates::from_results(&results);
    ExperimentReport {
        id: uuid::Uuid::new_v4().to_string(),
        name: format!("{}-{}", dataset.name, created_ms),
        dataset: dataset.name.clone(),
        created_ms,
        metadata: Value::Null,
        results,
        aggregates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[tokio::test]
    async fn evaluates_with_bounded_concurrency_and_keeps_order() {
        let mut ds = Dataset::new("sq");
        for i in 0..8 {
            ds.push(json!({"n": i}), Some(json!(i * i)));
        }
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let evaluators: Vec<Box<dyn Evaluator>> = vec![
            Box::new(ExactMatch::new()),
            Box::new(NumericTolerance::absolute(0.5)),
        ];
        let report = evaluate(
            |inputs: Value| {
                let (live, peak) = (live.clone(), peak.clone());
                async move {
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    live.fetch_sub(1, Ordering::SeqCst);
                    let n = inputs["n"].as_i64().unwrap();
                    if n == 7 {
                        Err("boom".to_string())
                    } else if n == 3 {
                        Ok(json!(10))
                    } else {
                        Ok(json!(n * n))
                    }
                }
            },
            &ds,
            &evaluators,
            3,
        )
        .await;
        assert!(peak.load(Ordering::SeqCst) <= 3);
        assert_eq!(report.results.len(), 8);
        assert!(report.results.iter().enumerate().all(|(i, r)| r.index == i));
        assert_eq!(report.aggregates.errors, 1);
        assert_eq!(report.results[7].error.as_deref(), Some("boom"));
        // 7 scored examples, one wrong (n = 3).
        let em = &report.aggregates.scores["exact_match"];
        assert_eq!(em.count, 7);
        assert!((em.mean - 6.0 / 7.0).abs() < 1e-9);
        assert_eq!(report.results[3].score("numeric"), Some(0.0));
        assert!(report.aggregates.p95_latency_ms >= 5);
        let back: ExperimentReport =
            serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
        assert_eq!(back, report);
    }
}
