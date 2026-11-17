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
//! For agents, the [`trajectory`] module scores the path as well as the
//! answer: which tools were called, in which order, and at what cost.
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
pub mod hub;
pub mod pairwise;
pub mod summary;
pub mod trajectory;

use std::{
    collections::BTreeMap,
    fmt::Display,
    future::Future,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use client::{Feedback, ScopeClient};
pub use dataset::{Dataset, Example};
pub use evaluators::{
    as_text, extract_number, get_path, parse_judge_reply, Contains, EvalInput, Evaluator,
    ExactMatch, JsonFieldEquals, LlmJudge, NumericTolerance, RegexMatch, Score,
};
pub use hub::{PromptHub, PromptPush, PromptVersion, PushResult};
pub use pairwise::{
    evaluate_pairwise, parse_preference, FnPairwise, LlmPairwiseJudge, PairwiseEvaluator,
    PairwiseInput, PairwiseReport, PairwiseResult, PairwiseScore, PairwiseStats, Preference,
};
pub use summary::{ClassificationF1, FnSummary, PassRate, SummaryEvaluator};
pub use trajectory::{
    json_subset, ArgumentMatch, EfficiencyMetric, NoRepeatedCalls, RelativeEfficiency, ToolStep,
    ToolUsage, Trajectory, TrajectoryBudget, TrajectoryEvaluator, TrajectoryInput, TrajectoryJudge,
    TrajectoryMatch,
};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("line {line}: {message}")]
    Parse { line: usize, message: String },
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid prompt: {0}")]
    Prompt(String),
}

/// Outcome of the target and every evaluator on one example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExampleResult {
    pub example_id: String,
    pub index: usize,
    /// Which repetition this is (0-based); see [`Experiment::repetitions`].
    #[serde(default)]
    pub repetition: usize,
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
    /// Scores of the summary evaluators (one per evaluator, whole experiment).
    #[serde(default)]
    pub summary_scores: Vec<Score>,
    /// How many times each example was run (1 unless set; 0 in old reports).
    #[serde(default)]
    pub repetitions: usize,
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
        for s in &self.summary_scores {
            out.push_str(&format!("\n  {:<24} {:.3} (summary)", s.key, s.score));
        }
        out
    }

    /// Summary score by key.
    pub fn summary_score(&self, key: &str) -> Option<f64> {
        self.summary_scores
            .iter()
            .find(|s| s.key == key)
            .map(|s| s.score)
    }
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Runs `target` on every example (at most `concurrency` at once), scores each
/// output with every evaluator and aggregates. Target errors are recorded on
/// the example (evaluators are skipped for it) and never abort the run.
///
/// For repetitions, summary evaluators, a name or metadata, use
/// [`Experiment`].
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
    run_experiment(target, dataset, evaluators, &[], &[], 1, concurrency).await
}

async fn run_experiment<F, Fut, E>(
    target: F,
    dataset: &Dataset,
    evaluators: &[Box<dyn Evaluator>],
    trajectory_evaluators: &[Box<dyn TrajectoryEvaluator>],
    summary_evaluators: &[Box<dyn SummaryEvaluator>],
    repetitions: usize,
    concurrency: usize,
) -> ExperimentReport
where
    F: Fn(Value) -> Fut,
    Fut: Future<Output = Result<Value, E>>,
    E: Display,
{
    let created_ms = now_ms();
    let target = &target;
    let repetitions = repetitions.max(1);
    let n = dataset.examples.len();
    let jobs = (0..repetitions).flat_map(|rep| {
        dataset
            .examples
            .iter()
            .enumerate()
            .map(move |(index, ex)| (rep, index, ex))
    });
    let mut results: Vec<ExampleResult> = futures::stream::iter(jobs)
        .map(|(repetition, index, ex)| async move {
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
                    if !trajectory_evaluators.is_empty() {
                        let trajectory = Trajectory::from_output(&output);
                        for e in trajectory_evaluators {
                            scores.push(match &trajectory {
                                Some(t) => {
                                    e.evaluate(TrajectoryInput {
                                        inputs: &ex.inputs,
                                        output: &output,
                                        trajectory: t,
                                        expected: ex.expected_trajectory.as_ref(),
                                        reference: ex.reference.as_ref(),
                                    })
                                    .await
                                }
                                None => Score::new(e.name(), 0.0).with_reason(
                                    "output has no trajectory (return Trajectory::into_output)",
                                ),
                            });
                        }
                    }
                    (Some(output), None, scores)
                }
                Err(e) => (None, Some(e.to_string()), Vec::new()),
            };
            ExampleResult {
                example_id: ex.id.clone(),
                index,
                repetition,
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
    results.sort_by_key(|r| r.repetition * n + r.index);
    let aggregates = Aggregates::from_results(&results);
    let summary_scores = summary_evaluators
        .iter()
        .map(|s| s.evaluate(&results))
        .collect();
    ExperimentReport {
        id: uuid::Uuid::new_v4().to_string(),
        name: format!("{}-{}", dataset.name, created_ms),
        dataset: dataset.name.clone(),
        created_ms,
        metadata: Value::Null,
        results,
        aggregates,
        summary_scores,
        repetitions,
    }
}

/// Builder for an experiment with every option: evaluators, summary
/// evaluators, repetitions, a concurrency limit, a name and metadata.
///
/// ```no_run
/// # async fn demo(ds: sinew_eval::Dataset) {
/// use serde_json::{json, Value};
/// use sinew_eval::{Contains, Experiment, PassRate};
///
/// let report = Experiment::new(&ds)
///     .evaluator(Contains::reference())
///     .summary_evaluator(PassRate::new("contains", 1.0))
///     .repetitions(3)          // run every example 3 times (variance)
///     .concurrency(4)          // at most 4 target calls in flight
///     .name("baseline")
///     .run(|inputs: Value| async move { Ok::<_, String>(json!("...")) })
///     .await;
/// # }
/// ```
pub struct Experiment<'a> {
    dataset: &'a Dataset,
    evaluators: Vec<Box<dyn Evaluator>>,
    trajectory_evaluators: Vec<Box<dyn TrajectoryEvaluator>>,
    summary: Vec<Box<dyn SummaryEvaluator>>,
    repetitions: usize,
    concurrency: usize,
    name: Option<String>,
    metadata: Value,
}

impl<'a> Experiment<'a> {
    pub fn new(dataset: &'a Dataset) -> Self {
        Self {
            dataset,
            evaluators: Vec::new(),
            trajectory_evaluators: Vec::new(),
            summary: Vec::new(),
            repetitions: 1,
            concurrency: 4,
            name: None,
            metadata: Value::Null,
        }
    }

    pub fn evaluator(mut self, e: impl Evaluator + 'static) -> Self {
        self.evaluators.push(Box::new(e));
        self
    }

    pub fn evaluators(mut self, list: Vec<Box<dyn Evaluator>>) -> Self {
        self.evaluators.extend(list);
        self
    }

    /// Adds a [`TrajectoryEvaluator`]. The target must return an output that
    /// carries a trajectory ([`Trajectory::into_output`]); the evaluator
    /// receives it with the example's `expected_trajectory`. Outputs without
    /// a trajectory score 0 with a reason.
    ///
    /// ```no_run
    /// # async fn demo(ds: sinew_eval::Dataset) {
    /// use serde_json::{json, Value};
    /// use sinew_eval::{Experiment, NoRepeatedCalls, Trajectory, TrajectoryMatch};
    ///
    /// let report = Experiment::new(&ds)
    ///     .trajectory_evaluator(TrajectoryMatch::subsequence())
    ///     .trajectory_evaluator(NoRepeatedCalls::new())
    ///     .run(|inputs: Value| async move {
    ///         let trajectory = Trajectory::new().tool("search", inputs).with_final_answer("...");
    ///         Ok::<_, String>(trajectory.into_output())
    ///     })
    ///     .await;
    /// println!("{:?}", report.mean("trajectory_subsequence"));
    /// # }
    /// ```
    pub fn trajectory_evaluator(mut self, e: impl TrajectoryEvaluator + 'static) -> Self {
        self.trajectory_evaluators.push(Box::new(e));
        self
    }

    /// Adds several [`TrajectoryEvaluator`]s.
    ///
    /// ```no_run
    /// # fn demo(ds: sinew_eval::Dataset) {
    /// use sinew_eval::{Experiment, ToolUsage, TrajectoryEvaluator, TrajectoryMatch};
    ///
    /// let list: Vec<Box<dyn TrajectoryEvaluator>> =
    ///     vec![Box::new(TrajectoryMatch::exact()), Box::new(ToolUsage::new().forbid("shell"))];
    /// let experiment = Experiment::new(&ds).trajectory_evaluators(list);
    /// # }
    /// ```
    pub fn trajectory_evaluators(mut self, list: Vec<Box<dyn TrajectoryEvaluator>>) -> Self {
        self.trajectory_evaluators.extend(list);
        self
    }

    pub fn summary_evaluator(mut self, s: impl SummaryEvaluator + 'static) -> Self {
        self.summary.push(Box::new(s));
        self
    }

    /// Runs every example `n` times (at least 1). Results carry their
    /// `repetition`; aggregates cover all of them.
    pub fn repetitions(mut self, n: usize) -> Self {
        self.repetitions = n.max(1);
        self
    }

    /// Maximum target calls in flight (default 4).
    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub async fn run<F, Fut, E>(self, target: F) -> ExperimentReport
    where
        F: Fn(Value) -> Fut,
        Fut: Future<Output = Result<Value, E>>,
        E: Display,
    {
        let mut report = run_experiment(
            target,
            self.dataset,
            &self.evaluators,
            &self.trajectory_evaluators,
            &self.summary,
            self.repetitions,
            self.concurrency,
        )
        .await;
        if let Some(n) = self.name {
            report.name = n;
        }
        report.metadata = self.metadata;
        report
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

    #[tokio::test]
    async fn trajectory_evaluators_run_in_experiments() {
        use sinew_core::{mock::ScriptedModel, Message, ToolCall};
        let ds = Dataset::new("weather")
            .with_example(
                Example::new(json!({"city": "Paris"}))
                    .with_id("paris")
                    .with_reference(json!("sunny"))
                    .with_expected_trajectory(
                        Trajectory::new().tool("get_weather", json!({"city": "Paris"})),
                    ),
            )
            .with_example(
                Example::new(json!({"city": "Lyon"}))
                    .with_id("lyon")
                    .with_expected_trajectory(
                        Trajectory::new().tool("get_weather", json!({"city": "Lyon"})),
                    ),
            )
            .with_example(Example::new(json!({"city": "none"})).with_id("plain"));
        let judge =
            Arc::new(ScriptedModel::new((0..3).map(|_| {
                Message::assistant(r#"{"score": 1, "reasoning": "fine"}"#)
            })));
        let report = Experiment::new(&ds)
            .evaluator(JsonFieldEquals::expecting("answer", json!("sunny")))
            .trajectory_evaluator(TrajectoryMatch::exact().arguments(ArgumentMatch::Subset))
            .trajectory_evaluator(NoRepeatedCalls::new())
            .trajectory_evaluator(TrajectoryJudge::new(judge, "ok?"))
            .concurrency(1)
            .run(|inputs: Value| async move {
                let city = inputs["city"].as_str().unwrap_or_default().to_string();
                if city == "none" {
                    return Ok::<_, String>(json!("plain text"));
                }
                // Lyon loops: the same call twice.
                let n = if city == "Lyon" { 2 } else { 1 };
                let mut messages = vec![Message::user(format!("weather in {city}?"))];
                for i in 0..n {
                    messages.push(Message::assistant_with_tools(
                        "",
                        vec![ToolCall {
                            id: format!("c{i}"),
                            name: "get_weather".into(),
                            arguments: json!({"city": city, "units": "metric"}),
                        }],
                    ));
                    messages.push(Message::tool_result(format!("c{i}"), "sunny"));
                }
                messages.push(Message::assistant("sunny"));
                Ok(Trajectory::from_messages(&messages).into_output())
            })
            .await;
        let paris = &report.results[0];
        assert_eq!(paris.score("json_field:answer"), Some(1.0));
        assert_eq!(paris.score("trajectory_exact"), Some(1.0));
        assert_eq!(paris.score("no_repeated_calls"), Some(1.0));
        assert_eq!(paris.score("trajectory_judge"), Some(1.0));
        let lyon = &report.results[1];
        assert_eq!(lyon.score("trajectory_exact"), Some(0.0));
        assert_eq!(lyon.score("no_repeated_calls"), Some(0.0));
        let plain = &report.results[2];
        let s = plain
            .scores
            .iter()
            .find(|s| s.key == "trajectory_exact")
            .unwrap();
        assert_eq!(s.score, 0.0);
        assert!(s.reason.as_deref().unwrap().contains("no trajectory"));
        assert!((report.mean("trajectory_exact").unwrap() - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(report.aggregates.scores["trajectory_judge"].count, 3);
    }

    #[tokio::test]
    async fn repetitions_and_summary_evaluators() {
        let mut ds = Dataset::new("labels");
        ds.push(json!("good movie"), Some(json!("pos")));
        ds.push(json!("bad movie"), Some(json!("neg")));
        ds.push(json!("great"), Some(json!("pos")));
        let calls = Arc::new(AtomicUsize::new(0));
        let report = Experiment::new(&ds)
            .evaluator(ExactMatch::new())
            .summary_evaluator(PassRate::new("exact_match", 1.0))
            .summary_evaluator(ClassificationF1::new("pos"))
            .summary_evaluator(FnSummary::new("count", |r: &[ExampleResult]| {
                Score::new("ignored", r.len() as f64 / 10.0)
            }))
            .repetitions(2)
            .concurrency(2)
            .name("rule-based")
            .metadata(json!({"v": 1}))
            .run(|inputs: Value| {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    let t = inputs.as_str().unwrap_or_default();
                    // "great" is misclassified.
                    Ok::<_, String>(json!(if t.contains("good") { "pos" } else { "neg" }))
                }
            })
            .await;
        assert_eq!(calls.load(Ordering::SeqCst), 6);
        assert_eq!(report.repetitions, 2);
        assert_eq!(report.results.len(), 6);
        assert_eq!(report.results[3].repetition, 1);
        assert_eq!(report.results[3].index, 0);
        assert_eq!(report.name, "rule-based");
        assert!((report.summary_score("pass_rate:exact_match").unwrap() - 4.0 / 6.0).abs() < 1e-9);
        // tp 2, fp 0, fn 2 → precision 1, recall 0.5, F1 2/3.
        assert!((report.summary_score("f1").unwrap() - 2.0 / 3.0).abs() < 1e-9);
        assert!((report.summary_score("count").unwrap() - 0.6).abs() < 1e-9);
        assert!(report.summary().contains("(summary)"));

        // Pairwise against a perfect variant.
        let perfect = Experiment::new(&ds)
            .repetitions(2)
            .name("perfect")
            .run(|inputs: Value| async move {
                let t = inputs.as_str().unwrap_or_default();
                Ok::<_, String>(json!(if t.contains("bad") { "neg" } else { "pos" }))
            })
            .await;
        let evaluators: Vec<Box<dyn PairwiseEvaluator>> = vec![Box::new(FnPairwise::new(
            "correct",
            |i: PairwiseInput<'_>| {
                let r = i.reference.cloned();
                match (Some(i.a.clone()) == r, Some(i.b.clone()) == r) {
                    (true, false) => Some(Preference::A),
                    (false, true) => Some(Preference::B),
                    _ => Some(Preference::Tie),
                }
            },
        ))];
        let pw = evaluate_pairwise(&report, &perfect, &evaluators, 3).await;
        assert_eq!(pw.results.len(), 6);
        let s = &pw.summary["correct"];
        assert_eq!((s.wins_a, s.wins_b, s.ties), (0, 2, 4));
        assert!((s.win_rate_b - 4.0 / 6.0).abs() < 1e-9);
        assert_eq!(pw.experiment_b.name, "perfect");
        let back: PairwiseReport =
            serde_json::from_str(&serde_json::to_string(&pw).unwrap()).unwrap();
        assert_eq!(back, pw);
        assert!(pw.summary_text().contains("B wins 2"));
    }
}
