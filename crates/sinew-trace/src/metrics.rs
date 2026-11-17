//! Prometheus metrics aggregated from finished runs.
//!
//! [`MetricsSink`] is a [`TraceSink`] that turns every finished [`Run`] into
//! counters and histograms, and renders them in the Prometheus text
//! exposition format, version 0.0.4 ([`CONTENT_TYPE`]). It has no
//! dependencies beyond the standard library: a small registry with counters,
//! fixed-bucket [`Histogram`]s and a [`TextEncoder`] that servers can also use
//! to append their own gauges.
//!
//! | Metric | Type | Labels |
//! |---|---|---|
//! | `sinew_runs_total` | counter | `project`, `kind`, `name`, `status` |
//! | `sinew_run_duration_seconds` | histogram | `project`, `kind`, `name` |
//! | `sinew_errors_total` | counter | `project`, `kind`, `name` |
//! | `sinew_tool_calls_total` | counter | `project`, `tool`, `status` |
//! | `sinew_llm_tokens_total` | counter | `project`, `model`, `direction` (`input`, `output`, `cached`) |
//! | `sinew_llm_cost_usd_total` | counter | `project`, `model` |
//! | `sinew_llm_time_to_first_token_seconds` | histogram | `project`, `model` |
//! | `sinew_metrics_label_values_folded_total` | counter | none |
//!
//! Label values:
//!
//! - `kind` is the run kind (`graph`, `node`, `llm`, `tool`, `chain`) and
//!   `status` is `ok` or `error` (the run has an `error`).
//! - `name` is the run name, except for `llm` runs where it is the model
//!   (`metadata.model`, `gen_ai.response.model` or `gen_ai.request.model`,
//!   falling back to the run name). `model` uses the same rule.
//! - `project` comes from [`RunContext::project`], then `metadata.project`,
//!   then the sink's default ([`MetricsSink::project`], `default` if unset).
//!
//! Cardinality is bounded: each label (`project`, `model`, `tool`, and `name`
//! per run kind) keeps at most [`MetricsSink::max_label_values`] distinct
//! values (default [`DEFAULT_MAX_LABEL_VALUES`]); later values are folded
//! into [`OTHER_LABEL`] and counted in
//! `sinew_metrics_label_values_folded_total`. Values longer than 128
//! characters are truncated.
//!
//! Inputs for the LLM metrics:
//!
//! - tokens: [`Run::input_tokens`], [`Run::output_tokens`] and
//!   `metadata.cached_input_tokens` (recorded by [`traced_chat`](crate::traced_chat)
//!   and [`traced_chat_stream`](crate::traced_chat_stream) when the provider
//!   reports prompt-cache reads);
//! - cost: [`RunContext::cost_usd`], else `metadata.cost_usd`, else the
//!   `sinew-core` [price table](sinew_core::pricing) applied to the usage of
//!   an `llm` run with a known model;
//! - time to first token: `metadata.ttft_ms` (recorded by
//!   [`traced_chat_stream`](crate::traced_chat_stream)).
//!
//! Runs without an end time are ignored. Counters only grow; they reset when
//! the process restarts, which Prometheus' `rate()` handles.
//!
//! ```
//! use std::sync::Arc;
//! use serde_json::json;
//! use sinew_trace::{metrics::MetricsSink, RunKind, Tracer};
//!
//! let metrics = Arc::new(MetricsSink::new().project("support-bot"));
//! let tracer = Tracer::new().with_shared_sink(metrics.clone());
//! tracer.start("search", RunKind::Tool, None, json!({})).finish(json!("ok"));
//!
//! let text = metrics.render();
//! assert!(text.contains(
//!     r#"sinew_tool_calls_total{project="support-bot",tool="search",status="ok"} 1"#
//! ));
//! ```

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Write as _,
    sync::Mutex,
};

use serde_json::Value;
use sinew_core::Usage;

use crate::{Run, RunKind, TraceSink};

/// `Content-Type` of the Prometheus text exposition format 0.0.4.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Upper bounds (seconds) of `sinew_run_duration_seconds`.
pub const DURATION_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

/// Upper bounds (seconds) of `sinew_llm_time_to_first_token_seconds`.
pub const TTFT_BUCKETS: &[f64] = &[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];

/// Default number of distinct values kept per label.
pub const DEFAULT_MAX_LABEL_VALUES: usize = 100;

/// Value that replaces label values beyond the cardinality cap.
pub const OTHER_LABEL: &str = "other";

const MAX_LABEL_LEN: usize = 128;

/// Prometheus metric type, written on the `# TYPE` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
}

impl MetricType {
    /// The name used in the exposition format.
    pub fn as_str(self) -> &'static str {
        match self {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
        }
    }
}

/// A histogram with fixed bucket upper bounds.
///
/// A value `v` falls in the first bucket whose bound `le` satisfies
/// `v <= le`, or in the implicit `+Inf` bucket. NaN values are ignored.
///
/// ```
/// use sinew_trace::metrics::Histogram;
///
/// let mut h = Histogram::new(&[0.1, 1.0]);
/// for v in [0.05, 0.1, 0.5, 3.0] {
///     h.observe(v);
/// }
/// assert_eq!(h.count(), 4);
/// assert_eq!(h.cumulative(), vec![(0.1, 2), (1.0, 3), (f64::INFINITY, 4)]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    bounds: Vec<f64>,
    /// Per-bucket counts (not cumulative); the last one is `+Inf`.
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    /// A histogram with these upper bounds. They are sorted, deduplicated,
    /// and non-finite bounds are dropped (`+Inf` is always implicit).
    pub fn new(bounds: &[f64]) -> Self {
        let mut b: Vec<f64> = bounds.iter().copied().filter(|v| v.is_finite()).collect();
        b.sort_by(f64::total_cmp);
        b.dedup();
        let n = b.len();
        Self {
            bounds: b,
            counts: vec![0; n + 1],
            sum: 0.0,
            count: 0,
        }
    }

    /// Records one value.
    pub fn observe(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }
        let i = self.bounds.partition_point(|b| *b < value);
        self.counts[i] += 1;
        self.sum += value;
        self.count += 1;
    }

    /// Bucket upper bounds, without `+Inf`.
    pub fn bounds(&self) -> &[f64] {
        &self.bounds
    }

    /// Number of observed values.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Sum of observed values.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// `(le, cumulative count)` per bucket, ending with `(+Inf, count)`.
    pub fn cumulative(&self) -> Vec<(f64, u64)> {
        let mut acc = 0;
        self.bounds
            .iter()
            .copied()
            .chain(std::iter::once(f64::INFINITY))
            .zip(&self.counts)
            .map(|(le, c)| {
                acc += c;
                (le, acc)
            })
            .collect()
    }
}

/// Writes the Prometheus text exposition format (0.0.4).
///
/// Call [`TextEncoder::header`] once per metric family, then add its
/// samples. Label values and help text are escaped.
///
/// ```
/// use sinew_trace::metrics::{MetricType, TextEncoder};
///
/// let mut enc = TextEncoder::new();
/// enc.header("queue_depth", "Jobs waiting.", MetricType::Gauge);
/// enc.sample("queue_depth", &[("queue", "default")], 3.0);
/// assert_eq!(
///     enc.finish(),
///     "# HELP queue_depth Jobs waiting.\n# TYPE queue_depth gauge\nqueue_depth{queue=\"default\"} 3\n"
/// );
/// ```
#[derive(Debug, Default, Clone)]
pub struct TextEncoder {
    out: String,
}

impl TextEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes the `# HELP` and `# TYPE` lines of a metric family.
    pub fn header(&mut self, name: &str, help: &str, kind: MetricType) {
        let help = help.replace('\\', "\\\\").replace('\n', "\\n");
        let _ = writeln!(self.out, "# HELP {name} {help}");
        let _ = writeln!(self.out, "# TYPE {name} {}", kind.as_str());
    }

    /// Writes one sample line.
    pub fn sample(&mut self, name: &str, labels: &[(&str, &str)], value: f64) {
        self.out.push_str(name);
        self.labels(labels, None);
        self.out.push(' ');
        push_value(&mut self.out, value);
        self.out.push('\n');
    }

    /// Writes the `_bucket`, `_sum` and `_count` samples of a histogram.
    pub fn histogram(&mut self, name: &str, labels: &[(&str, &str)], h: &Histogram) {
        let mut le = String::new();
        for (bound, n) in h.cumulative() {
            le.clear();
            push_value(&mut le, bound);
            let _ = write!(self.out, "{name}_bucket");
            self.labels(labels, Some(&le));
            let _ = writeln!(self.out, " {n}");
        }
        let _ = write!(self.out, "{name}_sum");
        self.labels(labels, None);
        self.out.push(' ');
        push_value(&mut self.out, h.sum());
        self.out.push('\n');
        let _ = write!(self.out, "{name}_count");
        self.labels(labels, None);
        let _ = writeln!(self.out, " {}", h.count());
    }

    /// The text written so far.
    pub fn as_str(&self) -> &str {
        &self.out
    }

    /// The finished exposition.
    pub fn finish(self) -> String {
        self.out
    }

    fn labels(&mut self, labels: &[(&str, &str)], le: Option<&str>) {
        if labels.is_empty() && le.is_none() {
            return;
        }
        self.out.push('{');
        let mut first = true;
        for (k, v) in labels.iter().copied().chain(le.map(|l| ("le", l))) {
            if !first {
                self.out.push(',');
            }
            first = false;
            self.out.push_str(k);
            self.out.push_str("=\"");
            for c in v.chars() {
                match c {
                    '\\' => self.out.push_str("\\\\"),
                    '"' => self.out.push_str("\\\""),
                    '\n' => self.out.push_str("\\n"),
                    c => self.out.push(c),
                }
            }
            self.out.push('"');
        }
        self.out.push('}');
    }
}

fn push_value(out: &mut String, v: f64) {
    if v.is_nan() {
        out.push_str("NaN");
    } else if v == f64::INFINITY {
        out.push_str("+Inf");
    } else if v == f64::NEG_INFINITY {
        out.push_str("-Inf");
    } else {
        let _ = write!(out, "{v}");
    }
}

/// Per-run extras for [`MetricsSink::record_with`].
///
/// ```
/// use sinew_trace::metrics::RunContext;
///
/// let ctx = RunContext::new().project("prod").cost_usd(Some(0.0042));
/// # let _ = ctx;
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunContext {
    project: Option<String>,
    cost_usd: Option<Option<f64>>,
}

impl RunContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// The project the run belongs to; overrides `metadata.project` and the
    /// sink's default.
    pub fn project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// The run's cost in USD, computed by the caller (for example with its
    /// own price table). `None` records no cost. When this is not called,
    /// the sink uses `metadata.cost_usd` or the `sinew-core` price table.
    pub fn cost_usd(mut self, cost: Option<f64>) -> Self {
        self.cost_usd = Some(cost);
        self
    }
}

type Key<const N: usize> = [String; N];

#[derive(Default)]
struct State {
    runs: BTreeMap<Key<4>, f64>,
    durations: BTreeMap<Key<3>, Histogram>,
    errors: BTreeMap<Key<3>, f64>,
    tools: BTreeMap<Key<3>, f64>,
    tokens: BTreeMap<Key<3>, f64>,
    cost: BTreeMap<Key<2>, f64>,
    ttft: BTreeMap<Key<2>, Histogram>,
    seen: HashMap<String, HashSet<String>>,
    folded: u64,
}

impl State {
    /// Bounds the cardinality of one label.
    fn guard(&mut self, label: &str, value: &str, max: usize) -> String {
        let value = match value.char_indices().nth(MAX_LABEL_LEN) {
            Some((i, _)) => &value[..i],
            None => value,
        };
        let set = self.seen.entry(label.to_string()).or_default();
        if set.contains(value) {
            return value.to_string();
        }
        if set.len() < max {
            set.insert(value.to_string());
            return value.to_string();
        }
        self.folded += 1;
        OTHER_LABEL.to_string()
    }
}

/// A [`TraceSink`] that aggregates finished runs into Prometheus metrics.
/// See the [module documentation](self) for the metrics and labels.
pub struct MetricsSink {
    state: Mutex<State>,
    project: String,
    max_label_values: usize,
}

impl Default for MetricsSink {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsSink {
    /// An empty sink: project `default`, at most
    /// [`DEFAULT_MAX_LABEL_VALUES`] values per label.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::default()),
            project: "default".into(),
            max_label_values: DEFAULT_MAX_LABEL_VALUES,
        }
    }

    /// The `project` label of runs that carry no project of their own.
    pub fn project(mut self, project: impl Into<String>) -> Self {
        self.project = project.into();
        self
    }

    /// The maximum number of distinct values per label (at least 1); later
    /// values become [`OTHER_LABEL`].
    pub fn max_label_values(mut self, max: usize) -> Self {
        self.max_label_values = max.max(1);
        self
    }

    /// Aggregates `run` with caller-provided extras (project, cost).
    pub fn record_with(&self, run: &Run, ctx: &RunContext) {
        let Some(duration_ms) = run.duration_ms() else {
            return;
        };
        let max = self.max_label_values;
        let kind = kind_str(run.kind);
        let status = if run.error.is_some() { "error" } else { "ok" };
        let model = (run.kind == RunKind::Llm).then(|| run_model(run));
        let raw_project = ctx
            .project
            .as_deref()
            .or_else(|| run.metadata.get("project").and_then(Value::as_str))
            .filter(|p| !p.is_empty())
            .unwrap_or(&self.project);

        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let project = s.guard("project", raw_project, max);
        let name = s.guard(
            &format!("name:{kind}"),
            model.as_deref().unwrap_or(&run.name),
            max,
        );
        let base = [project.clone(), kind.to_string(), name.clone()];

        *s.runs
            .entry([
                project.clone(),
                kind.to_string(),
                name.clone(),
                status.to_string(),
            ])
            .or_default() += 1.0;
        s.durations
            .entry(base.clone())
            .or_insert_with(|| Histogram::new(DURATION_BUCKETS))
            .observe(duration_ms as f64 / 1000.0);
        if run.error.is_some() {
            *s.errors.entry(base).or_default() += 1.0;
        }

        if run.kind == RunKind::Tool {
            let tool = s.guard("tool", &run.name, max);
            *s.tools
                .entry([project.clone(), tool, status.to_string()])
                .or_default() += 1.0;
        }

        let Some(raw_model) = model else {
            return;
        };
        let model = s.guard("model", &raw_model, max);
        let cached = meta_u32(&run.metadata, "cached_input_tokens");
        for (direction, n) in [
            ("input", run.input_tokens),
            ("output", run.output_tokens),
            ("cached", cached),
        ] {
            if let Some(n) = n {
                *s.tokens
                    .entry([project.clone(), model.clone(), direction.to_string()])
                    .or_default() += f64::from(n);
            }
        }
        let cost = match ctx.cost_usd {
            Some(c) => c,
            None => run_cost(run, &raw_model, cached),
        };
        if let Some(c) = cost.filter(|c| c.is_finite() && *c >= 0.0) {
            *s.cost.entry([project.clone(), model.clone()]).or_default() += c;
        }
        if let Some(ttft) = run
            .metadata
            .get("ttft_ms")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && *v >= 0.0)
        {
            s.ttft
                .entry([project, model])
                .or_insert_with(|| Histogram::new(TTFT_BUCKETS))
                .observe(ttft / 1000.0);
        }
    }

    /// Appends every metric family to `enc`.
    pub fn encode(&self, enc: &mut TextEncoder) {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        counter_family(
            enc,
            "sinew_runs_total",
            "Finished runs.",
            &["project", "kind", "name", "status"],
            &s.runs,
        );
        enc.header(
            "sinew_run_duration_seconds",
            "Run duration in seconds.",
            MetricType::Histogram,
        );
        for (k, h) in &s.durations {
            enc.histogram(
                "sinew_run_duration_seconds",
                &labels(&["project", "kind", "name"], k),
                h,
            );
        }
        counter_family(
            enc,
            "sinew_errors_total",
            "Runs that finished with an error.",
            &["project", "kind", "name"],
            &s.errors,
        );
        counter_family(
            enc,
            "sinew_tool_calls_total",
            "Tool calls.",
            &["project", "tool", "status"],
            &s.tools,
        );
        counter_family(
            enc,
            "sinew_llm_tokens_total",
            "LLM tokens by direction (input, output, cached; cached is a subset of input).",
            &["project", "model", "direction"],
            &s.tokens,
        );
        counter_family(
            enc,
            "sinew_llm_cost_usd_total",
            "Estimated LLM cost in US dollars.",
            &["project", "model"],
            &s.cost,
        );
        enc.header(
            "sinew_llm_time_to_first_token_seconds",
            "Time from the LLM request to the first streamed token, in seconds.",
            MetricType::Histogram,
        );
        for (k, h) in &s.ttft {
            enc.histogram(
                "sinew_llm_time_to_first_token_seconds",
                &labels(&["project", "model"], k),
                h,
            );
        }
        enc.header(
            "sinew_metrics_label_values_folded_total",
            "Label values replaced by \"other\" because a label reached its cardinality cap.",
            MetricType::Counter,
        );
        enc.sample(
            "sinew_metrics_label_values_folded_total",
            &[],
            s.folded as f64,
        );
    }

    /// The metrics in the text exposition format.
    pub fn render(&self) -> String {
        let mut enc = TextEncoder::new();
        self.encode(&mut enc);
        enc.finish()
    }
}

impl TraceSink for MetricsSink {
    fn record(&self, run: &Run) {
        self.record_with(run, &RunContext::default());
    }
}

fn labels<'a, const N: usize>(names: &[&'a str], values: &'a Key<N>) -> Vec<(&'a str, &'a str)> {
    names
        .iter()
        .copied()
        .zip(values.iter().map(String::as_str))
        .collect()
}

fn counter_family<const N: usize>(
    enc: &mut TextEncoder,
    name: &str,
    help: &str,
    names: &[&str],
    series: &BTreeMap<Key<N>, f64>,
) {
    enc.header(name, help, MetricType::Counter);
    for (k, v) in series {
        enc.sample(name, &labels(names, k), *v);
    }
}

fn kind_str(kind: RunKind) -> &'static str {
    match kind {
        RunKind::Graph => "graph",
        RunKind::Node => "node",
        RunKind::Llm => "llm",
        RunKind::Tool => "tool",
        RunKind::Chain => "chain",
    }
}

fn run_model(run: &Run) -> String {
    ["model", "gen_ai.response.model", "gen_ai.request.model"]
        .iter()
        .find_map(|k| {
            run.metadata
                .get(*k)
                .and_then(Value::as_str)
                .filter(|m| !m.is_empty())
        })
        .unwrap_or(&run.name)
        .to_string()
}

fn meta_u32(meta: &Value, key: &str) -> Option<u32> {
    meta.get(key)
        .and_then(Value::as_u64)
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
}

fn run_cost(run: &Run, model: &str, cached: Option<u32>) -> Option<f64> {
    if let Some(c) = run.metadata.get("cost_usd").and_then(Value::as_f64) {
        return Some(c);
    }
    if run.input_tokens.is_none() && run.output_tokens.is_none() {
        return None;
    }
    let usage = Usage {
        input_tokens: run.input_tokens.unwrap_or(0),
        output_tokens: run.output_tokens.unwrap_or(0),
        cached_input_tokens: cached.unwrap_or(0),
        cache_creation_input_tokens: meta_u32(&run.metadata, "cache_creation_input_tokens")
            .unwrap_or(0),
        ..Default::default()
    };
    usage.cost(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(kind: RunKind, name: &str, ms: u64) -> Run {
        Run {
            id: "r".into(),
            trace_id: "t".into(),
            parent_id: None,
            name: name.into(),
            kind,
            start_ms: 1_000,
            end_ms: Some(1_000 + ms),
            inputs: json!({}),
            outputs: None,
            error: None,
            input_tokens: None,
            output_tokens: None,
            metadata: json!({}),
        }
    }

    #[test]
    fn histogram_bucket_math() {
        let mut h = Histogram::new(&[1.0, 0.5, f64::INFINITY, 0.5, 2.0]);
        assert_eq!(h.bounds(), &[0.5, 1.0, 2.0]);
        // Bounds are inclusive upper limits.
        for v in [0.0, 0.5, 0.500_001, 1.0, 2.0, 2.5, 100.0, f64::NAN] {
            h.observe(v);
        }
        assert_eq!(h.count(), 7);
        assert!((h.sum() - 106.500_001).abs() < 1e-9);
        assert_eq!(
            h.cumulative(),
            vec![(0.5, 2), (1.0, 4), (2.0, 5), (f64::INFINITY, 7)]
        );
        let empty = Histogram::new(&[]);
        assert_eq!(empty.cumulative(), vec![(f64::INFINITY, 0)]);
    }

    #[test]
    fn encoder_escapes_and_formats() {
        let mut enc = TextEncoder::new();
        enc.header("m", "a \\ b\nc", MetricType::Gauge);
        enc.sample("m", &[("l", "x\"y\\z\nw")], f64::NAN);
        enc.sample("m", &[], f64::INFINITY);
        enc.sample("m", &[("a", "1"), ("b", "2")], -0.25);
        assert_eq!(
            enc.as_str(),
            "# HELP m a \\\\ b\\nc\n# TYPE m gauge\n\
             m{l=\"x\\\"y\\\\z\\nw\"} NaN\n\
             m +Inf\n\
             m{a=\"1\",b=\"2\"} -0.25\n"
        );
    }

    #[test]
    fn exposition_golden() {
        let sink = MetricsSink::new().project("demo");
        let mut llm = run(RunKind::Llm, "openai", 1_500);
        llm.input_tokens = Some(1_000);
        llm.output_tokens = Some(200);
        llm.metadata = json!({"model": "gpt-4o-mini", "ttft_ms": 300, "cached_input_tokens": 400});
        sink.record(&llm);
        let mut tool = run(RunKind::Tool, "search", 20);
        tool.error = Some("timeout".into());
        sink.record(&tool);
        // Unfinished runs are ignored.
        let mut open = run(RunKind::Node, "x", 0);
        open.end_ms = None;
        sink.record(&open);

        // gpt-4o-mini from the sinew-core price table, with 400 cached prompt tokens.
        let cost = Usage {
            input_tokens: 1_000,
            output_tokens: 200,
            cached_input_tokens: 400,
            ..Default::default()
        }
        .cost("gpt-4o-mini")
        .unwrap();
        let expected = format!(
            r#"# HELP sinew_runs_total Finished runs.
# TYPE sinew_runs_total counter
sinew_runs_total{{project="demo",kind="llm",name="gpt-4o-mini",status="ok"}} 1
sinew_runs_total{{project="demo",kind="tool",name="search",status="error"}} 1
# HELP sinew_run_duration_seconds Run duration in seconds.
# TYPE sinew_run_duration_seconds histogram
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="0.005"}} 0
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="0.01"}} 0
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="0.025"}} 0
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="0.05"}} 0
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="0.1"}} 0
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="0.25"}} 0
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="0.5"}} 0
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="1"}} 0
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="2.5"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="5"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="10"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="30"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="60"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="120"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="300"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="llm",name="gpt-4o-mini",le="+Inf"}} 1
sinew_run_duration_seconds_sum{{project="demo",kind="llm",name="gpt-4o-mini"}} 1.5
sinew_run_duration_seconds_count{{project="demo",kind="llm",name="gpt-4o-mini"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="0.005"}} 0
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="0.01"}} 0
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="0.025"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="0.05"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="0.1"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="0.25"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="0.5"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="1"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="2.5"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="5"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="10"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="30"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="60"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="120"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="300"}} 1
sinew_run_duration_seconds_bucket{{project="demo",kind="tool",name="search",le="+Inf"}} 1
sinew_run_duration_seconds_sum{{project="demo",kind="tool",name="search"}} 0.02
sinew_run_duration_seconds_count{{project="demo",kind="tool",name="search"}} 1
# HELP sinew_errors_total Runs that finished with an error.
# TYPE sinew_errors_total counter
sinew_errors_total{{project="demo",kind="tool",name="search"}} 1
# HELP sinew_tool_calls_total Tool calls.
# TYPE sinew_tool_calls_total counter
sinew_tool_calls_total{{project="demo",tool="search",status="error"}} 1
# HELP sinew_llm_tokens_total LLM tokens by direction (input, output, cached; cached is a subset of input).
# TYPE sinew_llm_tokens_total counter
sinew_llm_tokens_total{{project="demo",model="gpt-4o-mini",direction="cached"}} 400
sinew_llm_tokens_total{{project="demo",model="gpt-4o-mini",direction="input"}} 1000
sinew_llm_tokens_total{{project="demo",model="gpt-4o-mini",direction="output"}} 200
# HELP sinew_llm_cost_usd_total Estimated LLM cost in US dollars.
# TYPE sinew_llm_cost_usd_total counter
sinew_llm_cost_usd_total{{project="demo",model="gpt-4o-mini"}} {cost}
# HELP sinew_llm_time_to_first_token_seconds Time from the LLM request to the first streamed token, in seconds.
# TYPE sinew_llm_time_to_first_token_seconds histogram
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="0.05"}} 0
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="0.1"}} 0
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="0.25"}} 0
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="0.5"}} 1
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="0.75"}} 1
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="1"}} 1
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="1.5"}} 1
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="2"}} 1
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="3"}} 1
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="5"}} 1
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="10"}} 1
sinew_llm_time_to_first_token_seconds_bucket{{project="demo",model="gpt-4o-mini",le="+Inf"}} 1
sinew_llm_time_to_first_token_seconds_sum{{project="demo",model="gpt-4o-mini"}} 0.3
sinew_llm_time_to_first_token_seconds_count{{project="demo",model="gpt-4o-mini"}} 1
# HELP sinew_metrics_label_values_folded_total Label values replaced by "other" because a label reached its cardinality cap.
# TYPE sinew_metrics_label_values_folded_total counter
sinew_metrics_label_values_folded_total 0
"#
        );
        assert_eq!(sink.render(), expected);
    }

    #[test]
    fn cardinality_cap_folds_into_other() {
        let sink = MetricsSink::new().max_label_values(2);
        for name in ["a", "b", "c", "d", "a"] {
            sink.record(&run(RunKind::Tool, name, 1));
        }
        let long = "x".repeat(300);
        sink.record_with(
            &run(RunKind::Node, &long, 1),
            &RunContext::new().project("p2"),
        );
        let text = sink.render();
        assert!(
            text.contains(r#"sinew_tool_calls_total{project="default",tool="a",status="ok"} 2"#)
        );
        assert!(
            text.contains(r#"sinew_tool_calls_total{project="default",tool="b",status="ok"} 1"#)
        );
        assert!(
            text.contains(
                r#"sinew_tool_calls_total{project="default",tool="other",status="ok"} 2"#
            ),
            "{text}"
        );
        assert!(!text.contains("tool=\"c\""));
        // `name` is capped per kind: the node name is still kept, truncated.
        let truncated = format!("name=\"{}\"", "x".repeat(MAX_LABEL_LEN));
        assert!(text.contains(&truncated));
        assert!(!text.contains(&format!("name=\"{}\"", "x".repeat(MAX_LABEL_LEN + 1))));
        // 2 tool names + 2 `name:tool` values folded.
        assert!(
            text.contains("sinew_metrics_label_values_folded_total 4"),
            "{text}"
        );
    }

    #[test]
    fn context_overrides_project_and_cost() {
        let sink = MetricsSink::new();
        let mut llm = run(RunKind::Llm, "local", 10);
        llm.input_tokens = Some(10);
        llm.output_tokens = Some(5);
        llm.metadata = json!({"project": "meta-project", "cost_usd": 0.5});
        sink.record(&llm);
        sink.record_with(&llm, &RunContext::new().project("ctx").cost_usd(None));
        sink.record_with(&llm, &RunContext::new().project("ctx").cost_usd(Some(0.25)));
        let text = sink.render();
        assert!(
            text.contains(r#"sinew_llm_cost_usd_total{project="meta-project",model="local"} 0.5"#)
        );
        assert!(text.contains(r#"sinew_llm_cost_usd_total{project="ctx",model="local"} 0.25"#));
        // Unknown model without explicit cost: no cost series.
        let other = MetricsSink::new();
        let mut unknown = run(RunKind::Llm, "my-local-model", 10);
        unknown.input_tokens = Some(10);
        other.record(&unknown);
        assert!(!other.render().contains("sinew_llm_cost_usd_total{"));
    }
}
