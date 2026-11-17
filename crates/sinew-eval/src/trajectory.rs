//! Agent trajectory evaluation: score the path an agent took, not only its
//! final answer.
//!
//! A [`Trajectory`] is the ordered list of tool calls an agent made (name and
//! arguments, plus the result when known), the number of model calls, token
//! usage and cost when known, and the final answer. Build one from the
//! agent's messages ([`Trajectory::from_messages`],
//! [`Trajectory::from_last_turn`]) or from a trace ([`Trajectory::from_runs`]).
//!
//! [`TrajectoryEvaluator`]s score a trajectory, optionally against the
//! expected trajectory stored on the dataset [`Example`](crate::Example):
//!
//! | Evaluator | Key | Passes when |
//! |---|---|---|
//! | [`TrajectoryMatch::exact`] | `trajectory_exact` | same tool calls in the same order |
//! | [`TrajectoryMatch::unordered`] | `trajectory_unordered` | same tool calls in any order |
//! | [`TrajectoryMatch::subsequence`] | `trajectory_subsequence` | expected calls appear in order, extra calls allowed |
//! | [`TrajectoryMatch::superset`] | `trajectory_superset` | every expected call was made (any order, extras allowed) |
//! | [`TrajectoryMatch::subset`] | `trajectory_subset` | every call made is one of the expected calls |
//! | [`ToolUsage`] | `tool_usage` | required tools used, forbidden tools not used |
//! | [`TrajectoryBudget`] | `trajectory_budget` | steps, tool calls, tokens and cost within limits |
//! | [`RelativeEfficiency`] | `trajectory_efficiency` | `min(1, reference / actual)` for one metric |
//! | [`NoRepeatedCalls`] | `no_repeated_calls` | no identical call is repeated |
//! | [`TrajectoryJudge`] | `trajectory_judge` | graded by any [`ChatModel`] |
//!
//! In an [`Experiment`](crate::Experiment), the target returns
//! [`Trajectory::into_output`] (an object with `answer` and `trajectory`)
//! and trajectory evaluators are added with
//! [`Experiment::trajectory_evaluator`](crate::Experiment::trajectory_evaluator).
//!
//! ```
//! # async fn demo() {
//! use serde_json::json;
//! use sinew_eval::{Trajectory, TrajectoryEvaluator, TrajectoryInput, TrajectoryMatch};
//!
//! let actual = Trajectory::new()
//!     .tool("search", json!({"q": "weather Paris", "lang": "en"}))
//!     .tool("get_weather", json!({"city": "Paris"}))
//!     .with_final_answer("Sunny, 21 °C.");
//! let expected = Trajectory::new().tool("get_weather", json!({"city": "Paris"}));
//!
//! let score = TrajectoryMatch::subsequence()
//!     .evaluate(TrajectoryInput::new(&actual).with_expected(&expected))
//!     .await;
//! assert_eq!(score.score, 1.0);
//! # }
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{ChatModel, ChatRequest, Message, Role, StructuredMode, StructuredOutput, Usage};
use sinew_trace::{Run, RunKind};

use crate::evaluators::{as_text, Score};

/// One tool call of a trajectory.
///
/// In an expected trajectory, `arguments` set to `null` matches any
/// arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolStep {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
    /// The tool result text, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// The tool error, when the call failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolStep {
    /// A step with a name and arguments.
    ///
    /// ```
    /// let step = sinew_eval::ToolStep::new("search", serde_json::json!({"q": "rust"}));
    /// assert_eq!(step.name, "search");
    /// ```
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
            result: None,
            error: None,
        }
    }

    /// A step that matches any arguments when used in an expected trajectory.
    ///
    /// ```
    /// let step = sinew_eval::ToolStep::named("search");
    /// assert!(step.arguments.is_null());
    /// ```
    pub fn named(name: impl Into<String>) -> Self {
        Self::new(name, Value::Null)
    }

    /// Sets the result text.
    ///
    /// ```
    /// let step = sinew_eval::ToolStep::named("clock").with_result("12:00");
    /// assert_eq!(step.result.as_deref(), Some("12:00"));
    /// ```
    pub fn with_result(mut self, result: impl Into<String>) -> Self {
        self.result = Some(result.into());
        self
    }

    /// Sets the error text.
    ///
    /// ```
    /// let step = sinew_eval::ToolStep::named("clock").with_error("timeout");
    /// assert_eq!(step.error.as_deref(), Some("timeout"));
    /// ```
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }
}

/// The path an agent took: tool calls in order, model calls, usage, cost and
/// the final answer. Unknown usage and cost are `None`.
///
/// Serialized form (used in datasets as `expected_trajectory` and in target
/// outputs under `trajectory`):
///
/// ```json
/// {"tool_calls": [{"name": "search", "arguments": {"q": "rust"}}],
///  "model_calls": 2, "input_tokens": 120, "output_tokens": 30, "final_answer": "..."}
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    #[serde(default)]
    pub tool_calls: Vec<ToolStep>,
    /// Model calls (agent steps).
    #[serde(default)]
    pub model_calls: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Estimated cost in USD.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_answer: Option<String>,
}

impl Trajectory {
    /// An empty trajectory; add steps with [`Trajectory::tool`].
    ///
    /// ```
    /// assert!(sinew_eval::Trajectory::new().tool_calls.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a tool call.
    ///
    /// ```
    /// use serde_json::json;
    /// let t = sinew_eval::Trajectory::new().tool("add", json!({"a": 1, "b": 2}));
    /// assert_eq!(t.tool_names(), vec!["add"]);
    /// ```
    pub fn tool(mut self, name: impl Into<String>, arguments: Value) -> Self {
        self.tool_calls.push(ToolStep::new(name, arguments));
        self
    }

    /// Appends a step.
    ///
    /// ```
    /// use sinew_eval::{ToolStep, Trajectory};
    /// let t = Trajectory::new().step(ToolStep::named("search"));
    /// assert_eq!(t.tool_calls.len(), 1);
    /// ```
    pub fn step(mut self, step: ToolStep) -> Self {
        self.tool_calls.push(step);
        self
    }

    /// Sets the number of model calls.
    ///
    /// ```
    /// assert_eq!(sinew_eval::Trajectory::new().with_model_calls(3).model_calls, 3);
    /// ```
    pub fn with_model_calls(mut self, n: u32) -> Self {
        self.model_calls = n;
        self
    }

    /// Sets token usage.
    ///
    /// ```
    /// let t = sinew_eval::Trajectory::new().with_usage(100, 20);
    /// assert_eq!(t.total_tokens(), Some(120));
    /// ```
    pub fn with_usage(mut self, input_tokens: u64, output_tokens: u64) -> Self {
        self.input_tokens = Some(input_tokens);
        self.output_tokens = Some(output_tokens);
        self
    }

    /// Sets the cost in USD.
    ///
    /// ```
    /// assert_eq!(sinew_eval::Trajectory::new().with_cost(0.01).cost_usd, Some(0.01));
    /// ```
    pub fn with_cost(mut self, usd: f64) -> Self {
        self.cost_usd = Some(usd);
        self
    }

    /// Sets the final answer.
    ///
    /// ```
    /// let t = sinew_eval::Trajectory::new().with_final_answer("42");
    /// assert_eq!(t.final_answer.as_deref(), Some("42"));
    /// ```
    pub fn with_final_answer(mut self, answer: impl Into<String>) -> Self {
        self.final_answer = Some(answer.into());
        self
    }

    /// Tool names in call order.
    ///
    /// ```
    /// use serde_json::json;
    /// let t = sinew_eval::Trajectory::new().tool("a", json!({})).tool("b", json!({}));
    /// assert_eq!(t.tool_names(), vec!["a", "b"]);
    /// ```
    pub fn tool_names(&self) -> Vec<&str> {
        self.tool_calls.iter().map(|s| s.name.as_str()).collect()
    }

    /// `input_tokens + output_tokens`, or `None` when neither is known.
    ///
    /// ```
    /// assert_eq!(sinew_eval::Trajectory::new().total_tokens(), None);
    /// ```
    pub fn total_tokens(&self) -> Option<u64> {
        match (self.input_tokens, self.output_tokens) {
            (None, None) => None,
            (i, o) => Some(i.unwrap_or(0) + o.unwrap_or(0)),
        }
    }

    /// Builds a trajectory from a conversation (`AgentState.messages`, the
    /// harness thread messages, ...).
    ///
    /// Every assistant message counts as one model call; its tool calls are
    /// appended in order, with the result (or error, for results starting
    /// with `error: `) taken from the tool message that answers the call id.
    /// The final answer is the text of the last assistant message when it
    /// has no tool calls. Usage and cost stay unknown; set them with
    /// [`Trajectory::with_usage`] and [`Trajectory::with_cost`].
    ///
    /// ```
    /// use serde_json::json;
    /// use sinew_core::{Message, ToolCall};
    /// use sinew_eval::Trajectory;
    ///
    /// let messages = vec![
    ///     Message::user("2 + 3?"),
    ///     Message::assistant_with_tools("", vec![ToolCall {
    ///         id: "c1".into(), name: "add".into(), arguments: json!({"a": 2, "b": 3}),
    ///     }]),
    ///     Message::tool_result("c1", "5"),
    ///     Message::assistant("5"),
    /// ];
    /// let t = Trajectory::from_messages(&messages);
    /// assert_eq!(t.tool_names(), vec!["add"]);
    /// assert_eq!(t.tool_calls[0].result.as_deref(), Some("5"));
    /// assert_eq!(t.model_calls, 2);
    /// assert_eq!(t.final_answer.as_deref(), Some("5"));
    /// ```
    pub fn from_messages(messages: &[Message]) -> Self {
        let mut t = Self::new();
        let mut last_assistant: Option<&Message> = None;
        for (i, m) in messages.iter().enumerate() {
            if m.role != Role::Assistant {
                continue;
            }
            t.model_calls += 1;
            last_assistant = Some(m);
            for call in &m.tool_calls {
                let mut step = ToolStep::new(call.name.clone(), call.arguments.clone());
                let answer = messages[i + 1..].iter().find(|r| {
                    r.role == Role::Tool && r.tool_call_id.as_deref() == Some(call.id.as_str())
                });
                if let Some(r) = answer {
                    match r.content.strip_prefix("error: ") {
                        Some(e) => step.error = Some(e.to_string()),
                        None => step.result = Some(r.content.clone()),
                    }
                }
                t.tool_calls.push(step);
            }
        }
        t.final_answer = last_assistant
            .filter(|m| m.tool_calls.is_empty())
            .map(Message::text);
        t
    }

    /// Like [`Trajectory::from_messages`], over the messages after the last
    /// user message only (the latest turn of a multi-turn thread).
    ///
    /// ```
    /// use sinew_core::Message;
    /// let messages = vec![
    ///     Message::user("hi"), Message::assistant("hello"),
    ///     Message::user("bye"), Message::assistant("goodbye"),
    /// ];
    /// let t = sinew_eval::Trajectory::from_last_turn(&messages);
    /// assert_eq!(t.model_calls, 1);
    /// assert_eq!(t.final_answer.as_deref(), Some("goodbye"));
    /// ```
    pub fn from_last_turn(messages: &[Message]) -> Self {
        let start = messages
            .iter()
            .rposition(|m| m.role == Role::User)
            .map_or(0, |i| i + 1);
        Self::from_messages(&messages[start..])
    }

    /// Builds a trajectory from the runs of one trace (for example
    /// `MemorySink::runs()` after a traced agent run), ordered by start time.
    ///
    /// `tool` runs become steps (inputs as arguments, outputs as result,
    /// `error` as error). Every `llm` run is a model call; tokens are summed
    /// over the runs that report them. The cost is the sum of
    /// `metadata.cost_usd`, or of the built-in price table applied to the
    /// run's tokens and model, and stays `None` if any model call has no
    /// known cost. The final answer is the content of the last `llm` run's
    /// output message when it has no tool calls.
    ///
    /// ```
    /// use serde_json::json;
    /// use sinew_trace::{MemorySink, RunKind, Tracer};
    /// use std::sync::Arc;
    ///
    /// let sink = Arc::new(MemorySink::new());
    /// let tracer = Tracer::new().with_shared_sink(sink.clone());
    /// tracer.start("search", RunKind::Tool, None, json!({"q": "rust"})).finish(json!("found"));
    /// let t = sinew_eval::Trajectory::from_runs(&sink.runs());
    /// assert_eq!(t.tool_names(), vec!["search"]);
    /// assert_eq!(t.tool_calls[0].result.as_deref(), Some("found"));
    /// ```
    pub fn from_runs(runs: &[Run]) -> Self {
        let mut runs: Vec<&Run> = runs.iter().collect();
        runs.sort_by_key(|r| r.start_ms);
        let mut t = Self::new();
        let mut cost = Some(0.0);
        let mut last_llm: Option<&Run> = None;
        for r in runs {
            match r.kind {
                RunKind::Tool => t.tool_calls.push(ToolStep {
                    name: r.name.clone(),
                    arguments: r.inputs.clone(),
                    result: if r.error.is_some() {
                        None
                    } else {
                        r.outputs.as_ref().map(as_text)
                    },
                    error: r.error.clone(),
                }),
                RunKind::Llm => {
                    t.model_calls += 1;
                    last_llm = Some(r);
                    if let Some(n) = r.input_tokens {
                        *t.input_tokens.get_or_insert(0) += u64::from(n);
                    }
                    if let Some(n) = r.output_tokens {
                        *t.output_tokens.get_or_insert(0) += u64::from(n);
                    }
                    let run_cost = r
                        .metadata
                        .get("cost_usd")
                        .and_then(Value::as_f64)
                        .filter(|c| c.is_finite() && *c >= 0.0)
                        .or_else(|| {
                            let model = r
                                .metadata
                                .get("model")
                                .and_then(Value::as_str)
                                .unwrap_or(&r.name);
                            Usage::new(r.input_tokens?, r.output_tokens.unwrap_or(0)).cost(model)
                        });
                    cost = match (cost, run_cost) {
                        (Some(a), Some(b)) => Some(a + b),
                        _ => None,
                    };
                }
                _ => {}
            }
        }
        t.cost_usd = if t.model_calls == 0 { None } else { cost };
        t.final_answer = last_llm
            .and_then(|r| r.outputs.as_ref())
            .and_then(|o| serde_json::from_value::<Message>(o.clone()).ok())
            .filter(|m| m.tool_calls.is_empty())
            .map(|m| m.text());
        t
    }

    /// The target output convention for experiments:
    /// `{"answer": <final answer or null>, "trajectory": <this trajectory>}`.
    /// Output evaluators can read the answer with `JsonFieldEquals::new("answer")`.
    ///
    /// ```
    /// let out = sinew_eval::Trajectory::new().with_final_answer("42").into_output();
    /// assert_eq!(out["answer"], "42");
    /// assert!(sinew_eval::Trajectory::from_output(&out).is_some());
    /// ```
    pub fn into_output(self) -> Value {
        json!({
            "answer": self.final_answer,
            "trajectory": self,
        })
    }

    /// Reads a trajectory from a target output: the `trajectory` field of an
    /// object, or the object itself when it has a `tool_calls` field.
    ///
    /// ```
    /// use serde_json::json;
    /// use sinew_eval::Trajectory;
    /// assert!(Trajectory::from_output(&json!({"tool_calls": []})).is_some());
    /// assert!(Trajectory::from_output(&json!("just text")).is_none());
    /// ```
    pub fn from_output(output: &Value) -> Option<Self> {
        let obj = output.as_object()?;
        let v = match obj.get("trajectory") {
            Some(t) => t,
            None if obj.contains_key("tool_calls") => output,
            None => return None,
        };
        serde_json::from_value(v.clone()).ok()
    }

    /// A numbered, human-readable rendering (used by [`TrajectoryJudge`]).
    /// Tool results longer than `max_result_chars` characters are cut.
    ///
    /// ```
    /// use serde_json::json;
    /// let t = sinew_eval::Trajectory::new().tool("add", json!({"a": 1}));
    /// assert!(t.to_text(100).contains("1. add({\"a\":1})"));
    /// ```
    pub fn to_text(&self, max_result_chars: usize) -> String {
        let mut out = String::new();
        if self.tool_calls.is_empty() {
            out.push_str("(no tool calls)\n");
        }
        for (i, s) in self.tool_calls.iter().enumerate() {
            out.push_str(&format!("{}. {}({})", i + 1, s.name, s.arguments));
            if let Some(e) = &s.error {
                out.push_str(&format!(" -> error: {}", truncate(e, max_result_chars)));
            } else if let Some(r) = &s.result {
                out.push_str(&format!(" -> {}", truncate(r, max_result_chars)));
            }
            out.push('\n');
        }
        out.push_str(&format!("model calls: {}", self.model_calls));
        if let Some(n) = self.total_tokens() {
            out.push_str(&format!(", tokens: {n}"));
        }
        if let Some(c) = self.cost_usd {
            out.push_str(&format!(", cost: ${c:.6}"));
        }
        out.push('\n');
        if let Some(a) = &self.final_answer {
            out.push_str(&format!("final answer: {a}\n"));
        }
        out
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max).collect();
        t.push('…');
        t
    }
}

/// `true` when `expected` is contained in `actual`: objects match when every
/// key of `expected` is present in `actual` with a matching value (extra keys
/// in `actual` are allowed, recursively), arrays match element by element and
/// must have the same length, numbers compare by value (`1` matches `1.0`),
/// and other values must be equal.
///
/// ```
/// use serde_json::json;
/// use sinew_eval::json_subset;
/// assert!(json_subset(&json!({"city": "Paris"}), &json!({"city": "Paris", "units": "metric"})));
/// assert!(!json_subset(&json!({"city": "Paris", "units": "metric"}), &json!({"city": "Paris"})));
/// assert!(json_subset(&json!([1, {"a": 2}]), &json!([1.0, {"a": 2, "b": 3}])));
/// ```
pub fn json_subset(expected: &Value, actual: &Value) -> bool {
    match (expected, actual) {
        (Value::Object(e), Value::Object(a)) => e
            .iter()
            .all(|(k, ev)| a.get(k).is_some_and(|av| json_subset(ev, av))),
        (Value::Array(e), Value::Array(a)) => {
            e.len() == a.len() && e.iter().zip(a).all(|(x, y)| json_subset(x, y))
        }
        (Value::Number(e), Value::Number(a)) => e == a || e.as_f64() == a.as_f64(),
        (e, a) => e == a,
    }
}

fn json_equal(a: &Value, b: &Value) -> bool {
    json_subset(a, b) && json_subset(b, a)
}

/// Arguments sent as a JSON string by some providers are parsed first.
fn normalize(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s.trim()).unwrap_or_else(|_| v.clone()),
        other => other.clone(),
    }
}

/// How tool arguments are compared by [`TrajectoryMatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArgumentMatch {
    /// Compare tool names only (default).
    #[default]
    Ignore,
    /// The expected arguments must be a [`json_subset`] of the actual ones.
    Subset,
    /// Arguments must be equal (numbers by value).
    Exact,
}

impl ArgumentMatch {
    /// Whether `actual` satisfies the `expected` step. An expected step with
    /// `null` arguments matches any arguments.
    ///
    /// ```
    /// use serde_json::json;
    /// use sinew_eval::{ArgumentMatch, ToolStep};
    /// let expected = ToolStep::new("get_weather", json!({"city": "Paris"}));
    /// let actual = ToolStep::new("get_weather", json!({"city": "Paris", "units": "c"}));
    /// assert!(ArgumentMatch::Subset.matches(&expected, &actual));
    /// assert!(!ArgumentMatch::Exact.matches(&expected, &actual));
    /// ```
    pub fn matches(self, expected: &ToolStep, actual: &ToolStep) -> bool {
        if expected.name != actual.name {
            return false;
        }
        if self == ArgumentMatch::Ignore || expected.arguments.is_null() {
            return true;
        }
        let (e, a) = (normalize(&expected.arguments), normalize(&actual.arguments));
        match self {
            ArgumentMatch::Ignore => true,
            ArgumentMatch::Subset => json_subset(&e, &a),
            ArgumentMatch::Exact => json_equal(&e, &a),
        }
    }
}

/// What a [`TrajectoryEvaluator`] sees for one example.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryInput<'a> {
    pub inputs: &'a Value,
    /// The whole target output.
    pub output: &'a Value,
    pub trajectory: &'a Trajectory,
    /// The example's expected trajectory.
    pub expected: Option<&'a Trajectory>,
    /// The example's reference output.
    pub reference: Option<&'a Value>,
}

impl<'a> TrajectoryInput<'a> {
    /// An input with only a trajectory (`inputs` and `output` are `null`).
    ///
    /// ```
    /// let t = sinew_eval::Trajectory::new();
    /// let input = sinew_eval::TrajectoryInput::new(&t);
    /// assert!(input.expected.is_none());
    /// ```
    pub fn new(trajectory: &'a Trajectory) -> Self {
        Self {
            inputs: &Value::Null,
            output: &Value::Null,
            trajectory,
            expected: None,
            reference: None,
        }
    }

    /// Sets the expected trajectory.
    ///
    /// ```
    /// let (a, e) = (sinew_eval::Trajectory::new(), sinew_eval::Trajectory::new());
    /// assert!(sinew_eval::TrajectoryInput::new(&a).with_expected(&e).expected.is_some());
    /// ```
    pub fn with_expected(mut self, expected: &'a Trajectory) -> Self {
        self.expected = Some(expected);
        self
    }

    /// Sets the example inputs.
    ///
    /// ```
    /// let (t, q) = (sinew_eval::Trajectory::new(), serde_json::json!("question"));
    /// assert_eq!(sinew_eval::TrajectoryInput::new(&t).with_inputs(&q).inputs, "question");
    /// ```
    pub fn with_inputs(mut self, inputs: &'a Value) -> Self {
        self.inputs = inputs;
        self
    }
}

/// Scores one agent trajectory. Implementations must not panic; return a
/// zero score with a reason instead.
///
/// ```
/// use async_trait::async_trait;
/// use sinew_eval::{Score, TrajectoryEvaluator, TrajectoryInput};
///
/// struct AtMostThreeTools;
///
/// #[async_trait]
/// impl TrajectoryEvaluator for AtMostThreeTools {
///     fn name(&self) -> &str { "at_most_three_tools" }
///     async fn evaluate(&self, i: TrajectoryInput<'_>) -> Score {
///         Score::pass(self.name(), i.trajectory.tool_calls.len() <= 3)
///     }
/// }
/// ```
#[async_trait]
pub trait TrajectoryEvaluator: Send + Sync {
    /// Key under which scores are reported.
    fn name(&self) -> &str;
    async fn evaluate(&self, input: TrajectoryInput<'_>) -> Score;
}

fn no_expected(key: &str) -> Score {
    Score::new(key, 0.0).with_reason("example has no expected trajectory")
}

/// Maximum bipartite matching (Kuhn). Returns, for each left index, the
/// matched right index.
fn max_matching(
    left: usize,
    right: usize,
    edge: impl Fn(usize, usize) -> bool,
) -> Vec<Option<usize>> {
    fn try_assign(
        l: usize,
        adj: &[Vec<usize>],
        seen: &mut [bool],
        right_of: &mut [Option<usize>],
    ) -> bool {
        for &r in &adj[l] {
            if seen[r] {
                continue;
            }
            seen[r] = true;
            if right_of[r].is_none_or(|other| try_assign(other, adj, seen, right_of)) {
                right_of[r] = Some(l);
                return true;
            }
        }
        false
    }
    let adj: Vec<Vec<usize>> = (0..left)
        .map(|l| (0..right).filter(|&r| edge(l, r)).collect())
        .collect();
    let mut right_of: Vec<Option<usize>> = vec![None; right];
    for l in 0..left {
        let mut seen = vec![false; right];
        try_assign(l, &adj, &mut seen, &mut right_of);
    }
    let mut left_of = vec![None; left];
    for (r, l) in right_of.iter().enumerate() {
        if let Some(l) = l {
            left_of[*l] = Some(r);
        }
    }
    left_of
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchMode {
    Exact,
    Unordered,
    Subsequence,
    Superset,
    Subset,
}

/// Compares the tool calls with an expected trajectory: the example's
/// `expected_trajectory`, or a fixed one given with
/// [`TrajectoryMatch::expecting`]. Binary score; the reason shows both
/// sequences on failure.
///
/// Tool names are compared by default; [`TrajectoryMatch::arguments`]
/// compares arguments too.
///
/// ```
/// # async fn demo() {
/// use serde_json::json;
/// use sinew_eval::*;
///
/// let actual = Trajectory::new()
///     .tool("get_weather", json!({"city": "Paris", "units": "metric"}))
///     .tool("search", json!({"q": "Paris events"}));
/// let expected = Trajectory::new()
///     .tool("search", json!({"q": "Paris events"}))
///     .tool("get_weather", json!({"city": "Paris"}));
/// let input = TrajectoryInput::new(&actual).with_expected(&expected);
///
/// assert_eq!(TrajectoryMatch::exact().evaluate(input).await.score, 0.0);
/// let unordered = TrajectoryMatch::unordered().arguments(ArgumentMatch::Subset);
/// assert_eq!(unordered.evaluate(input).await.score, 1.0);
/// # }
/// ```
pub struct TrajectoryMatch {
    mode: MatchMode,
    args: ArgumentMatch,
    expected: Option<Trajectory>,
    key: String,
}

impl TrajectoryMatch {
    fn with_mode(mode: MatchMode, key: &str) -> Self {
        Self {
            mode,
            args: ArgumentMatch::Ignore,
            expected: None,
            key: key.into(),
        }
    }

    /// Same calls, same order, nothing extra. Key `trajectory_exact`.
    ///
    /// ```
    /// let e = sinew_eval::TrajectoryMatch::exact();
    /// # use sinew_eval::TrajectoryEvaluator;
    /// assert_eq!(e.name(), "trajectory_exact");
    /// ```
    pub fn exact() -> Self {
        Self::with_mode(MatchMode::Exact, "trajectory_exact")
    }

    /// Same calls (as a multiset), any order. Key `trajectory_unordered`.
    ///
    /// ```
    /// # use sinew_eval::TrajectoryEvaluator;
    /// assert_eq!(sinew_eval::TrajectoryMatch::unordered().name(), "trajectory_unordered");
    /// ```
    pub fn unordered() -> Self {
        Self::with_mode(MatchMode::Unordered, "trajectory_unordered")
    }

    /// The expected calls appear in order; extra calls before, between and
    /// after them are allowed. Key `trajectory_subsequence`.
    ///
    /// ```
    /// # use sinew_eval::TrajectoryEvaluator;
    /// assert_eq!(sinew_eval::TrajectoryMatch::subsequence().name(), "trajectory_subsequence");
    /// ```
    pub fn subsequence() -> Self {
        Self::with_mode(MatchMode::Subsequence, "trajectory_subsequence")
    }

    /// Every expected call was made, in any order; extra calls are allowed.
    /// An expected call listed twice must be made twice. Key
    /// `trajectory_superset`.
    ///
    /// ```
    /// # use sinew_eval::TrajectoryEvaluator;
    /// assert_eq!(sinew_eval::TrajectoryMatch::superset().name(), "trajectory_superset");
    /// ```
    pub fn superset() -> Self {
        Self::with_mode(MatchMode::Superset, "trajectory_superset")
    }

    /// Every call made matches one of the expected calls (repeats allowed);
    /// expected calls may be missing. Key `trajectory_subset`.
    ///
    /// ```
    /// # use sinew_eval::TrajectoryEvaluator;
    /// assert_eq!(sinew_eval::TrajectoryMatch::subset().name(), "trajectory_subset");
    /// ```
    pub fn subset() -> Self {
        Self::with_mode(MatchMode::Subset, "trajectory_subset")
    }

    /// How arguments are compared (default [`ArgumentMatch::Ignore`]).
    ///
    /// ```
    /// use sinew_eval::{ArgumentMatch, TrajectoryMatch};
    /// let e = TrajectoryMatch::exact().arguments(ArgumentMatch::Subset);
    /// ```
    pub fn arguments(mut self, args: ArgumentMatch) -> Self {
        self.args = args;
        self
    }

    /// Compares with a fixed trajectory instead of the example's.
    ///
    /// ```
    /// use sinew_eval::{ToolStep, Trajectory, TrajectoryMatch};
    /// let e = TrajectoryMatch::superset().expecting(Trajectory::new().step(ToolStep::named("search")));
    /// ```
    pub fn expecting(mut self, expected: Trajectory) -> Self {
        self.expected = Some(expected);
        self
    }

    /// Reports scores under a custom key.
    ///
    /// ```
    /// # use sinew_eval::TrajectoryEvaluator;
    /// let e = sinew_eval::TrajectoryMatch::exact().named("tools_in_order");
    /// assert_eq!(e.name(), "tools_in_order");
    /// ```
    pub fn named(mut self, key: impl Into<String>) -> Self {
        self.key = key.into();
        self
    }

    fn check(&self, expected: &[ToolStep], actual: &[ToolStep]) -> Result<(), String> {
        let m = |e: &ToolStep, a: &ToolStep| self.args.matches(e, a);
        match self.mode {
            MatchMode::Exact => {
                if expected.len() != actual.len() {
                    return Err(format!(
                        "expected {} tool calls, got {}",
                        expected.len(),
                        actual.len()
                    ));
                }
                match expected.iter().zip(actual).position(|(e, a)| !m(e, a)) {
                    Some(i) => Err(format!("call {} does not match", i + 1)),
                    None => Ok(()),
                }
            }
            MatchMode::Unordered => {
                if expected.len() != actual.len() {
                    return Err(format!(
                        "expected {} tool calls, got {}",
                        expected.len(),
                        actual.len()
                    ));
                }
                let matched = max_matching(expected.len(), actual.len(), |e, a| {
                    m(&expected[e], &actual[a])
                });
                match matched.iter().position(Option::is_none) {
                    Some(i) => Err(format!("expected call `{}` not matched", expected[i].name)),
                    None => Ok(()),
                }
            }
            MatchMode::Subsequence => {
                let mut rest = actual.iter();
                for e in expected {
                    if !rest.any(|a| m(e, a)) {
                        return Err(format!("expected call `{}` not found in order", e.name));
                    }
                }
                Ok(())
            }
            MatchMode::Superset => {
                let matched = max_matching(expected.len(), actual.len(), |e, a| {
                    m(&expected[e], &actual[a])
                });
                let missing: Vec<&str> = matched
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.is_none())
                    .map(|(i, _)| expected[i].name.as_str())
                    .collect();
                if missing.is_empty() {
                    Ok(())
                } else {
                    Err(format!("missing calls: {}", missing.join(", ")))
                }
            }
            MatchMode::Subset => {
                let extra: Vec<&str> = actual
                    .iter()
                    .filter(|a| !expected.iter().any(|e| m(e, a)))
                    .map(|a| a.name.as_str())
                    .collect();
                if extra.is_empty() {
                    Ok(())
                } else {
                    Err(format!("unexpected calls: {}", extra.join(", ")))
                }
            }
        }
    }
}

#[async_trait]
impl TrajectoryEvaluator for TrajectoryMatch {
    fn name(&self) -> &str {
        &self.key
    }
    async fn evaluate(&self, i: TrajectoryInput<'_>) -> Score {
        let Some(expected) = self.expected.as_ref().or(i.expected) else {
            return no_expected(self.name());
        };
        match self.check(&expected.tool_calls, &i.trajectory.tool_calls) {
            Ok(()) => Score::pass(self.name(), true),
            Err(why) => Score::pass(self.name(), false).with_reason(format!(
                "{why}; expected [{}], got [{}]",
                expected.tool_names().join(", "),
                i.trajectory.tool_names().join(", ")
            )),
        }
    }
}

/// Required tools were used and forbidden tools were not, by name. Binary
/// score; needs no expected trajectory.
///
/// ```
/// # async fn demo() {
/// use serde_json::json;
/// use sinew_eval::*;
///
/// let t = Trajectory::new().tool("search", json!({"q": "x"}));
/// let usage = ToolUsage::new().require("search").forbid("delete_file");
/// assert_eq!(usage.evaluate(TrajectoryInput::new(&t)).await.score, 1.0);
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct ToolUsage {
    required: Vec<String>,
    forbidden: Vec<String>,
    key: Option<String>,
}

impl ToolUsage {
    /// No requirements yet (always passes).
    ///
    /// ```
    /// let u = sinew_eval::ToolUsage::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// The tool must be called at least once.
    ///
    /// ```
    /// let u = sinew_eval::ToolUsage::new().require("search");
    /// ```
    pub fn require(mut self, tool: impl Into<String>) -> Self {
        self.required.push(tool.into());
        self
    }

    /// The tool must not be called.
    ///
    /// ```
    /// let u = sinew_eval::ToolUsage::new().forbid("shell");
    /// ```
    pub fn forbid(mut self, tool: impl Into<String>) -> Self {
        self.forbidden.push(tool.into());
        self
    }

    /// Reports scores under a custom key (default `tool_usage`).
    ///
    /// ```
    /// # use sinew_eval::TrajectoryEvaluator;
    /// assert_eq!(sinew_eval::ToolUsage::new().named("safe_tools").name(), "safe_tools");
    /// ```
    pub fn named(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }
}

#[async_trait]
impl TrajectoryEvaluator for ToolUsage {
    fn name(&self) -> &str {
        self.key.as_deref().unwrap_or("tool_usage")
    }
    async fn evaluate(&self, i: TrajectoryInput<'_>) -> Score {
        let names = i.trajectory.tool_names();
        let missing: Vec<&str> = self
            .required
            .iter()
            .filter(|r| !names.contains(&r.as_str()))
            .map(String::as_str)
            .collect();
        let used: Vec<&str> = self
            .forbidden
            .iter()
            .filter(|f| names.contains(&f.as_str()))
            .map(String::as_str)
            .collect();
        let mut problems = Vec::new();
        if !missing.is_empty() {
            problems.push(format!("required tools not used: {}", missing.join(", ")));
        }
        if !used.is_empty() {
            problems.push(format!("forbidden tools used: {}", used.join(", ")));
        }
        if problems.is_empty() {
            Score::pass(self.name(), true)
        } else {
            Score::pass(self.name(), false).with_reason(problems.join("; "))
        }
    }
}

/// Steps (model calls), tool calls, tokens and cost stay within the limits
/// that are set. Binary score. A limit on tokens or cost fails when the
/// trajectory does not report them.
///
/// ```
/// # async fn demo() {
/// use serde_json::json;
/// use sinew_eval::*;
///
/// let t = Trajectory::new()
///     .tool("search", json!({}))
///     .with_model_calls(2)
///     .with_usage(800, 100);
/// let budget = TrajectoryBudget::new().max_steps(4).max_tool_calls(2).max_tokens(1_000);
/// assert_eq!(budget.evaluate(TrajectoryInput::new(&t)).await.score, 1.0);
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct TrajectoryBudget {
    max_steps: Option<u32>,
    max_tool_calls: Option<usize>,
    max_tokens: Option<u64>,
    max_cost_usd: Option<f64>,
    key: Option<String>,
}

impl TrajectoryBudget {
    /// No limits yet (always passes).
    ///
    /// ```
    /// let b = sinew_eval::TrajectoryBudget::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// At most `n` model calls.
    ///
    /// ```
    /// let b = sinew_eval::TrajectoryBudget::new().max_steps(5);
    /// ```
    pub fn max_steps(mut self, n: u32) -> Self {
        self.max_steps = Some(n);
        self
    }

    /// At most `n` tool calls.
    ///
    /// ```
    /// let b = sinew_eval::TrajectoryBudget::new().max_tool_calls(3);
    /// ```
    pub fn max_tool_calls(mut self, n: usize) -> Self {
        self.max_tool_calls = Some(n);
        self
    }

    /// At most `n` input plus output tokens.
    ///
    /// ```
    /// let b = sinew_eval::TrajectoryBudget::new().max_tokens(10_000);
    /// ```
    pub fn max_tokens(mut self, n: u64) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// At most `usd` dollars.
    ///
    /// ```
    /// let b = sinew_eval::TrajectoryBudget::new().max_cost(0.05);
    /// ```
    pub fn max_cost(mut self, usd: f64) -> Self {
        self.max_cost_usd = Some(usd);
        self
    }

    /// Reports scores under a custom key (default `trajectory_budget`).
    ///
    /// ```
    /// # use sinew_eval::TrajectoryEvaluator;
    /// assert_eq!(sinew_eval::TrajectoryBudget::new().named("cheap").name(), "cheap");
    /// ```
    pub fn named(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }
}

#[async_trait]
impl TrajectoryEvaluator for TrajectoryBudget {
    fn name(&self) -> &str {
        self.key.as_deref().unwrap_or("trajectory_budget")
    }
    async fn evaluate(&self, i: TrajectoryInput<'_>) -> Score {
        let t = i.trajectory;
        let mut problems = Vec::new();
        if let Some(max) = self.max_steps {
            if t.model_calls > max {
                problems.push(format!("{} steps > {max}", t.model_calls));
            }
        }
        if let Some(max) = self.max_tool_calls {
            if t.tool_calls.len() > max {
                problems.push(format!("{} tool calls > {max}", t.tool_calls.len()));
            }
        }
        if let Some(max) = self.max_tokens {
            match t.total_tokens() {
                Some(n) if n > max => problems.push(format!("{n} tokens > {max}")),
                Some(_) => {}
                None => problems.push("token usage unknown".into()),
            }
        }
        if let Some(max) = self.max_cost_usd {
            match t.cost_usd {
                Some(c) if c > max => problems.push(format!("cost ${c:.6} > ${max:.6}")),
                Some(_) => {}
                None => problems.push("cost unknown".into()),
            }
        }
        if problems.is_empty() {
            Score::pass(self.name(), true)
        } else {
            Score::pass(self.name(), false).with_reason(problems.join("; "))
        }
    }
}

/// The quantity [`RelativeEfficiency`] compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EfficiencyMetric {
    /// Model calls.
    Steps,
    /// Tool calls (default).
    #[default]
    ToolCalls,
    /// Input plus output tokens.
    Tokens,
    /// Cost in USD.
    Cost,
}

impl EfficiencyMetric {
    fn value(self, t: &Trajectory) -> Option<f64> {
        match self {
            EfficiencyMetric::Steps => Some(f64::from(t.model_calls)),
            EfficiencyMetric::ToolCalls => Some(t.tool_calls.len() as f64),
            EfficiencyMetric::Tokens => t.total_tokens().map(|n| n as f64),
            EfficiencyMetric::Cost => t.cost_usd,
        }
    }
}

/// Efficiency relative to the expected (reference) trajectory:
/// `min(1, reference / actual)` for one [`EfficiencyMetric`]. Using as many
/// tool calls as the reference or fewer scores 1; twice as many scores 0.5.
/// A metric unknown on either side scores 0 with a reason.
///
/// ```
/// # async fn demo() {
/// use serde_json::json;
/// use sinew_eval::*;
///
/// let actual = Trajectory::new().tool("a", json!({})).tool("a", json!({})).tool("b", json!({}))
///     .tool("b", json!({}));
/// let expected = Trajectory::new().tool("a", json!({})).tool("b", json!({}));
/// let s = RelativeEfficiency::new()
///     .evaluate(TrajectoryInput::new(&actual).with_expected(&expected))
///     .await;
/// assert_eq!(s.score, 0.5);
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct RelativeEfficiency {
    metric: EfficiencyMetric,
    expected: Option<Trajectory>,
    key: Option<String>,
}

impl RelativeEfficiency {
    /// Compares tool calls with the example's expected trajectory.
    ///
    /// ```
    /// let e = sinew_eval::RelativeEfficiency::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// The metric to compare (default [`EfficiencyMetric::ToolCalls`]).
    ///
    /// ```
    /// use sinew_eval::{EfficiencyMetric, RelativeEfficiency};
    /// let e = RelativeEfficiency::new().metric(EfficiencyMetric::Tokens);
    /// ```
    pub fn metric(mut self, metric: EfficiencyMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Compares with a fixed reference trajectory instead of the example's.
    ///
    /// ```
    /// use sinew_eval::{RelativeEfficiency, Trajectory};
    /// let e = RelativeEfficiency::new().expecting(Trajectory::new().with_model_calls(2));
    /// ```
    pub fn expecting(mut self, reference: Trajectory) -> Self {
        self.expected = Some(reference);
        self
    }

    /// Reports scores under a custom key (default `trajectory_efficiency`).
    ///
    /// ```
    /// # use sinew_eval::TrajectoryEvaluator;
    /// assert_eq!(sinew_eval::RelativeEfficiency::new().named("eff").name(), "eff");
    /// ```
    pub fn named(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }
}

#[async_trait]
impl TrajectoryEvaluator for RelativeEfficiency {
    fn name(&self) -> &str {
        self.key.as_deref().unwrap_or("trajectory_efficiency")
    }
    async fn evaluate(&self, i: TrajectoryInput<'_>) -> Score {
        let Some(expected) = self.expected.as_ref().or(i.expected) else {
            return no_expected(self.name());
        };
        let (Some(reference), Some(actual)) =
            (self.metric.value(expected), self.metric.value(i.trajectory))
        else {
            return Score::new(self.name(), 0.0).with_reason(format!(
                "{:?} unknown in trajectory or reference",
                self.metric
            ));
        };
        let score = if actual <= reference {
            1.0
        } else {
            reference / actual
        };
        Score::new(self.name(), score).with_reason(format!(
            "{:?}: {actual} vs reference {reference}",
            self.metric
        ))
    }
}

/// No identical call (same tool name and equal arguments) is made more than
/// the allowed number of times (default once). Binary score; catches agents
/// stuck in a loop.
///
/// ```
/// # async fn demo() {
/// use serde_json::json;
/// use sinew_eval::*;
///
/// let t = Trajectory::new()
///     .tool("search", json!({"q": "x"}))
///     .tool("search", json!({"q": "x"}));
/// let s = NoRepeatedCalls::new().evaluate(TrajectoryInput::new(&t)).await;
/// assert_eq!(s.score, 0.0);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct NoRepeatedCalls {
    allowed: usize,
    key: Option<String>,
}

impl NoRepeatedCalls {
    /// Every identical call may appear once.
    ///
    /// ```
    /// let e = sinew_eval::NoRepeatedCalls::new();
    /// ```
    pub fn new() -> Self {
        Self {
            allowed: 1,
            key: None,
        }
    }

    /// An identical call may appear up to `n` times (at least 1).
    ///
    /// ```
    /// let e = sinew_eval::NoRepeatedCalls::new().allowed(2);
    /// ```
    pub fn allowed(mut self, n: usize) -> Self {
        self.allowed = n.max(1);
        self
    }

    /// Reports scores under a custom key (default `no_repeated_calls`).
    ///
    /// ```
    /// # use sinew_eval::TrajectoryEvaluator;
    /// assert_eq!(sinew_eval::NoRepeatedCalls::new().named("loops").name(), "loops");
    /// ```
    pub fn named(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }
}

impl Default for NoRepeatedCalls {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TrajectoryEvaluator for NoRepeatedCalls {
    fn name(&self) -> &str {
        self.key.as_deref().unwrap_or("no_repeated_calls")
    }
    async fn evaluate(&self, i: TrajectoryInput<'_>) -> Score {
        let calls = &i.trajectory.tool_calls;
        let same = |a: &ToolStep, b: &ToolStep| {
            a.name == b.name && json_equal(&normalize(&a.arguments), &normalize(&b.arguments))
        };
        for (n, c) in calls.iter().enumerate() {
            // Report each distinct call once, at its first occurrence.
            if calls[..n].iter().any(|p| same(p, c)) {
                continue;
            }
            let count = calls[n..].iter().filter(|o| same(o, c)).count();
            if count > self.allowed {
                return Score::pass(self.name(), false).with_reason(format!(
                    "`{}({})` called {count} times (allowed {})",
                    c.name, c.arguments, self.allowed
                ));
            }
        }
        Score::pass(self.name(), true)
    }
}

#[derive(Deserialize)]
struct Grade {
    score: f64,
    reasoning: String,
}

/// LLM-judged trajectory quality: any [`ChatModel`] grades the trajectory
/// against a rubric through [`StructuredOutput`] and returns
/// `{"score": 0..1, "reasoning": "..."}`. Scores on a 0–10 or 0–100 scale
/// are scaled down. A model error or an invalid reply scores 0 with the
/// reason.
///
/// The prompt contains the rubric, the example inputs, the expected
/// trajectory (when there is one) and the actual trajectory rendered with
/// [`Trajectory::to_text`].
///
/// ```
/// # async fn demo() {
/// use std::sync::Arc;
/// use serde_json::json;
/// use sinew_core::{mock::ScriptedModel, Message};
/// use sinew_eval::*;
///
/// let judge_model = Arc::new(ScriptedModel::new([Message::assistant(
///     r#"{"score": 0.9, "reasoning": "Direct path, one lookup."}"#,
/// )]));
/// let judge = TrajectoryJudge::new(judge_model, "Did the agent use the fewest tools needed?");
/// let t = Trajectory::new().tool("lookup", json!({"id": 7})).with_final_answer("found");
/// let s = judge.evaluate(TrajectoryInput::new(&t)).await;
/// assert_eq!(s.score, 0.9);
/// # }
/// ```
pub struct TrajectoryJudge {
    model: Arc<dyn ChatModel>,
    rubric: String,
    key: String,
    mode: StructuredMode,
    max_result_chars: usize,
}

impl TrajectoryJudge {
    /// A judge with a rubric. Key `trajectory_judge`.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # let model = Arc::new(sinew_core::mock::ScriptedModel::new([]));
    /// let judge = sinew_eval::TrajectoryJudge::new(model, "Was every tool call necessary?");
    /// ```
    pub fn new(model: Arc<dyn ChatModel>, rubric: impl Into<String>) -> Self {
        Self {
            model,
            rubric: rubric.into(),
            key: "trajectory_judge".into(),
            mode: StructuredMode::Tool,
            max_result_chars: 500,
        }
    }

    /// Reports scores under a custom key.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use sinew_eval::TrajectoryEvaluator;
    /// # let model = Arc::new(sinew_core::mock::ScriptedModel::new([]));
    /// let judge = sinew_eval::TrajectoryJudge::new(model, "...").named("path_quality");
    /// assert_eq!(judge.name(), "path_quality");
    /// ```
    pub fn named(mut self, key: impl Into<String>) -> Self {
        self.key = key.into();
        self
    }

    /// How the structured reply is requested (default
    /// [`StructuredMode::Tool`]; use [`StructuredMode::Json`] for models
    /// without forced tool choice).
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # let model = Arc::new(sinew_core::mock::ScriptedModel::new([]));
    /// let judge = sinew_eval::TrajectoryJudge::new(model, "...")
    ///     .mode(sinew_core::StructuredMode::Json);
    /// ```
    pub fn mode(mut self, mode: StructuredMode) -> Self {
        self.mode = mode;
        self
    }

    /// Tool results longer than `n` characters are cut in the prompt
    /// (default 500).
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # let model = Arc::new(sinew_core::mock::ScriptedModel::new([]));
    /// let judge = sinew_eval::TrajectoryJudge::new(model, "...").max_result_chars(200);
    /// ```
    pub fn max_result_chars(mut self, n: usize) -> Self {
        self.max_result_chars = n;
        self
    }

    fn prompt(&self, i: &TrajectoryInput<'_>) -> Vec<Message> {
        let expected = i
            .expected
            .map(|e| e.to_text(self.max_result_chars))
            .unwrap_or_else(|| "(none)\n".into());
        vec![
            Message::system(
                "You are a strict evaluator of AI agent trajectories. Grade the TRAJECTORY \
                 (the tool calls the agent made, in order, and its final answer) using the \
                 RUBRIC. Give a score from 0 to 1 and a short reasoning.",
            ),
            Message::user(format!(
                "RUBRIC:\n{}\n\nINPUT:\n{}\n\nEXPECTED TRAJECTORY:\n{}\nTRAJECTORY:\n{}",
                self.rubric,
                as_text(i.inputs),
                expected,
                i.trajectory.to_text(self.max_result_chars),
            )),
        ]
    }
}

#[async_trait]
impl TrajectoryEvaluator for TrajectoryJudge {
    fn name(&self) -> &str {
        &self.key
    }
    async fn evaluate(&self, i: TrajectoryInput<'_>) -> Score {
        let schema = json!({
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "0 (bad) to 1 (excellent)"},
                "reasoning": {"type": "string"}
            },
            "required": ["score", "reasoning"]
        });
        let req = ChatRequest::new(self.prompt(&i)).with_temperature(0.0);
        let out = StructuredOutput::new("trajectory_grade", schema)
            .with_description("Grade of the agent trajectory.")
            .with_mode(self.mode)
            .generate::<Grade, _>(self.model.as_ref(), req)
            .await;
        match out {
            Ok(g) => {
                let score = if g.score > 10.0 {
                    g.score / 100.0
                } else if g.score > 1.0 {
                    g.score / 10.0
                } else {
                    g.score
                };
                Score::new(self.name(), score).with_reason(g.reasoning)
            }
            Err(e) => Score::new(self.name(), 0.0).with_reason(format!("judge failed: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sinew_core::{mock::ScriptedModel, ToolCall};
    use sinew_trace::{MemorySink, Tracer};

    fn t(names: &[&str]) -> Trajectory {
        names
            .iter()
            .fold(Trajectory::new(), |t, n| t.step(ToolStep::named(*n)))
    }

    async fn score(e: &dyn TrajectoryEvaluator, actual: &Trajectory, expected: &Trajectory) -> f64 {
        e.evaluate(TrajectoryInput::new(actual).with_expected(expected))
            .await
            .score
    }

    #[test]
    fn json_subset_rules() {
        assert!(json_subset(&json!({}), &json!({"a": 1})));
        assert!(json_subset(
            &json!({"a": {"b": 1}}),
            &json!({"a": {"b": 1.0, "c": 2}})
        ));
        assert!(!json_subset(&json!({"a": 1}), &json!({"a": 2})));
        assert!(!json_subset(&json!([1]), &json!([1, 2])));
        assert!(!json_subset(&json!("1"), &json!(1)));
        assert!(json_equal(
            &json!({"a": 1, "b": [2]}),
            &json!({"b": [2.0], "a": 1})
        ));
        assert!(!json_equal(&json!({"a": 1}), &json!({"a": 1, "b": 2})));
    }

    #[test]
    fn argument_matching_parses_string_arguments() {
        let e = ToolStep::new("f", json!({"x": 1}));
        let a = ToolStep::new("f", json!("{\"x\": 1, \"y\": 2}"));
        assert!(ArgumentMatch::Subset.matches(&e, &a));
        assert!(!ArgumentMatch::Exact.matches(&e, &a));
        assert!(ArgumentMatch::Exact.matches(&ToolStep::named("f"), &a));
        assert!(!ArgumentMatch::Ignore.matches(&ToolStep::named("g"), &a));
    }

    #[tokio::test]
    async fn exact_match() {
        let e = TrajectoryMatch::exact();
        assert_eq!(score(&e, &t(&["a", "b"]), &t(&["a", "b"])).await, 1.0);
        assert_eq!(score(&e, &t(&["b", "a"]), &t(&["a", "b"])).await, 0.0);
        assert_eq!(score(&e, &t(&["a", "b", "c"]), &t(&["a", "b"])).await, 0.0);
        assert_eq!(score(&e, &t(&[]), &t(&[])).await, 1.0);
        let s = e
            .evaluate(TrajectoryInput::new(&t(&["b"])).with_expected(&t(&["a"])))
            .await;
        assert_eq!(
            s.reason.as_deref(),
            Some("call 1 does not match; expected [a], got [b]")
        );
        // No expected trajectory at all.
        let s = e.evaluate(TrajectoryInput::new(&t(&["a"]))).await;
        assert_eq!(s.score, 0.0);
        assert!(s.reason.unwrap().contains("no expected trajectory"));
        // A fixed expectation wins over the example's.
        let fixed = TrajectoryMatch::exact().expecting(t(&["a"]));
        assert_eq!(score(&fixed, &t(&["a"]), &t(&["z"])).await, 1.0);
    }

    #[tokio::test]
    async fn exact_match_with_arguments() {
        let actual = Trajectory::new()
            .tool("get_weather", json!({"city": "Paris", "units": "metric"}))
            .tool("search", json!({"q": "events"}));
        let expected = Trajectory::new()
            .tool("get_weather", json!({"city": "Paris"}))
            .step(ToolStep::named("search"));
        let names = TrajectoryMatch::exact();
        let subset = TrajectoryMatch::exact().arguments(ArgumentMatch::Subset);
        let exact = TrajectoryMatch::exact().arguments(ArgumentMatch::Exact);
        assert_eq!(score(&names, &actual, &expected).await, 1.0);
        assert_eq!(score(&subset, &actual, &expected).await, 1.0);
        assert_eq!(score(&exact, &actual, &expected).await, 0.0);
        let wrong_city = Trajectory::new()
            .tool("get_weather", json!({"city": "Lyon"}))
            .tool("search", json!({}));
        assert_eq!(score(&subset, &wrong_city, &expected).await, 0.0);
    }

    #[tokio::test]
    async fn unordered_match_uses_a_full_matching() {
        let e = TrajectoryMatch::unordered().arguments(ArgumentMatch::Subset);
        assert_eq!(score(&e, &t(&["b", "a"]), &t(&["a", "b"])).await, 1.0);
        assert_eq!(score(&e, &t(&["a", "a"]), &t(&["a", "b"])).await, 0.0);
        assert_eq!(score(&e, &t(&["a"]), &t(&["a", "b"])).await, 0.0);
        // A greedy matcher would pair the generic expectation with the first
        // call and then fail; the matching finds the assignment.
        let expected = Trajectory::new()
            .step(ToolStep::named("f"))
            .tool("f", json!({"x": 1}));
        let actual = Trajectory::new()
            .tool("f", json!({"x": 1}))
            .tool("f", json!({"x": 2}));
        assert_eq!(score(&e, &actual, &expected).await, 1.0);
    }

    #[tokio::test]
    async fn subsequence_match() {
        let e = TrajectoryMatch::subsequence();
        assert_eq!(
            score(&e, &t(&["x", "a", "y", "b", "z"]), &t(&["a", "b"])).await,
            1.0
        );
        assert_eq!(score(&e, &t(&["b", "a"]), &t(&["a", "b"])).await, 0.0);
        assert_eq!(score(&e, &t(&["a"]), &t(&["a", "a"])).await, 0.0);
        assert_eq!(score(&e, &t(&["a", "a"]), &t(&["a", "a"])).await, 1.0);
        assert_eq!(score(&e, &t(&["q"]), &t(&[])).await, 1.0);
    }

    #[tokio::test]
    async fn superset_and_subset() {
        let sup = TrajectoryMatch::superset();
        assert_eq!(
            score(&sup, &t(&["c", "b", "a"]), &t(&["a", "b"])).await,
            1.0
        );
        assert_eq!(score(&sup, &t(&["a"]), &t(&["a", "a"])).await, 0.0);
        let s = sup
            .evaluate(TrajectoryInput::new(&t(&["a"])).with_expected(&t(&["a", "b"])))
            .await;
        assert!(s.reason.unwrap().starts_with("missing calls: b"));

        let sub = TrajectoryMatch::subset();
        assert_eq!(score(&sub, &t(&["a", "a"]), &t(&["a", "b"])).await, 1.0);
        assert_eq!(score(&sub, &t(&[]), &t(&["a"])).await, 1.0);
        let s = sub
            .evaluate(TrajectoryInput::new(&t(&["a", "rm"])).with_expected(&t(&["a"])))
            .await;
        assert_eq!(s.score, 0.0);
        assert!(s.reason.unwrap().starts_with("unexpected calls: rm"));
    }

    #[tokio::test]
    async fn tool_usage() {
        let e = ToolUsage::new().require("search").forbid("shell");
        let traj = t(&["search", "read"]);
        assert_eq!(e.evaluate(TrajectoryInput::new(&traj)).await.score, 1.0);
        let traj = t(&["shell"]);
        let s = e.evaluate(TrajectoryInput::new(&traj)).await;
        assert_eq!(s.score, 0.0);
        assert_eq!(
            s.reason.as_deref(),
            Some("required tools not used: search; forbidden tools used: shell")
        );
    }

    #[tokio::test]
    async fn budget() {
        let traj = t(&["a", "b"]).with_model_calls(3);
        let ok = TrajectoryBudget::new().max_steps(3).max_tool_calls(2);
        assert_eq!(ok.evaluate(TrajectoryInput::new(&traj)).await.score, 1.0);
        let tight = TrajectoryBudget::new().max_steps(2).max_tokens(10);
        let s = tight.evaluate(TrajectoryInput::new(&traj)).await;
        assert_eq!(s.score, 0.0);
        assert_eq!(
            s.reason.as_deref(),
            Some("3 steps > 2; token usage unknown")
        );
        let traj = traj.with_usage(50, 10).with_cost(0.02);
        let money = TrajectoryBudget::new().max_tokens(100).max_cost(0.01);
        let s = money.evaluate(TrajectoryInput::new(&traj)).await;
        assert_eq!(s.score, 0.0);
        assert!(s.reason.unwrap().starts_with("cost $0.020000"));
        assert_eq!(
            TrajectoryBudget::new()
                .max_cost(0.05)
                .evaluate(TrajectoryInput::new(&traj))
                .await
                .score,
            1.0
        );
    }

    #[tokio::test]
    async fn relative_efficiency() {
        let e = RelativeEfficiency::new();
        assert_eq!(score(&e, &t(&["a"]), &t(&["a", "b"])).await, 1.0);
        assert_eq!(score(&e, &t(&["a", "b", "c", "d"]), &t(&["a"])).await, 0.25);
        let tokens = RelativeEfficiency::new().metric(EfficiencyMetric::Tokens);
        let s = tokens
            .evaluate(TrajectoryInput::new(&t(&[])).with_expected(&t(&[]).with_usage(10, 0)))
            .await;
        assert_eq!(s.score, 0.0);
        assert!(s.reason.unwrap().contains("unknown"));
        let a = t(&[]).with_usage(300, 100);
        let r = t(&[]).with_usage(100, 100);
        assert_eq!(score(&tokens, &a, &r).await, 0.5);
        let steps = RelativeEfficiency::new()
            .metric(EfficiencyMetric::Steps)
            .expecting(t(&[]).with_model_calls(2));
        let a = t(&[]).with_model_calls(3);
        let s = steps.evaluate(TrajectoryInput::new(&a)).await;
        assert!((s.score - 2.0 / 3.0).abs() < 1e-9);
        let cost = RelativeEfficiency::new().metric(EfficiencyMetric::Cost);
        assert_eq!(
            score(&cost, &t(&[]).with_cost(0.01), &t(&[]).with_cost(0.02)).await,
            1.0
        );
        assert_eq!(
            e.evaluate(TrajectoryInput::new(&a)).await.score,
            0.0,
            "no reference"
        );
    }

    #[tokio::test]
    async fn no_repeated_calls() {
        let e = NoRepeatedCalls::new();
        let distinct = Trajectory::new()
            .tool("s", json!({"q": 1}))
            .tool("s", json!({"q": 2}));
        assert_eq!(e.evaluate(TrajectoryInput::new(&distinct)).await.score, 1.0);
        let looped = Trajectory::new()
            .tool("s", json!({"q": 1, "p": 2}))
            .tool("t", json!({}))
            .tool("s", json!({"p": 2, "q": 1.0}));
        let s = e.evaluate(TrajectoryInput::new(&looped)).await;
        assert_eq!(s.score, 0.0);
        assert!(s.reason.unwrap().contains("called 2 times (allowed 1)"));
        assert_eq!(
            NoRepeatedCalls::new()
                .allowed(2)
                .evaluate(TrajectoryInput::new(&looped))
                .await
                .score,
            1.0
        );
    }

    #[tokio::test]
    async fn judge_with_scripted_model() {
        let model = Arc::new(ScriptedModel::new([
            Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "g1".into(),
                    name: "trajectory_grade".into(),
                    arguments: json!({"score": 7, "reasoning": "One redundant search."}),
                }],
            ),
            Message::assistant("no idea"),
            Message::assistant("still no JSON"),
        ]));
        let judge = TrajectoryJudge::new(model.clone(), "Was every call needed?").named("path");
        let actual = t(&["search", "search", "answer"]).with_final_answer("done");
        let expected = t(&["search", "answer"]);
        let inputs = json!({"question": "q?"});
        let s = judge
            .evaluate(
                TrajectoryInput::new(&actual)
                    .with_expected(&expected)
                    .with_inputs(&inputs),
            )
            .await;
        assert_eq!(s.key, "path");
        assert!((s.score - 0.7).abs() < 1e-9);
        assert_eq!(s.reason.as_deref(), Some("One redundant search."));
        let req = &model.requests()[0];
        assert_eq!(req.tools[0].name, "trajectory_grade");
        let prompt = &req.messages.last().unwrap().content;
        assert!(prompt.contains("Was every call needed?"));
        assert!(prompt.contains("1. search(null)\n2. search(null)\n3. answer(null)"));
        assert!(prompt.contains("EXPECTED TRAJECTORY:\n1. search(null)\n2. answer(null)"));
        assert!(prompt.contains("final answer: done") && prompt.contains("q?"));

        // An invalid reply (after one retry) scores 0 with the reason.
        let s = judge.evaluate(TrajectoryInput::new(&actual)).await;
        assert_eq!(s.score, 0.0);
        assert!(s.reason.unwrap().starts_with("judge failed"));
    }

    fn call(id: &str, name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: args,
        }
    }

    #[test]
    fn from_messages_pairs_results_and_errors() {
        let messages = vec![
            Message::system("sys"),
            Message::user("weather in Paris and Lyon?"),
            Message::assistant_with_tools(
                "",
                vec![
                    call("c1", "get_weather", json!({"city": "Paris"})),
                    call("c2", "get_weather", json!({"city": "Lyon"})),
                ],
            ),
            Message::tool_result("c2", "error: service down"),
            Message::tool_result("c1", "sunny"),
            Message::assistant_with_tools("", vec![call("c3", "search", json!({"q": "Lyon"}))]),
        ];
        let t = Trajectory::from_messages(&messages);
        assert_eq!(t.tool_names(), vec!["get_weather", "get_weather", "search"]);
        assert_eq!(t.tool_calls[0].result.as_deref(), Some("sunny"));
        assert_eq!(t.tool_calls[1].error.as_deref(), Some("service down"));
        assert_eq!(t.tool_calls[2].result, None);
        assert_eq!(t.model_calls, 2);
        assert_eq!(t.final_answer, None, "last assistant message calls tools");
        assert_eq!(t.total_tokens(), None);

        let mut more = messages.clone();
        more.push(Message::tool_result("c3", "rain"));
        more.push(Message::assistant("Paris sunny, Lyon rain."));
        more.push(Message::user("thanks"));
        more.push(Message::assistant("You're welcome."));
        let all = Trajectory::from_messages(&more);
        assert_eq!(all.model_calls, 4);
        let last = Trajectory::from_last_turn(&more);
        assert!(last.tool_calls.is_empty());
        assert_eq!(last.final_answer.as_deref(), Some("You're welcome."));
        assert_eq!(Trajectory::from_last_turn(&[]), Trajectory::new());
    }

    #[test]
    fn from_runs_reads_tools_usage_cost_and_answer() {
        let sink = Arc::new(MemorySink::new());
        let tracer = Tracer::new().with_shared_sink(sink.clone());
        let root = tracer.start("agent", RunKind::Chain, None, json!({}));
        let parent = root.reference();
        let mut llm = tracer.start("gpt-4o-mini", RunKind::Llm, Some(&parent), json!({}));
        llm.set_usage(1_000, 100);
        llm.set_metadata("model", json!("gpt-4o-mini"));
        llm.finish(json!(Message::assistant_with_tools(
            "",
            vec![call("c1", "search", json!({"q": "x"}))]
        )));
        tracer
            .start("search", RunKind::Tool, Some(&parent), json!({"q": "x"}))
            .finish(json!("result text"));
        tracer
            .start("fetch", RunKind::Tool, Some(&parent), json!({"url": "u"}))
            .fail("404");
        let mut llm = tracer.start("local", RunKind::Llm, Some(&parent), json!({}));
        llm.set_usage(10, 5);
        llm.set_metadata("cost_usd", json!(0.5));
        llm.finish(json!(Message::assistant("final")));
        root.finish(json!("final"));

        let mut runs = sink.runs();
        // Same millisecond starts keep the emitted order; make the order
        // explicit so the test does not depend on timing.
        for (i, r) in runs.iter_mut().enumerate() {
            r.start_ms = i as u64;
        }
        runs.reverse();
        let t = Trajectory::from_runs(&runs);
        assert_eq!(t.tool_names(), vec!["search", "fetch"]);
        assert_eq!(t.tool_calls[0].arguments, json!({"q": "x"}));
        assert_eq!(t.tool_calls[0].result.as_deref(), Some("result text"));
        assert_eq!(t.tool_calls[1].error.as_deref(), Some("404"));
        assert_eq!(t.tool_calls[1].result, None);
        assert_eq!(t.model_calls, 2);
        assert_eq!((t.input_tokens, t.output_tokens), (Some(1_010), Some(105)));
        let priced = Usage::new(1_000, 100).cost("gpt-4o-mini").unwrap();
        assert!((t.cost_usd.unwrap() - (priced + 0.5)).abs() < 1e-12);
        assert_eq!(t.final_answer.as_deref(), Some("final"));

        // An unpriced model call makes the total cost unknown.
        let mut unknown = runs.clone();
        for r in &mut unknown {
            if r.name == "local" {
                r.metadata = json!({"model": "my-local-model"});
            }
        }
        assert_eq!(Trajectory::from_runs(&unknown).cost_usd, None);
        assert_eq!(Trajectory::from_runs(&[]), Trajectory::new());
    }

    #[test]
    fn output_round_trip_and_text() {
        let traj = Trajectory::new()
            .tool("search", json!({"q": "x"}))
            .step(ToolStep::new("fetch", json!({})).with_error("timeout"))
            .with_model_calls(3)
            .with_usage(10, 2)
            .with_cost(0.001)
            .with_final_answer("ok");
        let out = traj.clone().into_output();
        assert_eq!(out["answer"], "ok");
        assert_eq!(Trajectory::from_output(&out), Some(traj.clone()));
        assert_eq!(
            Trajectory::from_output(&serde_json::to_value(&traj).unwrap()),
            Some(traj.clone())
        );
        assert_eq!(Trajectory::from_output(&json!({"answer": "x"})), None);
        let parsed: Trajectory =
            serde_json::from_str(r#"{"tool_calls": [{"name": "search"}]}"#).unwrap();
        assert_eq!(parsed.tool_calls[0].arguments, Value::Null);
        let text = traj.to_text(3);
        assert!(text.contains("2. fetch({}) -> error: tim…"));
        assert!(text.contains("model calls: 3, tokens: 12, cost: $0.001000"));
    }
}
