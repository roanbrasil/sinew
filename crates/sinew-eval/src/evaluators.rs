//! The [`Evaluator`] trait and the built-in evaluators.

use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinew_core::{ChatModel, ChatRequest, Message};

/// The result of one evaluator on one example. `score` is in `[0, 1]`
/// (1 = pass for the binary evaluators).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Score {
    pub key: String,
    pub score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl Score {
    pub fn new(key: impl Into<String>, score: f64) -> Self {
        Self {
            key: key.into(),
            score: if score.is_finite() {
                score.clamp(0.0, 1.0)
            } else {
                0.0
            },
            reason: None,
        }
    }

    pub fn pass(key: impl Into<String>, ok: bool) -> Self {
        Self::new(key, if ok { 1.0 } else { 0.0 })
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// What an evaluator sees for one example.
#[derive(Debug, Clone, Copy)]
pub struct EvalInput<'a> {
    pub inputs: &'a Value,
    pub output: &'a Value,
    pub reference: Option<&'a Value>,
}

/// Scores one target output. Implementations must not panic; return a zero
/// score with a reason instead.
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// Key under which scores are reported (one column per evaluator).
    fn name(&self) -> &str;
    async fn evaluate(&self, input: EvalInput<'_>) -> Score;
}

/// Renders a JSON value as text: strings as-is, everything else as JSON.
pub fn as_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Looks up a dotted path (`answer.value`, `items.0.name`) or a JSON pointer
/// (`/answer/value`). An empty path returns the value itself.
pub fn get_path<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(v);
    }
    if path.starts_with('/') {
        return v.pointer(path);
    }
    path.split('.').try_fold(v, |cur, seg| match cur {
        Value::Array(a) => seg.parse::<usize>().ok().and_then(|i| a.get(i)),
        Value::Object(o) => o.get(seg),
        _ => None,
    })
}

fn no_reference(key: &str) -> Score {
    Score::new(key, 0.0).with_reason("example has no reference output")
}

/// Output equals the reference. Strings are compared after trimming unless
/// [`ExactMatch::strict`] is used; other JSON values must be equal.
pub struct ExactMatch {
    trim: bool,
}

impl ExactMatch {
    pub fn new() -> Self {
        Self { trim: true }
    }
    pub fn strict() -> Self {
        Self { trim: false }
    }
}

impl Default for ExactMatch {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Evaluator for ExactMatch {
    fn name(&self) -> &str {
        "exact_match"
    }
    async fn evaluate(&self, i: EvalInput<'_>) -> Score {
        let Some(reference) = i.reference else {
            return no_reference(self.name());
        };
        let ok = match (i.output, reference) {
            (Value::String(a), Value::String(b)) if self.trim => a.trim() == b.trim(),
            (a, b) => a == b,
        };
        Score::pass(self.name(), ok)
    }
}

/// Output text contains a fixed needle, or the reference text when no
/// needle is given.
pub struct Contains {
    needle: Option<String>,
    ignore_case: bool,
}

impl Contains {
    /// Checks for `needle`.
    pub fn new(needle: impl Into<String>) -> Self {
        Self {
            needle: Some(needle.into()),
            ignore_case: false,
        }
    }
    /// Checks for the example's reference text.
    pub fn reference() -> Self {
        Self {
            needle: None,
            ignore_case: false,
        }
    }
    pub fn ignore_case(mut self) -> Self {
        self.ignore_case = true;
        self
    }
}

#[async_trait]
impl Evaluator for Contains {
    fn name(&self) -> &str {
        "contains"
    }
    async fn evaluate(&self, i: EvalInput<'_>) -> Score {
        let needle = match (&self.needle, i.reference) {
            (Some(n), _) => n.clone(),
            (None, Some(r)) => as_text(r),
            (None, None) => return no_reference(self.name()),
        };
        let hay = as_text(i.output);
        let ok = if self.ignore_case {
            hay.to_lowercase().contains(&needle.to_lowercase())
        } else {
            hay.contains(&needle)
        };
        Score::pass(self.name(), ok)
    }
}

/// Output text matches a regular expression.
pub struct RegexMatch {
    re: Regex,
}

impl RegexMatch {
    pub fn new(pattern: &str) -> Result<Self, regex::Error> {
        Ok(Self {
            re: Regex::new(pattern)?,
        })
    }
}

#[async_trait]
impl Evaluator for RegexMatch {
    fn name(&self) -> &str {
        "regex"
    }
    async fn evaluate(&self, i: EvalInput<'_>) -> Score {
        Score::pass(self.name(), self.re.is_match(&as_text(i.output)))
    }
}

/// A field of the output equals an expected value, or the same field of the
/// reference. Paths are dotted (`a.b.0`) or JSON pointers (`/a/b/0`).
/// String outputs are parsed as JSON first, so models that answer with a JSON
/// string work too.
pub struct JsonFieldEquals {
    path: String,
    expected: Option<Value>,
    key: String,
}

impl JsonFieldEquals {
    /// Compares `output[path]` with `reference[path]`.
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        Self {
            key: format!("json_field:{path}"),
            path,
            expected: None,
        }
    }
    /// Compares `output[path]` with a fixed value.
    pub fn expecting(path: impl Into<String>, expected: Value) -> Self {
        Self {
            expected: Some(expected),
            ..Self::new(path)
        }
    }
}

fn parse_if_string(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s.trim()).unwrap_or_else(|_| v.clone()),
        other => other.clone(),
    }
}

#[async_trait]
impl Evaluator for JsonFieldEquals {
    fn name(&self) -> &str {
        &self.key
    }
    async fn evaluate(&self, i: EvalInput<'_>) -> Score {
        let output = parse_if_string(i.output);
        let Some(actual) = get_path(&output, &self.path) else {
            return Score::new(self.name(), 0.0)
                .with_reason(format!("output has no field `{}`", self.path));
        };
        let expected = match (&self.expected, i.reference) {
            (Some(e), _) => e.clone(),
            (None, Some(r)) => match get_path(&parse_if_string(r), &self.path) {
                Some(v) => v.clone(),
                None => {
                    return Score::new(self.name(), 0.0)
                        .with_reason(format!("reference has no field `{}`", self.path))
                }
            },
            (None, None) => return no_reference(self.name()),
        };
        let ok = *actual == expected;
        let s = Score::pass(self.name(), ok);
        if ok {
            s
        } else {
            s.with_reason(format!("expected {expected}, got {actual}"))
        }
    }
}

/// The output (or a field of it) is a number within `tolerance` of the
/// reference. Numbers inside strings ("The answer is 42.") are extracted.
pub struct NumericTolerance {
    tolerance: f64,
    relative: bool,
    path: String,
}

impl NumericTolerance {
    /// Absolute tolerance: `|output - reference| <= tolerance`.
    pub fn absolute(tolerance: f64) -> Self {
        Self {
            tolerance,
            relative: false,
            path: String::new(),
        }
    }
    /// Relative tolerance: `|output - reference| <= tolerance * |reference|`.
    pub fn relative(tolerance: f64) -> Self {
        Self {
            relative: true,
            ..Self::absolute(tolerance)
        }
    }
    /// Reads the number from a field of output and reference.
    pub fn at(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }
}

/// Extracts a number: JSON numbers directly, strings by their last number.
pub fn extract_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => {
            let re = Regex::new(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?").ok()?;
            re.find_iter(&s.replace(['_', '\u{a0}'], ""))
                .last()
                .and_then(|m| m.as_str().replace(',', ".").parse().ok())
        }
        _ => None,
    }
}

#[async_trait]
impl Evaluator for NumericTolerance {
    fn name(&self) -> &str {
        "numeric"
    }
    async fn evaluate(&self, i: EvalInput<'_>) -> Score {
        let Some(reference) = i.reference else {
            return no_reference(self.name());
        };
        let get = |v: &Value| get_path(&parse_if_string(v), &self.path).and_then(extract_number);
        let (Some(a), Some(b)) = (get(i.output), get(reference)) else {
            return Score::new(self.name(), 0.0).with_reason("no number found in output");
        };
        let limit = if self.relative {
            self.tolerance * b.abs()
        } else {
            self.tolerance
        };
        let diff = (a - b).abs();
        Score::pass(self.name(), diff <= limit)
            .with_reason(format!("got {a}, expected {b}, diff {diff}"))
    }
}

/// LLM-as-judge: asks any [`ChatModel`] to grade the output against a rubric
/// and answer with `{"score": 0..1, "reason": "..."}`.
pub struct LlmJudge {
    model: Arc<dyn ChatModel>,
    rubric: String,
    key: String,
}

impl LlmJudge {
    pub fn new(model: Arc<dyn ChatModel>, rubric: impl Into<String>) -> Self {
        Self {
            model,
            rubric: rubric.into(),
            key: "llm_judge".into(),
        }
    }

    /// Reports scores under a custom key (e.g. `helpfulness`).
    pub fn named(mut self, key: impl Into<String>) -> Self {
        self.key = key.into();
        self
    }

    fn prompt(&self, i: &EvalInput<'_>) -> Vec<Message> {
        let reference = i.reference.map(as_text).unwrap_or_else(|| "(none)".into());
        vec![
            Message::system(
                "You are a strict evaluator. Grade the RESPONSE using the RUBRIC. \
                 Reply with only a JSON object: {\"score\": <number from 0 to 1>, \
                 \"reason\": \"<one sentence>\"}.",
            ),
            Message::user(format!(
                "RUBRIC:\n{}\n\nINPUT:\n{}\n\nREFERENCE:\n{}\n\nRESPONSE:\n{}",
                self.rubric,
                as_text(i.inputs),
                reference,
                as_text(i.output)
            )),
        ]
    }
}

/// Parses a judge reply: a JSON object (possibly wrapped in prose or a code
/// fence) with a numeric `score` (0..1, or 0..10 / 0..100 scaled down).
pub fn parse_judge_reply(text: &str) -> Option<(f64, Option<String>)> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    let v: Value = serde_json::from_str(text.get(start..=end)?).ok()?;
    let raw = match &v["score"] {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse().ok()?,
        Value::Bool(b) => f64::from(u8::from(*b)),
        _ => return None,
    };
    let score = if raw > 10.0 {
        raw / 100.0
    } else if raw > 1.0 {
        raw / 10.0
    } else {
        raw
    };
    let reason = v["reason"].as_str().map(str::to_string);
    Some((score, reason))
}

#[async_trait]
impl Evaluator for LlmJudge {
    fn name(&self) -> &str {
        &self.key
    }
    async fn evaluate(&self, i: EvalInput<'_>) -> Score {
        let req = ChatRequest::new(self.prompt(&i)).with_temperature(0.0);
        match self.model.chat(req).await {
            Ok(resp) => match parse_judge_reply(&resp.message.content) {
                Some((score, reason)) => {
                    let s = Score::new(self.name(), score);
                    match reason {
                        Some(r) => s.with_reason(r),
                        None => s,
                    }
                }
                None => Score::new(self.name(), 0.0).with_reason(format!(
                    "unparseable judge reply: {}",
                    resp.message.content.chars().take(200).collect::<String>()
                )),
            },
            Err(e) => Score::new(self.name(), 0.0).with_reason(format!("judge failed: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sinew_core::mock::ScriptedModel;

    async fn run(e: &dyn Evaluator, output: Value, reference: Option<Value>) -> Score {
        e.evaluate(EvalInput {
            inputs: &json!({}),
            output: &output,
            reference: reference.as_ref(),
        })
        .await
    }

    #[tokio::test]
    async fn exact_and_contains() {
        assert_eq!(
            run(&ExactMatch::new(), json!(" 4 "), Some(json!("4")))
                .await
                .score,
            1.0
        );
        assert_eq!(
            run(&ExactMatch::strict(), json!(" 4 "), Some(json!("4")))
                .await
                .score,
            0.0
        );
        assert_eq!(
            run(&ExactMatch::new(), json!("4"), None).await.score,
            0.0,
            "no reference fails"
        );
        assert_eq!(
            run(&Contains::new("Paris"), json!("It is Paris."), None)
                .await
                .score,
            1.0
        );
        assert_eq!(
            run(
                &Contains::reference().ignore_case(),
                json!("PARIS!"),
                Some(json!("paris"))
            )
            .await
            .score,
            1.0
        );
    }

    #[tokio::test]
    async fn regex_json_numeric() {
        let re = RegexMatch::new(r"^\d{3}$").unwrap();
        assert_eq!(run(&re, json!("111"), None).await.score, 1.0);
        assert_eq!(run(&re, json!("11a"), None).await.score, 0.0);

        let jf = JsonFieldEquals::new("answer.value");
        let out = json!("{\"answer\": {\"value\": 7}}");
        assert_eq!(
            run(&jf, out, Some(json!({"answer": {"value": 7}})))
                .await
                .score,
            1.0
        );
        let jf = JsonFieldEquals::expecting("/items/0", json!("a"));
        assert_eq!(run(&jf, json!({"items": ["b"]}), None).await.score, 0.0);

        let num = NumericTolerance::absolute(0.01);
        assert_eq!(
            run(&num, json!("37 × 3 = 111."), Some(json!(111)))
                .await
                .score,
            1.0
        );
        assert_eq!(run(&num, json!(110.5), Some(json!(111))).await.score, 0.0);
        let rel = NumericTolerance::relative(0.01).at("v");
        assert_eq!(
            run(&rel, json!({"v": 100.5}), Some(json!({"v": 100})))
                .await
                .score,
            1.0
        );
    }

    #[tokio::test]
    async fn llm_judge_parses_score_and_reason() {
        let model = Arc::new(ScriptedModel::new([
            Message::assistant("Sure:\n```json\n{\"score\": 8, \"reason\": \"mostly right\"}\n```"),
            Message::assistant("I cannot grade this."),
        ]));
        let judge = LlmJudge::new(model.clone(), "Is it correct?").named("correctness");
        let s = run(&judge, json!("111"), Some(json!("111"))).await;
        assert_eq!(s.key, "correctness");
        assert!((s.score - 0.8).abs() < 1e-9);
        assert_eq!(s.reason.as_deref(), Some("mostly right"));
        let prompt = &model.requests()[0].messages[1].content;
        assert!(prompt.contains("Is it correct?") && prompt.contains("111"));

        let s = run(&judge, json!("x"), None).await;
        assert_eq!(s.score, 0.0);
        assert!(s.reason.unwrap().starts_with("unparseable"));
    }
}
