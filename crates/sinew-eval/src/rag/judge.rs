//! LLM-judged RAG metrics: faithfulness, answer relevance, context
//! precision, context recall and citation accuracy.

use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use sinew_core::{ChatModel, ChatRequest, Message, StructuredMode, StructuredOutput};
use sinew_rag::Embeddings;

use super::{cosine, Meter, MetricResult, RagMetric, RagSample};

/// Shared settings of the judge metrics.
#[derive(Clone)]
struct Judge {
    model: Arc<dyn ChatModel>,
    key: String,
    mode: StructuredMode,
    max_context_chars: usize,
}

impl Judge {
    fn new(model: Arc<dyn ChatModel>, key: &str) -> Self {
        Self {
            model,
            key: key.into(),
            mode: StructuredMode::Tool,
            max_context_chars: 2000,
        }
    }

    fn contexts(&self, sample: &RagSample) -> String {
        sample
            .contexts
            .iter()
            .enumerate()
            .map(|(i, c)| {
                format!(
                    "[{}] {}",
                    i + 1,
                    cut(c.content.trim(), self.max_context_chars)
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn fail(&self, meter: &Meter<'_>, what: &str, e: String) -> MetricResult {
        MetricResult::new(&self.key, 0.0, format!("judge failed ({what}): {e}"))
            .with_usage(meter.usage())
    }
}

fn cut(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n).collect();
        t.push_str(" ...");
        t
    }
}

/// One structured call at temperature 0. A reply that fails the schema only
/// because a nested array or object arrived as a JSON-encoded string (some
/// local models do this in tool mode) is decoded leniently; any other
/// invalid reply gets one corrective retry.
pub(crate) async fn ask<T: DeserializeOwned>(
    model: &dyn ChatModel,
    mode: StructuredMode,
    name: &str,
    schema: Value,
    system: &str,
    user: String,
) -> Result<T, String> {
    let req =
        ChatRequest::new(vec![Message::system(system), Message::user(user)]).with_temperature(0.0);
    let so = StructuredOutput::new(name, schema).with_mode(mode);
    let rescue = |r: sinew_core::Result<T>| match r {
        Ok(v) => Ok(v),
        Err(sinew_core::Error::Structured { message, raw }) => {
            lenient(&raw).ok_or_else(|| sinew_core::Error::Structured { message, raw }.to_string())
        }
        Err(e) => Err(e.to_string()),
    };
    match so
        .clone()
        .with_retries(0)
        .generate::<T, _>(model, req.clone())
        .await
    {
        Err(sinew_core::Error::Structured { raw, .. }) if lenient::<T>(&raw).is_none() => {
            rescue(so.generate::<T, _>(model, req).await)
        }
        first => rescue(first),
    }
}

/// Parses `raw` after decoding top-level string fields that hold a JSON
/// array or object.
fn lenient<T: DeserializeOwned>(raw: &str) -> Option<T> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    let mut v: Value = serde_json::from_str(raw.get(start..=end)?).ok()?;
    if let Value::Object(o) = &mut v {
        for field in o.values_mut() {
            if let Value::String(s) = field {
                if let Ok(inner @ (Value::Array(_) | Value::Object(_))) =
                    serde_json::from_str::<Value>(s.trim())
                {
                    *field = inner;
                }
            }
        }
    }
    serde_json::from_value(v).ok()
}

fn verdicts_schema(field: &str, description: &str) -> Value {
    json!({
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "number", "description": "the item number, from 1"},
                        field: {"type": "boolean", "description": description},
                        "reason": {"type": "string"}
                    },
                    "required": ["index", field]
                }
            }
        },
        "required": ["verdicts"]
    })
}

#[derive(Deserialize)]
struct Verdicts {
    verdicts: Vec<Verdict>,
}

#[derive(Deserialize)]
struct Verdict {
    index: f64,
    #[serde(alias = "supported", alias = "useful", alias = "attributed")]
    verdict: bool,
    #[serde(default)]
    reason: String,
}

/// `(verdict, reason)` for items `1..=n`; items the judge skipped are
/// `false` with the reason "no verdict".
fn by_index(v: Verdicts, n: usize) -> Vec<(bool, String)> {
    let mut out = vec![(false, "no verdict from the judge".to_string()); n];
    for x in v.verdicts {
        let i = x.index.round();
        if i >= 1.0 && (i as usize) <= n {
            out[i as usize - 1] = (x.verdict, x.reason);
        }
    }
    out
}

fn scale(raw: f64) -> f64 {
    if raw > 10.0 {
        raw / 100.0
    } else if raw > 1.0 {
        raw / 10.0
    } else {
        raw
    }
}

macro_rules! judge_builders {
    ($t:ident, $key:literal) => {
        impl $t {
            #[doc = concat!(
                "Reports scores under a custom key (default `", $key, "`).\n\n```\n",
                "# use std::sync::Arc;\n",
                "use sinew_eval::rag::{RagMetric, ", stringify!($t), "};\n",
                "let model = Arc::new(sinew_core::mock::ScriptedModel::new([]));\n",
                "let m = ", stringify!($t), "::new(model).named(\"custom\");\n",
                "assert_eq!(m.key(), \"custom\");\n```"
            )]
            pub fn named(mut self, key: impl Into<String>) -> Self {
                self.judge.key = key.into();
                self
            }

            #[doc = concat!(
                "How structured replies are requested (default [`StructuredMode::Tool`]; ",
                "use [`StructuredMode::Json`] for models without forced tool choice).\n\n```\n",
                "# use std::sync::Arc;\n",
                "let model = Arc::new(sinew_core::mock::ScriptedModel::new([]));\n",
                "let m = sinew_eval::rag::", stringify!($t),
                "::new(model).mode(sinew_core::StructuredMode::Json);\n```"
            )]
            pub fn mode(mut self, mode: StructuredMode) -> Self {
                self.judge.mode = mode;
                self
            }

            #[doc = concat!(
                "Contexts longer than `n` characters are cut in the prompt (default 2000).",
                "\n\n```\n",
                "# use std::sync::Arc;\n",
                "let model = Arc::new(sinew_core::mock::ScriptedModel::new([]));\n",
                "let m = sinew_eval::rag::", stringify!($t), "::new(model).max_context_chars(500);\n```"
            )]
            pub fn max_context_chars(mut self, n: usize) -> Self {
                self.judge.max_context_chars = n.max(1);
                self
            }
        }
    };
}

/// Share of the answer's claims that the retrieved contexts support
/// (RAGAS faithfulness). Two judge calls: the answer is split into atomic
/// claims, then each claim is checked against the contexts. An answer
/// without claims (for example "I don't know") scores 1; claims with no
/// contexts score 0 without the second call.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use std::sync::Arc;
/// use sinew_core::{mock::ScriptedModel, Message};
/// use sinew_eval::rag::{Faithfulness, RagMetric, RagSample};
///
/// let judge = Arc::new(ScriptedModel::new([
///     Message::assistant(r#"{"claims": ["Paris is the capital of France.", "Paris has 30 million people."]}"#),
///     Message::assistant(r#"{"verdicts": [{"index": 1, "supported": true}, {"index": 2, "supported": false}]}"#),
/// ]));
/// let sample = RagSample::new("Capital of France?", "Paris, a city of 30 million.")
///     .with_contexts(["Paris is the capital of France."]);
/// let r = Faithfulness::new(judge).measure(&sample).await;
/// assert_eq!(r.score, 0.5);
/// assert_eq!(r.usage.calls, 2);
/// # }
/// ```
#[derive(Clone)]
pub struct Faithfulness {
    judge: Judge,
}

impl Faithfulness {
    /// Key `faithfulness`.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let m = sinew_eval::rag::Faithfulness::new(Arc::new(sinew_core::mock::ScriptedModel::new([])));
    /// ```
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            judge: Judge::new(model, "faithfulness"),
        }
    }
}

judge_builders!(Faithfulness, "faithfulness");

#[derive(Deserialize)]
struct Claims {
    claims: Vec<String>,
}

#[async_trait]
impl RagMetric for Faithfulness {
    fn key(&self) -> &str {
        &self.judge.key
    }

    async fn measure(&self, s: &RagSample) -> MetricResult {
        let j = &self.judge;
        let meter = Meter::new(j.model.as_ref());
        let schema = json!({
            "type": "object",
            "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
            "required": ["claims"]
        });
        let claims: Claims = match ask(
            &meter,
            j.mode,
            "answer_claims",
            schema,
            "Break the ANSWER into short, self-contained factual claims. Replace pronouns with \
             what they refer to, keep each claim to one fact, and do not add information that is \
             not in the answer. Ignore citation markers such as [1]. Return an empty list when \
             the answer makes no factual claim (for example, when it says it does not know).",
            format!("QUESTION:\n{}\n\nANSWER:\n{}", s.question, s.answer),
        )
        .await
        {
            Ok(c) => c,
            Err(e) => return j.fail(&meter, "claims", e),
        };
        let claims: Vec<String> = claims
            .claims
            .into_iter()
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect();
        if claims.is_empty() {
            return MetricResult::new(&j.key, 1.0, "the answer makes no factual claims")
                .with_details(json!({"claims": []}))
                .with_usage(meter.usage());
        }
        let verdicts = if s.contexts.is_empty() {
            vec![(false, "no retrieved context".to_string()); claims.len()]
        } else {
            let list: String = claims
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{}. {c}\n", i + 1))
                .collect();
            match ask::<Verdicts>(
                &meter,
                j.mode,
                "claim_verdicts",
                verdicts_schema("supported", "true if the context supports the claim"),
                "For each numbered CLAIM, decide whether the CONTEXT supports it. A claim is \
                 supported only when it can be inferred directly from the context; general \
                 knowledge does not count. Give one verdict per claim with its number.",
                format!("CONTEXT:\n{}\n\nCLAIMS:\n{list}", j.contexts(s)),
            )
            .await
            {
                Ok(v) => by_index(v, claims.len()),
                Err(e) => return j.fail(&meter, "verdicts", e),
            }
        };
        let supported = verdicts.iter().filter(|(v, _)| *v).count();
        let unsupported: Vec<&str> = claims
            .iter()
            .zip(&verdicts)
            .filter(|(_, (v, _))| !v)
            .map(|(c, _)| c.as_str())
            .collect();
        let mut reason = format!("{supported} of {} claims supported", claims.len());
        if !unsupported.is_empty() {
            reason.push_str(&format!("; unsupported: {}", unsupported.join(" | ")));
        }
        let details: Vec<Value> = claims
            .iter()
            .zip(&verdicts)
            .map(|(c, (v, r))| json!({"claim": c, "supported": v, "reason": r}))
            .collect();
        MetricResult::new(&j.key, supported as f64 / claims.len() as f64, reason)
            .with_details(json!({ "claims": details }))
            .with_usage(meter.usage())
    }
}

/// How directly the answer addresses the question. Correctness is not
/// judged (use [`Faithfulness`] and the reference-based metrics).
///
/// - [`AnswerRelevance::new`]: one judge call that returns a 0–1 score
///   (0–10 and 0–100 scales are scaled down);
/// - [`AnswerRelevance::from_generated_questions`] (RAGAS): the judge writes
///   `n` questions the answer would answer, and the score is the mean cosine
///   similarity (negative values count as 0) between their embeddings and
///   the question's. An answer the judge marks as noncommittal scores 0.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use std::sync::Arc;
/// use sinew_core::{mock::ScriptedModel, Message};
/// use sinew_eval::rag::{AnswerRelevance, RagMetric, RagSample};
///
/// let judge = Arc::new(ScriptedModel::new([Message::assistant(
///     r#"{"score": 0.5, "reasoning": "answers half of the question"}"#,
/// )]));
/// let r = AnswerRelevance::new(judge)
///     .measure(&RagSample::new("Capital and population of France?", "Paris."))
///     .await;
/// assert_eq!(r.score, 0.5);
/// # }
/// ```
#[derive(Clone)]
pub struct AnswerRelevance {
    judge: Judge,
    questions: Option<(Arc<dyn Embeddings>, usize)>,
}

impl AnswerRelevance {
    /// Judge-scored, key `answer_relevance`.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let m = sinew_eval::rag::AnswerRelevance::new(Arc::new(sinew_core::mock::ScriptedModel::new([])));
    /// ```
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            judge: Judge::new(model, "answer_relevance"),
            questions: None,
        }
    }

    /// Scored by the similarity of `n` generated questions (at least 1) to
    /// the original question, key `answer_relevance`.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// use sinew_eval::rag::AnswerRelevance;
    /// use sinew_rag::FakeEmbeddings;
    ///
    /// let model = Arc::new(sinew_core::mock::ScriptedModel::new([]));
    /// let m = AnswerRelevance::from_generated_questions(model, Arc::new(FakeEmbeddings::new(64)), 3);
    /// ```
    pub fn from_generated_questions(
        model: Arc<dyn ChatModel>,
        embeddings: Arc<dyn Embeddings>,
        n: usize,
    ) -> Self {
        Self {
            judge: Judge::new(model, "answer_relevance"),
            questions: Some((embeddings, n.max(1))),
        }
    }
}

judge_builders!(AnswerRelevance, "answer_relevance");

#[derive(Deserialize)]
struct Grade {
    score: f64,
    #[serde(default, alias = "reason")]
    reasoning: String,
}

#[derive(Deserialize)]
struct Generated {
    questions: Vec<String>,
    #[serde(default)]
    noncommittal: bool,
}

#[async_trait]
impl RagMetric for AnswerRelevance {
    fn key(&self) -> &str {
        &self.judge.key
    }

    async fn measure(&self, s: &RagSample) -> MetricResult {
        let j = &self.judge;
        let meter = Meter::new(j.model.as_ref());
        let Some((embeddings, n)) = &self.questions else {
            let schema = json!({
                "type": "object",
                "properties": {
                    "score": {"type": "number", "description": "0 (does not address the question) to 1 (addresses it fully)"},
                    "reasoning": {"type": "string"}
                },
                "required": ["score", "reasoning"]
            });
            return match ask::<Grade>(
                &meter,
                j.mode,
                "answer_relevance",
                schema,
                "Rate how well the ANSWER addresses the QUESTION. Score 1 when it answers the \
                 question directly and completely, 0.5 when it answers part of it or mixes in much \
                 unrelated content, and 0 when it does not address the question or declines to \
                 answer. Judge relevance only; do not judge whether the answer is correct.",
                format!("QUESTION:\n{}\n\nANSWER:\n{}", s.question, s.answer),
            )
            .await
            {
                Ok(g) => MetricResult::new(&j.key, scale(g.score), g.reasoning)
                    .with_details(json!({"method": "judge"}))
                    .with_usage(meter.usage()),
                Err(e) => j.fail(&meter, "grade", e),
            };
        };
        let schema = json!({
            "type": "object",
            "properties": {
                "questions": {"type": "array", "items": {"type": "string"}},
                "noncommittal": {"type": "boolean", "description": "true if the answer is evasive, vague or says it does not know"}
            },
            "required": ["questions", "noncommittal"]
        });
        let generated: Generated = match ask(
            &meter,
            j.mode,
            "answer_questions",
            schema,
            &format!(
                "Write {n} different questions that the ANSWER answers. Each question must be \
                 answerable from the answer alone. Also set noncommittal to true when the answer is \
                 evasive, vague or says it does not know."
            ),
            format!("ANSWER:\n{}", s.answer),
        )
        .await
        {
            Ok(g) => g,
            Err(e) => return j.fail(&meter, "questions", e),
        };
        let questions: Vec<String> = generated
            .questions
            .into_iter()
            .map(|q| q.trim().to_string())
            .filter(|q| !q.is_empty())
            .collect();
        if generated.noncommittal {
            return MetricResult::new(&j.key, 0.0, "the judge marked the answer as noncommittal")
                .with_details(json!({"method": "generated_questions", "questions": questions, "noncommittal": true}))
                .with_usage(meter.usage());
        }
        if questions.is_empty() {
            return MetricResult::new(&j.key, 0.0, "the judge generated no questions")
                .with_usage(meter.usage());
        }
        let mut texts = vec![s.question.clone()];
        texts.extend(questions.iter().cloned());
        let vectors = match embeddings.embed_documents(&texts).await {
            Ok(v) if v.len() == texts.len() => v,
            Ok(_) => {
                return MetricResult::new(&j.key, 0.0, "embeddings returned the wrong count")
                    .with_usage(meter.usage())
            }
            Err(e) => {
                return MetricResult::new(&j.key, 0.0, format!("embeddings failed: {e}"))
                    .with_usage(meter.usage())
            }
        };
        let sims: Vec<f64> = vectors[1..]
            .iter()
            .map(|v| cosine(&vectors[0], v).max(0.0))
            .collect();
        let mean = sims.iter().sum::<f64>() / sims.len() as f64;
        MetricResult::new(
            &j.key,
            mean,
            format!(
                "mean similarity of {} generated questions: {mean:.3}",
                sims.len()
            ),
        )
        .with_details(json!({
            "method": "generated_questions",
            "questions": questions,
            "similarities": sims,
            "noncommittal": false,
        }))
        .with_usage(meter.usage())
    }
}

/// Whether the useful contexts are ranked first (RAGAS context precision).
/// One judge call marks each context as useful or not for arriving at the
/// reference answer (or at the answer when the example has no reference);
/// the score is the average precision
/// `Σ_k (precision@k × useful_k) / Σ_k useful_k`. No useful context scores 0.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use std::sync::Arc;
/// use sinew_core::{mock::ScriptedModel, Message};
/// use sinew_eval::rag::{ContextPrecision, RagMetric, RagSample};
///
/// let judge = Arc::new(ScriptedModel::new([Message::assistant(
///     r#"{"verdicts": [{"index": 1, "useful": false}, {"index": 2, "useful": true}]}"#,
/// )]));
/// let sample = RagSample::new("Capital of France?", "Paris")
///     .with_contexts(["Rome is in Italy.", "Paris is the capital of France."])
///     .with_reference("Paris");
/// let r = ContextPrecision::new(judge).measure(&sample).await;
/// assert_eq!(r.score, 0.5); // the only useful context is at rank 2
/// # }
/// ```
#[derive(Clone)]
pub struct ContextPrecision {
    judge: Judge,
}

impl ContextPrecision {
    /// Key `context_precision`.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let m = sinew_eval::rag::ContextPrecision::new(Arc::new(sinew_core::mock::ScriptedModel::new([])));
    /// ```
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            judge: Judge::new(model, "context_precision"),
        }
    }
}

judge_builders!(ContextPrecision, "context_precision");

/// Average precision of binary relevance judgements in rank order.
fn average_precision(useful: &[bool]) -> f64 {
    let mut hits = 0usize;
    let mut sum = 0.0;
    for (i, u) in useful.iter().enumerate() {
        if *u {
            hits += 1;
            sum += hits as f64 / (i + 1) as f64;
        }
    }
    if hits == 0 {
        0.0
    } else {
        sum / hits as f64
    }
}

#[async_trait]
impl RagMetric for ContextPrecision {
    fn key(&self) -> &str {
        &self.judge.key
    }

    async fn measure(&self, s: &RagSample) -> MetricResult {
        let j = &self.judge;
        if s.contexts.is_empty() {
            return MetricResult::new(&j.key, 0.0, "no retrieved context");
        }
        let meter = Meter::new(j.model.as_ref());
        let (target, against) = match &s.reference {
            Some(r) => (r.as_str(), "reference"),
            None => (s.answer.as_str(), "answer"),
        };
        let verdicts = match ask::<Verdicts>(
            &meter,
            j.mode,
            "context_verdicts",
            verdicts_schema(
                "useful",
                "true if the passage helps to arrive at the answer",
            ),
            "For each numbered CONTEXT passage, decide whether it was useful to arrive at the \
             given ANSWER to the QUESTION: it contains information the answer relies on. Give one \
             verdict per passage with its number.",
            format!(
                "QUESTION:\n{}\n\nANSWER:\n{target}\n\nCONTEXT:\n{}",
                s.question,
                j.contexts(s)
            ),
        )
        .await
        {
            Ok(v) => by_index(v, s.contexts.len()),
            Err(e) => return j.fail(&meter, "verdicts", e),
        };
        let useful: Vec<bool> = verdicts.iter().map(|(v, _)| *v).collect();
        let ranks: Vec<usize> = useful
            .iter()
            .enumerate()
            .filter(|(_, u)| **u)
            .map(|(i, _)| i + 1)
            .collect();
        let score = average_precision(&useful);
        let reason = if ranks.is_empty() {
            format!(
                "none of {} contexts useful (against the {against})",
                useful.len()
            )
        } else {
            format!(
                "useful contexts at ranks {ranks:?} of {} (against the {against})",
                useful.len()
            )
        };
        let details: Vec<Value> = verdicts
            .iter()
            .enumerate()
            .map(|(i, (v, r))| json!({"rank": i + 1, "id": s.contexts[i].id, "useful": v, "reason": r}))
            .collect();
        MetricResult::new(&j.key, score, reason)
            .with_details(json!({"against": against, "contexts": details}))
            .with_usage(meter.usage())
    }
}

/// Share of the reference answer's statements that can be attributed to
/// the retrieved contexts (RAGAS context recall). One judge call splits the
/// reference into statements and attributes each. Needs a reference; no
/// contexts scores 0 without a call.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use std::sync::Arc;
/// use sinew_core::{mock::ScriptedModel, Message};
/// use sinew_eval::rag::{ContextRecall, RagMetric, RagSample};
///
/// let judge = Arc::new(ScriptedModel::new([Message::assistant(
///     r#"{"statements": [
///         {"statement": "Paris is the capital.", "attributed": true},
///         {"statement": "It has 2 million inhabitants.", "attributed": false}]}"#,
/// )]));
/// let sample = RagSample::new("Tell me about Paris", "...")
///     .with_contexts(["Paris is the capital of France."])
///     .with_reference("Paris is the capital. It has 2 million inhabitants.");
/// assert_eq!(ContextRecall::new(judge).measure(&sample).await.score, 0.5);
/// # }
/// ```
#[derive(Clone)]
pub struct ContextRecall {
    judge: Judge,
}

impl ContextRecall {
    /// Key `context_recall`.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let m = sinew_eval::rag::ContextRecall::new(Arc::new(sinew_core::mock::ScriptedModel::new([])));
    /// ```
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            judge: Judge::new(model, "context_recall"),
        }
    }
}

judge_builders!(ContextRecall, "context_recall");

#[derive(Deserialize)]
struct Statements {
    statements: Vec<Statement>,
}

#[derive(Deserialize)]
struct Statement {
    statement: String,
    attributed: bool,
    #[serde(default)]
    reason: String,
}

#[async_trait]
impl RagMetric for ContextRecall {
    fn key(&self) -> &str {
        &self.judge.key
    }

    async fn measure(&self, s: &RagSample) -> MetricResult {
        let j = &self.judge;
        let Some(reference) = &s.reference else {
            return MetricResult::new(&j.key, 0.0, "example has no reference answer");
        };
        if s.contexts.is_empty() {
            return MetricResult::new(&j.key, 0.0, "no retrieved context");
        }
        let meter = Meter::new(j.model.as_ref());
        let schema = json!({
            "type": "object",
            "properties": {
                "statements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "statement": {"type": "string"},
                            "attributed": {"type": "boolean", "description": "true if the context contains this statement"},
                            "reason": {"type": "string"}
                        },
                        "required": ["statement", "attributed"]
                    }
                }
            },
            "required": ["statements"]
        });
        let st: Statements = match ask(
            &meter,
            j.mode,
            "reference_statements",
            schema,
            "Split the REFERENCE answer into its individual factual statements. For each \
             statement, decide whether it can be attributed to the CONTEXT: the context states it \
             or it follows directly from the context.",
            format!(
                "QUESTION:\n{}\n\nREFERENCE:\n{reference}\n\nCONTEXT:\n{}",
                s.question,
                j.contexts(s)
            ),
        )
        .await
        {
            Ok(v) => v,
            Err(e) => return j.fail(&meter, "statements", e),
        };
        if st.statements.is_empty() {
            return MetricResult::new(
                &j.key,
                0.0,
                "the judge found no statements in the reference",
            )
            .with_usage(meter.usage());
        }
        let total = st.statements.len();
        let found = st.statements.iter().filter(|x| x.attributed).count();
        let missing: Vec<&str> = st
            .statements
            .iter()
            .filter(|x| !x.attributed)
            .map(|x| x.statement.as_str())
            .collect();
        let mut reason = format!("{found} of {total} reference statements found in the contexts");
        if !missing.is_empty() {
            reason.push_str(&format!("; missing: {}", missing.join(" | ")));
        }
        let details: Vec<Value> = st
            .statements
            .iter()
            .map(|x| json!({"statement": x.statement, "attributed": x.attributed, "reason": x.reason}))
            .collect();
        MetricResult::new(&j.key, found as f64 / total as f64, reason)
            .with_details(json!({ "statements": details }))
            .with_usage(meter.usage())
    }
}

/// Share of the answer's `[n]` citations whose passage supports the cited
/// sentence. Citations follow the [`RagChain`](sinew_rag::RagChain)
/// convention: passages are numbered from 1 in context order, and a marker
/// (`[2]`, `[1][3]` or `[1, 3]`) cites the sentence it ends, including a
/// marker placed right after the sentence's final period. A citation of a
/// passage that does not exist is unsupported without asking the judge; the
/// rest are checked in one call. An answer without citations scores 0.
/// The details also list the sentences that cite nothing.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use std::sync::Arc;
/// use sinew_core::{mock::ScriptedModel, Message};
/// use sinew_eval::rag::{CitationAccuracy, RagMetric, RagSample};
///
/// let judge = Arc::new(ScriptedModel::new([Message::assistant(
///     r#"{"verdicts": [{"index": 1, "supported": true}]}"#,
/// )]));
/// let sample = RagSample::new("Capital of France?", "Paris is the capital [1]. It is old [5].")
///     .with_contexts(["Paris is the capital of France."]);
/// let r = CitationAccuracy::new(judge).measure(&sample).await;
/// assert_eq!(r.score, 0.5); // [5] does not exist
/// # }
/// ```
#[derive(Clone)]
pub struct CitationAccuracy {
    judge: Judge,
}

impl CitationAccuracy {
    /// Key `citation_accuracy`.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let m = sinew_eval::rag::CitationAccuracy::new(Arc::new(sinew_core::mock::ScriptedModel::new([])));
    /// ```
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            judge: Judge::new(model, "citation_accuracy"),
        }
    }
}

judge_builders!(CitationAccuracy, "citation_accuracy");

/// Splits an answer into sentences and the passage numbers each cites.
/// Markers right after a sentence's terminator belong to that sentence.
pub(crate) fn cited_sentences(answer: &str) -> Vec<(String, Vec<usize>)> {
    let Ok(marker) = Regex::new(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]") else {
        return Vec::new();
    };
    let Ok(leading) = Regex::new(r"^\s*(?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+") else {
        return Vec::new();
    };
    // 1. Raw sentences.
    let mut raw: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = answer.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        cur.push(*c);
        let end = match c {
            '\n' => true,
            '.' | '!' | '?' => chars.get(i + 1).is_none_or(|n| n.is_whitespace()),
            _ => false,
        };
        if end {
            raw.push(std::mem::take(&mut cur));
        }
    }
    raw.push(cur);
    // 2. Move leading markers to the previous sentence.
    let mut sentences: Vec<String> = Vec::new();
    for s in raw {
        let (lead, rest) = match leading.find(&s) {
            Some(m) => (m.as_str().to_string(), s[m.end()..].to_string()),
            None => (String::new(), s),
        };
        if !lead.is_empty() {
            match sentences.last_mut() {
                Some(prev) => prev.push_str(&lead),
                None => sentences.push(lead),
            }
        }
        sentences.push(rest);
    }
    sentences
        .into_iter()
        .filter_map(|s| {
            let cites: Vec<usize> = marker
                .captures_iter(&s)
                .flat_map(|c| {
                    c[1].split([',', ';'])
                        .filter_map(|n| n.trim().parse().ok())
                        .collect::<Vec<usize>>()
                })
                .collect();
            let text = marker.replace_all(&s, "");
            let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
            let text = text.replace(" .", ".");
            if text
                .trim_matches(|c: char| c.is_ascii_punctuation())
                .is_empty()
            {
                None
            } else {
                Some((text, cites))
            }
        })
        .collect()
}

#[async_trait]
impl RagMetric for CitationAccuracy {
    fn key(&self) -> &str {
        &self.judge.key
    }

    async fn measure(&self, s: &RagSample) -> MetricResult {
        let j = &self.judge;
        let sentences = cited_sentences(&s.answer);
        let uncited: Vec<&str> = sentences
            .iter()
            .filter(|(_, c)| c.is_empty())
            .map(|(t, _)| t.as_str())
            .collect();
        // (sentence, passage number)
        let citations: Vec<(&str, usize)> = sentences
            .iter()
            .flat_map(|(t, cs)| {
                let mut seen: Vec<usize> = Vec::new();
                cs.iter()
                    .filter(move |n| {
                        let new = !seen.contains(n);
                        seen.push(**n);
                        new
                    })
                    .map(move |n| (t.as_str(), *n))
                    .collect::<Vec<_>>()
            })
            .collect();
        if citations.is_empty() {
            return MetricResult::new(&j.key, 0.0, "the answer cites no passages")
                .with_details(json!({"citations": [], "uncited_sentences": uncited}));
        }
        let n = s.contexts.len();
        let valid: Vec<usize> = (0..citations.len())
            .filter(|&i| (1..=n).contains(&citations[i].1))
            .collect();
        let meter = Meter::new(j.model.as_ref());
        let mut verdicts: Vec<(bool, String)> = citations
            .iter()
            .map(|(_, p)| {
                (
                    false,
                    format!("passage [{p}] does not exist ({n} retrieved)"),
                )
            })
            .collect();
        if !valid.is_empty() {
            let mut used: Vec<usize> = valid.iter().map(|&i| citations[i].1).collect();
            used.sort_unstable();
            used.dedup();
            let passages: Vec<String> = used
                .iter()
                .map(|p| {
                    format!(
                        "[{p}] {}",
                        cut(s.contexts[p - 1].content.trim(), j.max_context_chars)
                    )
                })
                .collect();
            let list: String = valid
                .iter()
                .enumerate()
                .map(|(k, &i)| {
                    format!(
                        "{}. STATEMENT: {} CITES: [{}]\n",
                        k + 1,
                        citations[i].0,
                        citations[i].1
                    )
                })
                .collect();
            match ask::<Verdicts>(
                &meter,
                j.mode,
                "citation_verdicts",
                verdicts_schema(
                    "supported",
                    "true if the cited passage supports the statement",
                ),
                "For each numbered CITATION, decide whether the cited PASSAGE supports the \
                 STATEMENT: the statement can be inferred from that passage alone. Give one \
                 verdict per citation with its number.",
                format!("PASSAGES:\n{}\n\nCITATIONS:\n{list}", passages.join("\n\n")),
            )
            .await
            {
                Ok(v) => {
                    for (k, verdict) in by_index(v, valid.len()).into_iter().enumerate() {
                        verdicts[valid[k]] = verdict;
                    }
                }
                Err(e) => return j.fail(&meter, "verdicts", e),
            }
        }
        let supported = verdicts.iter().filter(|(v, _)| *v).count();
        let mut reason = format!("{supported} of {} citations supported", citations.len());
        if !uncited.is_empty() {
            reason.push_str(&format!("; {} sentences cite nothing", uncited.len()));
        }
        let details: Vec<Value> = citations
            .iter()
            .zip(&verdicts)
            .map(|((t, p), (v, r))| json!({"sentence": t, "passage": p, "supported": v, "reason": r}))
            .collect();
        MetricResult::new(&j.key, supported as f64 / citations.len() as f64, reason)
            .with_details(json!({"citations": details, "uncited_sentences": uncited}))
            .with_usage(meter.usage())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sentences_and_citations() {
        let s = cited_sentences("Paris is the capital [1]. It is old. [2][3] Pi is 3.14 [1, 2].");
        assert_eq!(
            s,
            vec![
                ("Paris is the capital.".to_string(), vec![1]),
                ("It is old.".to_string(), vec![2, 3]),
                ("Pi is 3.14.".to_string(), vec![1, 2]),
            ]
        );
        assert!(cited_sentences("").is_empty());
    }

    #[test]
    fn average_precision_by_rank() {
        assert_eq!(average_precision(&[true, false]), 1.0);
        assert_eq!(average_precision(&[false, true]), 0.5);
        // (1/1 + 2/3) / 2
        assert!((average_precision(&[true, false, true]) - (1.0 + 2.0 / 3.0) / 2.0).abs() < 1e-12);
        assert_eq!(average_precision(&[false, false]), 0.0);
    }
}
