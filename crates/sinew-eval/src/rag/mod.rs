//! RAG evaluation: metrics over a question, the retrieved contexts, the
//! answer and (optionally) a reference answer and the ids of the relevant
//! documents, plus synthetic test-set generation from a corpus.
//!
//! Enabled by the `rag` feature (it depends on `sinew-rag` for
//! [`Embeddings`](sinew_rag::Embeddings) and [`Document`]).
//!
//! | Metric | Key | Needs | Measures |
//! |---|---|---|---|
//! | [`Faithfulness`] | `faithfulness` | judge model, contexts | share of the answer's claims supported by the contexts |
//! | [`AnswerRelevance`] | `answer_relevance` | judge model (+ embeddings) | how directly the answer addresses the question |
//! | [`ContextPrecision`] | `context_precision` | judge model, contexts, reference (or answer) | useful contexts ranked first (average precision) |
//! | [`ContextRecall`] | `context_recall` | judge model, contexts, reference | share of reference statements found in the contexts |
//! | [`CitationAccuracy`] | `citation_accuracy` | judge model, contexts, `[n]` citations | share of citations whose passage supports the cited sentence |
//! | [`AnswerSimilarity`] | `answer_similarity` | embeddings, reference | cosine similarity of answer and reference |
//! | [`RetrievalMetric`] | `hit_rate@k`, `recall@k`, `precision@k`, `mrr`, `ndcg@k` | context ids, relevant ids | ranking quality, no model |
//!
//! Every metric returns a [`MetricResult`] with a score in `[0, 1]`, a
//! reason, structured `details` and the [`JudgeUsage`] (calls, tokens, cost
//! when the price is known) of its model calls. Every metric is also an
//! [`Evaluator`], so it runs inside an
//! [`Experiment`](crate::Experiment): the target returns
//! [`RagSample::to_output`] (or [`rag_output`] for a
//! [`RagAnswer`]) and the sample is read back with
//! [`RagSample::from_eval`]. In that case the usage is appended to the
//! score's reason.
//!
//! [`TestsetGenerator`] writes question / reference answer / source id
//! triples from documents with any chat model and exports them as a
//! [`Dataset`](crate::Dataset).
//!
//! ```
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! use sinew_eval::rag::{RagMetric, RagSample, RetrievalMetric};
//!
//! let sample = RagSample::new("Where is the Eiffel Tower?", "In Paris.")
//!     .with_context("paris", "The Eiffel Tower is in Paris.")
//!     .with_context("rome", "The Colosseum is in Rome.")
//!     .with_relevant_ids(["paris"]);
//! let mrr = RetrievalMetric::mrr().measure(&sample).await;
//! assert_eq!(mrr.score, 1.0);
//! # }
//! ```

mod embedding;
mod judge;
mod retrieval;
mod testset;

use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{
    stream::ChatStream, Capabilities, ChatModel, ChatRequest, ChatResponse, Result as CoreResult,
};
use sinew_rag::{Document, RagAnswer};

use crate::{as_text, EvalInput, Evaluator, Score};

pub use embedding::AnswerSimilarity;
pub use judge::{AnswerRelevance, CitationAccuracy, ContextPrecision, ContextRecall, Faithfulness};
pub use retrieval::{
    hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, RetrievalMetric,
};
pub use testset::{QuestionType, SyntheticExample, Testset, TestsetGenerator};

/// One retrieved passage: its text and, when known, the id of the document
/// it came from (used by the retrieval metrics).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RagContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub content: String,
}

/// What the RAG metrics look at for one question.
///
/// ```
/// use sinew_eval::rag::RagSample;
///
/// let sample = RagSample::new("Capital of France?", "Paris [1].")
///     .with_context("fr", "Paris is the capital of France.")
///     .with_reference("Paris")
///     .with_relevant_ids(["fr"]);
/// assert_eq!(sample.context_ids(), vec!["fr"]);
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RagSample {
    pub question: String,
    /// Retrieved passages, best first (numbered from 1 for citations).
    #[serde(default)]
    pub contexts: Vec<RagContext>,
    pub answer: String,
    /// The reference ("ground truth") answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// Ids of the documents that are relevant to the question.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relevant_ids: Vec<String>,
}

impl RagSample {
    /// A sample with a question and an answer, no contexts yet.
    ///
    /// ```
    /// let s = sinew_eval::rag::RagSample::new("q", "a");
    /// assert!(s.contexts.is_empty() && s.reference.is_none());
    /// ```
    pub fn new(question: impl Into<String>, answer: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            answer: answer.into(),
            ..Default::default()
        }
    }

    /// Appends a retrieved passage with the id of its document.
    ///
    /// ```
    /// let s = sinew_eval::rag::RagSample::new("q", "a").with_context("doc-1", "text");
    /// assert_eq!(s.contexts[0].id.as_deref(), Some("doc-1"));
    /// ```
    pub fn with_context(mut self, id: impl Into<String>, content: impl Into<String>) -> Self {
        self.contexts.push(RagContext {
            id: Some(id.into()),
            content: content.into(),
        });
        self
    }

    /// Appends passages without ids.
    ///
    /// ```
    /// let s = sinew_eval::rag::RagSample::new("q", "a").with_contexts(["one", "two"]);
    /// assert_eq!(s.contexts.len(), 2);
    /// ```
    pub fn with_contexts<I, S>(mut self, contexts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.contexts
            .extend(contexts.into_iter().map(|c| RagContext {
                id: None,
                content: c.into(),
            }));
        self
    }

    /// Sets the reference answer.
    ///
    /// ```
    /// let s = sinew_eval::rag::RagSample::new("q", "a").with_reference("a");
    /// assert_eq!(s.reference.as_deref(), Some("a"));
    /// ```
    pub fn with_reference(mut self, reference: impl Into<String>) -> Self {
        self.reference = Some(reference.into());
        self
    }

    /// Sets the ids of the relevant documents.
    ///
    /// ```
    /// let s = sinew_eval::rag::RagSample::new("q", "a").with_relevant_ids(["d1", "d2"]);
    /// assert_eq!(s.relevant_ids, vec!["d1", "d2"]);
    /// ```
    pub fn with_relevant_ids<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.relevant_ids = ids.into_iter().map(Into::into).collect();
        self
    }

    /// A sample from retrieved documents. A context's id is the document id,
    /// else its `source` metadata.
    ///
    /// ```
    /// use sinew_eval::rag::RagSample;
    /// use sinew_rag::Document;
    ///
    /// let docs = [Document::new("Paris is in France.").with_metadata("source", "fr.md")];
    /// let s = RagSample::from_documents("Where is Paris?", &docs, "In France [1].");
    /// assert_eq!(s.context_ids(), vec!["fr.md"]);
    /// ```
    pub fn from_documents(
        question: impl Into<String>,
        documents: &[Document],
        answer: impl Into<String>,
    ) -> Self {
        Self {
            question: question.into(),
            contexts: documents
                .iter()
                .map(|d| RagContext {
                    id: d.id.clone().or_else(|| d.source().map(str::to_string)),
                    content: d.content.clone(),
                })
                .collect(),
            answer: answer.into(),
            ..Default::default()
        }
    }

    /// A sample from a [`RagChain`](sinew_rag::RagChain) answer and its
    /// sources (see [`from_documents`](Self::from_documents) for ids).
    ///
    /// ```
    /// use sinew_core::Usage;
    /// use sinew_eval::rag::RagSample;
    /// use sinew_rag::{Document, RagAnswer};
    ///
    /// let out = RagAnswer {
    ///     answer: "In France [1].".into(),
    ///     sources: vec![Document::new("Paris is in France.").with_id("fr")],
    ///     usage: Usage::default(),
    /// };
    /// let s = RagSample::from_answer("Where is Paris?", &out);
    /// assert_eq!(s.context_ids(), vec!["fr"]);
    /// ```
    pub fn from_answer(question: impl Into<String>, answer: &RagAnswer) -> Self {
        Self::from_documents(question, &answer.sources, answer.answer.clone())
    }

    /// The ids of the retrieved contexts, in rank order (contexts without an
    /// id are skipped).
    ///
    /// ```
    /// let s = sinew_eval::rag::RagSample::new("q", "a").with_context("x", "...");
    /// assert_eq!(s.context_ids(), vec!["x"]);
    /// ```
    pub fn context_ids(&self) -> Vec<&str> {
        self.contexts
            .iter()
            .filter_map(|c| c.id.as_deref())
            .collect()
    }

    /// The target output an [`Experiment`](crate::Experiment) should return:
    /// `{"answer": ..., "contexts": [{"id": ..., "content": ...}]}`.
    ///
    /// ```
    /// let s = sinew_eval::rag::RagSample::new("q", "a").with_context("x", "text");
    /// let out = s.to_output();
    /// assert_eq!(out["answer"], "a");
    /// assert_eq!(out["contexts"][0]["id"], "x");
    /// ```
    pub fn to_output(&self) -> Value {
        json!({"answer": self.answer, "contexts": self.contexts})
    }

    /// Reads a sample from what an evaluator sees.
    ///
    /// - question: `inputs` when it is a string, else `inputs.question`,
    ///   `inputs.query` or `inputs.input`, else the inputs as JSON text;
    /// - answer and contexts: `output.answer` and `output.contexts` (strings
    ///   or `{"id", "content"}` objects; `output.sources` with documents is
    ///   read too). A string output that is not a JSON object is the answer;
    /// - reference: the reference string, or `reference.answer`;
    ///   relevant ids: `reference.relevant_ids` or `reference.source_ids`.
    ///
    /// ```
    /// use serde_json::json;
    /// use sinew_eval::{rag::RagSample, EvalInput};
    ///
    /// let inputs = json!({"question": "Where is Paris?"});
    /// let output = json!({"answer": "France", "contexts": [{"id": "fr", "content": "..."}]});
    /// let reference = json!({"answer": "In France", "source_ids": ["fr"]});
    /// let s = RagSample::from_eval(EvalInput { inputs: &inputs, output: &output, reference: Some(&reference) });
    /// assert_eq!(s.question, "Where is Paris?");
    /// assert_eq!(s.reference.as_deref(), Some("In France"));
    /// assert_eq!(s.relevant_ids, vec!["fr"]);
    /// ```
    pub fn from_eval(input: EvalInput<'_>) -> Self {
        let question = match input.inputs {
            Value::String(s) => s.clone(),
            v => ["question", "query", "input"]
                .iter()
                .find_map(|k| v.get(*k).map(as_text))
                .unwrap_or_else(|| as_text(v)),
        };
        let output = match input.output {
            Value::String(s) => match serde_json::from_str::<Value>(s.trim()) {
                Ok(v @ Value::Object(_)) => v,
                _ => Value::String(s.clone()),
            },
            v => v.clone(),
        };
        let (answer, contexts) = match &output {
            Value::Object(o) => {
                let answer = o.get("answer").map(as_text).unwrap_or_default();
                let list = o
                    .get("contexts")
                    .or_else(|| o.get("sources"))
                    .and_then(Value::as_array);
                let contexts = list
                    .map(|l| l.iter().filter_map(read_context).collect())
                    .unwrap_or_default();
                (answer, contexts)
            }
            other => (as_text(other), Vec::new()),
        };
        let (reference, relevant_ids) = match input.reference {
            None | Some(Value::Null) => (None, Vec::new()),
            Some(Value::String(s)) => (Some(s.clone()), Vec::new()),
            Some(Value::Object(o)) => {
                let ids = o
                    .get("relevant_ids")
                    .or_else(|| o.get("source_ids"))
                    .and_then(Value::as_array)
                    .map(|a| a.iter().map(as_text).collect())
                    .unwrap_or_default();
                let reference = o.get("answer").or_else(|| o.get("reference")).map(as_text);
                (reference, ids)
            }
            Some(other) => (Some(as_text(other)), Vec::new()),
        };
        Self {
            question,
            contexts,
            answer,
            reference,
            relevant_ids,
        }
    }
}

fn read_context(v: &Value) -> Option<RagContext> {
    match v {
        Value::String(s) => Some(RagContext {
            id: None,
            content: s.clone(),
        }),
        Value::Object(o) => {
            let content = ["content", "page_content", "text"]
                .iter()
                .find_map(|k| o.get(*k).and_then(Value::as_str))?
                .to_string();
            let id = o
                .get("id")
                .and_then(Value::as_str)
                .or_else(|| o.get("source").and_then(Value::as_str))
                .or_else(|| {
                    o.get("metadata")
                        .and_then(|m| m.get("source"))
                        .and_then(Value::as_str)
                })
                .map(str::to_string);
            Some(RagContext { id, content })
        }
        _ => None,
    }
}

/// The experiment output for a [`RagAnswer`]: the same shape as
/// [`RagSample::to_output`].
///
/// ```
/// use sinew_core::Usage;
/// use sinew_rag::{Document, RagAnswer};
///
/// let out = RagAnswer {
///     answer: "Paris [1].".into(),
///     sources: vec![Document::new("Paris is the capital.").with_id("fr")],
///     usage: Usage::default(),
/// };
/// let v = sinew_eval::rag::rag_output(&out);
/// assert_eq!(v["contexts"][0]["id"], "fr");
/// ```
pub fn rag_output(answer: &RagAnswer) -> Value {
    RagSample::from_answer("", answer).to_output()
}

/// Model calls made by one metric (or one test-set generation).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct JudgeUsage {
    pub calls: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Estimated cost in USD from the built-in price table; `None` when no
    /// call was made or any call's model has no known price.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

impl JudgeUsage {
    /// Input plus output tokens.
    ///
    /// ```
    /// let u = sinew_eval::rag::JudgeUsage { input_tokens: 10, output_tokens: 5, ..Default::default() };
    /// assert_eq!(u.total_tokens(), 15);
    /// ```
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// `"2 calls, 150 tokens, $0.000120"` (the cost only when known).
    ///
    /// ```
    /// let u = sinew_eval::rag::JudgeUsage { calls: 1, input_tokens: 10, output_tokens: 5, cost_usd: None };
    /// assert_eq!(u.summary(), "1 call, 15 tokens");
    /// ```
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} call{}, {} tokens",
            self.calls,
            if self.calls == 1 { "" } else { "s" },
            self.total_tokens()
        );
        if let Some(c) = self.cost_usd {
            s.push_str(&format!(", ${c:.6}"));
        }
        s
    }
}

/// The result of one RAG metric on one sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricResult {
    pub key: String,
    /// In `[0, 1]`.
    pub score: f64,
    pub reason: String,
    /// Metric-specific breakdown (claims and verdicts, per-context
    /// relevance, similarities, ...).
    #[serde(default)]
    pub details: Value,
    #[serde(default)]
    pub usage: JudgeUsage,
}

impl MetricResult {
    /// A result without model calls. The score is clamped to `[0, 1]`
    /// (non-finite becomes 0).
    ///
    /// ```
    /// let r = sinew_eval::rag::MetricResult::new("m", 1.5, "too high");
    /// assert_eq!(r.score, 1.0);
    /// ```
    pub fn new(key: impl Into<String>, score: f64, reason: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            // `+ 0.0` turns -0.0 (an empty float sum) into 0.0.
            score: if score.is_finite() {
                score.clamp(0.0, 1.0) + 0.0
            } else {
                0.0
            },
            reason: reason.into(),
            details: Value::Null,
            usage: JudgeUsage::default(),
        }
    }

    /// Sets the details.
    ///
    /// ```
    /// let r = sinew_eval::rag::MetricResult::new("m", 1.0, "ok").with_details(serde_json::json!({"n": 1}));
    /// assert_eq!(r.details["n"], 1);
    /// ```
    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }

    /// Sets the usage.
    ///
    /// ```
    /// use sinew_eval::rag::{JudgeUsage, MetricResult};
    /// let r = MetricResult::new("m", 1.0, "ok").with_usage(JudgeUsage { calls: 2, ..Default::default() });
    /// assert_eq!(r.usage.calls, 2);
    /// ```
    pub fn with_usage(mut self, usage: JudgeUsage) -> Self {
        self.usage = usage;
        self
    }

    /// The experiment [`Score`]: same key and score; the reason ends with
    /// the judge usage when the metric called a model.
    ///
    /// ```
    /// use sinew_eval::rag::{JudgeUsage, MetricResult};
    /// let usage = JudgeUsage { calls: 1, input_tokens: 7, output_tokens: 3, cost_usd: None };
    /// let s = MetricResult::new("faithfulness", 0.5, "1 of 2 claims supported").with_usage(usage).to_score();
    /// assert_eq!(s.reason.as_deref(), Some("1 of 2 claims supported [judge: 1 call, 10 tokens]"));
    /// ```
    pub fn to_score(&self) -> Score {
        let reason = if self.usage.calls > 0 {
            format!("{} [judge: {}]", self.reason, self.usage.summary())
        } else {
            self.reason.clone()
        };
        Score::new(self.key.clone(), self.score).with_reason(reason)
    }
}

/// A metric over a [`RagSample`]. Implement it for your own metric and wrap
/// it in [`RagEvaluator`] to use it in an experiment; the built-in metrics
/// implement [`Evaluator`] directly.
#[async_trait]
pub trait RagMetric: Send + Sync {
    /// Key under which the score is reported.
    fn key(&self) -> &str;
    /// Scores one sample. Must not panic: return a zero score with a reason.
    async fn measure(&self, sample: &RagSample) -> MetricResult;
}

/// Runs any [`RagMetric`] as an [`Evaluator`] (the sample is read with
/// [`RagSample::from_eval`]).
///
/// ```
/// use async_trait::async_trait;
/// use sinew_eval::{rag::{MetricResult, RagEvaluator, RagMetric, RagSample}, Evaluator};
///
/// struct NonEmpty;
/// #[async_trait]
/// impl RagMetric for NonEmpty {
///     fn key(&self) -> &str { "non_empty" }
///     async fn measure(&self, s: &RagSample) -> MetricResult {
///         MetricResult::new("non_empty", f64::from(u8::from(!s.answer.is_empty())), "")
///     }
/// }
/// let e: Box<dyn Evaluator> = Box::new(RagEvaluator(NonEmpty));
/// assert_eq!(e.name(), "non_empty");
/// ```
pub struct RagEvaluator<M>(pub M);

#[async_trait]
impl<M: RagMetric> Evaluator for RagEvaluator<M> {
    fn name(&self) -> &str {
        self.0.key()
    }
    async fn evaluate(&self, input: EvalInput<'_>) -> Score {
        self.0
            .measure(&RagSample::from_eval(input))
            .await
            .to_score()
    }
}

macro_rules! rag_evaluator {
    ($($t:ty),* $(,)?) => {$(
        #[async_trait]
        impl Evaluator for $t {
            fn name(&self) -> &str {
                RagMetric::key(self)
            }
            async fn evaluate(&self, input: EvalInput<'_>) -> Score {
                RagMetric::measure(self, &RagSample::from_eval(input)).await.to_score()
            }
        }
    )*};
}

rag_evaluator!(
    Faithfulness,
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    CitationAccuracy,
    AnswerSimilarity,
    RetrievalMetric,
);

/// Wraps a model for the duration of one metric and records the usage of
/// every call.
pub(crate) struct Meter<'a> {
    inner: &'a dyn ChatModel,
    state: Mutex<(JudgeUsage, bool)>,
}

impl<'a> Meter<'a> {
    pub(crate) fn new(inner: &'a dyn ChatModel) -> Self {
        Self {
            inner,
            state: Mutex::new((JudgeUsage::default(), true)),
        }
    }

    pub(crate) fn usage(&self) -> JudgeUsage {
        let (mut u, priced) = *self.state.lock().unwrap_or_else(|e| e.into_inner());
        if u.calls == 0 || !priced {
            u.cost_usd = None;
        }
        u
    }
}

#[async_trait]
impl ChatModel for Meter<'_> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    async fn chat(&self, request: ChatRequest) -> CoreResult<ChatResponse> {
        let resp = self.inner.chat(request).await?;
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.0.calls += 1;
        st.0.input_tokens += u64::from(resp.usage.input_tokens);
        st.0.output_tokens += u64::from(resp.usage.output_tokens);
        match resp.cost() {
            Some(c) => st.0.cost_usd = Some(st.0.cost_usd.unwrap_or(0.0) + c),
            None => st.1 = false,
        }
        Ok(resp)
    }
    async fn chat_stream(&self, request: ChatRequest) -> CoreResult<ChatStream> {
        self.inner.chat_stream(request).await
    }
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
}

/// Cosine similarity in `[-1, 1]`; 0 for empty or zero vectors and for
/// vectors of different lengths.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}
