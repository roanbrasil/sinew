//! Embedding-based RAG metrics.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use sinew_rag::Embeddings;

use super::{cosine, MetricResult, RagMetric, RagSample};

/// Semantic similarity of the answer and the reference answer: the cosine
/// similarity of their embeddings, negative values counted as 0. Needs a
/// reference. With [`threshold`](Self::threshold) the score is 1 when the
/// similarity reaches it and 0 otherwise.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use std::sync::Arc;
/// use sinew_eval::rag::{AnswerSimilarity, RagMetric, RagSample};
/// use sinew_rag::FakeEmbeddings;
///
/// let m = AnswerSimilarity::new(Arc::new(FakeEmbeddings::new(256)));
/// let same = RagSample::new("q", "Paris is the capital").with_reference("Paris is the capital");
/// assert!((m.measure(&same).await.score - 1.0).abs() < 1e-6);
/// # }
/// ```
#[derive(Clone)]
pub struct AnswerSimilarity {
    embeddings: Arc<dyn Embeddings>,
    key: String,
    threshold: Option<f64>,
}

impl AnswerSimilarity {
    /// Key `answer_similarity`.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let m = sinew_eval::rag::AnswerSimilarity::new(Arc::new(sinew_rag::FakeEmbeddings::new(64)));
    /// ```
    pub fn new(embeddings: Arc<dyn Embeddings>) -> Self {
        Self {
            embeddings,
            key: "answer_similarity".into(),
            threshold: None,
        }
    }

    /// Reports scores under a custom key.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// use sinew_eval::rag::{AnswerSimilarity, RagMetric};
    /// let m = AnswerSimilarity::new(Arc::new(sinew_rag::FakeEmbeddings::new(64))).named("semantic");
    /// assert_eq!(m.key(), "semantic");
    /// ```
    pub fn named(mut self, key: impl Into<String>) -> Self {
        self.key = key.into();
        self
    }

    /// Pass/fail: 1 when the similarity is at least `t`, else 0.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let m = sinew_eval::rag::AnswerSimilarity::new(Arc::new(sinew_rag::FakeEmbeddings::new(64)))
    ///     .threshold(0.8);
    /// ```
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = Some(t);
        self
    }
}

#[async_trait]
impl RagMetric for AnswerSimilarity {
    fn key(&self) -> &str {
        &self.key
    }

    async fn measure(&self, s: &RagSample) -> MetricResult {
        let Some(reference) = &s.reference else {
            return MetricResult::new(&self.key, 0.0, "example has no reference answer");
        };
        let texts = vec![s.answer.clone(), reference.clone()];
        let v = match self.embeddings.embed_documents(&texts).await {
            Ok(v) if v.len() == 2 => v,
            Ok(_) => {
                return MetricResult::new(&self.key, 0.0, "embeddings returned the wrong count")
            }
            Err(e) => return MetricResult::new(&self.key, 0.0, format!("embeddings failed: {e}")),
        };
        let sim = cosine(&v[0], &v[1]).max(0.0);
        let (score, reason) = match self.threshold {
            Some(t) => (
                f64::from(u8::from(sim >= t)),
                format!("similarity {sim:.3} (threshold {t})"),
            ),
            None => (sim, format!("similarity {sim:.3}")),
        };
        MetricResult::new(&self.key, score, reason)
            .with_details(json!({"similarity": sim, "model": self.embeddings.model()}))
    }
}
