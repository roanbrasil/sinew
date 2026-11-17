//! Ranking metrics over retrieved ids and relevant ids. No model calls.
//!
//! Relevance is binary: a retrieved id is relevant when it is in the
//! relevant set. Repeated retrieved ids (several chunks of the same source,
//! when ids are sources) count once, at their first rank.

use async_trait::async_trait;
use serde_json::json;

use super::{MetricResult, RagMetric, RagSample};

fn dedup<'a>(retrieved: &[&'a str]) -> Vec<&'a str> {
    let mut out: Vec<&str> = Vec::with_capacity(retrieved.len());
    for id in retrieved {
        if !out.contains(id) {
            out.push(id);
        }
    }
    out
}

fn is_relevant(id: &str, relevant: &[&str]) -> bool {
    relevant.contains(&id)
}

/// 1 when at least one relevant id is among the first `k` retrieved, else 0.
///
/// ```
/// use sinew_eval::rag::hit_rate_at_k;
/// assert_eq!(hit_rate_at_k(&["a", "b", "c"], &["c"], 2), 0.0);
/// assert_eq!(hit_rate_at_k(&["a", "b", "c"], &["c"], 3), 1.0);
/// ```
pub fn hit_rate_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let hit = dedup(retrieved)
        .iter()
        .take(k)
        .any(|id| is_relevant(id, relevant));
    f64::from(u8::from(hit))
}

/// Share of the relevant ids found among the first `k` retrieved
/// (0 when there are no relevant ids).
///
/// ```
/// use sinew_eval::rag::recall_at_k;
/// assert_eq!(recall_at_k(&["a", "b", "c"], &["a", "c", "z"], 3), 2.0 / 3.0);
/// ```
pub fn recall_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let relevant = dedup(relevant);
    if relevant.is_empty() {
        return 0.0;
    }
    let found = dedup(retrieved)
        .iter()
        .take(k)
        .filter(|id| is_relevant(id, &relevant))
        .count();
    found as f64 / relevant.len() as f64
}

/// Relevant ids among the first `k` retrieved, divided by `k` (fewer than
/// `k` results still divide by `k`).
///
/// ```
/// use sinew_eval::rag::precision_at_k;
/// assert_eq!(precision_at_k(&["a", "b", "c", "d"], &["a", "d"], 4), 0.5);
/// assert_eq!(precision_at_k(&["a"], &["a"], 2), 0.5);
/// ```
pub fn precision_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let found = dedup(retrieved)
        .iter()
        .take(k)
        .filter(|id| is_relevant(id, relevant))
        .count();
    found as f64 / k as f64
}

/// `1 / rank` of the first relevant id (ranks from 1), 0 when none is
/// retrieved. Averaged over examples this is the mean reciprocal rank.
///
/// ```
/// use sinew_eval::rag::reciprocal_rank;
/// assert_eq!(reciprocal_rank(&["a", "b", "c"], &["b", "c"]), 0.5);
/// ```
pub fn reciprocal_rank(retrieved: &[&str], relevant: &[&str]) -> f64 {
    dedup(retrieved)
        .iter()
        .position(|id| is_relevant(id, relevant))
        .map_or(0.0, |i| 1.0 / (i + 1) as f64)
}

/// Normalised discounted cumulative gain at `k` with binary relevance:
/// `DCG = Σ rel_i / log2(i + 1)` over ranks `i = 1..k`, divided by the DCG
/// of an ideal ranking (all relevant ids first).
///
/// ```
/// use sinew_eval::rag::ndcg_at_k;
/// assert_eq!(ndcg_at_k(&["a", "b"], &["a"], 2), 1.0);
/// let second = ndcg_at_k(&["b", "a"], &["a"], 2);
/// assert!((second - 1.0 / 3f64.log2()).abs() < 1e-12);
/// ```
pub fn ndcg_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let relevant = dedup(relevant);
    let gain = |rank: usize| 1.0 / ((rank + 1) as f64).log2();
    let dcg: f64 = dedup(retrieved)
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, id)| is_relevant(id, &relevant))
        .map(|(i, _)| gain(i + 1))
        .sum();
    let ideal: f64 = (1..=relevant.len().min(k)).map(gain).sum();
    if ideal == 0.0 {
        0.0
    } else {
        dcg / ideal
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    HitRate,
    Recall,
    Precision,
    Mrr,
    Ndcg,
}

/// A retrieval metric as a [`RagMetric`] and [`Evaluator`](crate::Evaluator):
/// compares the ids of the retrieved contexts ([`RagSample::context_ids`])
/// with [`RagSample::relevant_ids`]. Examples without relevant ids score 0
/// with a reason.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use sinew_eval::rag::{RagMetric, RagSample, RetrievalMetric};
///
/// let s = RagSample::new("q", "a")
///     .with_context("x", "...")
///     .with_context("y", "...")
///     .with_relevant_ids(["y"]);
/// assert_eq!(RetrievalMetric::hit_rate(1).measure(&s).await.score, 0.0);
/// assert_eq!(RetrievalMetric::recall(2).measure(&s).await.score, 1.0);
/// assert_eq!(RetrievalMetric::mrr().key(), "mrr");
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct RetrievalMetric {
    kind: Kind,
    k: usize,
    key: String,
}

impl RetrievalMetric {
    fn with(kind: Kind, k: usize, key: String) -> Self {
        Self { kind, k, key }
    }

    /// Hit rate at `k`, key `hit_rate@k`.
    ///
    /// ```
    /// use sinew_eval::rag::RagMetric;
    /// assert_eq!(sinew_eval::rag::RetrievalMetric::hit_rate(5).key(), "hit_rate@5");
    /// ```
    pub fn hit_rate(k: usize) -> Self {
        Self::with(Kind::HitRate, k, format!("hit_rate@{k}"))
    }

    /// Recall at `k`, key `recall@k`.
    ///
    /// ```
    /// use sinew_eval::rag::RagMetric;
    /// assert_eq!(sinew_eval::rag::RetrievalMetric::recall(5).key(), "recall@5");
    /// ```
    pub fn recall(k: usize) -> Self {
        Self::with(Kind::Recall, k, format!("recall@{k}"))
    }

    /// Precision at `k`, key `precision@k`.
    ///
    /// ```
    /// use sinew_eval::rag::RagMetric;
    /// assert_eq!(sinew_eval::rag::RetrievalMetric::precision(3).key(), "precision@3");
    /// ```
    pub fn precision(k: usize) -> Self {
        Self::with(Kind::Precision, k, format!("precision@{k}"))
    }

    /// Reciprocal rank over all retrieved contexts, key `mrr` (its mean in
    /// the experiment aggregates is the MRR).
    ///
    /// ```
    /// use sinew_eval::rag::RagMetric;
    /// assert_eq!(sinew_eval::rag::RetrievalMetric::mrr().key(), "mrr");
    /// ```
    pub fn mrr() -> Self {
        Self::with(Kind::Mrr, usize::MAX, "mrr".into())
    }

    /// nDCG at `k`, key `ndcg@k`.
    ///
    /// ```
    /// use sinew_eval::rag::RagMetric;
    /// assert_eq!(sinew_eval::rag::RetrievalMetric::ndcg(10).key(), "ndcg@10");
    /// ```
    pub fn ndcg(k: usize) -> Self {
        Self::with(Kind::Ndcg, k, format!("ndcg@{k}"))
    }

    /// Reports scores under a custom key.
    ///
    /// ```
    /// use sinew_eval::rag::RagMetric;
    /// let m = sinew_eval::rag::RetrievalMetric::mrr().named("mrr_chunks");
    /// assert_eq!(m.key(), "mrr_chunks");
    /// ```
    pub fn named(mut self, key: impl Into<String>) -> Self {
        self.key = key.into();
        self
    }

    /// The score for explicit id lists.
    ///
    /// ```
    /// use sinew_eval::rag::RetrievalMetric;
    /// assert_eq!(RetrievalMetric::precision(2).compute(&["a", "b"], &["b"]), 0.5);
    /// ```
    pub fn compute(&self, retrieved: &[&str], relevant: &[&str]) -> f64 {
        match self.kind {
            Kind::HitRate => hit_rate_at_k(retrieved, relevant, self.k),
            Kind::Recall => recall_at_k(retrieved, relevant, self.k),
            Kind::Precision => precision_at_k(retrieved, relevant, self.k),
            Kind::Mrr => reciprocal_rank(retrieved, relevant),
            Kind::Ndcg => ndcg_at_k(retrieved, relevant, self.k),
        }
    }
}

#[async_trait]
impl RagMetric for RetrievalMetric {
    fn key(&self) -> &str {
        &self.key
    }

    async fn measure(&self, sample: &RagSample) -> MetricResult {
        if sample.relevant_ids.is_empty() {
            return MetricResult::new(&self.key, 0.0, "example has no relevant ids");
        }
        let retrieved = sample.context_ids();
        if retrieved.is_empty() {
            return MetricResult::new(&self.key, 0.0, "no retrieved context has an id");
        }
        let relevant: Vec<&str> = sample.relevant_ids.iter().map(String::as_str).collect();
        let score = self.compute(&retrieved, &relevant);
        let ranks: Vec<usize> = dedup(&retrieved)
            .iter()
            .enumerate()
            .filter(|(_, id)| is_relevant(id, &relevant))
            .map(|(i, _)| i + 1)
            .collect();
        let reason = if ranks.is_empty() {
            format!(
                "none of {} relevant ids retrieved in {} results",
                relevant.len(),
                retrieved.len()
            )
        } else {
            format!(
                "relevant ids at ranks {:?} of {} results ({} relevant)",
                ranks,
                dedup(&retrieved).len(),
                relevant.len()
            )
        };
        MetricResult::new(&self.key, score, reason)
            .with_details(json!({"retrieved": retrieved, "relevant": relevant, "ranks": ranks}))
    }
}
