use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use sinew_core::{ChatModel, ChatRequest, Message};

use super::Retriever;
use crate::{vectorstore::dot, Document, Embeddings, Result};

/// Post-processes retrieved documents for a query: rerank, filter, trim.
#[async_trait]
pub trait DocumentCompressor: Send + Sync {
    async fn compress(&self, docs: Vec<Document>, query: &str) -> Result<Vec<Document>>;
}

/// Runs a base retriever, then a [`DocumentCompressor`] (LangChain's
/// `ContextualCompressionRetriever`). Typical use: fetch 20 candidates
/// cheaply, keep the best 4 after reranking.
#[derive(Clone)]
pub struct ContextualCompressionRetriever {
    base: Arc<dyn Retriever>,
    compressor: Arc<dyn DocumentCompressor>,
}

impl ContextualCompressionRetriever {
    pub fn new(base: Arc<dyn Retriever>, compressor: Arc<dyn DocumentCompressor>) -> Self {
        Self { base, compressor }
    }
}

#[async_trait]
impl Retriever for ContextualCompressionRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<Document>> {
        let docs = self.base.retrieve(query).await?;
        if docs.is_empty() {
            return Ok(docs);
        }
        self.compressor.compress(docs, query).await
    }
}

/// Listwise reranker that asks a chat model to score every passage from 0
/// to 10 in one call, then keeps the `top_n` best (default 3) with their
/// score in `relevance_score` metadata.
///
/// Passages are truncated to `max_chars` (default 1,500) to bound the
/// prompt. If the reply cannot be parsed, the original order is kept, so a
/// weak model degrades to "no reranking" rather than an error.
#[derive(Clone)]
pub struct LlmReranker {
    model: Arc<dyn ChatModel>,
    top_n: usize,
    min_score: Option<f32>,
    max_chars: usize,
}

impl LlmReranker {
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            model,
            top_n: 3,
            min_score: None,
            max_chars: 1500,
        }
    }

    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    /// Drop passages scored below `s` (0-10).
    pub fn with_min_score(mut self, s: f32) -> Self {
        self.min_score = Some(s);
        self
    }

    pub fn with_max_chars(mut self, n: usize) -> Self {
        self.max_chars = n.max(1);
        self
    }

    fn prompt(&self, docs: &[Document], query: &str) -> String {
        let mut p = String::from(
            "Rate how relevant each passage is to the query, from 0 (unrelated) to 10 \
             (answers it directly). Reply with only a JSON array covering every passage, \
             like [{\"index\": 0, \"score\": 7}, {\"index\": 1, \"score\": 2}].\n\nQuery: ",
        );
        p.push_str(query);
        p.push_str("\n\nPassages:\n");
        for (i, d) in docs.iter().enumerate() {
            let text: String = d.content.chars().take(self.max_chars).collect();
            p.push_str(&format!("[{i}] {}\n", text.replace('\n', " ")));
        }
        p
    }
}

/// Extracts `(index, score)` pairs from a model reply. Accepts an array of
/// `{"index", "score"}` objects or a plain array of indices in rank order.
pub(crate) fn parse_scores(reply: &str, n: usize) -> Option<Vec<(usize, f32)>> {
    let start = reply.find('[')?;
    let end = reply.rfind(']')?;
    if end <= start {
        return None;
    }
    let v: Value = serde_json::from_str(&reply[start..=end]).ok()?;
    let arr = v.as_array()?;
    let mut out = Vec::new();
    for (rank, item) in arr.iter().enumerate() {
        let (idx, score) = match item {
            Value::Object(o) => {
                let idx = o
                    .get("index")
                    .or_else(|| o.get("id"))
                    .and_then(Value::as_u64)?;
                let score = o
                    .get("score")
                    .or_else(|| o.get("relevance"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                (idx as usize, score as f32)
            }
            Value::Number(x) => (x.as_u64()? as usize, (arr.len() - rank) as f32),
            _ => return None,
        };
        if idx < n && !out.iter().any(|(i, _)| *i == idx) {
            out.push((idx, score));
        }
    }
    (!out.is_empty()).then_some(out)
}

#[async_trait]
impl DocumentCompressor for LlmReranker {
    async fn compress(&self, docs: Vec<Document>, query: &str) -> Result<Vec<Document>> {
        let resp = self
            .model
            .chat(
                ChatRequest::new(vec![Message::user(self.prompt(&docs, query))])
                    .with_temperature(0.0),
            )
            .await?;
        let Some(mut scores) = parse_scores(&resp.message.content, docs.len()) else {
            let mut docs = docs;
            docs.truncate(self.top_n);
            return Ok(docs);
        };
        // Passages the model skipped rank last, in original order.
        for i in 0..docs.len() {
            if !scores.iter().any(|(j, _)| *j == i) {
                scores.push((i, f32::NEG_INFINITY));
            }
        }
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut slots: Vec<Option<Document>> = docs.into_iter().map(Some).collect();
        Ok(scores
            .into_iter()
            .filter(|(_, s)| self.min_score.is_none_or(|m| *s >= m))
            .take(self.top_n)
            .filter_map(|(i, s)| {
                slots[i].take().map(|mut d| {
                    if s.is_finite() {
                        d.metadata
                            .insert("relevance_score".into(), (s as f64).into());
                    }
                    d
                })
            })
            .collect())
    }
}

/// Keeps documents whose embedding similarity to the query is at least
/// `threshold`, best first (LangChain's `EmbeddingsFilter`). Useful after a
/// keyword retriever, or to drop weak vector hits.
#[derive(Clone)]
pub struct EmbeddingsFilter {
    embeddings: Arc<dyn Embeddings>,
    threshold: f32,
    k: Option<usize>,
}

impl EmbeddingsFilter {
    pub fn new(embeddings: Arc<dyn Embeddings>, threshold: f32) -> Self {
        Self {
            embeddings,
            threshold,
            k: None,
        }
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }
}

#[async_trait]
impl DocumentCompressor for EmbeddingsFilter {
    async fn compress(&self, docs: Vec<Document>, query: &str) -> Result<Vec<Document>> {
        let texts: Vec<String> = docs.iter().map(|d| d.content.clone()).collect();
        let (q, vs) = futures_util::future::try_join(
            self.embeddings.embed_query(query),
            self.embeddings.embed_documents(&texts),
        )
        .await?;
        let qn = dot(&q, &q).sqrt().max(f32::MIN_POSITIVE);
        let mut scored: Vec<(f32, Document)> = docs
            .into_iter()
            .zip(vs)
            .map(|(d, v)| {
                let vn = dot(&v, &v).sqrt().max(f32::MIN_POSITIVE);
                (dot(&q, &v) / (qn * vn), d)
            })
            .filter(|(s, _)| *s >= self.threshold)
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        if let Some(k) = self.k {
            scored.truncate(k);
        }
        Ok(scored.into_iter().map(|(_, d)| d).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeEmbeddings;
    use sinew_core::mock::ScriptedModel;

    fn docs() -> Vec<Document> {
        ["alpha", "beta", "gamma"]
            .into_iter()
            .map(Document::new)
            .collect()
    }

    #[test]
    fn parses_reranker_replies() {
        assert_eq!(
            parse_scores(
                "Sure: [{\"index\": 2, \"score\": 9}, {\"index\": 0, \"score\": 1}]",
                3
            ),
            Some(vec![(2, 9.0), (0, 1.0)])
        );
        assert_eq!(parse_scores("[1, 0]", 3), Some(vec![(1, 2.0), (0, 1.0)]));
        assert_eq!(parse_scores("no json", 3), None);
        assert_eq!(parse_scores("[7]", 3), None);
    }

    #[tokio::test]
    async fn llm_reranker_reorders_and_trims() {
        let model = Arc::new(ScriptedModel::new([
            Message::assistant(
                r#"[{"index":0,"score":2},{"index":1,"score":8},{"index":2,"score":5}]"#,
            ),
            Message::assistant("I cannot do that"),
        ]));
        let r = LlmReranker::new(model.clone()).with_top_n(2);
        let got = r.compress(docs(), "q").await.unwrap();
        assert_eq!(got[0].content, "beta");
        assert_eq!(got[1].content, "gamma");
        assert_eq!(got[0].metadata["relevance_score"], 8.0);
        assert!(model.requests()[0].messages[0]
            .content
            .contains("[2] gamma"));
        // Unparseable reply: original order.
        let got = r.compress(docs(), "q").await.unwrap();
        assert_eq!(got[0].content, "alpha");
        assert_eq!(got.len(), 2);
    }

    #[tokio::test]
    async fn embeddings_filter_and_compression_retriever() {
        struct Base;
        #[async_trait]
        impl Retriever for Base {
            async fn retrieve(&self, _: &str) -> Result<Vec<Document>> {
                Ok(vec![
                    Document::new("pasta recipes"),
                    Document::new("rust compiler errors"),
                    Document::new("the rust compiler"),
                ])
            }
        }
        let f = EmbeddingsFilter::new(Arc::new(FakeEmbeddings::new(128)), 0.3);
        let r = ContextualCompressionRetriever::new(Arc::new(Base), Arc::new(f));
        let got = r.retrieve("rust compiler").await.unwrap();
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|d| d.content.contains("rust")));
    }
}
