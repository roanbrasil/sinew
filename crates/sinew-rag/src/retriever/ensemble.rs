use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;

use super::Retriever;
use crate::{Document, Error, Result};

/// Weighted reciprocal rank fusion: every list contributes
/// `weight / (c + rank)` (rank starting at 1) to each of its documents;
/// documents are de-duplicated by id (or content) and sorted by the sum.
/// The fused score is stored as `rrf_score` metadata.
pub fn reciprocal_rank_fusion(lists: Vec<Vec<Document>>, weights: &[f32], c: f32) -> Vec<Document> {
    let mut scores: HashMap<String, (f32, usize)> = HashMap::new();
    let mut docs: Vec<Document> = Vec::new();
    for (li, list) in lists.into_iter().enumerate() {
        let w = weights.get(li).copied().unwrap_or(1.0);
        for (rank, doc) in list.into_iter().enumerate() {
            let add = w / (c + rank as f32 + 1.0);
            match scores.get_mut(doc.dedup_key()) {
                Some((s, _)) => *s += add,
                None => {
                    scores.insert(doc.dedup_key().to_string(), (add, docs.len()));
                    docs.push(doc);
                }
            }
        }
    }
    let mut ranked: Vec<(f32, usize)> = scores.into_values().collect();
    // Higher score first; ties keep first-seen order.
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut slots: Vec<Option<Document>> = docs.into_iter().map(Some).collect();
    ranked
        .into_iter()
        .filter_map(|(s, i)| {
            slots[i].take().map(|mut d| {
                d.metadata.insert("rrf_score".into(), (s as f64).into());
                d
            })
        })
        .collect()
}

/// Hybrid search: runs several retrievers concurrently and fuses their
/// rankings with weighted [`reciprocal_rank_fusion`] (LangChain's
/// `EnsembleRetriever`, `c` = 60).
///
/// ```no_run
/// # use std::sync::Arc;
/// # use sinew_rag::*;
/// # async fn demo(store: Arc<dyn VectorStore>, docs: Vec<Document>) -> sinew_rag::Result<()> {
/// let hybrid = EnsembleRetriever::new(vec![
///     Arc::new(Bm25Retriever::from_documents(docs)) as Arc<dyn Retriever>,
///     Arc::new(VectorStoreRetriever::new(store)),
/// ])
/// .with_weights(vec![0.4, 0.6])?
/// .with_k(5);
/// let hits = hybrid.retrieve("how do I resume a thread?").await?;
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct EnsembleRetriever {
    retrievers: Vec<Arc<dyn Retriever>>,
    weights: Vec<f32>,
    c: f32,
    k: Option<usize>,
}

impl EnsembleRetriever {
    /// Equal weights.
    pub fn new(retrievers: Vec<Arc<dyn Retriever>>) -> Self {
        let weights = vec![1.0; retrievers.len()];
        Self {
            retrievers,
            weights,
            c: 60.0,
            k: None,
        }
    }

    /// One weight per retriever.
    pub fn with_weights(mut self, weights: Vec<f32>) -> Result<Self> {
        if weights.len() != self.retrievers.len() {
            return Err(Error::Invalid(format!(
                "{} weights for {} retrievers",
                weights.len(),
                self.retrievers.len()
            )));
        }
        self.weights = weights;
        Ok(self)
    }

    /// RRF constant (default 60).
    pub fn with_c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Truncate the fused list (default: keep everything).
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }
}

#[async_trait]
impl Retriever for EnsembleRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<Document>> {
        let lists =
            futures_util::future::try_join_all(self.retrievers.iter().map(|r| r.retrieve(query)))
                .await?;
        let mut fused = reciprocal_rank_fusion(lists, &self.weights, self.c);
        if let Some(k) = self.k {
            fused.truncate(k);
        }
        Ok(fused)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(Vec<&'static str>);

    #[async_trait]
    impl Retriever for Fixed {
        async fn retrieve(&self, _: &str) -> Result<Vec<Document>> {
            Ok(self.0.iter().map(|s| Document::new(*s)).collect())
        }
    }

    #[tokio::test]
    async fn fuses_rankings() {
        let e = EnsembleRetriever::new(vec![
            Arc::new(Fixed(vec!["a", "b", "c"])),
            Arc::new(Fixed(vec!["c", "b", "d"])),
        ]);
        let got: Vec<String> = e
            .retrieve("q")
            .await
            .unwrap()
            .into_iter()
            .map(|d| d.content)
            .collect();
        // b: 1/62 + 1/62; a: 1/61; c: 1/63 + 1/61.
        assert_eq!(got, ["c", "b", "a", "d"]);

        let weighted = e.clone().with_weights(vec![10.0, 0.0]).unwrap().with_k(2);
        let got = weighted.retrieve("q").await.unwrap();
        assert_eq!(got[0].content, "a");
        assert_eq!(got.len(), 2);
        assert!(got[0].metadata["rrf_score"].as_f64().unwrap() > 0.0);
        assert!(e.with_weights(vec![1.0]).is_err());
    }
}
