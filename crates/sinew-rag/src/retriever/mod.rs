//! Retrievers: anything that turns a query into relevant documents.
//!
//! - [`VectorStoreRetriever`]: similarity, score threshold or MMR search;
//! - [`Bm25Retriever`]: keyword search (Okapi BM25, pure Rust);
//! - [`EnsembleRetriever`]: hybrid search, fusing several retrievers with
//!   weighted reciprocal rank fusion;
//! - [`MultiQueryRetriever`]: expands the question into several queries
//!   ([`QueryExpander`], e.g. [`LlmQueryExpander`]) and merges the results;
//! - [`ContextualCompressionRetriever`]: post-processes results with a
//!   [`DocumentCompressor`] such as [`LlmReranker`] or [`EmbeddingsFilter`];
//! - [`ParentDocumentRetriever`]: searches small chunks, returns their
//!   larger parent documents.

mod bm25;
mod compression;
mod ensemble;
mod multi_query;
mod parent;

use std::sync::Arc;

use async_trait::async_trait;

pub use bm25::Bm25Retriever;
pub use compression::{
    ContextualCompressionRetriever, DocumentCompressor, EmbeddingsFilter, LlmReranker,
};
pub use ensemble::{reciprocal_rank_fusion, EnsembleRetriever};
pub use multi_query::{LlmQueryExpander, MultiQueryRetriever, QueryExpander};
pub use parent::{DocStore, InMemoryDocStore, ParentDocumentRetriever};

use crate::{Document, Filter, Result, VectorStore};

/// Returns documents relevant to a query, best first.
#[async_trait]
pub trait Retriever: Send + Sync {
    async fn retrieve(&self, query: &str) -> Result<Vec<Document>>;
}

#[async_trait]
impl<T: Retriever + ?Sized> Retriever for Arc<T> {
    async fn retrieve(&self, query: &str) -> Result<Vec<Document>> {
        (**self).retrieve(query).await
    }
}

#[async_trait]
impl<T: Retriever + ?Sized> Retriever for Box<T> {
    async fn retrieve(&self, query: &str) -> Result<Vec<Document>> {
        (**self).retrieve(query).await
    }
}

/// How a [`VectorStoreRetriever`] searches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchType {
    /// Top `k` by cosine similarity.
    Similarity,
    /// Top `k` with similarity at least the threshold.
    SimilarityScoreThreshold(f32),
    /// Maximal marginal relevance over `fetch_k` candidates.
    Mmr { fetch_k: usize, lambda: f32 },
}

/// Retriever over a [`VectorStore`] (LangChain's `vectorstore.as_retriever()`).
///
/// With [`with_scores`](Self::with_scores) each document carries its
/// similarity in the `score` metadata field.
#[derive(Clone)]
pub struct VectorStoreRetriever {
    store: Arc<dyn VectorStore>,
    k: usize,
    search: SearchType,
    filter: Option<Filter>,
    scores: bool,
}

impl VectorStoreRetriever {
    /// Similarity search, `k` = 4.
    pub fn new(store: Arc<dyn VectorStore>) -> Self {
        Self {
            store,
            k: 4,
            search: SearchType::Similarity,
            filter: None,
            scores: false,
        }
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn with_search_type(mut self, search: SearchType) -> Self {
        self.search = search;
        self
    }

    /// MMR with `fetch_k` = 20 and `lambda` = 0.5 (LangChain defaults).
    pub fn mmr(self) -> Self {
        self.with_search_type(SearchType::Mmr {
            fetch_k: 20,
            lambda: 0.5,
        })
    }

    pub fn with_score_threshold(self, threshold: f32) -> Self {
        self.with_search_type(SearchType::SimilarityScoreThreshold(threshold))
    }

    pub fn with_filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_scores(mut self) -> Self {
        self.scores = true;
        self
    }

    pub fn store(&self) -> &Arc<dyn VectorStore> {
        &self.store
    }
}

#[async_trait]
impl Retriever for VectorStoreRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<Document>> {
        let filter = self.filter.as_ref();
        let scored = match self.search {
            SearchType::Mmr { fetch_k, lambda } => {
                return self
                    .store
                    .max_marginal_relevance_search(query, self.k, fetch_k, lambda, filter)
                    .await
            }
            SearchType::Similarity => {
                self.store
                    .similarity_search_with_score(query, self.k, filter)
                    .await?
            }
            SearchType::SimilarityScoreThreshold(t) => self
                .store
                .similarity_search_with_score(query, self.k, filter)
                .await?
                .into_iter()
                .filter(|(_, s)| *s >= t)
                .collect(),
        };
        Ok(scored
            .into_iter()
            .map(|(mut d, s)| {
                if self.scores {
                    d.metadata.insert("score".into(), (s as f64).into());
                }
                d
            })
            .collect())
    }
}

/// Removes later duplicates (same id, or same content when there is no id).
pub(crate) fn dedup(docs: Vec<Document>) -> Vec<Document> {
    let mut seen = std::collections::HashSet::new();
    docs.into_iter()
        .filter(|d| seen.insert(d.dedup_key().to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FakeEmbeddings, InMemoryVectorStore};

    #[tokio::test]
    async fn vector_retriever_modes() {
        let store = InMemoryVectorStore::new(FakeEmbeddings::new(64));
        store
            .add_documents(vec![
                Document::new("cats purr").with_metadata("kind", "pet"),
                Document::new("dogs bark").with_metadata("kind", "pet"),
                Document::new("cars honk").with_metadata("kind", "machine"),
            ])
            .await
            .unwrap();
        let store: Arc<dyn VectorStore> = Arc::new(store);
        let r = VectorStoreRetriever::new(store.clone())
            .with_k(1)
            .with_scores();
        let got = r.retrieve("cats").await.unwrap();
        assert_eq!(got[0].content, "cats purr");
        assert!(got[0].metadata["score"].as_f64().unwrap() > 0.3);

        let r = VectorStoreRetriever::new(store.clone()).with_score_threshold(0.99);
        assert!(r.retrieve("cats").await.unwrap().is_empty());

        let r = VectorStoreRetriever::new(store.clone())
            .with_filter(Filter::eq("kind", "machine"))
            .with_k(5);
        assert_eq!(r.retrieve("cats").await.unwrap().len(), 1);

        let r = VectorStoreRetriever::new(store).mmr().with_k(2);
        assert_eq!(r.retrieve("cats").await.unwrap().len(), 2);
    }
}
