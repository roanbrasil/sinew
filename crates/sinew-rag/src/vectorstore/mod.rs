//! Vector stores: index embedded documents and search them by similarity.
//!
//! | Store | Feature | Notes |
//! |---|---|---|
//! | [`InMemoryVectorStore`] | always | contiguous `f32` matrix, normalised vectors, heap top-k, multi-threaded scan for large stores |
//! | [`SqliteVectorStore`] | `sqlite` (default) | vectors as blobs, brute-force scan, metadata filters evaluated in Rust |
//! | [`QdrantVectorStore`] | `qdrant` | Qdrant REST API, filters translated to Qdrant filters |
//! | [`PgVectorStore`] | `pgvector` | Postgres + pgvector, cosine distance, filters translated to SQL on `jsonb` |
//!
//! Every store reports **cosine similarity** as the score (higher is more
//! similar, at most 1).

mod filter;
mod memory;
#[cfg(feature = "pgvector")]
mod pgvector;
#[cfg(feature = "qdrant")]
mod qdrant;
#[cfg(feature = "sqlite")]
mod sqlite;

use std::sync::Arc;

use async_trait::async_trait;

pub use filter::Filter;
pub use memory::InMemoryVectorStore;
#[cfg(feature = "pgvector")]
pub use pgvector::PgVectorStore;
#[cfg(feature = "qdrant")]
pub use qdrant::QdrantVectorStore;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteVectorStore;

use crate::{Document, Embeddings, Error, Result};

/// A search hit.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredDocument {
    pub document: Document,
    /// Cosine similarity to the query.
    pub score: f32,
    /// The stored (normalised) vector, when it was requested.
    pub vector: Option<Vec<f32>>,
}

/// Index of embedded documents.
///
/// Implementors provide storage and vector search
/// ([`add_vectors`](Self::add_vectors), [`delete`](Self::delete),
/// [`get`](Self::get), [`search_by_vector`](Self::search_by_vector)); text
/// search, batch indexing and MMR come for free.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// The model used to embed documents and queries.
    fn embeddings(&self) -> &Arc<dyn Embeddings>;

    /// Inserts or replaces documents with precomputed vectors (one per
    /// document). Documents without an id get a UUID. Returns the ids.
    async fn add_vectors(&self, docs: Vec<Document>, vectors: Vec<Vec<f32>>)
        -> Result<Vec<String>>;

    /// Removes documents by id; returns how many existed.
    async fn delete(&self, ids: &[String]) -> Result<usize>;

    /// Fetches documents by id (missing ids are skipped).
    async fn get(&self, ids: &[String]) -> Result<Vec<Document>>;

    /// Number of stored documents.
    async fn len(&self) -> Result<usize>;

    async fn is_empty(&self) -> Result<bool> {
        Ok(self.len().await? == 0)
    }

    /// The `k` most similar documents to `query`, best first, optionally
    /// restricted by a metadata filter. `with_vectors` also returns the
    /// stored vectors (needed for MMR).
    async fn search_by_vector(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        with_vectors: bool,
    ) -> Result<Vec<ScoredDocument>>;

    /// Embeds the documents in one batch call and stores them.
    async fn add_documents(&self, docs: Vec<Document>) -> Result<Vec<String>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = docs.iter().map(|d| d.content.clone()).collect();
        let vectors = self.embeddings().embed_documents(&texts).await?;
        self.add_vectors(docs, vectors).await
    }

    /// Indexes plain strings.
    async fn add_texts(&self, texts: Vec<String>) -> Result<Vec<String>> {
        self.add_documents(texts.into_iter().map(Document::new).collect())
            .await
    }

    async fn similarity_search_with_score(
        &self,
        query: &str,
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<(Document, f32)>> {
        let v = self.embeddings().embed_query(query).await?;
        Ok(self
            .search_by_vector(&v, k, filter, false)
            .await?
            .into_iter()
            .map(|h| (h.document, h.score))
            .collect())
    }

    async fn similarity_search(
        &self,
        query: &str,
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Document>> {
        Ok(self
            .similarity_search_with_score(query, k, filter)
            .await?
            .into_iter()
            .map(|(d, _)| d)
            .collect())
    }

    /// Maximal marginal relevance: fetches `fetch_k` candidates, then picks
    /// `k` that are relevant to the query but different from each other.
    /// `lambda` = 1 is pure relevance, 0 pure diversity (LangChain default
    /// 0.5).
    async fn max_marginal_relevance_search(
        &self,
        query: &str,
        k: usize,
        fetch_k: usize,
        lambda: f32,
        filter: Option<&Filter>,
    ) -> Result<Vec<Document>> {
        let q = self.embeddings().embed_query(query).await?;
        let hits = self
            .search_by_vector(&q, fetch_k.max(k), filter, true)
            .await?;
        let vectors: Vec<&[f32]> = hits
            .iter()
            .map(|h| h.vector.as_deref().unwrap_or(&[]))
            .collect();
        let picked = maximal_marginal_relevance(&q, &vectors, k, lambda);
        let mut hits: Vec<Option<ScoredDocument>> = hits.into_iter().map(Some).collect();
        Ok(picked
            .into_iter()
            .filter_map(|i| hits[i].take().map(|h| h.document))
            .collect())
    }
}

#[async_trait]
impl<T: VectorStore + ?Sized> VectorStore for Arc<T> {
    fn embeddings(&self) -> &Arc<dyn Embeddings> {
        (**self).embeddings()
    }
    async fn add_vectors(
        &self,
        docs: Vec<Document>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<String>> {
        (**self).add_vectors(docs, vectors).await
    }
    async fn delete(&self, ids: &[String]) -> Result<usize> {
        (**self).delete(ids).await
    }
    async fn get(&self, ids: &[String]) -> Result<Vec<Document>> {
        (**self).get(ids).await
    }
    async fn len(&self) -> Result<usize> {
        (**self).len().await
    }
    async fn search_by_vector(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        with_vectors: bool,
    ) -> Result<Vec<ScoredDocument>> {
        (**self)
            .search_by_vector(query, k, filter, with_vectors)
            .await
    }
}

/// Indices of `k` candidates chosen by maximal marginal relevance, in pick
/// order. Vectors need not be normalised.
pub fn maximal_marginal_relevance(
    query: &[f32],
    candidates: &[&[f32]],
    k: usize,
    lambda: f32,
) -> Vec<usize> {
    let k = k.min(candidates.len());
    if k == 0 {
        return Vec::new();
    }
    let norm = |v: &[f32]| dot(v, v).sqrt().max(f32::MIN_POSITIVE);
    let qn = norm(query);
    let norms: Vec<f32> = candidates.iter().map(|c| norm(c)).collect();
    let cos = |a: usize, b: usize| {
        if candidates[a].len() != candidates[b].len() {
            return 0.0;
        }
        dot(candidates[a], candidates[b]) / (norms[a] * norms[b])
    };
    let relevance: Vec<f32> = candidates
        .iter()
        .zip(&norms)
        .map(|(c, n)| {
            if c.len() == query.len() {
                dot(c, query) / (n * qn)
            } else {
                0.0
            }
        })
        .collect();
    // Best similarity of each candidate to anything already picked.
    let mut max_sim = vec![f32::NEG_INFINITY; candidates.len()];
    let mut picked = Vec::with_capacity(k);
    let mut used = vec![false; candidates.len()];
    for _ in 0..k {
        let mut best = None;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..candidates.len() {
            if used[i] {
                continue;
            }
            let redundancy = if picked.is_empty() { 0.0 } else { max_sim[i] };
            let s = lambda * relevance[i] - (1.0 - lambda) * redundancy;
            if best.is_none() || s > best_score {
                best = Some(i);
                best_score = s;
            }
        }
        let Some(b) = best else { break };
        used[b] = true;
        picked.push(b);
        for i in 0..candidates.len() {
            if !used[i] {
                max_sim[i] = max_sim[i].max(cos(i, b));
            }
        }
    }
    picked
}

/// Dot product, written so LLVM vectorises it (8 independent lanes).
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for ((s, xi), yi) in acc.iter_mut().zip(x).zip(y) {
            *s += xi * yi;
        }
    }
    let mut sum = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// Scales `v` to unit length in place (zero vectors stay zero).
#[inline]
pub(crate) fn normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt();
    if n > 0.0 && (n - 1.0).abs() > 1e-6 {
        let inv = 1.0 / n;
        v.iter_mut().for_each(|x| *x *= inv);
    }
}

pub(crate) fn check_lengths(docs: usize, vectors: usize) -> Result<()> {
    if docs != vectors {
        return Err(Error::Invalid(format!(
            "{docs} documents but {vectors} vectors"
        )));
    }
    Ok(())
}

/// Fixed-size top-k selection with a min-heap.
pub(crate) mod topk {
    use std::{cmp::Ordering, collections::BinaryHeap};

    #[derive(Clone, Copy, Debug)]
    pub struct Hit {
        pub score: f32,
        pub idx: usize,
    }

    impl PartialEq for Hit {
        fn eq(&self, o: &Self) -> bool {
            self.cmp(o) == Ordering::Equal
        }
    }
    impl Eq for Hit {}
    impl PartialOrd for Hit {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Hit {
        /// Reversed so `BinaryHeap` keeps the *worst* hit on top; ties
        /// prefer the lower index.
        fn cmp(&self, o: &Self) -> Ordering {
            o.score
                .total_cmp(&self.score)
                .then_with(|| self.idx.cmp(&o.idx))
        }
    }

    pub struct TopK {
        k: usize,
        heap: BinaryHeap<Hit>,
    }

    impl TopK {
        pub fn new(k: usize) -> Self {
            Self {
                k,
                heap: BinaryHeap::with_capacity(k + 1),
            }
        }

        #[inline]
        pub fn push(&mut self, score: f32, idx: usize) {
            if self.k == 0 || score.is_nan() {
                return;
            }
            if self.heap.len() < self.k {
                self.heap.push(Hit { score, idx });
            } else if let Some(mut worst) = self.heap.peek_mut() {
                if score > worst.score {
                    *worst = Hit { score, idx };
                }
            }
        }

        /// Whether `score` would enter the current top k.
        #[inline]
        pub fn accepts(&self, score: f32) -> bool {
            self.k > 0
                && !score.is_nan()
                && (self.heap.len() < self.k || self.heap.peek().is_some_and(|w| score > w.score))
        }

        /// Best first.
        pub fn into_sorted(self) -> Vec<Hit> {
            // Ascending by the reversed order = descending by score.
            self.heap.into_sorted_vec()
        }

        pub fn merge(mut self, other: TopK) -> Self {
            for h in other.heap {
                self.push(h.score, h.idx);
            }
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_naive() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..37).map(|i| (37 - i) as f32 * 0.2).collect();
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - naive).abs() < 1e-3);
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert_eq!(v, vec![0.6, 0.8]);
    }

    #[test]
    fn topk_keeps_best() {
        let mut t = topk::TopK::new(3);
        for (i, s) in [0.1, 0.9, 0.5, f32::NAN, 0.7, 0.2].into_iter().enumerate() {
            t.push(s, i);
        }
        let got: Vec<usize> = t.into_sorted().into_iter().map(|h| h.idx).collect();
        assert_eq!(got, vec![1, 4, 2]);
    }

    #[test]
    fn mmr_prefers_diverse_results() {
        let q = [1.0, 0.0];
        let a = [1.0, 0.0];
        let a2 = [0.99, 0.01];
        let b = [0.7, 0.7];
        let c: [&[f32]; 3] = [&a, &a2, &b];
        assert_eq!(maximal_marginal_relevance(&q, &c, 2, 0.3), vec![0, 2]);
        assert_eq!(maximal_marginal_relevance(&q, &c, 2, 1.0), vec![0, 1]);
        assert!(maximal_marginal_relevance(&q, &c, 0, 0.5).is_empty());
    }
}
