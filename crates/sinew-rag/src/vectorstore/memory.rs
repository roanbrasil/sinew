use std::{collections::HashMap, ops::Range, sync::Arc, sync::RwLock};

use async_trait::async_trait;

use super::{
    check_lengths, dot, normalize,
    topk::{Hit, TopK},
    Filter, ScoredDocument, VectorStore,
};
use crate::{document::ensure_ids, Document, Embeddings, Error, Result};

/// Fast in-process vector store.
///
/// Vectors are normalised on insert and kept in one contiguous row-major
/// `Vec<f32>`, so a search is a linear scan of dot products (cosine
/// similarity) with a bounded min-heap for the top `k`, with no allocation
/// per row. Stores larger than [`with_parallel_threshold`](Self::with_parallel_threshold)
/// floats are scanned by several scoped threads (no thread pool
/// dependency). Deletes swap the last row into the hole, so they are O(dim).
///
/// Cloning is cheap and shares the data.
#[derive(Clone)]
pub struct InMemoryVectorStore {
    embeddings: Arc<dyn Embeddings>,
    inner: Arc<RwLock<Inner>>,
    parallel_threshold: usize,
    threads: usize,
}

#[derive(Default)]
struct Inner {
    dim: usize,
    data: Vec<f32>,
    docs: Vec<Document>,
    index: HashMap<String, usize>,
}

impl InMemoryVectorStore {
    pub fn new(embeddings: impl Embeddings + 'static) -> Self {
        Self::from_arc(Arc::new(embeddings))
    }

    pub fn from_arc(embeddings: Arc<dyn Embeddings>) -> Self {
        Self {
            embeddings,
            inner: Arc::default(),
            parallel_threshold: 1 << 21,
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(8),
        }
    }

    /// Embeds and indexes `docs`.
    pub async fn from_documents(
        embeddings: impl Embeddings + 'static,
        docs: Vec<Document>,
    ) -> Result<Self> {
        let store = Self::new(embeddings);
        store.add_documents(docs).await?;
        Ok(store)
    }

    /// Scan with several threads once the store holds at least `floats`
    /// values (rows × dimensions). Default 2,097,152 (≈ 5,000 × 384).
    pub fn with_parallel_threshold(mut self, floats: usize) -> Self {
        self.parallel_threshold = floats;
        self
    }

    /// Maximum scan threads (default: available cores, at most 8).
    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads = n.max(1);
        self
    }

    /// Vector dimension (0 while empty).
    pub fn dimensions(&self) -> usize {
        self.inner.read().unwrap().dim
    }

    /// Synchronous search with a precomputed query vector, returning
    /// `(document index, score)` pairs: the hot path, exposed for
    /// benchmarks. The query need not be normalised.
    pub fn search_vector_sync(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<(Document, f32)>> {
        let inner = self.inner.read().unwrap();
        let hits = self.scan(&inner, query, k, filter)?;
        Ok(hits
            .into_iter()
            .map(|h| (inner.docs[h.idx].clone(), h.score))
            .collect())
    }

    fn scan(
        &self,
        inner: &Inner,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Hit>> {
        let n = inner.docs.len();
        if n == 0 || k == 0 {
            return Ok(Vec::new());
        }
        if query.len() != inner.dim {
            return Err(Error::Dimensions {
                expected: inner.dim,
                got: query.len(),
            });
        }
        let mut q = query.to_vec();
        normalize(&mut q);
        let threads = if n * inner.dim >= self.parallel_threshold {
            self.threads.min(n)
        } else {
            1
        };
        if threads <= 1 {
            return Ok(scan_range(inner, &q, 0..n, k, filter).into_sorted());
        }
        let per = n.div_ceil(threads);
        let q = &q;
        let merged = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let range = (t * per).min(n)..((t + 1) * per).min(n);
                    s.spawn(move || scan_range(inner, q, range, k, filter))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("scan thread panicked"))
                .reduce(TopK::merge)
                .unwrap_or_else(|| TopK::new(k))
        });
        Ok(merged.into_sorted())
    }
}

fn scan_range(
    inner: &Inner,
    q: &[f32],
    rows: Range<usize>,
    k: usize,
    filter: Option<&Filter>,
) -> TopK {
    let dim = inner.dim;
    let mut top = TopK::new(k);
    let start = rows.start;
    let data = &inner.data[rows.start * dim..rows.end * dim];
    match filter {
        None => {
            for (i, row) in data.chunks_exact(dim).enumerate() {
                top.push(dot(q, row), start + i);
            }
        }
        Some(f) => {
            for (i, row) in data.chunks_exact(dim).enumerate() {
                if f.matches(&inner.docs[start + i].metadata) {
                    top.push(dot(q, row), start + i);
                }
            }
        }
    }
    top
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    fn embeddings(&self) -> &Arc<dyn Embeddings> {
        &self.embeddings
    }

    async fn add_vectors(
        &self,
        mut docs: Vec<Document>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<String>> {
        check_lengths(docs.len(), vectors.len())?;
        let ids = ensure_ids(&mut docs);
        let mut inner = self.inner.write().unwrap();
        if let Some(v) = vectors.first() {
            if inner.dim == 0 {
                if v.is_empty() {
                    return Err(Error::Invalid("empty embedding".into()));
                }
                inner.dim = v.len();
            }
        }
        let dim = inner.dim;
        if let Some(bad) = vectors.iter().find(|v| v.len() != dim) {
            return Err(Error::Dimensions {
                expected: dim,
                got: bad.len(),
            });
        }
        inner.data.reserve(vectors.len() * dim);
        for ((doc, mut v), id) in docs.into_iter().zip(vectors).zip(&ids) {
            normalize(&mut v);
            match inner.index.get(id).copied() {
                Some(row) => {
                    inner.data[row * dim..(row + 1) * dim].copy_from_slice(&v);
                    inner.docs[row] = doc;
                }
                None => {
                    let row = inner.docs.len();
                    inner.data.extend_from_slice(&v);
                    inner.docs.push(doc);
                    inner.index.insert(id.clone(), row);
                }
            }
        }
        Ok(ids)
    }

    async fn delete(&self, ids: &[String]) -> Result<usize> {
        let mut inner = self.inner.write().unwrap();
        let dim = inner.dim;
        let mut removed = 0;
        for id in ids {
            let Some(row) = inner.index.remove(id) else {
                continue;
            };
            removed += 1;
            let last = inner.docs.len() - 1;
            if row != last {
                inner
                    .data
                    .copy_within(last * dim..(last + 1) * dim, row * dim);
                if let Some(moved) = inner.docs[last].id.clone() {
                    inner.index.insert(moved, row);
                }
            }
            inner.docs.swap_remove(row);
            inner.data.truncate(last * dim);
        }
        if inner.docs.is_empty() {
            inner.dim = 0;
        }
        Ok(removed)
    }

    async fn get(&self, ids: &[String]) -> Result<Vec<Document>> {
        let inner = self.inner.read().unwrap();
        Ok(ids
            .iter()
            .filter_map(|id| inner.index.get(id).map(|&i| inner.docs[i].clone()))
            .collect())
    }

    async fn len(&self) -> Result<usize> {
        Ok(self.inner.read().unwrap().docs.len())
    }

    async fn search_by_vector(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        with_vectors: bool,
    ) -> Result<Vec<ScoredDocument>> {
        let inner = self.inner.read().unwrap();
        let hits = self.scan(&inner, query, k, filter)?;
        let dim = inner.dim;
        Ok(hits
            .into_iter()
            .map(|h| ScoredDocument {
                document: inner.docs[h.idx].clone(),
                score: h.score,
                vector: with_vectors.then(|| inner.data[h.idx * dim..(h.idx + 1) * dim].to_vec()),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeEmbeddings;

    fn store() -> InMemoryVectorStore {
        InMemoryVectorStore::new(FakeEmbeddings::new(64))
    }

    #[tokio::test]
    async fn add_search_upsert_delete() {
        let s = store();
        let ids = s
            .add_documents(vec![
                Document::new("rust borrow checker").with_metadata("lang", "rust"),
                Document::new("python garbage collector").with_metadata("lang", "py"),
                Document::new("rust async runtimes")
                    .with_id("r2")
                    .with_metadata("lang", "rust"),
            ])
            .await
            .unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[2], "r2");
        assert_eq!(s.len().await.unwrap(), 3);

        let hits = s
            .similarity_search_with_score("rust borrow", 2, None)
            .await
            .unwrap();
        assert_eq!(hits[0].0.content, "rust borrow checker");
        assert!(hits[0].1 > hits[1].1 && hits[0].1 <= 1.0001);

        let f = Filter::eq("lang", "py");
        let only_py = s.similarity_search("rust", 3, Some(&f)).await.unwrap();
        assert_eq!(only_py.len(), 1);
        assert_eq!(only_py[0].metadata["lang"], "py");

        // Upsert keeps the count and replaces content.
        s.add_documents(vec![Document::new("rust embedded").with_id("r2")])
            .await
            .unwrap();
        assert_eq!(s.len().await.unwrap(), 3);
        assert_eq!(
            s.get(&["r2".into()]).await.unwrap()[0].content,
            "rust embedded"
        );

        // Delete the first row: the last row moves into its slot.
        assert_eq!(s.delete(&[ids[0].clone(), "nope".into()]).await.unwrap(), 1);
        assert_eq!(s.len().await.unwrap(), 2);
        let got = s.get(&["r2".into(), ids[1].clone()]).await.unwrap();
        assert_eq!(got.len(), 2);
        let top = s.similarity_search("rust embedded", 1, None).await.unwrap();
        assert_eq!(top[0].id.as_deref(), Some("r2"));
    }

    #[tokio::test]
    async fn dimension_mismatch_is_an_error() {
        let s = store();
        s.add_vectors(vec![Document::new("a")], vec![vec![1.0, 0.0]])
            .await
            .unwrap();
        assert!(matches!(
            s.add_vectors(vec![Document::new("b")], vec![vec![1.0]])
                .await,
            Err(Error::Dimensions { .. })
        ));
        assert!(s.search_by_vector(&[1.0], 1, None, false).await.is_err());
        assert!(s
            .add_vectors(vec![Document::new("c")], vec![])
            .await
            .is_err());
    }

    #[tokio::test]
    async fn parallel_scan_matches_serial() {
        let e = FakeEmbeddings::new(32);
        let docs: Vec<Document> = (0..2000)
            .map(|i| Document::new(format!("doc {i} topic {}", i % 37)).with_metadata("i", i))
            .collect();
        let serial = InMemoryVectorStore::new(FakeEmbeddings::new(32)).with_threads(1);
        serial.add_documents(docs.clone()).await.unwrap();
        let parallel = InMemoryVectorStore::new(FakeEmbeddings::new(32))
            .with_threads(4)
            .with_parallel_threshold(0);
        parallel.add_documents(docs).await.unwrap();
        let q = e.embed("topic 5 doc 42");
        let a = serial.search_vector_sync(&q, 10, None).unwrap();
        let b = parallel.search_vector_sync(&q, 10, None).unwrap();
        let ids =
            |v: &[(Document, f32)]| v.iter().map(|(d, _)| d.content.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b));
        let f = Filter::lt("i", 100.0);
        let c = parallel.search_vector_sync(&q, 500, Some(&f)).unwrap();
        assert_eq!(c.len(), 100);
    }

    #[tokio::test]
    async fn mmr_search_returns_k_distinct() {
        let s = store();
        s.add_texts(vec![
            "apples are red".into(),
            "apples are red fruit".into(),
            "bananas are yellow".into(),
        ])
        .await
        .unwrap();
        let out = s
            .max_marginal_relevance_search("apples red", 2, 3, 0.2, None)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_ne!(out[0].content, out[1].content);
        assert!(out.iter().any(|d| d.content.contains("bananas")));
    }
}
