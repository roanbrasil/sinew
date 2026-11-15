use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;

use super::Embeddings;
use crate::Result;

/// Key-value storage for embeddings (LangChain's `ByteStore` behind
/// `CacheBackedEmbeddings`). Implement it to back the cache with Redis, a
/// database or files.
#[async_trait]
pub trait EmbeddingCache: Send + Sync {
    /// One entry per key, `None` for misses.
    async fn get_many(&self, keys: &[String]) -> Vec<Option<Vec<f32>>>;
    async fn put_many(&self, entries: Vec<(String, Vec<f32>)>);
}

/// In-process cache, optionally bounded (oldest entries are evicted first).
#[derive(Default)]
pub struct InMemoryEmbeddingCache {
    inner: Mutex<CacheInner>,
    capacity: Option<usize>,
}

#[derive(Default)]
struct CacheInner {
    map: HashMap<String, Arc<[f32]>>,
    order: VecDeque<String>,
}

impl InMemoryEmbeddingCache {
    /// Unbounded.
    pub fn new() -> Self {
        Self::default()
    }

    /// Keeps at most `capacity` vectors.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::default(),
            capacity: Some(capacity.max(1)),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl EmbeddingCache for InMemoryEmbeddingCache {
    async fn get_many(&self, keys: &[String]) -> Vec<Option<Vec<f32>>> {
        let inner = self.inner.lock().unwrap();
        keys.iter()
            .map(|k| inner.map.get(k).map(|v| v.to_vec()))
            .collect()
    }

    async fn put_many(&self, entries: Vec<(String, Vec<f32>)>) {
        let mut inner = self.inner.lock().unwrap();
        for (k, v) in entries {
            if inner.map.insert(k.clone(), v.into()).is_none() {
                inner.order.push_back(k);
            }
        }
        if let Some(cap) = self.capacity {
            while inner.map.len() > cap {
                match inner.order.pop_front() {
                    Some(old) => {
                        inner.map.remove(&old);
                    }
                    None => break,
                }
            }
        }
    }
}

/// Wraps an embedding model with a cache keyed by model name and text.
///
/// Only cache misses reach the inner model, de-duplicated and in one batch.
/// Queries are cached too unless [`without_query_cache`](Self::without_query_cache).
pub struct CachedEmbeddings<E> {
    inner: E,
    cache: Arc<dyn EmbeddingCache>,
    namespace: String,
    cache_queries: bool,
}

impl<E: Embeddings> CachedEmbeddings<E> {
    /// Uses a fresh unbounded [`InMemoryEmbeddingCache`].
    pub fn new(inner: E) -> Self {
        Self::with_cache(inner, Arc::new(InMemoryEmbeddingCache::new()))
    }

    pub fn with_cache(inner: E, cache: Arc<dyn EmbeddingCache>) -> Self {
        let namespace = inner.model().to_string();
        Self {
            inner,
            cache,
            namespace,
            cache_queries: true,
        }
    }

    /// Overrides the key prefix (defaults to the model name).
    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    pub fn without_query_cache(mut self) -> Self {
        self.cache_queries = false;
        self
    }

    pub fn inner(&self) -> &E {
        &self.inner
    }

    fn key(&self, text: &str) -> String {
        let mut k = String::with_capacity(self.namespace.len() + 1 + text.len());
        k.push_str(&self.namespace);
        k.push('\u{0}');
        k.push_str(text);
        k
    }
}

#[async_trait]
impl<E: Embeddings> Embeddings for CachedEmbeddings<E> {
    fn model(&self) -> &str {
        self.inner.model()
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let keys: Vec<String> = texts.iter().map(|t| self.key(t)).collect();
        let cached = self.cache.get_many(&keys).await;
        let mut out: Vec<Option<Vec<f32>>> = cached;
        // De-duplicated misses, in first-seen order.
        let mut miss_index: HashMap<&str, usize> = HashMap::new();
        let mut misses: Vec<String> = Vec::new();
        for (i, v) in out.iter().enumerate() {
            if v.is_none() && !miss_index.contains_key(texts[i].as_str()) {
                miss_index.insert(texts[i].as_str(), misses.len());
                misses.push(texts[i].clone());
            }
        }
        if !misses.is_empty() {
            let fresh = self.inner.embed_documents(&misses).await?;
            for (i, slot) in out.iter_mut().enumerate() {
                if slot.is_none() {
                    *slot = Some(fresh[miss_index[texts[i].as_str()]].clone());
                }
            }
            let entries = misses
                .iter()
                .map(|t| self.key(t))
                .zip(fresh)
                .collect::<Vec<_>>();
            self.cache.put_many(entries).await;
        }
        Ok(out.into_iter().map(|v| v.unwrap_or_default()).collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        if !self.cache_queries {
            return self.inner.embed_query(text).await;
        }
        let key = format!("{}\u{0}query\u{0}{text}", self.namespace);
        if let Some(Some(v)) = self.cache.get_many(std::slice::from_ref(&key)).await.pop() {
            return Ok(v);
        }
        let v = self.inner.embed_query(text).await?;
        self.cache.put_many(vec![(key, v.clone())]).await;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeEmbeddings;

    #[tokio::test]
    async fn only_misses_reach_the_model() {
        let e = CachedEmbeddings::new(FakeEmbeddings::new(16));
        let a: Vec<String> = vec!["a".into(), "b".into(), "a".into()];
        let first = e.embed_documents(&a).await.unwrap();
        assert_eq!(e.inner().texts_embedded(), 2);
        assert_eq!(first[0], first[2]);
        let b: Vec<String> = vec!["b".into(), "c".into()];
        let second = e.embed_documents(&b).await.unwrap();
        assert_eq!(e.inner().texts_embedded(), 3);
        assert_eq!(second[0], first[1]);
        e.embed_query("q").await.unwrap();
        e.embed_query("q").await.unwrap();
        assert_eq!(e.inner().texts_embedded(), 4);
    }

    #[tokio::test]
    async fn bounded_cache_evicts_oldest() {
        let c = InMemoryEmbeddingCache::with_capacity(2);
        c.put_many(vec![
            ("a".into(), vec![1.0]),
            ("b".into(), vec![2.0]),
            ("c".into(), vec![3.0]),
        ])
        .await;
        assert_eq!(c.len(), 2);
        let got = c.get_many(&["a".into(), "c".into()]).await;
        assert_eq!(got, vec![None, Some(vec![3.0])]);
    }
}
