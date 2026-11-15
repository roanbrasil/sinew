use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use super::Embeddings;
use crate::Result;

/// Deterministic, offline embeddings for tests and demos.
///
/// Each word from [`english_tokenize`](crate::english_tokenize) (lower-cased,
/// stop words removed, lightly stemmed), and each pair of adjacent words, is
/// hashed into one of `dimensions` buckets with a ±1 sign, and the vector is
/// L2-normalised.
/// Texts that share words therefore get similar vectors, so similarity
/// search behaves sensibly without any model. It also counts how many texts
/// it embedded, which is handy to test caching and batching.
#[derive(Debug)]
pub struct FakeEmbeddings {
    dimensions: usize,
    calls: AtomicUsize,
    texts: AtomicUsize,
}

impl FakeEmbeddings {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions: dimensions.max(1),
            calls: AtomicUsize::new(0),
            texts: AtomicUsize::new(0),
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Number of `embed_documents` / `embed_query` calls so far.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    /// Number of texts embedded so far.
    pub fn texts_embedded(&self) -> usize {
        self.texts.load(Ordering::Relaxed)
    }

    /// The vector for `text` (synchronous).
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dimensions];
        let mut prev: Option<u64> = None;
        for word in crate::english_tokenize(text) {
            let h = fnv1a(word.as_bytes());
            self.add(&mut v, h, 1.0);
            if let Some(p) = prev {
                self.add(&mut v, h.rotate_left(17) ^ p, 0.5);
            }
            prev = Some(h);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }

    fn add(&self, v: &mut [f32], h: u64, w: f32) {
        let i = (h % self.dimensions as u64) as usize;
        v[i] += if h >> 63 == 0 { w } else { -w };
    }
}

impl Default for FakeEmbeddings {
    fn default() -> Self {
        Self::new(256)
    }
}

pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[async_trait]
impl Embeddings for FakeEmbeddings {
    fn model(&self) -> &str {
        "fake"
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.texts.fetch_add(texts.len(), Ordering::Relaxed);
        Ok(texts.iter().map(|t| self.embed(t)).collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.texts.fetch_add(1, Ordering::Relaxed);
        Ok(self.embed(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[tokio::test]
    async fn deterministic_and_similarity_aware() {
        let e = FakeEmbeddings::new(128);
        let a = e.embed_query("Rust ownership and borrowing").await.unwrap();
        let b = e.embed_query("rust ownership").await.unwrap();
        let c = e.embed_query("baking sourdough bread").await.unwrap();
        assert_eq!(a, e.embed("Rust ownership and borrowing"));
        assert!(dot(&a, &b) > dot(&a, &c));
        assert!((dot(&a, &a) - 1.0).abs() < 1e-5);
        assert_eq!(e.calls(), 3);
        assert!(e.embed("").iter().all(|x| *x == 0.0));
    }
}
