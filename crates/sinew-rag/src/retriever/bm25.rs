use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;

use super::Retriever;
use crate::{vectorstore::topk::TopK, Document, Result};

type Tokenizer = Arc<dyn Fn(&str) -> Vec<String> + Send + Sync>;

/// Lower-cased alphanumeric words.
pub(crate) fn default_tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Keyword retriever with Okapi BM25 scoring over an inverted index.
///
/// IDF is `ln(1 + (N - n + 0.5) / (n + 0.5))` (never negative), with the
/// usual `k1` = 1.5 and `b` = 0.75. Only documents sharing at least one
/// term with the query are returned. Pair it with a vector retriever in an
/// [`EnsembleRetriever`](super::EnsembleRetriever) for hybrid search.
#[derive(Clone)]
pub struct Bm25Retriever {
    docs: Vec<Document>,
    vocab: HashMap<String, u32>,
    /// term id -> (document index, term frequency)
    postings: Vec<Vec<(u32, u32)>>,
    doc_len: Vec<u32>,
    avgdl: f32,
    tokenizer: Tokenizer,
    k: usize,
    k1: f32,
    b: f32,
    scores: bool,
}

impl Bm25Retriever {
    pub fn from_documents(docs: Vec<Document>) -> Self {
        Self::from_documents_with_tokenizer(docs, default_tokenize)
    }

    /// Indexes with a custom tokenizer (stemming, stop words, n-grams ...);
    /// queries use the same tokenizer.
    pub fn from_documents_with_tokenizer<F>(docs: Vec<Document>, tokenizer: F) -> Self
    where
        F: Fn(&str) -> Vec<String> + Send + Sync + 'static,
    {
        let tokenizer: Tokenizer = Arc::new(tokenizer);
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut postings: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut doc_len = Vec::with_capacity(docs.len());
        let mut tf: HashMap<u32, u32> = HashMap::new();
        for (i, d) in docs.iter().enumerate() {
            let tokens = tokenizer(&d.content);
            doc_len.push(tokens.len() as u32);
            tf.clear();
            for t in tokens {
                let next = vocab.len() as u32;
                let id = *vocab.entry(t).or_insert(next);
                if id as usize == postings.len() {
                    postings.push(Vec::new());
                }
                *tf.entry(id).or_default() += 1;
            }
            for (&term, &count) in &tf {
                postings[term as usize].push((i as u32, count));
            }
        }
        let total: u64 = doc_len.iter().map(|&l| l as u64).sum();
        let avgdl = if docs.is_empty() {
            0.0
        } else {
            total as f32 / docs.len() as f32
        };
        Self {
            docs,
            vocab,
            postings,
            doc_len,
            avgdl,
            tokenizer,
            k: 4,
            k1: 1.5,
            b: 0.75,
            scores: false,
        }
    }

    /// Results per query (default 4).
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// BM25 term-saturation (`k1`) and length-normalisation (`b`) parameters.
    pub fn with_params(mut self, k1: f32, b: f32) -> Self {
        self.k1 = k1;
        self.b = b;
        self
    }

    /// Adds the BM25 score as `score` metadata.
    pub fn with_scores(mut self) -> Self {
        self.scores = true;
        self
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// The top `k` documents and their BM25 scores.
    pub fn search(&self, query: &str, k: usize) -> Vec<(&Document, f32)> {
        let n = self.docs.len() as f32;
        let mut scores = vec![0f32; self.docs.len()];
        let mut touched: Vec<u32> = Vec::new();
        for token in (self.tokenizer)(query) {
            let Some(&term) = self.vocab.get(&token) else {
                continue;
            };
            let posting = &self.postings[term as usize];
            let df = posting.len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for &(doc, f) in posting {
                let f = f as f32;
                let len_norm = if self.avgdl > 0.0 {
                    self.doc_len[doc as usize] as f32 / self.avgdl
                } else {
                    1.0
                };
                let denom = f + self.k1 * (1.0 - self.b + self.b * len_norm);
                let s = &mut scores[doc as usize];
                if *s == 0.0 {
                    touched.push(doc);
                }
                *s += idf * f * (self.k1 + 1.0) / denom;
            }
        }
        let mut top = TopK::new(k);
        for d in touched {
            top.push(scores[d as usize], d as usize);
        }
        top.into_sorted()
            .into_iter()
            .filter(|h| h.score > 0.0)
            .map(|h| (&self.docs[h.idx], h.score))
            .collect()
    }
}

#[async_trait]
impl Retriever for Bm25Retriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<Document>> {
        Ok(self
            .search(query, self.k)
            .into_iter()
            .map(|(d, s)| {
                let mut d = d.clone();
                if self.scores {
                    d.metadata.insert("score".into(), (s as f64).into());
                }
                d
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<Document> {
        [
            "the quick brown fox jumps over the lazy dog",
            "a fast brown fox",
            "lorem ipsum dolor sit amet",
            "the dog sleeps all day the dog dreams",
        ]
        .into_iter()
        .map(Document::new)
        .collect()
    }

    #[test]
    fn ranks_by_bm25() {
        let r = Bm25Retriever::from_documents(corpus());
        let hits = r.search("brown fox", 10);
        assert_eq!(hits.len(), 2);
        // The shorter document wins on length normalisation.
        assert_eq!(hits[0].0.content, "a fast brown fox");
        let hits = r.search("DOG", 10);
        assert_eq!(hits[0].0.content, "the dog sleeps all day the dog dreams");
        assert!(r.search("unknown words", 10).is_empty());
        assert!(Bm25Retriever::from_documents(vec![])
            .search("x", 3)
            .is_empty());
    }

    #[tokio::test]
    async fn retrieve_respects_k_and_tokenizer() {
        let r = Bm25Retriever::from_documents(corpus())
            .with_k(1)
            .with_scores();
        let got = r.retrieve("dog").await.unwrap();
        assert_eq!(got.len(), 1);
        assert!(got[0].metadata["score"].as_f64().unwrap() > 0.0);

        // Character trigrams make the search typo-tolerant.
        let trigrams = |s: &str| {
            let c: Vec<char> = s.to_lowercase().chars().collect();
            c.windows(3).map(|w| w.iter().collect()).collect()
        };
        let r = Bm25Retriever::from_documents_with_tokenizer(corpus(), trigrams).with_k(1);
        assert_eq!(
            r.retrieve("lorem ipsun").await.unwrap()[0].content,
            "lorem ipsum dolor sit amet"
        );
    }
}
