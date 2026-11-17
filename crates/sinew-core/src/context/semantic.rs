//! Keep the older turns most relevant to the latest user message.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{message_groups, render_transcript, retain_mask, ContextBudget, ContextStrategy};
use crate::message::{Message, Role};
use crate::{Error, Result};

/// Turns texts into embedding vectors, for [`SemanticSelect`].
///
/// Defined here so the core crate does not depend on `sinew-rag`; wrap any
/// `sinew_rag` embeddings model with `sinew_rag::embeddings::ContextEmbedder`,
/// or a function with [`embedder_fn`].
#[async_trait]
pub trait TextEmbedder: Send + Sync {
    /// One vector per input text, in order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[async_trait]
impl<T: TextEmbedder + ?Sized> TextEmbedder for Arc<T> {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        (**self).embed(texts).await
    }
}

/// A [`TextEmbedder`] backed by a synchronous function; see [`embedder_fn`].
pub struct EmbedderFn<F>(F);

/// Wraps a function `&[String] -> Result<Vec<Vec<f32>>>` as a
/// [`TextEmbedder`] (a local model, or a toy embedding in tests).
///
/// ```
/// use sinew_core::context::{embedder_fn, TextEmbedder};
///
/// let lengths = embedder_fn(|texts: &[String]| {
///     Ok(texts.iter().map(|t| vec![t.len() as f32, 1.0]).collect())
/// });
/// # let _ = lengths;
/// ```
pub fn embedder_fn<F>(f: F) -> EmbedderFn<F>
where
    F: Fn(&[String]) -> Result<Vec<Vec<f32>>> + Send + Sync,
{
    EmbedderFn(f)
}

#[async_trait]
impl<F> TextEmbedder for EmbedderFn<F>
where
    F: Fn(&[String]) -> Result<Vec<Vec<f32>>> + Send + Sync,
{
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        (self.0)(texts)
    }
}

const CACHE_ENTRIES: usize = 4_096;
const GROUP_TEXT_CHARS: usize = 2_000;

/// Keeps the recent messages, plus the older groups whose text is most
/// similar (cosine similarity of embeddings) to the latest user message;
/// drops the other older groups. System messages are always kept.
///
/// Useful when a long session wanders across topics and the current
/// question relates to something discussed far back. Groups (see
/// [`message_groups`]) are scored and kept whole, in their original order.
/// Group embeddings are cached by text, so each older group is embedded
/// once. Without a user message the input is returned unchanged.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use sinew_core::context::{embedder_fn, ContextBudget, ContextStrategy, SemanticSelect};
/// use sinew_core::Message;
///
/// // Toy embedding: does the text mention "rust"?
/// let embed = embedder_fn(|texts: &[String]| {
///     Ok(texts.iter().map(|t| vec![t.contains("rust") as u8 as f32, 1.0]).collect())
/// });
/// let msgs = vec![
///     Message::user("tell me about rust lifetimes"),
///     Message::assistant("rust lifetimes name borrow scopes"),
///     Message::user("and the weather?"),
///     Message::assistant("sunny"),
///     Message::user("back to rust: what is 'static?"),
/// ];
/// let select = SemanticSelect::new(embed).keep_recent(1).top_k(2);
/// let out = select.apply(msgs, &ContextBudget::new(1_000)).await.unwrap();
/// let text: Vec<&str> = out.iter().map(|m| m.content.as_str()).collect();
/// assert_eq!(text[2], "back to rust: what is 'static?");
/// assert!(text[0].contains("rust") && text[1].contains("rust"));
/// # }
/// ```
pub struct SemanticSelect {
    embedder: Arc<dyn TextEmbedder>,
    keep_recent: usize,
    top_k: usize,
    min_score: Option<f32>,
    cache: Mutex<HashMap<u64, Vec<f32>>>,
}

impl std::fmt::Debug for SemanticSelect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticSelect")
            .field("keep_recent", &self.keep_recent)
            .field("top_k", &self.top_k)
            .field("min_score", &self.min_score)
            .finish_non_exhaustive()
    }
}

impl SemanticSelect {
    /// Keeps the 6 most recent messages and the 4 most similar older groups.
    pub fn new(embedder: impl TextEmbedder + 'static) -> Self {
        Self::from_arc(Arc::new(embedder))
    }

    /// Like [`new`](Self::new) with a shared embedder.
    pub fn from_arc(embedder: Arc<dyn TextEmbedder>) -> Self {
        Self {
            embedder,
            keep_recent: 6,
            top_k: 4,
            min_score: None,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Recent non-system messages always kept (whole groups, at least the
    /// newest group; default 6).
    pub fn keep_recent(mut self, n: usize) -> Self {
        self.keep_recent = n;
        self
    }

    /// Older groups kept by similarity (default 4).
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Only keep older groups with at least this cosine similarity.
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    async fn embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let keys: Vec<u64> = texts
            .iter()
            .map(|t| {
                let mut h = DefaultHasher::new();
                t.hash(&mut h);
                h.finish()
            })
            .collect();
        let missing: Vec<usize> = {
            let cache = self.cache.lock().expect("cache lock");
            (0..texts.len())
                .filter(|&i| !cache.contains_key(&keys[i]))
                .collect()
        };
        if !missing.is_empty() {
            let input: Vec<String> = missing.iter().map(|&i| texts[i].clone()).collect();
            let vectors = self.embedder.embed(&input).await?;
            if vectors.len() != input.len() {
                return Err(Error::InvalidResponse(format!(
                    "embedder returned {} vectors for {} texts",
                    vectors.len(),
                    input.len()
                )));
            }
            let mut cache = self.cache.lock().expect("cache lock");
            if cache.len() + vectors.len() > CACHE_ENTRIES {
                cache.clear();
            }
            for (&i, v) in missing.iter().zip(vectors) {
                cache.insert(keys[i], v);
            }
        }
        let cache = self.cache.lock().expect("cache lock");
        // A text embedded in this call may have been evicted by a concurrent
        // clear; fall back to an empty vector (score 0) in that case.
        Ok(keys
            .iter()
            .map(|k| cache.get(k).cloned().unwrap_or_default())
            .collect())
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[async_trait]
impl ContextStrategy for SemanticSelect {
    fn name(&self) -> &str {
        "semantic_select"
    }

    async fn apply(&self, messages: Vec<Message>, _budget: &ContextBudget) -> Result<Vec<Message>> {
        let Some(query) = messages
            .iter()
            .rposition(|m| m.role == Role::User)
            .map(|i| messages[i].text())
        else {
            return Ok(messages);
        };
        let groups: Vec<_> = message_groups(&messages)
            .into_iter()
            .filter(|g| messages[g.start].role != Role::System)
            .collect();
        // Recent groups.
        let mut keep: Vec<bool> = messages.iter().map(|m| m.role == Role::System).collect();
        let mut recent = 0usize;
        let mut older = groups.len();
        for (n, g) in groups.iter().enumerate().rev() {
            if recent + g.len() > self.keep_recent && n + 1 < groups.len() {
                break;
            }
            recent += g.len();
            older = n;
        }
        for g in &groups[older..] {
            keep[g.clone()].iter_mut().for_each(|k| *k = true);
        }
        let candidates = &groups[..older];
        if !candidates.is_empty() && self.top_k > 0 {
            let mut texts = vec![query];
            texts.extend(
                candidates
                    .iter()
                    .map(|g| render_transcript(&messages[g.clone()], Some(GROUP_TEXT_CHARS))),
            );
            let vectors = self.embeddings(&texts).await?;
            let mut scored: Vec<(usize, f32)> = vectors[1..]
                .iter()
                .map(|v| cosine(&vectors[0], v))
                .enumerate()
                .filter(|(_, s)| self.min_score.is_none_or(|min| *s >= min))
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (i, _) in scored.into_iter().take(self.top_k) {
                keep[candidates[i].clone()]
                    .iter_mut()
                    .for_each(|k| *k = true);
            }
        }
        Ok(retain_mask(messages, &keep))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::check_tool_pairing;
    use crate::context::testkit::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn keeps_pairs_and_caches_embeddings() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let embed = embedder_fn(move |texts: &[String]| {
            counted.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|t| vec![t.contains("grep") as u8 as f32, t.len() as f32 / 1e4])
                .collect())
        });
        let select = SemanticSelect::new(embed).keep_recent(8).top_k(10);
        let msgs = transcript(11, 500);
        let out = select
            .apply(msgs.clone(), &ContextBudget::new(1_000))
            .await
            .unwrap();
        check_tool_pairing(&out).unwrap();
        assert!(out.len() < msgs.len());
        let first = calls.load(Ordering::SeqCst);
        select
            .apply(msgs, &ContextBudget::new(1_000))
            .await
            .unwrap();
        // Second call: only the query is new... and it is cached too.
        assert_eq!(calls.load(Ordering::SeqCst), first);
    }

    #[tokio::test]
    async fn wrong_vector_count_is_an_error() {
        let embed = embedder_fn(|_: &[String]| Ok(vec![]));
        let msgs = vec![
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
        ];
        let err = SemanticSelect::new(embed)
            .keep_recent(1)
            .apply(msgs, &ContextBudget::new(10))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidResponse(_)));
    }
}
