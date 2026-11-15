//! Long-term memory shared across threads (LangGraph's `BaseStore`).
//!
//! A [`Store`] holds JSON [`Item`]s addressed by a hierarchical namespace
//! (e.g. `["memories", user_id]`) and a key. Checkpoints belong to one thread;
//! the store is for what must outlive it: user preferences, facts learned in
//! earlier conversations, shared documents.
//!
//! Search filters by namespace prefix and by exact value of top-level fields,
//! and, when the store has an [`IndexConfig`], ranks by semantic similarity
//! using an [`Embedder`] you supply (any embedding model or service).
//!
//! Nodes reach the graph's store through
//! [`NodeContext::store`](crate::NodeContext::store); attach one with
//! [`CompiledGraph::with_store`](crate::CompiledGraph::with_store).

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{checkpoint::now_millis, BoxError};

/// Errors returned by stores.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("invalid namespace: {0}")]
    InvalidNamespace(String),
    #[error("embedding failed: {0}")]
    Embedding(BoxError),
    #[error("store backend error: {0}")]
    Backend(String),
    #[error("serialization failed: {0}")]
    Serde(#[from] serde_json::Error),
}

/// A stored value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub namespace: Vec<String>,
    pub key: String,
    pub value: Value,
    /// Unix time in milliseconds.
    pub created_at: u64,
    pub updated_at: u64,
}

/// A search hit. `score` is the cosine similarity to the query when the
/// search was semantic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchItem {
    pub item: Item,
    pub score: Option<f32>,
}

/// Parameters of [`Store::search`].
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    /// Natural-language query for semantic ranking. Ignored by stores
    /// without an [`IndexConfig`].
    pub query: Option<String>,
    /// Every pair must equal the item's top-level field of that name.
    pub filter: Map<String, Value>,
    pub limit: usize,
    pub offset: usize,
}

impl SearchQuery {
    pub fn new() -> Self {
        Self {
            limit: 10,
            ..Self::default()
        }
    }

    pub fn query(mut self, q: impl Into<String>) -> Self {
        self.query = Some(q.into());
        self
    }

    pub fn filter(mut self, field: impl Into<String>, value: Value) -> Self {
        self.filter.insert(field.into(), value);
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }
}

/// Options for [`Store::put_with`].
#[derive(Debug, Clone, Default)]
pub struct PutOptions {
    /// Embed this item for semantic search (default: when the store has an
    /// index). `Some(false)` skips embedding.
    pub index: Option<bool>,
    /// Remove the item after this long.
    pub ttl: Option<Duration>,
}

/// Turns texts into vectors for semantic search. Implement it over any
/// embedding model; [`FnEmbedder`] wraps a closure.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BoxError>;
}

/// [`Embedder`] from a synchronous closure (tests, local hashing models).
pub struct FnEmbedder<F>(pub F);

#[async_trait]
impl<F> Embedder for FnEmbedder<F>
where
    F: Fn(&str) -> Vec<f32> + Send + Sync,
{
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BoxError> {
        Ok(texts.iter().map(|t| (self.0)(t)).collect())
    }
}

/// Semantic-search configuration of a store.
#[derive(Clone)]
pub struct IndexConfig {
    pub embedder: Arc<dyn Embedder>,
    /// Top-level fields (or dotted paths like `"profile.bio"`) whose text is
    /// embedded. Empty: the whole value (a string value as is, anything else
    /// as JSON).
    pub fields: Vec<String>,
}

impl IndexConfig {
    pub fn new(embedder: impl Embedder + 'static) -> Self {
        Self {
            embedder: Arc::new(embedder),
            fields: Vec::new(),
        }
    }

    pub fn with_fields<I, T>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.fields = fields.into_iter().map(Into::into).collect();
        self
    }

    /// The text embedded for `value`, or `None` when it has none of the
    /// configured fields.
    pub fn text_of(&self, value: &Value) -> Option<String> {
        if self.fields.is_empty() {
            return Some(match value {
                Value::String(s) => s.clone(),
                v => v.to_string(),
            });
        }
        let parts: Vec<String> = self
            .fields
            .iter()
            .filter_map(|f| {
                let mut cur = value;
                for seg in f.split('.') {
                    cur = cur.get(seg)?;
                }
                Some(match cur {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                })
            })
            .collect();
        (!parts.is_empty()).then(|| parts.join("\n"))
    }
}

impl std::fmt::Debug for IndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexConfig")
            .field("fields", &self.fields)
            .finish()
    }
}

/// Namespaced key-value storage with search. See the [module docs](self).
#[async_trait]
pub trait Store: Send + Sync {
    /// Inserts or replaces an item.
    async fn put_with(
        &self,
        namespace: &[&str],
        key: &str,
        value: Value,
        options: PutOptions,
    ) -> Result<(), StoreError>;

    async fn get(&self, namespace: &[&str], key: &str) -> Result<Option<Item>, StoreError>;

    async fn delete(&self, namespace: &[&str], key: &str) -> Result<(), StoreError>;

    /// Items whose namespace starts with `prefix`, filtered and (with a
    /// query and an index) ranked by similarity; otherwise newest first.
    async fn search(
        &self,
        prefix: &[&str],
        query: SearchQuery,
    ) -> Result<Vec<SearchItem>, StoreError>;

    /// Distinct namespaces starting with `prefix`, truncated to `max_depth`
    /// segments when given.
    async fn list_namespaces(
        &self,
        prefix: &[&str],
        max_depth: Option<usize>,
    ) -> Result<Vec<Vec<String>>, StoreError>;

    /// Inserts or replaces an item with default options.
    async fn put(&self, namespace: &[&str], key: &str, value: Value) -> Result<(), StoreError> {
        self.put_with(namespace, key, value, PutOptions::default())
            .await
    }
}

/// Checks a namespace: at least one segment, none empty.
pub fn validate_namespace(ns: &[&str]) -> Result<(), StoreError> {
    if ns.is_empty() || ns.iter().any(|s| s.is_empty()) {
        return Err(StoreError::InvalidNamespace(format!("{ns:?}")));
    }
    Ok(())
}

/// Whether `value` matches every `filter` pair (top-level fields).
pub fn matches_filter(value: &Value, filter: &Map<String, Value>) -> bool {
    filter.iter().all(|(k, v)| value.get(k) == Some(v))
}

/// Cosine similarity (0 for mismatched or zero vectors).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
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

/// Ranks candidates (item, optional embedding) for a search: filters,
/// scores against `query_vec` when given, sorts, and pages.
pub fn rank(
    candidates: Vec<(Item, Option<Vec<f32>>)>,
    query: &SearchQuery,
    query_vec: Option<&[f32]>,
) -> Vec<SearchItem> {
    let mut hits: Vec<SearchItem> = candidates
        .into_iter()
        .filter(|(item, _)| matches_filter(&item.value, &query.filter))
        .map(|(item, emb)| {
            let score = query_vec.map(|q| emb.as_deref().map_or(0.0, |e| cosine(q, e)));
            SearchItem { item, score }
        })
        .collect();
    if query_vec.is_some() {
        hits.sort_by(|a, b| {
            b.score
                .unwrap_or(0.0)
                .partial_cmp(&a.score.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        hits.sort_by(|a, b| b.item.updated_at.cmp(&a.item.updated_at));
    }
    let limit = if query.limit == 0 {
        usize::MAX
    } else {
        query.limit
    };
    hits.into_iter().skip(query.offset).take(limit).collect()
}

struct Entry {
    item: Item,
    embedding: Option<Vec<f32>>,
    expires_at: Option<u64>,
}

/// Process-local [`Store`].
#[derive(Default)]
pub struct InMemoryStore {
    items: Mutex<BTreeMap<(Vec<String>, String), Entry>>,
    index: Option<IndexConfig>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables semantic search.
    pub fn with_index(mut self, index: IndexConfig) -> Self {
        self.index = Some(index);
        self
    }

    fn purge_expired(map: &mut BTreeMap<(Vec<String>, String), Entry>) {
        let now = now_millis();
        map.retain(|_, e| e.expires_at.is_none_or(|t| t > now));
    }
}

fn owned(ns: &[&str]) -> Vec<String> {
    ns.iter().map(|s| s.to_string()).collect()
}

fn has_prefix(ns: &[String], prefix: &[&str]) -> bool {
    ns.len() >= prefix.len() && ns.iter().zip(prefix).all(|(a, b)| a == b)
}

#[async_trait]
impl Store for InMemoryStore {
    async fn put_with(
        &self,
        namespace: &[&str],
        key: &str,
        value: Value,
        options: PutOptions,
    ) -> Result<(), StoreError> {
        validate_namespace(namespace)?;
        let embedding = match (&self.index, options.index) {
            (Some(ix), None | Some(true)) => match ix.text_of(&value) {
                Some(text) => ix
                    .embedder
                    .embed(vec![text])
                    .await
                    .map_err(StoreError::Embedding)?
                    .pop(),
                None => None,
            },
            _ => None,
        };
        let now = now_millis();
        let ns = owned(namespace);
        let mut map = self.items.lock().unwrap();
        let created_at = map
            .get(&(ns.clone(), key.to_string()))
            .map_or(now, |e| e.item.created_at);
        map.insert(
            (ns.clone(), key.to_string()),
            Entry {
                item: Item {
                    namespace: ns,
                    key: key.to_string(),
                    value,
                    created_at,
                    updated_at: now,
                },
                embedding,
                expires_at: options.ttl.map(|t| now + t.as_millis() as u64),
            },
        );
        Ok(())
    }

    async fn get(&self, namespace: &[&str], key: &str) -> Result<Option<Item>, StoreError> {
        let mut map = self.items.lock().unwrap();
        Self::purge_expired(&mut map);
        Ok(map
            .get(&(owned(namespace), key.to_string()))
            .map(|e| e.item.clone()))
    }

    async fn delete(&self, namespace: &[&str], key: &str) -> Result<(), StoreError> {
        self.items
            .lock()
            .unwrap()
            .remove(&(owned(namespace), key.to_string()));
        Ok(())
    }

    async fn search(
        &self,
        prefix: &[&str],
        query: SearchQuery,
    ) -> Result<Vec<SearchItem>, StoreError> {
        let query_vec = match (&self.index, &query.query) {
            (Some(ix), Some(q)) => ix
                .embedder
                .embed(vec![q.clone()])
                .await
                .map_err(StoreError::Embedding)?
                .pop(),
            _ => None,
        };
        let candidates = {
            let mut map = self.items.lock().unwrap();
            Self::purge_expired(&mut map);
            map.values()
                .filter(|e| has_prefix(&e.item.namespace, prefix))
                .map(|e| (e.item.clone(), e.embedding.clone()))
                .collect()
        };
        Ok(rank(candidates, &query, query_vec.as_deref()))
    }

    async fn list_namespaces(
        &self,
        prefix: &[&str],
        max_depth: Option<usize>,
    ) -> Result<Vec<Vec<String>>, StoreError> {
        let mut map = self.items.lock().unwrap();
        Self::purge_expired(&mut map);
        let set: BTreeSet<Vec<String>> = map
            .keys()
            .filter(|(ns, _)| has_prefix(ns, prefix))
            .map(|(ns, _)| match max_depth {
                Some(d) => ns.iter().take(d).cloned().collect(),
                None => ns.clone(),
            })
            .collect();
        Ok(set.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Bag-of-letters embedding: good enough to rank by shared vocabulary.
    pub(crate) fn letters(text: &str) -> Vec<f32> {
        let mut v = vec![0f32; 26];
        for c in text.to_lowercase().chars() {
            if c.is_ascii_lowercase() {
                v[(c as u8 - b'a') as usize] += 1.0;
            }
        }
        v
    }

    #[tokio::test]
    async fn put_get_delete_and_prefix_search() {
        let s = InMemoryStore::new();
        s.put(
            &["users", "ana"],
            "lang",
            json!({"v": "pt", "kind": "pref"}),
        )
        .await
        .unwrap();
        s.put(
            &["users", "bob"],
            "lang",
            json!({"v": "en", "kind": "pref"}),
        )
        .await
        .unwrap();
        s.put(&["docs"], "x", json!({"kind": "doc"})).await.unwrap();
        assert_eq!(
            s.get(&["users", "ana"], "lang")
                .await
                .unwrap()
                .unwrap()
                .value["v"],
            "pt"
        );
        let users = s.search(&["users"], SearchQuery::new()).await.unwrap();
        assert_eq!(users.len(), 2);
        let en = s
            .search(&[], SearchQuery::new().filter("v", json!("en")))
            .await
            .unwrap();
        assert_eq!(en.len(), 1);
        assert_eq!(en[0].item.namespace, vec!["users", "bob"]);
        assert_eq!(
            s.list_namespaces(&["users"], None).await.unwrap(),
            vec![vec!["users", "ana"], vec!["users", "bob"]]
        );
        assert_eq!(
            s.list_namespaces(&[], Some(1)).await.unwrap(),
            vec![vec!["docs"], vec!["users"]]
        );
        s.delete(&["users", "ana"], "lang").await.unwrap();
        assert!(s.get(&["users", "ana"], "lang").await.unwrap().is_none());
        assert!(s.put(&[], "k", json!(1)).await.is_err());
    }

    #[tokio::test]
    async fn semantic_search_ranks_by_similarity() {
        let s = InMemoryStore::new()
            .with_index(IndexConfig::new(FnEmbedder(letters)).with_fields(["text"]));
        for (k, t) in [("a", "zebra zoo"), ("b", "apple banana"), ("c", "pizza")] {
            s.put(&["m"], k, json!({"text": t})).await.unwrap();
        }
        let hits = s
            .search(&["m"], SearchQuery::new().query("bananas and apples"))
            .await
            .unwrap();
        assert_eq!(hits[0].item.key, "b");
        assert!(hits[0].score.unwrap() > hits[2].score.unwrap());
    }

    #[tokio::test]
    async fn ttl_expires_items() {
        let s = InMemoryStore::new();
        s.put_with(
            &["t"],
            "k",
            json!(1),
            PutOptions {
                ttl: Some(Duration::from_millis(0)),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(2)).await;
        assert!(s.get(&["t"], "k").await.unwrap().is_none());
    }
}
