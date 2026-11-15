use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;

use super::Retriever;
use crate::{splitter::TextSplitter, Document, Filter, Result, VectorStore};

/// Key-value storage for parent documents.
#[async_trait]
pub trait DocStore: Send + Sync {
    async fn put(&self, docs: Vec<(String, Document)>) -> Result<()>;
    /// One entry per id, `None` when missing.
    async fn get(&self, ids: &[String]) -> Result<Vec<Option<Document>>>;
    async fn delete(&self, ids: &[String]) -> Result<()>;
}

/// [`DocStore`] in memory.
#[derive(Default, Clone)]
pub struct InMemoryDocStore {
    map: Arc<RwLock<HashMap<String, Document>>>,
}

impl InMemoryDocStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.map.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl DocStore for InMemoryDocStore {
    async fn put(&self, docs: Vec<(String, Document)>) -> Result<()> {
        self.map.write().unwrap().extend(docs);
        Ok(())
    }
    async fn get(&self, ids: &[String]) -> Result<Vec<Option<Document>>> {
        let m = self.map.read().unwrap();
        Ok(ids.iter().map(|id| m.get(id).cloned()).collect())
    }
    async fn delete(&self, ids: &[String]) -> Result<()> {
        let mut m = self.map.write().unwrap();
        for id in ids {
            m.remove(id);
        }
        Ok(())
    }
}

/// Indexes small chunks for precise matching but returns the larger
/// documents they came from, so the model sees enough context (LangChain's
/// `ParentDocumentRetriever`).
///
/// [`add_documents`](Self::add_documents) optionally splits the input into
/// parents with `parent_splitter`, splits each parent into children with
/// `child_splitter`, embeds the children into the vector store (with a
/// `parent_id` metadata field) and keeps the parents in the [`DocStore`].
/// A query searches `k * fetch_factor` children and returns up to `k`
/// distinct parents in rank order.
#[derive(Clone)]
pub struct ParentDocumentRetriever {
    store: Arc<dyn VectorStore>,
    docstore: Arc<dyn DocStore>,
    child_splitter: Arc<dyn TextSplitter>,
    parent_splitter: Option<Arc<dyn TextSplitter>>,
    k: usize,
    fetch_factor: usize,
    filter: Option<Filter>,
}

impl ParentDocumentRetriever {
    pub fn new(
        store: Arc<dyn VectorStore>,
        docstore: Arc<dyn DocStore>,
        child_splitter: impl TextSplitter + 'static,
    ) -> Self {
        Self {
            store,
            docstore,
            child_splitter: Arc::new(child_splitter),
            parent_splitter: None,
            k: 4,
            fetch_factor: 4,
            filter: None,
        }
    }

    /// Splits inputs into medium-sized parents first (otherwise whole input
    /// documents are the parents).
    pub fn with_parent_splitter(mut self, s: impl TextSplitter + 'static) -> Self {
        self.parent_splitter = Some(Arc::new(s));
        self
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Children fetched per wanted parent (default 4).
    pub fn with_fetch_factor(mut self, f: usize) -> Self {
        self.fetch_factor = f.max(1);
        self
    }

    pub fn with_filter(mut self, f: Filter) -> Self {
        self.filter = Some(f);
        self
    }

    /// Indexes documents; returns the parent ids.
    pub async fn add_documents(&self, docs: Vec<Document>) -> Result<Vec<String>> {
        let mut parents = match &self.parent_splitter {
            Some(s) => s.split_documents(&docs),
            None => docs,
        };
        let ids = crate::document::ensure_ids(&mut parents);
        let mut children = Vec::new();
        for p in &parents {
            for mut c in self.child_splitter.split_documents(std::slice::from_ref(p)) {
                c.metadata
                    .insert("parent_id".into(), p.id.clone().unwrap_or_default().into());
                children.push(c);
            }
        }
        self.store.add_documents(children).await?;
        self.docstore
            .put(ids.iter().cloned().zip(parents).collect())
            .await?;
        Ok(ids)
    }
}

#[async_trait]
impl Retriever for ParentDocumentRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<Document>> {
        let children = self
            .store
            .similarity_search(query, self.k * self.fetch_factor, self.filter.as_ref())
            .await?;
        let mut parent_ids: Vec<String> = Vec::new();
        for c in &children {
            if let Some(pid) = c.metadata.get("parent_id").and_then(|v| v.as_str()) {
                if !parent_ids.iter().any(|p| p == pid) {
                    parent_ids.push(pid.to_string());
                    if parent_ids.len() == self.k {
                        break;
                    }
                }
            }
        }
        Ok(self
            .docstore
            .get(&parent_ids)
            .await?
            .into_iter()
            .flatten()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FakeEmbeddings, InMemoryVectorStore, RecursiveCharacterTextSplitter};

    #[tokio::test]
    async fn returns_parents_of_matching_children() {
        let store: Arc<dyn VectorStore> =
            Arc::new(InMemoryVectorStore::new(FakeEmbeddings::new(128)));
        let docstore = Arc::new(InMemoryDocStore::new());
        let r = ParentDocumentRetriever::new(
            store.clone(),
            docstore.clone(),
            RecursiveCharacterTextSplitter::new(30, 0),
        )
        .with_k(1);
        let ids = r
            .add_documents(vec![
                Document::new(
                    "Checkpoints persist graph state. SQLite and Postgres backends survive restarts.",
                )
                .with_id("cp"),
                Document::new("Streaming emits tokens as they arrive from the provider.").with_id("st"),
            ])
            .await
            .unwrap();
        assert_eq!(ids, ["cp", "st"]);
        assert_eq!(docstore.len(), 2);
        assert!(store.len().await.unwrap() > 2);
        let got = r.retrieve("postgres backends").await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id.as_deref(), Some("cp"));
        assert!(got[0].content.starts_with("Checkpoints"));
    }
}
