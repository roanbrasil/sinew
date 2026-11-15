use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{check_lengths, Filter, ScoredDocument, VectorStore};
use crate::{document::ensure_ids, Document, Embeddings, Error, Metadata, Result};

/// Namespace for deriving Qdrant point UUIDs from arbitrary document ids.
const NS: uuid::Uuid = uuid::Uuid::from_u128(0x5e1e_7a90_0b1d_4c7e_9a3f_2d6c_8b4e_1f00);

/// [Qdrant](https://qdrant.tech) over its REST API (feature `qdrant`).
///
/// Points use cosine distance. The payload holds `page_content`, `metadata`
/// and `doc_id` (LangChain's layout plus the original id); point ids are the
/// document id when it is a UUID, otherwise a UUIDv5 derived from it, so
/// re-adding a document replaces it. The collection is created on the first
/// insert if it does not exist. Filters become Qdrant filters on
/// `metadata.*`.
#[derive(Clone)]
pub struct QdrantVectorStore {
    url: String,
    collection: String,
    api_key: Option<String>,
    embeddings: Arc<dyn Embeddings>,
    client: reqwest::Client,
    batch_size: usize,
    ready: Arc<AtomicBool>,
}

#[derive(Deserialize)]
struct Envelope<T> {
    result: T,
}

#[derive(Deserialize)]
struct Point {
    #[serde(default)]
    score: Option<f32>,
    #[serde(default)]
    payload: Option<Value>,
    #[serde(default)]
    vector: Option<Value>,
}

impl QdrantVectorStore {
    /// `url` like `http://127.0.0.1:6333`.
    pub fn new(
        url: impl Into<String>,
        collection: impl Into<String>,
        embeddings: impl Embeddings + 'static,
    ) -> Self {
        Self {
            url: url.into().trim_end_matches('/').to_string(),
            collection: collection.into(),
            api_key: None,
            embeddings: Arc::new(embeddings),
            client: reqwest::Client::new(),
            batch_size: 256,
            ready: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Qdrant Cloud / secured instances (`api-key` header).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Points per upsert request (default 256).
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    fn req(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let r = self.client.request(method, format!("{}{path}", self.url));
        match &self.api_key {
            Some(k) => r.header("api-key", k),
            None => r,
        }
    }

    async fn send<T: serde::de::DeserializeOwned>(
        &self,
        req: reqwest::RequestBuilder,
        body: Option<&Value>,
    ) -> Result<T> {
        let req = match body {
            Some(b) => req.json(b),
            None => req,
        };
        let resp = req.send().await?;
        let status = resp.status();
        let bytes = resp.bytes().await?;
        if !status.is_success() {
            return Err(Error::Provider {
                provider: "qdrant".into(),
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        let env: Envelope<T> = serde_json::from_slice(&bytes)?;
        Ok(env.result)
    }

    /// Creates the collection (cosine, `dim` dimensions) unless it exists.
    pub async fn ensure_collection(&self, dim: usize) -> Result<()> {
        if self.ready.load(Ordering::Acquire) {
            return Ok(());
        }
        let path = format!("/collections/{}", self.collection);
        let exists = self
            .req(reqwest::Method::GET, &path)
            .send()
            .await?
            .status()
            .is_success();
        if !exists {
            let body = json!({"vectors": {"size": dim, "distance": "Cosine"}});
            let res: Result<Value> = self
                .send(self.req(reqwest::Method::PUT, &path), Some(&body))
                .await;
            // Another writer may have created it concurrently.
            if let Err(e) = res {
                let again = self.req(reqwest::Method::GET, &path).send().await?;
                if !again.status().is_success() {
                    return Err(e);
                }
            }
        }
        self.ready.store(true, Ordering::Release);
        Ok(())
    }

    /// Drops the whole collection.
    pub async fn delete_collection(&self) -> Result<()> {
        let path = format!("/collections/{}", self.collection);
        let _: Value = self
            .send(self.req(reqwest::Method::DELETE, &path), None)
            .await?;
        self.ready.store(false, Ordering::Release);
        Ok(())
    }

    fn point_id(id: &str) -> String {
        match uuid::Uuid::parse_str(id) {
            Ok(u) => u.to_string(),
            Err(_) => uuid::Uuid::new_v5(&NS, id.as_bytes()).to_string(),
        }
    }

    fn to_doc(p: &Point) -> Document {
        let payload = p.payload.as_ref();
        let get = |k: &str| payload.and_then(|pl| pl.get(k));
        Document {
            id: get("doc_id").and_then(Value::as_str).map(String::from),
            content: get("page_content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            metadata: get("metadata")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_else(Metadata::new),
        }
    }
}

fn vector_of(v: &Value) -> Option<Vec<f32>> {
    // Unnamed vectors come back as an array.
    v.as_array().map(|a| {
        a.iter()
            .filter_map(|x| x.as_f64().map(|f| f as f32))
            .collect()
    })
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    fn embeddings(&self) -> &Arc<dyn Embeddings> {
        &self.embeddings
    }

    async fn add_vectors(
        &self,
        mut docs: Vec<Document>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<String>> {
        check_lengths(docs.len(), vectors.len())?;
        let Some(first) = vectors.first() else {
            return Ok(Vec::new());
        };
        self.ensure_collection(first.len()).await?;
        let ids = ensure_ids(&mut docs);
        let path = format!("/collections/{}/points?wait=true", self.collection);
        let points: Vec<Value> = docs
            .iter()
            .zip(&vectors)
            .zip(&ids)
            .map(|((d, v), id)| {
                json!({
                    "id": Self::point_id(id),
                    "vector": v,
                    "payload": {"page_content": d.content, "metadata": d.metadata, "doc_id": id},
                })
            })
            .collect();
        for chunk in points.chunks(self.batch_size) {
            let body = json!({ "points": chunk });
            let _: Value = self
                .send(self.req(reqwest::Method::PUT, &path), Some(&body))
                .await?;
        }
        Ok(ids)
    }

    async fn delete(&self, ids: &[String]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let existing = self.get(ids).await?.len();
        let path = format!("/collections/{}/points/delete?wait=true", self.collection);
        let pids: Vec<String> = ids.iter().map(|i| Self::point_id(i)).collect();
        let _: Value = self
            .send(
                self.req(reqwest::Method::POST, &path),
                Some(&json!({ "points": pids })),
            )
            .await?;
        Ok(existing)
    }

    async fn get(&self, ids: &[String]) -> Result<Vec<Document>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let path = format!("/collections/{}/points", self.collection);
        let pids: Vec<String> = ids.iter().map(|i| Self::point_id(i)).collect();
        let body = json!({ "ids": pids, "with_payload": true, "with_vector": false });
        let points: Vec<Point> = match self
            .send(self.req(reqwest::Method::POST, &path), Some(&body))
            .await
        {
            Ok(p) => p,
            Err(Error::Provider { status: 404, .. }) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut by_id: std::collections::HashMap<String, Document> = points
            .iter()
            .map(Self::to_doc)
            .filter_map(|d| d.id.clone().map(|id| (id, d)))
            .collect();
        Ok(ids.iter().filter_map(|id| by_id.remove(id)).collect())
    }

    async fn len(&self) -> Result<usize> {
        let path = format!("/collections/{}/points/count", self.collection);
        #[derive(Deserialize)]
        struct Count {
            count: usize,
        }
        match self
            .send::<Count>(
                self.req(reqwest::Method::POST, &path),
                Some(&json!({"exact": true})),
            )
            .await
        {
            Ok(c) => Ok(c.count),
            Err(Error::Provider { status: 404, .. }) => Ok(0),
            Err(e) => Err(e),
        }
    }

    async fn search_by_vector(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        with_vectors: bool,
    ) -> Result<Vec<ScoredDocument>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let path = format!("/collections/{}/points/search", self.collection);
        let mut body = json!({
            "vector": query,
            "limit": k,
            "with_payload": true,
            "with_vector": with_vectors,
        });
        if let Some(f) = filter {
            body["filter"] = f.to_qdrant("metadata.");
        }
        let points: Vec<Point> = match self
            .send(self.req(reqwest::Method::POST, &path), Some(&body))
            .await
        {
            Ok(p) => p,
            Err(Error::Provider { status: 404, .. }) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        Ok(points
            .iter()
            .map(|p| ScoredDocument {
                document: Self::to_doc(p),
                score: p.score.unwrap_or_default(),
                vector: p.vector.as_ref().and_then(vector_of),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_ids_are_stable_uuids() {
        let u = "6f1c2a8e-3b7d-4c55-9a0e-1d2b3c4d5e6f";
        assert_eq!(QdrantVectorStore::point_id(u), u);
        let a = QdrantVectorStore::point_id("docs/a.md:3");
        assert_eq!(a, QdrantVectorStore::point_id("docs/a.md:3"));
        assert_ne!(a, QdrantVectorStore::point_id("docs/a.md:4"));
        assert!(uuid::Uuid::parse_str(&a).is_ok());
    }
}
