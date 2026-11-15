//! Native Gemini embeddings (`batchEmbedContents`).

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{batched, post_json, BatchFuture, Embeddings};
use crate::{Error, Result};

#[derive(Clone)]
enum Auth {
    ApiKey(String),
    Bearer(String),
}

/// Gemini embedding models (default `gemini-embedding-001`) through the
/// Google AI Studio API, or Vertex AI with
/// [`with_base_url`](Self::with_base_url) +
/// [`with_bearer_token`](Self::with_bearer_token).
///
/// Documents are embedded with task type `RETRIEVAL_DOCUMENT` and queries
/// with `RETRIEVAL_QUERY`, 100 texts per request.
#[derive(Clone)]
pub struct GeminiEmbeddings {
    base_url: String,
    auth: Auth,
    model: String,
    dimensions: Option<u32>,
    batch_size: usize,
    concurrency: usize,
    client: reqwest::Client,
}

#[derive(Deserialize)]
struct Response {
    embeddings: Vec<Values>,
}

#[derive(Deserialize)]
struct Values {
    values: Vec<f32>,
}

impl GeminiEmbeddings {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
            auth: Auth::ApiKey(api_key.into()),
            model: model.into(),
            dimensions: None,
            batch_size: 100,
            concurrency: 4,
            client: reqwest::Client::new(),
        }
    }

    /// Reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| Error::Config("GEMINI_API_KEY not set".into()))?;
        Ok(Self::new(key, model))
    }

    /// Base URL up to (not including) `/models`.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// OAuth access token instead of an API key (Vertex AI).
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.auth = Auth::Bearer(token.into());
        self
    }

    /// `outputDimensionality` (e.g. 768 for `gemini-embedding-001`).
    pub fn with_dimensions(mut self, n: u32) -> Self {
        self.dimensions = Some(n);
        self
    }

    /// Texts per request (Gemini accepts at most 100).
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.clamp(1, 100);
        self
    }

    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    fn model_path(&self) -> String {
        if self.model.starts_with("models/") {
            self.model.clone()
        } else {
            format!("models/{}", self.model)
        }
    }

    pub(crate) fn encode(&self, texts: &[String], task: &str) -> Value {
        let model = self.model_path();
        let requests: Vec<Value> = texts
            .iter()
            .map(|t| {
                let mut r = json!({
                    "model": model,
                    "content": { "parts": [{ "text": t }] },
                    "taskType": task,
                });
                if let Some(d) = self.dimensions {
                    r["outputDimensionality"] = json!(d);
                }
                r
            })
            .collect();
        json!({ "requests": requests })
    }

    async fn embed_batch(&self, texts: &[String], task: &str) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/{}:batchEmbedContents", self.base_url, self.model_path());
        let req = self.client.post(url);
        let req = match &self.auth {
            Auth::ApiKey(k) => req.header("x-goog-api-key", k),
            Auth::Bearer(t) => req.bearer_auth(t),
        };
        let resp: Response = post_json("gemini", req, &self.encode(texts, task)).await?;
        if resp.embeddings.len() != texts.len() {
            return Err(Error::InvalidResponse(format!(
                "gemini returned {} embeddings for {} inputs",
                resp.embeddings.len(),
                texts.len()
            )));
        }
        Ok(resp.embeddings.into_iter().map(|e| e.values).collect())
    }
}

#[async_trait]
impl Embeddings for GeminiEmbeddings {
    fn model(&self) -> &str {
        &self.model
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batches: Vec<BatchFuture<'_>> = texts
            .chunks(self.batch_size)
            .map(|b| Box::pin(self.embed_batch(b, "RETRIEVAL_DOCUMENT")) as BatchFuture<'_>)
            .collect();
        batched(batches, self.concurrency).await
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self
            .embed_batch(&[text.to_string()], "RETRIEVAL_QUERY")
            .await?;
        v.pop()
            .ok_or_else(|| Error::InvalidResponse("no embedding returned".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_batch_request() {
        let g = GeminiEmbeddings::new("k", "gemini-embedding-001").with_dimensions(768);
        let body = g.encode(&["a".into(), "b".into()], "RETRIEVAL_DOCUMENT");
        let reqs = body["requests"].as_array().unwrap();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0]["model"], "models/gemini-embedding-001");
        assert_eq!(reqs[1]["content"]["parts"][0]["text"], "b");
        assert_eq!(reqs[0]["outputDimensionality"], 768);
        assert_eq!(reqs[0]["taskType"], "RETRIEVAL_DOCUMENT");
    }
}
