//! Text embedding models.
//!
//! - [`OpenAiEmbeddings`]: any OpenAI-compatible `/embeddings` endpoint
//!   (OpenAI, Mistral, Together, Fireworks, NVIDIA NIM, Ollama, LM Studio,
//!   llama.cpp, vLLM, SGLang, LocalAI, TEI ...), with the same base-URL and
//!   API-key environment variables as the chat presets in `sinew-core`;
//! - [`GeminiEmbeddings`]: native Gemini `batchEmbedContents`;
//! - [`FakeEmbeddings`]: deterministic feature-hashing vectors for tests and
//!   offline demos;
//! - [`CachedEmbeddings`]: wraps any model with an [`EmbeddingCache`].
//!
//! [`embeddings_from_spec`] builds any of the real ones from
//! `"provider:model"`, e.g. `"ollama:bge-m3"` or
//! `"openai:text-embedding-3-small"`.

mod cache;
mod fake;
mod gemini;
mod openai;

use std::sync::Arc;

use async_trait::async_trait;

pub use cache::{CachedEmbeddings, EmbeddingCache, InMemoryEmbeddingCache};
pub use fake::FakeEmbeddings;
pub use gemini::GeminiEmbeddings;
pub use openai::OpenAiEmbeddings;

use crate::{Error, Result};

/// Turns text into vectors.
///
/// `embed_documents` is the batch entry point used for indexing; providers
/// split large inputs into batches and send them concurrently.
/// `embed_query` embeds a search query (some models embed queries and
/// documents differently).
#[async_trait]
pub trait Embeddings: Send + Sync {
    /// Model identifier (also used as the cache namespace).
    fn model(&self) -> &str;

    /// Embeds many texts; the result has one vector per input, in order.
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Embeds one search query.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed_documents(&[text.to_string()]).await?;
        v.pop()
            .ok_or_else(|| Error::InvalidResponse("no embedding returned".into()))
    }
}

#[async_trait]
impl<T: Embeddings + ?Sized> Embeddings for Arc<T> {
    fn model(&self) -> &str {
        (**self).model()
    }
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_documents(texts).await
    }
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_query(text).await
    }
}

#[async_trait]
impl<T: Embeddings + ?Sized> Embeddings for Box<T> {
    fn model(&self) -> &str {
        (**self).model()
    }
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_documents(texts).await
    }
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_query(text).await
    }
}

/// Adapts any [`Embeddings`] model to
/// [`sinew_core::context::TextEmbedder`], for
/// [`SemanticSelect`](sinew_core::context::SemanticSelect) context selection.
/// Errors other than [`Error::Core`] become `sinew_core::Error::Other`.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use sinew_core::context::{SemanticSelect, TextEmbedder};
/// use sinew_rag::embeddings::{ContextEmbedder, FakeEmbeddings};
///
/// let embedder = ContextEmbedder(FakeEmbeddings::new(64));
/// let vectors = embedder.embed(&["hello".to_string()]).await.unwrap();
/// assert_eq!(vectors[0].len(), 64);
/// let _select = SemanticSelect::new(embedder);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ContextEmbedder<E>(pub E);

#[async_trait]
impl<E: Embeddings> sinew_core::context::TextEmbedder for ContextEmbedder<E> {
    async fn embed(&self, texts: &[String]) -> sinew_core::Result<Vec<Vec<f32>>> {
        self.0.embed_documents(texts).await.map_err(|e| match e {
            Error::Core(e) => e,
            other => sinew_core::Error::Other(other.to_string()),
        })
    }
}

/// Builds an embedding model from `"provider:model"`.
///
/// Providers: `gemini` (`GEMINI_API_KEY`), the hosted OpenAI-compatible
/// services `openai`, `mistral`, `together`, `fireworks`, `nvidia`,
/// `openrouter`, `huggingface` (each reads the same key variable as its chat
/// preset), and the local servers `ollama`, `lmstudio`, `llamacpp`, `vllm`,
/// `sglang`, `localai`, `jan`, `llamafile`, `tgi`/`tei` (address from
/// `OLLAMA_HOST`, `LMSTUDIO_HOST`, ... or the usual default port).
/// `"http://host:port/v1|model"` targets a custom endpoint.
pub fn embeddings_from_spec(spec: &str) -> Result<Arc<dyn Embeddings>> {
    if let Some(model) = spec
        .strip_prefix("gemini:")
        .or_else(|| spec.strip_prefix("google:"))
    {
        return Ok(Arc::new(GeminiEmbeddings::from_env(model)?));
    }
    Ok(Arc::new(OpenAiEmbeddings::from_spec(spec)?))
}

pub(crate) async fn post_json<T: serde::de::DeserializeOwned>(
    provider: &str,
    req: reqwest::RequestBuilder,
    body: &serde_json::Value,
) -> Result<T> {
    let resp = req.json(body).send().await?;
    let status = resp.status();
    let bytes = resp.bytes().await?;
    if !status.is_success() {
        return Err(Error::Provider {
            provider: provider.into(),
            status: status.as_u16(),
            body: String::from_utf8_lossy(&bytes).into_owned(),
        });
    }
    Ok(serde_json::from_slice(&bytes)?)
}

pub(crate) type BatchFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>;

/// Runs batch requests with at most `concurrency` in flight and
/// concatenates the results in order.
pub(crate) async fn batched(
    batches: Vec<BatchFuture<'_>>,
    concurrency: usize,
) -> Result<Vec<Vec<f32>>> {
    use futures_util::{StreamExt, TryStreamExt};
    let results: Vec<Vec<Vec<f32>>> = futures_util::stream::iter(batches)
        .buffered(concurrency.max(1))
        .try_collect()
        .await?;
    Ok(results.into_iter().flatten().collect())
}
