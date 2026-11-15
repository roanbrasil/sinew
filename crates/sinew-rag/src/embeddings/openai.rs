//! OpenAI-compatible `/embeddings` client.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use super::{batched, post_json, BatchFuture, Embeddings};
use crate::{Error, Result};

/// A hosted service: name, aliases, base URL, API-key variable. Same names
/// and variables as the chat presets in `sinew-core`.
const HOSTED: &[(&str, &[&str], &str, &str)] = &[
    ("openai", &[], "https://api.openai.com/v1", "OPENAI_API_KEY"),
    (
        "mistral",
        &[],
        "https://api.mistral.ai/v1",
        "MISTRAL_API_KEY",
    ),
    (
        "together",
        &[],
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
    ),
    (
        "fireworks",
        &[],
        "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_API_KEY",
    ),
    (
        "nvidia",
        &["nim"],
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA_API_KEY",
    ),
    (
        "openrouter",
        &[],
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
    ),
    (
        "huggingface",
        &["hf"],
        "https://router.huggingface.co/v1",
        "HF_TOKEN",
    ),
];

/// A local server: name, aliases, default address, address variable.
const LOCAL: &[(&str, &[&str], &str, &str)] = &[
    ("ollama", &[], "http://127.0.0.1:11434", "OLLAMA_HOST"),
    (
        "lmstudio",
        &["lm-studio", "lm_studio"],
        "http://127.0.0.1:1234",
        "LMSTUDIO_HOST",
    ),
    (
        "llamacpp",
        &["llama.cpp", "llama-cpp", "llama_cpp"],
        "http://127.0.0.1:8080",
        "LLAMACPP_HOST",
    ),
    ("vllm", &[], "http://127.0.0.1:8000", "VLLM_HOST"),
    ("sglang", &[], "http://127.0.0.1:30000", "SGLANG_HOST"),
    ("localai", &[], "http://127.0.0.1:8080", "LOCALAI_HOST"),
    ("jan", &[], "http://127.0.0.1:1337", "JAN_HOST"),
    ("llamafile", &[], "http://127.0.0.1:8080", "LLAMAFILE_HOST"),
    ("tgi", &["tei"], "http://127.0.0.1:8080", "TGI_HOST"),
];

pub(crate) fn local_url(host: &str) -> String {
    let host = if host.starts_with("http") {
        host.to_string()
    } else {
        format!("http://{host}")
    };
    let host = host.trim_end_matches('/');
    if host.ends_with("/v1") {
        host.to_string()
    } else {
        format!("{host}/v1")
    }
}

/// Embeddings from any server that speaks the OpenAI `/embeddings` format.
///
/// Inputs are sent in batches (`batch_size`, default 256) with up to
/// `concurrency` (default 4) requests in flight over one pooled HTTP client.
#[derive(Clone)]
pub struct OpenAiEmbeddings {
    provider: String,
    base_url: String,
    api_key: Option<String>,
    model: String,
    dimensions: Option<u32>,
    batch_size: usize,
    concurrency: usize,
    client: reqwest::Client,
}

#[derive(Deserialize)]
struct Response {
    data: Vec<Item>,
}

#[derive(Deserialize)]
struct Item {
    embedding: Vec<f32>,
    #[serde(default)]
    index: Option<usize>,
}

impl OpenAiEmbeddings {
    /// `base_url` up to and including `/v1`, e.g. `http://127.0.0.1:11434/v1`.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: "openai-compatible".into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: None,
            model: model.into(),
            dimensions: None,
            batch_size: 256,
            concurrency: 4,
            client: reqwest::Client::new(),
        }
    }

    /// OpenAI itself; reads `OPENAI_API_KEY`.
    pub fn openai(model: impl Into<String>) -> Result<Self> {
        Self::hosted("openai", model)
    }

    /// Local Ollama (`OLLAMA_HOST` or 127.0.0.1:11434), e.g. `bge-m3`,
    /// `nomic-embed-text`, `mxbai-embed-large`.
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::local("ollama", model)
    }

    /// LM Studio (`LMSTUDIO_HOST` or 127.0.0.1:1234).
    pub fn lm_studio(model: impl Into<String>) -> Self {
        Self::local("lmstudio", model)
    }

    /// vLLM (`VLLM_HOST` or 127.0.0.1:8000).
    pub fn vllm(model: impl Into<String>) -> Self {
        Self::local("vllm", model)
    }

    /// llama.cpp `llama-server --embeddings` (`LLAMACPP_HOST` or 127.0.0.1:8080).
    pub fn llama_cpp(model: impl Into<String>) -> Self {
        Self::local("llamacpp", model)
    }

    /// Builds from `"provider:model"` (see
    /// [`embeddings_from_spec`](super::embeddings_from_spec)); everything
    /// after the first `:` is the model, so `"ollama:bge-m3:latest"` works.
    pub fn from_spec(spec: &str) -> Result<Self> {
        if spec.starts_with("http://") || spec.starts_with("https://") {
            let (url, model) = spec
                .split_once('|')
                .ok_or_else(|| Error::Config(format!("expected `url|model`, got `{spec}`")))?;
            let mut e = Self::new(url, model);
            if let Ok(k) = std::env::var("LOCAL_API_KEY") {
                e = e.with_api_key(k);
            }
            return Ok(e);
        }
        let (provider, model) = spec
            .split_once(':')
            .ok_or_else(|| Error::Config(format!("expected `provider:model`, got `{spec}`")))?;
        if model.is_empty() {
            return Err(Error::Config(format!("no model in `{spec}`")));
        }
        let p = provider.to_ascii_lowercase();
        if let Some(h) = HOSTED
            .iter()
            .find(|h| h.0 == p || h.1.contains(&p.as_str()))
        {
            return Self::hosted(h.0, model);
        }
        if let Some(l) = LOCAL.iter().find(|l| l.0 == p || l.1.contains(&p.as_str())) {
            return Ok(Self::local(l.0, model));
        }
        let known: Vec<&str> = std::iter::once("gemini")
            .chain(HOSTED.iter().map(|h| h.0))
            .chain(LOCAL.iter().map(|l| l.0))
            .collect();
        Err(Error::Config(format!(
            "unknown embeddings provider `{provider}`; known: {}",
            known.join(", ")
        )))
    }

    fn hosted(name: &str, model: impl Into<String>) -> Result<Self> {
        let (_, _, url, var) = HOSTED.iter().find(|h| h.0 == name).expect("known preset");
        let key = std::env::var(var).map_err(|_| Error::Config(format!("{var} not set")))?;
        Ok(Self::new(*url, model).with_api_key(key).named(name))
    }

    fn local(name: &str, model: impl Into<String>) -> Self {
        let (_, _, default, var) = LOCAL.iter().find(|l| l.0 == name).expect("known preset");
        let host = std::env::var(var).unwrap_or_else(|_| default.to_string());
        let mut e = Self::new(local_url(&host), model).named(name);
        if let Ok(k) = std::env::var("LOCAL_API_KEY") {
            e = e.with_api_key(k);
        }
        e
    }

    fn named(mut self, provider: &str) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Asks for shorter vectors (OpenAI `text-embedding-3-*` `dimensions`).
    pub fn with_dimensions(mut self, n: u32) -> Self {
        self.dimensions = Some(n);
        self
    }

    /// Inputs per request (default 256).
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    /// Requests in flight while embedding many batches (default 4).
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    /// Shares a configured `reqwest::Client` (timeouts, proxies, pool).
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Most servers reject empty strings.
        let input: Vec<&str> = texts
            .iter()
            .map(|t| if t.is_empty() { " " } else { t.as_str() })
            .collect();
        let mut body = json!({ "model": self.model, "input": input });
        if let Some(d) = self.dimensions {
            body["dimensions"] = json!(d);
        }
        let mut req = self.client.post(format!("{}/embeddings", self.base_url));
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        let resp: Response = post_json(&self.provider, req, &body).await?;
        if resp.data.len() != texts.len() {
            return Err(Error::InvalidResponse(format!(
                "{} returned {} embeddings for {} inputs",
                self.provider,
                resp.data.len(),
                texts.len()
            )));
        }
        let mut data = resp.data;
        if data.iter().all(|d| d.index.is_some()) {
            data.sort_by_key(|d| d.index);
        }
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl Embeddings for OpenAiEmbeddings {
    fn model(&self) -> &str {
        &self.model
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() <= self.batch_size {
            return self.embed_batch(texts).await;
        }
        let batches: Vec<BatchFuture<'_>> = texts
            .chunks(self.batch_size)
            .map(|b| Box::pin(self.embed_batch(b)) as BatchFuture<'_>)
            .collect();
        batched(batches, self.concurrency).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_resolves_local_servers() {
        let e = OpenAiEmbeddings::from_spec("ollama:bge-m3:latest").unwrap();
        assert_eq!(e.model(), "bge-m3:latest");
        assert!(e.base_url().ends_with("/v1"));
        for p in ["lmstudio", "llama.cpp", "vllm", "tei", "jan"] {
            assert_eq!(
                OpenAiEmbeddings::from_spec(&format!("{p}:m"))
                    .unwrap()
                    .model(),
                "m"
            );
        }
        let e = OpenAiEmbeddings::from_spec("http://10.0.0.5:8000/v1/|served").unwrap();
        assert_eq!(e.base_url(), "http://10.0.0.5:8000/v1");
    }

    #[test]
    fn spec_errors() {
        assert!(OpenAiEmbeddings::from_spec("nope:m")
            .err()
            .unwrap()
            .to_string()
            .contains("ollama"));
        assert!(OpenAiEmbeddings::from_spec("ollama:").is_err());
        assert!(OpenAiEmbeddings::from_spec("x").is_err());
        assert_eq!(local_url("box:1234"), "http://box:1234/v1");
    }
}
