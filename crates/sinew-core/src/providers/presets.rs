//! Ready-made [`OpenAiCompat`] endpoints for hosted and local servers that
//! speak the OpenAI Chat Completions format, plus [`from_spec`], which builds
//! any supported model from a `"provider:model"` string.

use std::sync::Arc;

use super::{Anthropic, Gemini, OpenAiCompat};
use crate::{ChatModel, Error, Result};

/// A hosted, OpenAI-compatible service.
struct Hosted {
    name: &'static str,
    aliases: &'static [&'static str],
    base_url: &'static str,
    key_var: &'static str,
}

const HOSTED: &[Hosted] = &[
    Hosted {
        name: "openai",
        aliases: &[],
        base_url: "https://api.openai.com/v1",
        key_var: "OPENAI_API_KEY",
    },
    Hosted {
        name: "xai",
        aliases: &["grok"],
        base_url: "https://api.x.ai/v1",
        key_var: "XAI_API_KEY",
    },
    Hosted {
        name: "groq",
        aliases: &[],
        base_url: "https://api.groq.com/openai/v1",
        key_var: "GROQ_API_KEY",
    },
    Hosted {
        name: "mistral",
        aliases: &[],
        base_url: "https://api.mistral.ai/v1",
        key_var: "MISTRAL_API_KEY",
    },
    Hosted {
        name: "deepseek",
        aliases: &[],
        base_url: "https://api.deepseek.com/v1",
        key_var: "DEEPSEEK_API_KEY",
    },
    Hosted {
        name: "openrouter",
        aliases: &[],
        base_url: "https://openrouter.ai/api/v1",
        key_var: "OPENROUTER_API_KEY",
    },
    Hosted {
        name: "together",
        aliases: &[],
        base_url: "https://api.together.xyz/v1",
        key_var: "TOGETHER_API_KEY",
    },
    Hosted {
        name: "fireworks",
        aliases: &[],
        base_url: "https://api.fireworks.ai/inference/v1",
        key_var: "FIREWORKS_API_KEY",
    },
    Hosted {
        name: "perplexity",
        aliases: &[],
        base_url: "https://api.perplexity.ai",
        key_var: "PERPLEXITY_API_KEY",
    },
    Hosted {
        name: "cerebras",
        aliases: &[],
        base_url: "https://api.cerebras.ai/v1",
        key_var: "CEREBRAS_API_KEY",
    },
    Hosted {
        name: "nvidia",
        aliases: &["nim"],
        base_url: "https://integrate.api.nvidia.com/v1",
        key_var: "NVIDIA_API_KEY",
    },
    Hosted {
        name: "huggingface",
        aliases: &["hf"],
        base_url: "https://router.huggingface.co/v1",
        key_var: "HF_TOKEN",
    },
];

/// A local server. `host_var` overrides the default address.
struct Local {
    name: &'static str,
    aliases: &'static [&'static str],
    default_host: &'static str,
    host_var: &'static str,
}

const LOCAL: &[Local] = &[
    Local {
        name: "ollama",
        aliases: &[],
        default_host: "http://127.0.0.1:11434",
        host_var: "OLLAMA_HOST",
    },
    Local {
        name: "lmstudio",
        aliases: &["lm-studio", "lm_studio"],
        default_host: "http://127.0.0.1:1234",
        host_var: "LMSTUDIO_HOST",
    },
    Local {
        name: "llamacpp",
        aliases: &["llama.cpp", "llama-cpp", "llama_cpp"],
        default_host: "http://127.0.0.1:8080",
        host_var: "LLAMACPP_HOST",
    },
    Local {
        name: "vllm",
        aliases: &[],
        default_host: "http://127.0.0.1:8000",
        host_var: "VLLM_HOST",
    },
    Local {
        name: "sglang",
        aliases: &[],
        default_host: "http://127.0.0.1:30000",
        host_var: "SGLANG_HOST",
    },
    Local {
        name: "localai",
        aliases: &[],
        default_host: "http://127.0.0.1:8080",
        host_var: "LOCALAI_HOST",
    },
    Local {
        name: "jan",
        aliases: &[],
        default_host: "http://127.0.0.1:1337",
        host_var: "JAN_HOST",
    },
    Local {
        name: "llamafile",
        aliases: &[],
        default_host: "http://127.0.0.1:8080",
        host_var: "LLAMAFILE_HOST",
    },
    Local {
        name: "tgi",
        aliases: &[],
        default_host: "http://127.0.0.1:8080",
        host_var: "TGI_HOST",
    },
];

fn matches(name: &str, primary: &str, aliases: &[&str]) -> bool {
    primary == name || aliases.contains(&name)
}

fn local_url(host: String) -> String {
    let host = if host.starts_with("http") {
        host
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

fn hosted(name: &str, model: impl Into<String>) -> Result<OpenAiCompat> {
    let h = HOSTED
        .iter()
        .find(|h| h.name == name)
        .expect("known preset");
    let key =
        std::env::var(h.key_var).map_err(|_| Error::Config(format!("{} not set", h.key_var)))?;
    Ok(OpenAiCompat::new(h.base_url, model).with_api_key(key))
}

fn local(name: &str, model: impl Into<String>) -> OpenAiCompat {
    let l = LOCAL.iter().find(|l| l.name == name).expect("known preset");
    let host = std::env::var(l.host_var).unwrap_or_else(|_| l.default_host.to_string());
    let mut m = OpenAiCompat::new(local_url(host), model);
    // Most local servers ignore the key; send one when the user set it.
    if let Ok(k) = std::env::var("LOCAL_API_KEY") {
        m = m.with_api_key(k);
    }
    m
}

/// Default Azure OpenAI `api-version` (latest GA data-plane version).
pub const AZURE_DEFAULT_API_VERSION: &str = "2024-10-21";

fn azure_url(endpoint: &str, deployment: &str) -> String {
    let endpoint = endpoint.trim_end_matches('/');
    let endpoint = if endpoint.starts_with("http") {
        endpoint.to_string()
    } else {
        // A bare resource name.
        format!("https://{endpoint}.openai.azure.com")
    };
    format!("{endpoint}/openai/deployments/{deployment}")
}

impl OpenAiCompat {
    /// Azure OpenAI: `POST {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...`
    /// with the key in the `api-key` header. `endpoint` is
    /// `https://{resource}.openai.azure.com` (or just the resource name).
    /// The deployment name is also used as the model name.
    pub fn azure(
        endpoint: &str,
        deployment: impl Into<String>,
        api_version: &str,
        api_key: impl Into<String>,
    ) -> Self {
        let deployment = deployment.into();
        OpenAiCompat::new(azure_url(endpoint, &deployment), deployment)
            .with_api_key(api_key)
            .with_api_key_header("api-key")
            .with_query("api-version", api_version)
    }

    /// Azure OpenAI from the environment: `AZURE_OPENAI_ENDPOINT`,
    /// `AZURE_OPENAI_API_KEY` and optionally `AZURE_OPENAI_API_VERSION`
    /// (default [`AZURE_DEFAULT_API_VERSION`]).
    pub fn azure_from_env(deployment: impl Into<String>) -> Result<Self> {
        let var = |k: &str| std::env::var(k).map_err(|_| Error::Config(format!("{k} not set")));
        let endpoint = var("AZURE_OPENAI_ENDPOINT")?;
        let key = var("AZURE_OPENAI_API_KEY")?;
        let version = std::env::var("AZURE_OPENAI_API_VERSION")
            .unwrap_or_else(|_| AZURE_DEFAULT_API_VERSION.to_string());
        Ok(Self::azure(&endpoint, deployment, &version, key))
    }

    /// xAI Grok (`XAI_API_KEY`).
    pub fn xai(model: impl Into<String>) -> Result<Self> {
        hosted("xai", model)
    }
    /// Groq (`GROQ_API_KEY`).
    pub fn groq(model: impl Into<String>) -> Result<Self> {
        hosted("groq", model)
    }
    /// Mistral AI (`MISTRAL_API_KEY`).
    pub fn mistral(model: impl Into<String>) -> Result<Self> {
        hosted("mistral", model)
    }
    /// DeepSeek (`DEEPSEEK_API_KEY`).
    pub fn deepseek(model: impl Into<String>) -> Result<Self> {
        hosted("deepseek", model)
    }
    /// OpenRouter (`OPENROUTER_API_KEY`); models look like `vendor/model`.
    pub fn openrouter(model: impl Into<String>) -> Result<Self> {
        hosted("openrouter", model)
    }
    /// Together AI (`TOGETHER_API_KEY`).
    pub fn together(model: impl Into<String>) -> Result<Self> {
        hosted("together", model)
    }
    /// Fireworks AI (`FIREWORKS_API_KEY`).
    pub fn fireworks(model: impl Into<String>) -> Result<Self> {
        hosted("fireworks", model)
    }
    /// Perplexity (`PERPLEXITY_API_KEY`).
    pub fn perplexity(model: impl Into<String>) -> Result<Self> {
        hosted("perplexity", model)
    }
    /// Cerebras (`CEREBRAS_API_KEY`).
    pub fn cerebras(model: impl Into<String>) -> Result<Self> {
        hosted("cerebras", model)
    }
    /// NVIDIA NIM hosted API (`NVIDIA_API_KEY`).
    pub fn nvidia(model: impl Into<String>) -> Result<Self> {
        hosted("nvidia", model)
    }
    /// Hugging Face Inference Providers router (`HF_TOKEN`).
    pub fn huggingface(model: impl Into<String>) -> Result<Self> {
        hosted("huggingface", model)
    }

    /// LM Studio server (`LMSTUDIO_HOST` or 127.0.0.1:1234).
    pub fn lm_studio(model: impl Into<String>) -> Self {
        local("lmstudio", model)
    }
    /// llama.cpp `llama-server` (`LLAMACPP_HOST` or 127.0.0.1:8080).
    pub fn llama_cpp(model: impl Into<String>) -> Self {
        local("llamacpp", model)
    }
    /// vLLM OpenAI server (`VLLM_HOST` or 127.0.0.1:8000).
    pub fn vllm(model: impl Into<String>) -> Self {
        local("vllm", model)
    }
    /// SGLang server (`SGLANG_HOST` or 127.0.0.1:30000).
    pub fn sglang(model: impl Into<String>) -> Self {
        local("sglang", model)
    }
    /// LocalAI (`LOCALAI_HOST` or 127.0.0.1:8080).
    pub fn localai(model: impl Into<String>) -> Self {
        local("localai", model)
    }
    /// Jan (`JAN_HOST` or 127.0.0.1:1337).
    pub fn jan(model: impl Into<String>) -> Self {
        local("jan", model)
    }
    /// llamafile (`LLAMAFILE_HOST` or 127.0.0.1:8080).
    pub fn llamafile(model: impl Into<String>) -> Self {
        local("llamafile", model)
    }
    /// Hugging Face Text Generation Inference (`TGI_HOST` or 127.0.0.1:8080).
    pub fn tgi(model: impl Into<String>) -> Self {
        local("tgi", model)
    }
}

/// Every provider name [`from_spec`] understands (aliases not included).
pub fn provider_names() -> Vec<&'static str> {
    let mut v = vec!["anthropic", "gemini", "azure"];
    v.extend(HOSTED.iter().map(|h| h.name));
    v.extend(LOCAL.iter().map(|l| l.name));
    v
}

/// Builds a model from `"provider:model"`, e.g. `"gemini:gemini-2.5-flash"`,
/// `"xai:grok-4"`, `"lmstudio:qwen3-8b"`, `"ollama:llama3.2"`,
/// `"azure:my-gpt4o-deployment"` (Azure OpenAI; see
/// [`OpenAiCompat::azure_from_env`]).
///
/// Anything after the first `:` is the model id, so Ollama tags such as
/// `"ollama:qwen3:4b"` work. An `http(s)://` URL as the provider part is
/// treated as a custom OpenAI-compatible endpoint:
/// `"http://10.0.0.5:8000/v1|my-model"` (separated by `|`).
pub fn from_spec(spec: &str) -> Result<Arc<dyn ChatModel>> {
    if spec.starts_with("http://") || spec.starts_with("https://") {
        let (url, model) = spec
            .split_once('|')
            .ok_or_else(|| Error::Config(format!("expected `url|model`, got `{spec}`")))?;
        return Ok(Arc::new(OpenAiCompat::new(url, model)));
    }
    let (provider, model) = spec
        .split_once(':')
        .ok_or_else(|| Error::Config(format!("expected `provider:model`, got `{spec}`")))?;
    let provider = provider.to_ascii_lowercase();
    let p = provider.as_str();
    if model.is_empty() {
        return Err(Error::Config(format!("no model in `{spec}`")));
    }
    match p {
        "anthropic" | "claude" => return Ok(Arc::new(Anthropic::from_env(model)?)),
        "gemini" | "google" => return Ok(Arc::new(Gemini::from_env(model)?)),
        "azure" | "azure-openai" | "azure_openai" => {
            return Ok(Arc::new(OpenAiCompat::azure_from_env(model)?))
        }
        _ => {}
    }
    if let Some(h) = HOSTED.iter().find(|h| matches(p, h.name, h.aliases)) {
        return Ok(Arc::new(hosted(h.name, model)?));
    }
    if let Some(l) = LOCAL.iter().find(|l| matches(p, l.name, l.aliases)) {
        return Ok(Arc::new(local(l.name, model)));
    }
    Err(Error::Config(format!(
        "unknown provider `{provider}`; known: {}",
        provider_names().join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_urls_are_normalised() {
        assert_eq!(
            local_url("127.0.0.1:1234".into()),
            "http://127.0.0.1:1234/v1"
        );
        assert_eq!(local_url("http://box:8000/".into()), "http://box:8000/v1");
        assert_eq!(local_url("http://box:8000/v1".into()), "http://box:8000/v1");
    }

    #[test]
    fn spec_resolves_local_providers_and_keeps_model_tags() {
        let m = from_spec("ollama:qwen3:4b").unwrap();
        assert_eq!(m.name(), "qwen3:4b");
        for p in [
            "lmstudio",
            "lm-studio",
            "llama.cpp",
            "vllm",
            "sglang",
            "localai",
            "jan",
            "llamafile",
            "tgi",
        ] {
            assert_eq!(from_spec(&format!("{p}:m")).unwrap().name(), "m", "{p}");
        }
    }

    #[test]
    fn spec_custom_url() {
        let m = from_spec("http://10.0.0.5:8000/v1|served").unwrap();
        assert_eq!(m.name(), "served");
        assert!(from_spec("http://x/v1").is_err());
    }

    #[test]
    fn spec_errors_are_helpful() {
        let e = from_spec("nope:m").err().unwrap().to_string();
        assert!(e.contains("unknown provider") && e.contains("gemini") && e.contains("xai"));
        assert!(from_spec("no-colon").is_err());
        assert!(from_spec("ollama:").is_err());
    }

    #[test]
    fn azure_preset_builds_deployment_url() {
        let m = OpenAiCompat::azure("https://res.openai.azure.com/", "gpt4o", "2024-10-21", "k");
        assert_eq!(
            m.base_url(),
            "https://res.openai.azure.com/openai/deployments/gpt4o"
        );
        assert_eq!(m.name(), "gpt4o");
        let m = OpenAiCompat::azure("res", "d", AZURE_DEFAULT_API_VERSION, "k");
        assert_eq!(
            m.base_url(),
            "https://res.openai.azure.com/openai/deployments/d"
        );
        assert!(provider_names().contains(&"azure"));
    }

    #[test]
    fn hosted_presets_need_their_key() {
        for h in HOSTED {
            assert!(h.base_url.starts_with("https://"), "{}", h.name);
            assert!(h.key_var.ends_with("_KEY") || h.key_var == "HF_TOKEN");
        }
        assert!(provider_names().contains(&"xai"));
        assert!(provider_names().contains(&"lmstudio"));
    }
}
