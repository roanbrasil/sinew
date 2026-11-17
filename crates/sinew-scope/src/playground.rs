//! Playground: run one prompt against several models side by side.
//!
//! - `GET /api/playground/models`: the providers and models the playground
//!   may use (the [`catalog`] filtered by the allowed providers, models
//!   registered with [`PlaygroundConfig::with_model`] and, for allowed local
//!   servers, the models they report on `/v1/models`).
//! - `POST /api/playground/run`: renders raw messages or a prompt-hub version
//!   with variables, runs every requested model concurrently and returns (or
//!   streams as Server-Sent Events) the output with latency, time to first
//!   token, tokens and cost per model. Each call is traced into the store
//!   under the `playground` project.
//! - `GET /api/runs/{id}/playground`: the inputs of a stored LLM run, to
//!   prefill the editor.
//!
//! Models are built on the server with the server's environment (API keys
//! and hosts), never with credentials from the browser. Which providers can
//! be used is a security boundary: by default only local providers
//! (Ollama, LM Studio, llama.cpp, vLLM, ...) are allowed, so a token holder
//! cannot spend the server's hosted-provider credits unless the operator
//! lists those providers with [`PlaygroundConfig::allow_providers`] (CLI:
//! `--playground-providers`). Custom `http(s)://` endpoint specs are always
//! refused, so the playground cannot be pointed at arbitrary URLs.

use std::{
    collections::{BTreeMap, HashMap},
    convert::Infallible,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{
    chat_streaming,
    providers::{capabilities_for, catalog, ProviderKind},
    ChatModel, ChatRequest, Message, PromptTemplate, ResponseFormat, Role, StreamEvent, ToolSpec,
};
use sinew_trace::{MemorySink, Run, RunKind, RunRef, TraceSink, Tracer};
use tokio::sync::mpsc;

use crate::{
    engine::{default_model_factory, ModelFactory},
    pricing::{run_model, Pricing},
    server::{parse_json, ApiError, ApiResult, AppState},
};

/// Wildcard accepted by [`PlaygroundConfig::allow_providers`] and
/// [`parse_providers`]: every provider in the catalog.
pub const ALL_PROVIDERS: &str = "*";

/// A model, or why it could not be built (reported per model).
type Built = Result<Arc<dyn ChatModel>, String>;

/// Settings of the playground endpoints (see the [module docs](self)).
///
/// The default allows the local providers of
/// [`catalog`] only, builds models with
/// [`from_spec`](sinew_core::providers::from_spec), traces into the
/// `playground` project, accepts up to 8 models per request and stops a model
/// call after 5 minutes.
///
/// ```
/// use std::sync::Arc;
/// use sinew_core::{mock::ScriptedModel, Message};
/// use sinew_scope::{playground::PlaygroundConfig, ServerConfig};
///
/// let playground = PlaygroundConfig::default()
///     .allow_providers(["ollama", "openai"])
///     .with_model(
///         "test:echo",
///         Arc::new(ScriptedModel::new([Message::assistant("hi")])),
///     );
/// assert!(playground.allows("openai"));
/// assert!(!playground.allows("anthropic"));
/// let config = ServerConfig { playground, ..Default::default() };
/// # let _ = config;
/// ```
#[derive(Clone)]
pub struct PlaygroundConfig {
    allowed: Vec<String>,
    factory: ModelFactory,
    models: BTreeMap<String, Arc<dyn ChatModel>>,
    project: String,
    max_models: usize,
    timeout: Duration,
    discover: bool,
}

impl Default for PlaygroundConfig {
    fn default() -> Self {
        Self {
            allowed: local_providers(),
            factory: default_model_factory(),
            models: BTreeMap::new(),
            project: "playground".into(),
            max_models: 8,
            timeout: Duration::from_secs(300),
            discover: true,
        }
    }
}

/// Names of the local providers in the catalog.
fn local_providers() -> Vec<String> {
    catalog()
        .into_iter()
        .filter(|p| p.kind == ProviderKind::Local)
        .map(|p| p.name.to_string())
        .collect()
}

/// The catalog name for a provider name or alias (`claude` → `anthropic`).
fn canonical_provider(name: &str) -> Option<&'static str> {
    let n = name.trim().to_ascii_lowercase();
    catalog()
        .into_iter()
        .find(|p| p.name == n || p.aliases.contains(&n.as_str()))
        .map(|p| p.name)
}

/// Parses a comma-separated provider list as given to
/// `--playground-providers`: catalog names or aliases, `*` / `all` for
/// every provider, `local` for the local ones, or `none` (or an empty
/// string) to allow none. Unknown names are an error.
///
/// ```
/// use sinew_scope::playground::parse_providers;
///
/// assert_eq!(parse_providers("ollama, claude").unwrap(), ["ollama", "anthropic"]);
/// assert_eq!(parse_providers("all").unwrap(), ["*"]);
/// assert!(parse_providers("none").unwrap().is_empty());
/// assert!(parse_providers("nope").is_err());
/// ```
pub fn parse_providers(list: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for item in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let names = match item.to_ascii_lowercase().as_str() {
            "*" | "all" => vec![ALL_PROVIDERS.to_string()],
            "local" => local_providers(),
            "none" => Vec::new(),
            other => vec![canonical_provider(other)
                .ok_or_else(|| {
                    format!(
                        "unknown provider `{other}`; known: {}",
                        sinew_core::providers::provider_names().join(", ")
                    )
                })?
                .to_string()],
        };
        for n in names {
            if !out.contains(&n) {
                out.push(n);
            }
        }
    }
    Ok(out)
}

impl PlaygroundConfig {
    /// Replaces the allowed providers. Names and aliases from the catalog
    /// are accepted (`claude` means `anthropic`); [`ALL_PROVIDERS`] allows
    /// every provider. An empty list disables every `from_spec` provider
    /// (models registered with [`with_model`](Self::with_model) still work).
    ///
    /// ```
    /// use sinew_scope::playground::PlaygroundConfig;
    ///
    /// let c = PlaygroundConfig::default().allow_providers(["claude"]);
    /// assert!(c.allows("anthropic") && !c.allows("ollama"));
    /// ```
    pub fn allow_providers<I, S>(mut self, providers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.allowed = providers
            .into_iter()
            .map(|p| {
                let p = p.as_ref().trim();
                canonical_provider(p)
                    .map(str::to_string)
                    .unwrap_or_else(|| p.to_ascii_lowercase())
            })
            .filter(|p| !p.is_empty())
            .collect();
        self
    }

    /// Whether `provider` (a catalog name or alias) may be used.
    ///
    /// ```
    /// use sinew_scope::playground::PlaygroundConfig;
    ///
    /// let c = PlaygroundConfig::default();
    /// assert!(c.allows("ollama") && c.allows("lm-studio"));
    /// assert!(!c.allows("openai"));
    /// ```
    pub fn allows(&self, provider: &str) -> bool {
        if self.allowed.iter().any(|a| a == ALL_PROVIDERS) {
            return true;
        }
        let name = canonical_provider(provider)
            .map(str::to_string)
            .unwrap_or_else(|| provider.trim().to_ascii_lowercase());
        self.allowed.contains(&name)
    }

    /// The allowed provider names.
    ///
    /// ```
    /// use sinew_scope::playground::PlaygroundConfig;
    ///
    /// assert!(PlaygroundConfig::default().allowed_providers().contains(&"ollama".to_string()));
    /// ```
    pub fn allowed_providers(&self) -> &[String] {
        &self.allowed
    }

    /// Serves `model` for the exact `spec` (for example `test:echo`). A
    /// registered model is always allowed and is listed by
    /// `GET /api/playground/models`; tests use it to run the playground with
    /// a scripted model.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use sinew_core::{mock::ScriptedModel, Message};
    /// use sinew_scope::playground::PlaygroundConfig;
    ///
    /// let c = PlaygroundConfig::default()
    ///     .with_model("test:a", Arc::new(ScriptedModel::new([Message::assistant("ok")])));
    /// # let _ = c;
    /// ```
    pub fn with_model(mut self, spec: impl Into<String>, model: Arc<dyn ChatModel>) -> Self {
        self.models.insert(spec.into(), model);
        self
    }

    /// Replaces the function that builds a model from an allowed
    /// `provider:model` spec (default: `sinew_core::providers::from_spec`).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use sinew_scope::playground::PlaygroundConfig;
    ///
    /// let c = PlaygroundConfig::default()
    ///     .with_model_factory(Arc::new(|spec: &str| Err(format!("no model for {spec}"))));
    /// # let _ = c;
    /// ```
    pub fn with_model_factory(mut self, factory: ModelFactory) -> Self {
        self.factory = factory;
        self
    }

    /// The project playground traces are stored under (default
    /// `playground`).
    ///
    /// ```
    /// use sinew_scope::playground::PlaygroundConfig;
    ///
    /// let c = PlaygroundConfig::default().project("sandbox");
    /// # let _ = c;
    /// ```
    pub fn project(mut self, project: impl Into<String>) -> Self {
        self.project = project.into();
        self
    }

    /// The most models one request may run (default 8).
    ///
    /// ```
    /// use sinew_scope::playground::PlaygroundConfig;
    ///
    /// let c = PlaygroundConfig::default().max_models(4);
    /// # let _ = c;
    /// ```
    pub fn max_models(mut self, n: usize) -> Self {
        self.max_models = n.max(1);
        self
    }

    /// Time limit for one model call (default 5 minutes).
    ///
    /// ```
    /// use std::time::Duration;
    /// use sinew_scope::playground::PlaygroundConfig;
    ///
    /// let c = PlaygroundConfig::default().timeout(Duration::from_secs(60));
    /// # let _ = c;
    /// ```
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Whether `GET /api/playground/models` asks allowed local servers for
    /// their models (`GET <host>/v1/models`, 1.5 s timeout; default on).
    ///
    /// ```
    /// use sinew_scope::playground::PlaygroundConfig;
    ///
    /// let c = PlaygroundConfig::default().discover_models(false);
    /// # let _ = c;
    /// ```
    pub fn discover_models(mut self, on: bool) -> Self {
        self.discover = on;
        self
    }

    /// Resolves a spec to a model, enforcing the allowed providers. The
    /// outer error rejects the whole request; the inner one (a model that
    /// cannot be built, e.g. a missing API key) is reported per model.
    fn resolve(&self, spec: &str) -> Result<Built, ApiError> {
        if let Some(m) = self.models.get(spec) {
            return Ok(Ok(m.clone()));
        }
        if spec.starts_with("http://") || spec.starts_with("https://") {
            return Err(ApiError(
                StatusCode::FORBIDDEN,
                format!("`{spec}`: custom endpoint URLs are not allowed in the playground"),
            ));
        }
        let Some((provider, model)) = spec.split_once(':') else {
            return Err(ApiError::bad_request(format!(
                "`{spec}`: expected `provider:model`"
            )));
        };
        if model.is_empty() {
            return Err(ApiError::bad_request(format!("`{spec}`: no model name")));
        }
        let Some(canonical) = canonical_provider(provider) else {
            return Err(ApiError::bad_request(format!(
                "`{spec}`: unknown provider `{provider}`"
            )));
        };
        if !self.allows(canonical) {
            return Err(ApiError(
                StatusCode::FORBIDDEN,
                format!(
                    "provider `{canonical}` is not allowed in the playground; \
                     allowed: {}. Start the server with \
                     `--playground-providers` to allow more",
                    if self.allowed.is_empty() {
                        "none".to_string()
                    } else {
                        self.allowed.join(", ")
                    }
                ),
            ));
        }
        Ok((self.factory)(spec))
    }
}

// ---------- GET /api/playground/models ----------

/// Base URL of a local server: the host variable or the default host,
/// normalised to `http://host:port`.
fn local_base(env_var: &str, default_host: &str) -> String {
    let host = std::env::var(env_var)
        .ok()
        .filter(|h| !h.trim().is_empty())
        .unwrap_or_else(|| default_host.to_string());
    let host = host.trim().trim_end_matches('/');
    let host = host.strip_suffix("/v1").unwrap_or(host);
    if host.starts_with("http://") || host.starts_with("https://") {
        host.to_string()
    } else {
        format!("http://{host}")
    }
}

/// Model ids a local OpenAI-compatible server reports, or an empty list.
async fn discover_local(base: String) -> Vec<String> {
    let client = sinew_core::providers::shared_client();
    let resp = client
        .get(format!("{base}/v1/models"))
        .timeout(Duration::from_millis(1500))
        .send()
        .await;
    let Ok(resp) = resp else {
        return Vec::new();
    };
    if !resp.status().is_success() {
        return Vec::new();
    }
    let Ok(body) = resp.json::<Value>().await else {
        return Vec::new();
    };
    let mut ids: Vec<String> = body["data"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m["id"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    ids.sort();
    ids.dedup();
    ids
}

/// `GET /api/playground/models`.
pub(crate) async fn models(State(state): State<AppState>) -> Json<Value> {
    let cfg = &state.playground;
    let allowed: Vec<_> = catalog()
        .into_iter()
        .filter(|p| cfg.allows(p.name))
        .collect();
    let discovered: Vec<Vec<String>> = futures_util::future::join_all(allowed.iter().map(|p| {
        let base =
            (cfg.discover && p.kind == ProviderKind::Local).then(|| local_base(p.env_var, p.url));
        async move {
            match base {
                Some(b) => discover_local(b).await,
                None => Vec::new(),
            }
        }
    }))
    .await;
    let mut models: Vec<Value> = cfg
        .models
        .iter()
        .map(|(spec, m)| {
            json!({
                "spec": spec,
                "provider": spec.split_once(':').map(|(p, _)| p).unwrap_or(""),
                "capabilities": m.capabilities(),
                "source": "registered",
            })
        })
        .collect();
    let providers: Vec<Value> = allowed
        .iter()
        .zip(&discovered)
        .map(|(p, ids)| {
            for id in ids {
                models.push(json!({
                    "spec": format!("{}:{id}", p.name),
                    "provider": p.name,
                    "capabilities": capabilities_for(p.name, id),
                    "source": "discovered",
                }));
            }
            let configured = p.kind == ProviderKind::Local
                || std::env::var(p.env_var).is_ok_and(|v| !v.trim().is_empty());
            json!({
                "name": p.name,
                "aliases": p.aliases,
                "kind": p.kind,
                "env_var": p.env_var,
                "configured": configured,
                "capabilities": p.capabilities,
            })
        })
        .collect();
    Json(json!({
        "project": cfg.project,
        "allowed_providers": cfg.allowed,
        "providers": providers,
        "models": models,
        "max_models": cfg.max_models,
    }))
}

// ---------- POST /api/playground/run ----------

/// Body of `POST /api/playground/run`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RunRequest {
    /// Chat messages; rendered as templates when `variables` is present.
    messages: Option<Vec<Message>>,
    /// A prompt-hub reference, `name` or `name@ref`.
    prompt: Option<String>,
    /// Values for `{variable}` placeholders (strings; other JSON values are
    /// used as their JSON text).
    variables: Option<BTreeMap<String, Value>>,
    /// `provider:model` specs.
    models: Vec<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    tools: Vec<ToolSpec>,
    response_format: Option<ResponseFormat>,
    /// Stream Server-Sent Events instead of returning one JSON body.
    stream: bool,
}

/// Everything needed to run the request, after validation.
struct Plan {
    messages: Vec<Message>,
    request: ChatRequest,
    models: Vec<(String, Built)>,
    /// The root run's inputs.
    inputs: Value,
    /// `name@commit` of a prompt-hub version.
    prompt: Option<String>,
}

/// One model's outcome, as returned to the client.
#[derive(Debug, Clone, Serialize)]
struct ModelResult {
    index: usize,
    spec: String,
    model: Option<String>,
    run_id: Option<String>,
    output: Option<Message>,
    error: Option<String>,
    latency_ms: u64,
    ttft_ms: Option<u64>,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    cost_usd: Option<f64>,
    stop_reason: Option<String>,
    /// Options not sent because the model does not support them.
    ignored: Vec<&'static str>,
}

/// `f32` → `f64` without the binary noise (`0.2` stays `0.2`).
fn t64(t: Option<f32>) -> Option<f64> {
    t.map(|t| (f64::from(t) * 1e6).round() / 1e6)
}

fn var_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

async fn plan(state: &AppState, req: RunRequest) -> Result<Plan, ApiError> {
    let vars: Option<HashMap<String, String>> = req
        .variables
        .as_ref()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), var_text(v))).collect());
    let mut models = req.models.clone();
    let mut temperature = req.temperature;
    let mut max_tokens = req.max_tokens;
    let mut prompt_ref = None;
    let messages = match (&req.prompt, &req.messages) {
        (Some(_), Some(m)) if !m.is_empty() => {
            return Err(ApiError::bad_request(
                "send either `messages` or `prompt`, not both",
            ))
        }
        (Some(p), _) => {
            let (name, reference) = p.split_once('@').unwrap_or((p.as_str(), "latest"));
            let v = state
                .store
                .resolve_prompt(name, reference)
                .await?
                .ok_or_else(|| ApiError::not_found("prompt version"))?;
            let empty = HashMap::new();
            let vars = vars.as_ref().unwrap_or(&empty);
            let borrowed: HashMap<&str, String> =
                vars.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            let messages = v
                .render_messages(&borrowed)
                .map_err(|e| ApiError::bad_request(format!("prompt {name}: {e}")))?;
            let cfg = &v.model_config;
            if models.is_empty() {
                match &cfg["model"] {
                    Value::String(s) => models.push(s.clone()),
                    Value::Array(a) => {
                        models.extend(a.iter().filter_map(Value::as_str).map(str::to_string))
                    }
                    _ => {}
                }
            }
            if temperature.is_none() {
                temperature = cfg["temperature"].as_f64().map(|t| t as f32);
            }
            if max_tokens.is_none() {
                max_tokens = cfg["max_tokens"]
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok());
            }
            prompt_ref = Some(format!("{}@{}", v.name, v.commit));
            messages
        }
        (None, Some(m)) if !m.is_empty() => match &vars {
            None => m.clone(),
            Some(vars) => {
                let borrowed: HashMap<&str, String> =
                    vars.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
                let mut out = Vec::with_capacity(m.len());
                for (i, msg) in m.iter().enumerate() {
                    let content = PromptTemplate::new(msg.content.clone())
                        .and_then(|t| t.render(&borrowed))
                        .map_err(|e| {
                            ApiError::bad_request(format!(
                                "message {i}: {e} (use {{{{ and }}}} for literal braces)"
                            ))
                        })?;
                    let mut msg = msg.clone();
                    msg.content = content;
                    out.push(msg);
                }
                out
            }
        },
        _ => {
            return Err(ApiError::bad_request(
                "`messages` (non-empty) or `prompt` is required",
            ))
        }
    };
    if models.is_empty() {
        return Err(ApiError::bad_request(
            "`models` is required (a list of provider:model specs)",
        ));
    }
    let cfg = &state.playground;
    if models.len() > cfg.max_models {
        return Err(ApiError::bad_request(format!(
            "at most {} models per request",
            cfg.max_models
        )));
    }
    let mut resolved = Vec::with_capacity(models.len());
    for spec in &models {
        let spec = spec.trim().to_string();
        let m = cfg.resolve(&spec)?;
        resolved.push((spec, m));
    }
    let request = ChatRequest {
        messages: messages.clone(),
        tools: req.tools.clone(),
        temperature,
        max_tokens,
        tool_choice: None,
        response_format: req.response_format.clone(),
    };
    let inputs = json!({
        "messages": messages,
        "prompt": req.prompt,
        "variables": req.variables,
        "models": models,
        "temperature": t64(temperature),
        "max_tokens": max_tokens,
        "tools": req.tools,
        "response_format": req.response_format,
    });
    Ok(Plan {
        messages,
        request,
        models: resolved,
        inputs,
        prompt: prompt_ref,
    })
}

fn sse_event(name: &str, data: &Value) -> Event {
    Event::default().event(name).data(data.to_string())
}

type Events = Option<mpsc::UnboundedSender<Event>>;

#[allow(clippy::too_many_arguments)]
async fn run_one(
    index: usize,
    spec: String,
    model: Built,
    base: ChatRequest,
    tracer: Tracer,
    parent: RunRef,
    events: Events,
    timeout: Duration,
    pricing: Arc<Pricing>,
) -> ModelResult {
    let mut result = ModelResult {
        index,
        spec: spec.clone(),
        model: None,
        run_id: None,
        output: None,
        error: None,
        latency_ms: 0,
        ttft_ms: None,
        input_tokens: None,
        output_tokens: None,
        cost_usd: None,
        stop_reason: None,
        ignored: Vec::new(),
    };
    let mut req = base;
    let model = match model {
        Ok(m) => m,
        Err(e) => {
            let mut run = tracer.start(
                spec.clone(),
                RunKind::Llm,
                Some(&parent),
                json!({ "messages": req.messages }),
            );
            run.set_metadata("spec", json!(spec));
            result.run_id = Some(run.reference().run_id);
            run.fail(&e);
            result.error = Some(e);
            return result;
        }
    };
    let caps = model.capabilities();
    if req.temperature.is_some() && !caps.sampling {
        req.temperature = None;
        result.ignored.push("temperature");
    }
    if !req.tools.is_empty() && !caps.tools {
        req.tools.clear();
        result.ignored.push("tools");
    }
    let tool_names: Vec<&str> = req.tools.iter().map(|t| t.name.as_str()).collect();
    let mut run = tracer.start(
        spec.clone(),
        RunKind::Llm,
        Some(&parent),
        json!({ "messages": req.messages, "tools": tool_names }),
    );
    run.set_metadata("spec", json!(spec));
    run.set_metadata("playground", json!(true));
    run.set_metadata(
        "invocation_params",
        json!({
            "temperature": t64(req.temperature),
            "max_tokens": req.max_tokens,
            "tools": req.tools,
            "response_format": req.response_format,
        }),
    );
    if !result.ignored.is_empty() {
        run.set_metadata("ignored_params", json!(result.ignored));
    }
    result.run_id = Some(run.reference().run_id);
    let started = Instant::now();
    let mut ttft: Option<u64> = None;
    let mut stop_reason: Option<String> = None;
    let call = chat_streaming(model.as_ref(), req, |ev| {
        let first = matches!(
            ev,
            StreamEvent::TextDelta { .. } | StreamEvent::ToolCallDelta { .. }
        );
        if first && ttft.is_none() {
            ttft = Some(started.elapsed().as_millis() as u64);
        }
        if let StreamEvent::Done {
            stop_reason: Some(r),
            ..
        } = ev
        {
            stop_reason = Some(r.clone());
        }
        if let (Some(tx), StreamEvent::TextDelta { text }) = (&events, ev) {
            let _ = tx.send(sse_event("delta", &json!({ "index": index, "text": text })));
        }
    });
    let outcome = tokio::time::timeout(timeout, call).await;
    result.latency_ms = started.elapsed().as_millis() as u64;
    result.ttft_ms = ttft;
    if let Some(r) = &stop_reason {
        run.set_metadata("stop_reason", json!(r));
    }
    result.stop_reason = stop_reason;
    match outcome {
        Ok(Ok(resp)) => {
            let name = if resp.model.is_empty() {
                spec.split_once(':')
                    .map_or(spec.as_str(), |(_, m)| m)
                    .to_string()
            } else {
                resp.model.clone()
            };
            run.set_usage(resp.usage.input_tokens, resp.usage.output_tokens);
            run.set_metadata("model", json!(name));
            if let Some(t) = ttft {
                run.set_metadata("ttft_ms", json!(t));
            }
            result.input_tokens = Some(resp.usage.input_tokens);
            result.output_tokens = Some(resp.usage.output_tokens);
            result.cost_usd = pricing.cost(
                &name,
                u64::from(resp.usage.input_tokens),
                u64::from(resp.usage.output_tokens),
            );
            result.model = Some(name);
            run.finish(json!(resp.message));
            result.output = Some(resp.message);
        }
        Ok(Err(e)) => {
            let msg = e.to_string();
            run.fail(&msg);
            result.error = Some(msg);
        }
        Err(_) => {
            let msg = format!("timed out after {} s", timeout.as_secs());
            run.fail(&msg);
            result.error = Some(msg);
        }
    }
    result
}

/// Runs every model, records the trace in the store and returns
/// `{trace_id, project, prompt, results}`.
async fn execute(state: AppState, plan: Plan, events: Events) -> Value {
    let cfg = state.playground.clone();
    let sink = Arc::new(MemorySink::new());
    let tracer = Tracer::new().with_shared_sink(sink.clone() as Arc<dyn TraceSink>);
    let mut root = tracer.start("playground", RunKind::Chain, None, plan.inputs.clone());
    root.set_metadata("playground", json!(true));
    root.set_metadata("tags", json!(["playground"]));
    if let Some(p) = &plan.prompt {
        root.set_metadata("prompt", json!(p));
    }
    let parent = root.reference();
    if let Some(tx) = &events {
        let _ = tx.send(sse_event(
            "start",
            &json!({
                "trace_id": parent.trace_id,
                "project": cfg.project,
                "messages": plan.messages,
                "models": plan.models.iter().map(|(s, _)| s).collect::<Vec<_>>(),
            }),
        ));
    }
    let pricing = Arc::new(state.store.pricing());
    let calls = plan.models.into_iter().enumerate().map(|(i, (spec, m))| {
        run_one(
            i,
            spec,
            m,
            plan.request.clone(),
            tracer.clone(),
            parent.clone(),
            events.clone(),
            cfg.timeout,
            pricing.clone(),
        )
    });
    let results = futures_util::future::join_all(calls).await;
    for r in &results {
        if let Some(tx) = &events {
            let _ = tx.send(sse_event("result", &json!(r)));
        }
    }
    let errors = results.iter().filter(|r| r.error.is_some()).count();
    root.set_metadata("models", json!(results.len()));
    root.set_metadata("errors", json!(errors));
    let summary: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "spec": r.spec,
                "content": r.output.as_ref().map(|m| m.content.as_str()),
                "error": r.error,
            })
        })
        .collect();
    root.finish(json!({ "results": summary }));
    let runs: Vec<(String, Run)> = sink
        .runs()
        .into_iter()
        .map(|r| (cfg.project.clone(), r))
        .collect();
    let trace_error = match state.store.ingest(&runs).await {
        Ok(_) => {
            if let Some(engine) = &state.engine {
                engine.submit(runs);
            }
            None
        }
        Err(e) => Some(e.to_string()),
    };
    let done = json!({
        "trace_id": parent.trace_id,
        "project": cfg.project,
        "prompt": plan.prompt,
        "results": results,
        "trace_error": trace_error,
    });
    if let Some(tx) = &events {
        let _ = tx.send(sse_event("done", &done));
    }
    done
}

/// `POST /api/playground/run`.
pub(crate) async fn run(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let req: RunRequest = serde_json::from_value(parse_json(&body)?)
        .map_err(|e| ApiError::bad_request(format!("invalid playground request: {e}")))?;
    let stream = req.stream;
    let plan = plan(&state, req).await?;
    if stream {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(execute(state, plan, Some(tx)));
        let events = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|e| (Ok::<_, Infallible>(e), rx))
        });
        return Ok(Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response());
    }
    // Spawned so the calls finish and are traced even if the client leaves.
    let done = tokio::spawn(execute(state, plan, None))
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(done).into_response())
}

// ---------- GET /api/runs/{id}/playground ----------

fn role_of(s: &str) -> Role {
    match s.to_ascii_lowercase().as_str() {
        "system" | "developer" => Role::System,
        "assistant" | "model" | "ai" => Role::Assistant,
        "tool" | "function" => Role::Tool,
        _ => Role::User,
    }
}

/// Converts one stored message (Sinew form, or OpenTelemetry GenAI form
/// with `parts`) to a [`Message`].
fn to_message(v: &Value) -> Message {
    if let Ok(m) = serde_json::from_value::<Message>(v.clone()) {
        return m;
    }
    let role = role_of(v["role"].as_str().unwrap_or("user"));
    let content = match &v["content"] {
        Value::String(s) => s.clone(),
        Value::Null => v["parts"]
            .as_array()
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| {
                        p["content"]
                            .as_str()
                            .or_else(|| p["text"].as_str())
                            .map(str::to_string)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default(),
        other => other.to_string(),
    };
    let mut m = match role {
        Role::System => Message::system(content),
        Role::Assistant => Message::assistant(content),
        _ => Message::user(content),
    };
    if role == Role::Tool {
        m.role = Role::Tool;
        m.tool_call_id = v["tool_call_id"].as_str().map(str::to_string);
    }
    m
}

fn is_message(v: &Value) -> bool {
    v.get("role").is_some_and(Value::is_string)
}

/// The chat messages of an LLM run's inputs.
fn extract_messages(inputs: &Value) -> Vec<Message> {
    let list = match inputs {
        Value::Array(a) if a.iter().all(is_message) => Some(a),
        Value::Object(o) => o
            .get("messages")
            .and_then(Value::as_array)
            .filter(|a| a.iter().all(is_message)),
        _ => None,
    };
    if let Some(list) = list {
        return list.iter().map(to_message).collect();
    }
    match inputs {
        Value::Null => Vec::new(),
        Value::String(s) => vec![Message::user(s.clone())],
        Value::Object(o) if o.get("prompt").is_some_and(Value::is_string) => {
            vec![Message::user(o["prompt"].as_str().unwrap_or_default())]
        }
        _ if is_message(inputs) => vec![to_message(inputs)],
        other => vec![Message::user(
            serde_json::to_string_pretty(other).unwrap_or_default(),
        )],
    }
}

fn num(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// `GET /api/runs/{id}/playground`.
pub(crate) async fn prefill(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    let stored = state
        .store
        .run(&id)
        .await?
        .ok_or_else(|| ApiError::not_found("run"))?;
    let run = &stored.run;
    if run.kind != RunKind::Llm {
        return Err(ApiError::bad_request(format!(
            "run {id} is not an LLM run; only LLM calls can be opened in the playground"
        )));
    }
    let meta = &run.metadata;
    let params = meta.get("invocation_params").filter(|p| p.is_object());
    let attrs = meta.get("attributes").filter(|a| a.is_object());
    let param = |key: &str, attr: &str| {
        num(params.and_then(|p| p.get(key)))
            .or_else(|| num(attrs.and_then(|a| a.get(attr))))
            .or_else(|| num(meta.get(attr)))
    };
    let temperature = param("temperature", "gen_ai.request.temperature");
    let max_tokens = param("max_tokens", "gen_ai.request.max_tokens").map(|n| n as u64);
    let model = stored.model.clone().or_else(|| run_model(run));
    let spec = meta
        .get("spec")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            let provider = attrs
                .and_then(|a| {
                    a.get("gen_ai.provider.name")
                        .or_else(|| a.get("gen_ai.system"))
                })
                .and_then(Value::as_str)
                .and_then(canonical_provider)
                .or_else(|| {
                    let m = model.as_deref()?;
                    let p = state.store.pricing().lookup(m)?.provider.clone()?;
                    canonical_provider(&p)
                })?;
            Some(format!("{provider}:{}", model.as_deref()?))
        });
    let tools = params
        .and_then(|p| p.get("tools"))
        .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()))
        .cloned();
    let tool_names: Vec<String> = run
        .inputs
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let response_format = params
        .and_then(|p| p.get("response_format"))
        .filter(|v| !v.is_null())
        .cloned();
    let allowed = spec.as_deref().map(|s| {
        state.playground.models.contains_key(s)
            || s.split_once(':')
                .is_some_and(|(p, _)| state.playground.allows(p))
    });
    Ok(Json(json!({
        "run_id": run.id,
        "trace_id": run.trace_id,
        "project": stored.project,
        "name": run.name,
        "model": model,
        "spec": spec,
        "spec_allowed": allowed,
        "messages": extract_messages(&run.inputs),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": tools,
        "tool_names": tool_names,
        "response_format": response_format,
        "prompt": meta.get("prompt"),
        "output": run.outputs,
        "error": run.error,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_from_sinew_and_otel_inputs() {
        let m = extract_messages(&json!({
            "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
            "tools": []
        }));
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].role, Role::System);
        let m = extract_messages(&json!([
            {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
        ]));
        assert_eq!(m[0].content, "hello");
        let m = extract_messages(&json!("plain"));
        assert_eq!(m[0].content, "plain");
        let m = extract_messages(&json!({"q": 1}));
        assert!(m[0].content.contains("\"q\""));
    }

    #[test]
    fn spec_policy() {
        let c = PlaygroundConfig::default();
        assert!(matches!(c.resolve("ollama:qwen3:4b"), Ok(Ok(_))));
        assert_eq!(
            c.resolve("openai:gpt-4.1").err().unwrap().0,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            c.resolve("http://10.0.0.1:8000/v1|m").err().unwrap().0,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            c.resolve("nope:m").err().unwrap().0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            c.resolve("ollama").err().unwrap().0,
            StatusCode::BAD_REQUEST
        );
        let none = PlaygroundConfig::default().allow_providers(Vec::<String>::new());
        assert_eq!(
            none.resolve("ollama:x").err().unwrap().0,
            StatusCode::FORBIDDEN
        );
        let all = PlaygroundConfig::default().allow_providers([ALL_PROVIDERS]);
        assert!(all.allows("openai") && all.allows("anthropic"));
    }

    #[test]
    fn local_base_normalises() {
        assert_eq!(
            local_base("SINEW_TEST_UNSET_HOST_VAR", "127.0.0.1:11434"),
            "http://127.0.0.1:11434"
        );
        assert_eq!(
            local_base("SINEW_TEST_UNSET_HOST_VAR", "http://box:8000/v1/"),
            "http://box:8000"
        );
    }
}
