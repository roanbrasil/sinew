//! The axum application: configuration, routes, middleware and serving.

use std::{
    collections::BTreeMap,
    future::Future,
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, MatchedPath, Path, Query, Request, State},
    http::{header, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    serve::ListenerExt,
    Json, Router,
};
use futures_util::StreamExt;
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use sinew_core::ChatModel;
use sinew_graph::{Checkpoint, Checkpointer};
use sinew_trace::{
    metrics::{MetricType, TextEncoder, CONTENT_TYPE},
    MetricsSink, RunKind, Tracer,
};
use tokio::{net::TcpListener, sync::oneshot};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use crate::{
    api::{
        merge, now_ms, AssistantInfo, OnDisconnect, Run, RunCreate, RunStatus, StateUpdate, Thread,
        ThreadCreate, ThreadQuery, ThreadState, ThreadStatus,
    },
    assistant::{Assistant, ErasedGraph, Modes, Outcome, StartSpec, Step},
    openai::{self, ChatBackend},
    runs::{ActiveGuard, Final, Refused, RunEntry, Runs},
    sse::SseBody,
    store::{MemoryStore, StoreError, ThreadStore},
};

/// Which webhook URLs runs may ask the server to call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum WebhookPolicy {
    /// Runs with a `webhook` are rejected (default: avoids SSRF).
    #[default]
    Disabled,
    /// Any `http(s)` URL.
    Any,
    /// URLs starting with one of these prefixes.
    Prefixes(Vec<String>),
}

impl WebhookPolicy {
    fn allows(&self, url: &str) -> bool {
        let http = url.starts_with("http://") || url.starts_with("https://");
        match self {
            WebhookPolicy::Disabled => false,
            WebhookPolicy::Any => http,
            WebhookPolicy::Prefixes(p) => http && p.iter().any(|p| url.starts_with(p.as_str())),
        }
    }
}

/// Cross-origin settings.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// `None` allows any origin.
    origins: Option<Vec<String>>,
    max_age: Duration,
}

impl CorsConfig {
    /// Any origin, method and header.
    pub fn permissive() -> Self {
        Self {
            origins: None,
            max_age: Duration::from_secs(3600),
        }
    }

    /// Only these origins (e.g. `https://app.example.com`).
    pub fn origins<I, T>(origins: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self {
            origins: Some(origins.into_iter().map(Into::into).collect()),
            max_age: Duration::from_secs(3600),
        }
    }

    fn layer(&self) -> CorsLayer {
        let base = CorsLayer::new()
            .allow_methods(Any)
            .allow_headers(Any)
            .max_age(self.max_age);
        match &self.origins {
            None => base.allow_origin(Any),
            Some(list) => base.allow_origin(AllowOrigin::list(
                list.iter().filter_map(|o| HeaderValue::from_str(o).ok()),
            )),
        }
    }
}

/// Server settings. Defaults: no auth, no CORS, no tracing, webhooks off,
/// 15 s SSE keep-alive, 10 s shutdown grace, 10 000 finished runs kept,
/// no run metrics, `/metrics` without auth.
#[derive(Clone)]
pub struct ServeConfig {
    pub api_keys: Vec<String>,
    pub cors: Option<CorsConfig>,
    pub tracer: Option<Tracer>,
    pub webhooks: WebhookPolicy,
    pub keep_alive: Option<Duration>,
    pub shutdown_grace: Duration,
    pub max_finished_runs: usize,
    pub body_limit: usize,
    /// Run metrics rendered by `GET /metrics`; registered on the tracer
    /// (one is created if `tracer` is `None`) when the server is built.
    pub metrics: Option<Arc<MetricsSink>>,
    /// Require the API keys on `GET /metrics` too.
    pub metrics_auth: bool,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            api_keys: Vec::new(),
            cors: None,
            tracer: None,
            webhooks: WebhookPolicy::Disabled,
            keep_alive: Some(Duration::from_secs(15)),
            shutdown_grace: Duration::from_secs(10),
            max_finished_runs: 10_000,
            body_limit: 16 * 1024 * 1024,
            metrics: None,
            metrics_auth: false,
        }
    }
}

/// Builds a [`Server`].
#[derive(Default)]
pub struct ServerBuilder {
    store: Option<Arc<dyn ThreadStore>>,
    assistants: Vec<Assistant>,
    models: Vec<(String, Arc<dyn ChatModel>)>,
    config: ServeConfig,
}

impl ServerBuilder {
    /// Registers a graph. Chat-capable assistants (see
    /// [`Assistant::agent`]) also appear in `/v1/models`.
    pub fn assistant(mut self, assistant: impl Into<Assistant>) -> Self {
        self.assistants.push(assistant.into());
        self
    }

    /// Exposes a plain chat model on the OpenAI façade under `name`.
    pub fn chat_model(mut self, name: impl Into<String>, model: Arc<dyn ChatModel>) -> Self {
        self.models.push((name.into(), model));
        self
    }

    /// Where threads and checkpoints live (default: [`MemoryStore`]).
    pub fn store(mut self, store: impl ThreadStore) -> Self {
        self.store = Some(Arc::new(store));
        self
    }

    pub fn shared_store(mut self, store: Arc<dyn ThreadStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Requires `Authorization: Bearer <key>` (or `x-api-key: <key>`) on
    /// everything except `/ok`, `/health`, `/ready` and (unless
    /// [`ServerBuilder::metrics_auth`] is set) `/metrics`. Can be called
    /// several times to accept several keys.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        if !key.is_empty() {
            self.config.api_keys.push(key);
        }
        self
    }

    pub fn cors(mut self, cors: CorsConfig) -> Self {
        self.config.cors = Some(cors);
        self
    }

    /// Records one `chain` run per HTTP request and attaches the tracer to
    /// every registered graph (graph, node, model and tool runs).
    pub fn tracer(mut self, tracer: Tracer) -> Self {
        self.config.tracer = Some(tracer);
        self
    }

    pub fn webhooks(mut self, policy: WebhookPolicy) -> Self {
        self.config.webhooks = policy;
        self
    }

    /// SSE keep-alive comment interval (`None` disables it).
    pub fn keep_alive(mut self, every: Option<Duration>) -> Self {
        self.config.keep_alive = every;
        self
    }

    /// How long running runs may continue after shutdown starts before
    /// they are cancelled.
    pub fn shutdown_grace(mut self, grace: Duration) -> Self {
        self.config.shutdown_grace = grace;
        self
    }

    /// Aggregates graph, node, model, tool and HTTP request runs into
    /// Prometheus metrics served at `GET /metrics`. The sink is added to the
    /// tracer set with [`ServerBuilder::tracer`], or to a new tracer if none
    /// is set (which also records one `chain` run per HTTP request).
    ///
    /// `GET /metrics` is always available and also reports
    /// `sinew_serve_queued_runs`, `sinew_serve_running_runs` and
    /// `sinew_serve_active_streams`. It does not require an API key unless
    /// [`ServerBuilder::metrics_auth`] is set.
    ///
    /// ```
    /// use sinew_serve::Server;
    /// use sinew_trace::MetricsSink;
    ///
    /// let server = Server::builder()
    ///     .metrics(MetricsSink::new().project("support-bot"))
    ///     .api_key("secret")
    ///     .metrics_auth(true)
    ///     .build();
    /// # let _ = server;
    /// ```
    pub fn metrics(self, sink: MetricsSink) -> Self {
        self.shared_metrics(Arc::new(sink))
    }

    /// Like [`ServerBuilder::metrics`], with a sink shared with other code.
    pub fn shared_metrics(mut self, sink: Arc<MetricsSink>) -> Self {
        self.config.metrics = Some(sink);
        self
    }

    /// Requires the API keys on `GET /metrics` (default: open, for an
    /// internal Prometheus scraper). No effect without API keys.
    pub fn metrics_auth(mut self, required: bool) -> Self {
        self.config.metrics_auth = required;
        self
    }

    /// Replaces the whole configuration.
    pub fn config(mut self, config: ServeConfig) -> Self {
        self.config = config;
        self
    }

    pub fn build(mut self) -> Server {
        if let Some(sink) = self.config.metrics.clone() {
            let tracer = self.config.tracer.take().unwrap_or_default();
            self.config.tracer = Some(tracer.with_shared_sink(sink));
        }
        let store = self.store.unwrap_or_else(|| Arc::new(MemoryStore::new()));
        let cp = store.checkpointer();
        let mut assistants = BTreeMap::new();
        let mut chat = BTreeMap::new();
        for a in self.assistants {
            let (info, graph) = a.bind(cp.clone(), self.config.tracer.clone());
            if info.chat {
                chat.insert(info.assistant_id.clone(), ChatBackend::Graph(graph.clone()));
            }
            assistants.insert(info.assistant_id.clone(), (info, graph));
        }
        for (name, model) in self.models {
            chat.insert(name, ChatBackend::Model(model));
        }
        Server {
            app: App(Arc::new(AppInner {
                runs: Runs::new(self.config.max_finished_runs),
                assistants,
                chat,
                store,
                cp,
                http: reqwest::Client::builder()
                    .timeout(Duration::from_secs(10))
                    .build()
                    .unwrap_or_default(),
                config: self.config,
                streams: Arc::new(AtomicUsize::new(0)),
            })),
        }
    }
}

pub(crate) struct AppInner {
    pub assistants: BTreeMap<String, (AssistantInfo, Arc<dyn ErasedGraph>)>,
    pub chat: BTreeMap<String, ChatBackend>,
    pub store: Arc<dyn ThreadStore>,
    pub cp: Arc<dyn Checkpointer>,
    pub runs: Arc<Runs>,
    pub config: ServeConfig,
    http: reqwest::Client,
    /// Open SSE responses.
    streams: Arc<AtomicUsize>,
}

#[derive(Clone)]
pub(crate) struct App(Arc<AppInner>);

impl std::ops::Deref for App {
    type Target = AppInner;
    fn deref(&self) -> &AppInner {
        &self.0
    }
}

/// A configured server. Cheap to clone.
#[derive(Clone)]
pub struct Server {
    app: App,
}

/// A server running in the background (tests, examples, benchmarks).
pub struct RunningServer {
    pub addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl RunningServer {
    /// `http://127.0.0.1:port`.
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Graceful shutdown: stops accepting, drains runs, waits for the server.
    pub async fn shutdown(mut self) -> std::io::Result<()> {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        (&mut self.task)
            .await
            .unwrap_or_else(|e| Err(std::io::Error::other(e)))
    }
}

impl Server {
    pub fn builder() -> ServerBuilder {
        ServerBuilder::default()
    }

    /// Registered assistants.
    pub fn assistants(&self) -> Vec<AssistantInfo> {
        self.app
            .assistants
            .values()
            .map(|(i, _)| i.clone())
            .collect()
    }

    /// The axum router, to nest in a bigger app or serve yourself.
    pub fn router(&self) -> Router {
        router(self.app.clone())
    }

    /// Serves on `listener` until `shutdown` resolves, then stops accepting,
    /// lets runs finish for the configured grace period (cancelling the
    /// rest) and returns once every connection is closed.
    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        let runs = self.app.runs.clone();
        let grace = self.app.config.shutdown_grace;
        let (tx, rx) = oneshot::channel();
        let signal = async move {
            shutdown.await;
            let _ = tx.send(tokio::spawn(async move { runs.shutdown(grace).await }));
        };
        // Small writes (tokens) go out immediately.
        let listener = listener.tap_io(|tcp| {
            let _ = tcp.set_nodelay(true);
        });
        axum::serve(listener, self.router())
            .with_graceful_shutdown(signal)
            .await?;
        if let Ok(drain) = rx.await {
            let _ = drain.await;
        }
        Ok(())
    }

    /// Binds `addr` and serves until Ctrl-C.
    pub async fn serve_addr(self, addr: SocketAddr) -> std::io::Result<()> {
        let listener = TcpListener::bind(addr).await?;
        eprintln!(
            "sinew-serve listening on http://{}{}",
            listener.local_addr()?,
            if self.app.config.api_keys.is_empty() {
                ""
            } else {
                "  (bearer auth on)"
            }
        );
        self.serve(listener, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
    }

    /// Binds `addr` (e.g. `127.0.0.1:0`) and serves in a background task.
    pub async fn spawn(self, addr: &str) -> std::io::Result<RunningServer> {
        let listener = TcpListener::bind(addr).await?;
        let addr = listener.local_addr()?;
        let (tx, rx) = oneshot::channel::<()>();
        let task = tokio::spawn(self.serve(listener, async move {
            let _ = rx.await;
        }));
        Ok(RunningServer {
            addr,
            shutdown: Some(tx),
            task,
        })
    }
}

/// A multi-threaded runtime set up for serving: named worker threads, all
/// drivers on. `workers = None` uses one thread per core.
pub fn runtime(workers: Option<usize>) -> std::io::Result<tokio::runtime::Runtime> {
    let mut b = tokio::runtime::Builder::new_multi_thread();
    b.enable_all().thread_name("sinew-serve");
    if let Some(n) = workers {
        b.worker_threads(n.max(1));
    }
    b.build()
}

// ---------------------------------------------------------------- errors

/// JSON error response: `{"error": "..."}` with a status code.
#[derive(Debug)]
pub(crate) struct ApiError(StatusCode, String);

impl ApiError {
    fn new(status: StatusCode, msg: impl Into<String>) -> Self {
        Self(status, msg.into())
    }
    fn bad_request(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, msg)
    }
    fn not_found(what: impl std::fmt::Display) -> Self {
        Self::new(StatusCode::NOT_FOUND, format!("{what} not found"))
    }
    fn conflict(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, msg)
    }
}

impl From<StoreError> for ApiError {
    fn from(e: StoreError) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

type ApiResult<T> = Result<Json<T>, ApiError>;

/// Parses a JSON body; an empty body means `T::default()`.
fn body_or_default<T: DeserializeOwned + Default>(b: &Bytes) -> Result<T, ApiError> {
    if b.iter().all(u8::is_ascii_whitespace) {
        return Ok(T::default());
    }
    serde_json::from_slice(b).map_err(|e| ApiError::bad_request(format!("invalid body: {e}")))
}

fn body<T: DeserializeOwned>(b: &Bytes) -> Result<T, ApiError> {
    serde_json::from_slice(b).map_err(|e| ApiError::bad_request(format!("invalid body: {e}")))
}

// ---------------------------------------------------------------- router

fn router(app: App) -> Router {
    let protected = Router::new()
        .route("/info", get(info))
        .route("/assistants", get(list_assistants))
        .route("/assistants/search", post(search_assistants))
        .route("/assistants/{id}", get(get_assistant))
        .route("/threads", get(list_threads).post(create_thread))
        .route("/threads/search", post(search_threads))
        .route(
            "/threads/{tid}",
            get(get_thread).delete(delete_thread).patch(patch_thread),
        )
        .route("/threads/{tid}/state", get(get_state).post(update_state))
        .route("/threads/{tid}/history", get(history))
        .route("/threads/{tid}/runs", get(list_runs).post(thread_run_bg))
        .route("/threads/{tid}/runs/wait", post(thread_run_wait))
        .route("/threads/{tid}/runs/stream", post(thread_run_stream))
        .route("/threads/{tid}/runs/{rid}", get(thread_get_run))
        .route("/threads/{tid}/runs/{rid}/cancel", post(thread_cancel_run))
        .route("/threads/{tid}/runs/{rid}/join", get(thread_join_run))
        .route("/threads/{tid}/runs/{rid}/stream", get(thread_join_stream))
        .route("/runs", post(run_bg))
        .route("/runs/wait", post(run_wait))
        .route("/runs/stream", post(run_stream))
        .route("/runs/{rid}", get(get_run))
        .route("/runs/{rid}/cancel", post(cancel_run))
        .route("/runs/{rid}/join", get(join_run))
        .route("/runs/{rid}/stream", get(join_stream))
        .route("/v1/models", get(openai::list_models))
        .route("/v1/chat/completions", post(openai::chat_completions));
    let protected = if app.config.metrics_auth {
        protected.route("/metrics", get(metrics))
    } else {
        protected
    }
    .route_layer(middleware::from_fn_with_state(app.clone(), auth));
    let public = Router::new()
        .route("/ok", get(health))
        .route("/health", get(health))
        .route("/ready", get(ready));
    let public = if app.config.metrics_auth {
        public
    } else {
        public.route("/metrics", get(metrics))
    };
    let mut r = public
        .merge(protected)
        .layer(DefaultBodyLimit::max(app.config.body_limit));
    if app.config.tracer.is_some() {
        r = r.layer(middleware::from_fn_with_state(app.clone(), trace_requests));
    }
    if let Some(cors) = &app.config.cors {
        r = r.layer(cors.layer());
    }
    r.with_state(app)
}

/// Compares in time independent of where the first difference is.
fn same_secret(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn auth(State(app): State<App>, req: Request, next: Next) -> Response {
    if app.config.api_keys.is_empty() || req.method() == Method::OPTIONS {
        return next.run(req).await;
    }
    let h = req.headers();
    let given = h
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| h.get("x-api-key").and_then(|v| v.to_str().ok()))
        .unwrap_or("");
    let mut ok = false;
    for k in &app.config.api_keys {
        ok |= same_secret(given.as_bytes(), k.as_bytes());
    }
    if ok {
        next.run(req).await
    } else {
        let mut resp =
            ApiError::new(StatusCode::UNAUTHORIZED, "missing or invalid API key").into_response();
        resp.headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        resp
    }
}

async fn trace_requests(State(app): State<App>, req: Request, next: Next) -> Response {
    let Some(tracer) = app.config.tracer.as_ref() else {
        return next.run(req).await;
    };
    let path = req.uri().path().to_string();
    if matches!(path.as_str(), "/ok" | "/health" | "/ready" | "/metrics") {
        return next.run(req).await;
    }
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| path.clone());
    let method = req.method().to_string();
    let mut run = tracer.start(
        format!("{method} {route}"),
        RunKind::Chain,
        None,
        json!({"method": method, "path": path}),
    );
    let t = Instant::now();
    let resp = next.run(req).await;
    let status = resp.status().as_u16();
    run.set_metadata("http.status", json!(status));
    run.set_metadata("latency_us", json!(t.elapsed().as_micros() as u64));
    if resp.status().is_server_error() {
        run.fail(format!("HTTP {status}"));
    } else {
        run.finish(json!({"status": status}));
    }
    resp
}

// ---------------------------------------------------------------- system

async fn health() -> Json<Value> {
    Json(json!({"ok": true}))
}

async fn ready(State(app): State<App>) -> Response {
    if app.runs.is_closing() {
        return ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "shutting down").into_response();
    }
    match app.store.ping().await {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => ApiError::new(StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response(),
    }
}

/// `GET /metrics`: Prometheus text exposition of the run metrics (when a
/// [`MetricsSink`] is configured) and the server gauges.
async fn metrics(State(app): State<App>) -> Response {
    let mut enc = TextEncoder::new();
    if let Some(sink) = &app.config.metrics {
        sink.encode(&mut enc);
    }
    let (queued, running) = app.runs.live_counts();
    for (name, help, value) in [
        (
            "sinew_serve_queued_runs",
            "Runs waiting for an earlier run on the same thread.",
            queued,
        ),
        (
            "sinew_serve_running_runs",
            "Runs currently executing.",
            running,
        ),
        (
            "sinew_serve_active_streams",
            "Open server-sent event streams.",
            app.streams.load(Ordering::Relaxed),
        ),
        (
            "sinew_serve_assistants",
            "Registered assistants.",
            app.assistants.len(),
        ),
    ] {
        enc.header(name, help, MetricType::Gauge);
        enc.sample(name, &[], value as f64);
    }
    enc.header(
        "sinew_serve_build_info",
        "Constant 1, labelled with the sinew-serve version.",
        MetricType::Gauge,
    );
    enc.sample(
        "sinew_serve_build_info",
        &[("version", env!("CARGO_PKG_VERSION"))],
        1.0,
    );
    ([(header::CONTENT_TYPE, CONTENT_TYPE)], enc.finish()).into_response()
}

async fn info(State(app): State<App>) -> Json<Value> {
    Json(json!({
        "name": "sinew-serve",
        "version": env!("CARGO_PKG_VERSION"),
        "assistants": app.assistants.keys().collect::<Vec<_>>(),
        "chat_models": app.chat.keys().collect::<Vec<_>>(),
    }))
}

// ---------------------------------------------------------------- assistants

#[derive(Deserialize, Default)]
struct Page {
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    offset: Option<usize>,
}

fn page_assistants(app: &App, p: &Page) -> Vec<AssistantInfo> {
    app.assistants
        .values()
        .skip(p.offset.unwrap_or(0))
        .take(p.limit.unwrap_or(usize::MAX))
        .map(|(i, _)| i.clone())
        .collect()
}

async fn list_assistants(
    State(app): State<App>,
    Query(p): Query<Page>,
) -> Json<Vec<AssistantInfo>> {
    Json(page_assistants(&app, &p))
}

async fn search_assistants(State(app): State<App>, b: Bytes) -> ApiResult<Vec<AssistantInfo>> {
    let p: Page = body_or_default(&b)?;
    Ok(Json(page_assistants(&app, &p)))
}

async fn get_assistant(State(app): State<App>, Path(id): Path<String>) -> ApiResult<AssistantInfo> {
    app.assistants
        .get(&id)
        .map(|(i, _)| Json(i.clone()))
        .ok_or_else(|| ApiError::not_found(format_args!("assistant `{id}`")))
}

impl App {
    fn graph(&self, id: &str) -> Result<Arc<dyn ErasedGraph>, ApiError> {
        self.assistants
            .get(id)
            .map(|(_, g)| g.clone())
            .ok_or_else(|| ApiError::not_found(format_args!("assistant `{id}`")))
    }

    async fn thread(&self, tid: &str) -> Result<Thread, ApiError> {
        let mut t = self
            .store
            .get_thread(tid)
            .await?
            .ok_or_else(|| ApiError::not_found(format_args!("thread `{tid}`")))?;
        if self.runs.thread_busy(tid) {
            t.status = ThreadStatus::Busy;
        }
        Ok(t)
    }
}

// ---------------------------------------------------------------- threads

async fn create_thread(State(app): State<App>, b: Bytes) -> Result<Response, ApiError> {
    let req: ThreadCreate = body_or_default(&b)?;
    if let Some(a) = &req.assistant_id {
        app.graph(a)?;
    }
    let now = now_ms();
    let t = Thread {
        thread_id: req
            .thread_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        assistant_id: req.assistant_id,
        status: ThreadStatus::Idle,
        metadata: req.metadata.unwrap_or_else(|| json!({})),
        created_at: now,
        updated_at: now,
    };
    if app.store.create_thread(&t).await? {
        return Ok((StatusCode::OK, Json(t)).into_response());
    }
    if req.if_exists.as_deref() == Some("do_nothing") {
        return Ok(Json(app.thread(&t.thread_id).await?).into_response());
    }
    Err(ApiError::conflict(format!(
        "thread `{}` already exists",
        t.thread_id
    )))
}

async fn list_threads(
    State(app): State<App>,
    Query(q): Query<ThreadQuery>,
) -> ApiResult<Vec<Thread>> {
    threads_page(&app, &q).await
}

async fn search_threads(State(app): State<App>, b: Bytes) -> ApiResult<Vec<Thread>> {
    let q: ThreadQuery = body_or_default(&b)?;
    threads_page(&app, &q).await
}

async fn threads_page(app: &App, q: &ThreadQuery) -> ApiResult<Vec<Thread>> {
    let Some(status) = q.status else {
        let mut list = app.store.list_threads(q).await?;
        for t in &mut list {
            if app.runs.thread_busy(&t.thread_id) {
                t.status = ThreadStatus::Busy;
            }
        }
        return Ok(Json(list));
    };
    // `busy` is not stored: it comes from the runs in flight. Filter on the
    // effective status here (the store would never match `busy`, and would
    // return busy threads for their stored status), then page.
    let (limit, offset) = crate::store::page(q);
    let wide = ThreadQuery {
        status: (status != ThreadStatus::Busy).then_some(status),
        limit: Some(crate::store::MAX_LIMIT),
        offset: None,
        ..q.clone()
    };
    let list = app
        .store
        .list_threads(&wide)
        .await?
        .into_iter()
        .map(|mut t| {
            if app.runs.thread_busy(&t.thread_id) {
                t.status = ThreadStatus::Busy;
            }
            t
        })
        .filter(|t| t.status == status)
        .skip(offset)
        .take(limit)
        .collect();
    Ok(Json(list))
}

async fn get_thread(State(app): State<App>, Path(tid): Path<String>) -> ApiResult<Thread> {
    Ok(Json(app.thread(&tid).await?))
}

#[derive(Deserialize)]
struct ThreadPatch {
    metadata: Value,
}

async fn patch_thread(
    State(app): State<App>,
    Path(tid): Path<String>,
    b: Bytes,
) -> ApiResult<Thread> {
    let p: ThreadPatch = body(&b)?;
    let mut t = app.thread(&tid).await?;
    t.metadata = merge(t.metadata, p.metadata);
    t.updated_at = now_ms();
    let mut stored = t.clone();
    if stored.status == ThreadStatus::Busy {
        if let Some(cur) = app.store.get_thread(&tid).await? {
            stored.status = cur.status;
        }
    }
    app.store.update_thread(&stored).await?;
    Ok(Json(t))
}

async fn delete_thread(
    State(app): State<App>,
    Path(tid): Path<String>,
) -> Result<StatusCode, ApiError> {
    app.runs.drop_thread(&tid);
    if app.store.delete_thread(&tid).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::not_found(format_args!("thread `{tid}`")))
    }
}

fn state_of(tid: &str, cp: Option<Checkpoint>) -> ThreadState {
    match cp {
        Some(cp) => ThreadState {
            thread_id: tid.to_string(),
            next: cp.next_nodes(),
            step: cp.step,
            values: cp.state,
        },
        None => ThreadState {
            thread_id: tid.to_string(),
            values: Value::Null,
            next: Vec::new(),
            step: 0,
        },
    }
}

async fn get_state(State(app): State<App>, Path(tid): Path<String>) -> ApiResult<ThreadState> {
    app.thread(&tid).await?;
    let cp = app
        .cp
        .latest(&tid)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(state_of(&tid, cp)))
}

async fn update_state(
    State(app): State<App>,
    Path(tid): Path<String>,
    b: Bytes,
) -> ApiResult<ThreadState> {
    let upd: StateUpdate = body(&b)?;
    let thread = app.thread(&tid).await?;
    if thread.status == ThreadStatus::Busy {
        return Err(ApiError::conflict("thread has a run in flight"));
    }
    let latest = app
        .cp
        .latest(&tid)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let mut cp = match latest {
        Some(mut cp) => {
            cp.state = merge(cp.state, upd.values);
            cp
        }
        // Seeding a fresh thread: a finished checkpoint, so the next run
        // with input merges over it.
        None => Checkpoint::new(tid.clone(), 0, Vec::new(), upd.values),
    };
    if let Some(a) = &thread.assistant_id {
        app.graph(a)?
            .validate(&cp.state)
            .map_err(ApiError::bad_request)?;
    }
    cp.thread_id.clone_from(&tid);
    app.cp
        .put(cp.clone())
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(state_of(&tid, Some(cp))))
}

#[derive(Deserialize)]
struct HistoryQuery {
    #[serde(default)]
    limit: Option<usize>,
}

async fn history(
    State(app): State<App>,
    Path(tid): Path<String>,
    Query(q): Query<HistoryQuery>,
) -> ApiResult<Vec<ThreadState>> {
    app.thread(&tid).await?;
    let all = app
        .cp
        .history(&tid)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(
        all.into_iter()
            .rev()
            .take(q.limit.unwrap_or(10))
            .map(|cp| state_of(&tid, Some(cp)))
            .collect(),
    ))
}

// ---------------------------------------------------------------- runs

async fn start_run(
    app: &App,
    thread_id: Option<String>,
    req: RunCreate,
    subscribe: bool,
) -> Result<
    (
        Arc<RunEntry>,
        Option<tokio::sync::mpsc::UnboundedReceiver<Bytes>>,
    ),
    ApiError,
> {
    if app.runs.is_closing() {
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "shutting down",
        ));
    }
    let graph = app.graph(&req.assistant_id)?;
    if let Some(url) = &req.webhook {
        if !app.config.webhooks.allows(url) {
            return Err(ApiError::bad_request(format!(
                "webhook `{url}` is not allowed"
            )));
        }
    }
    if let Some(tid) = &thread_id {
        let existing = app.store.get_thread(tid).await?;
        let mut thread = match existing {
            Some(t) => t,
            None if req.if_not_exists.as_deref() == Some("reject") => {
                return Err(ApiError::not_found(format_args!("thread `{tid}`")))
            }
            None => {
                let now = now_ms();
                let t = Thread {
                    thread_id: tid.clone(),
                    assistant_id: Some(req.assistant_id.clone()),
                    status: ThreadStatus::Idle,
                    metadata: json!({}),
                    created_at: now,
                    updated_at: now,
                };
                if !app.store.create_thread(&t).await? {
                    // Lost a creation race; use the winner's record.
                    app.store.get_thread(tid).await?.unwrap_or(t)
                } else {
                    t
                }
            }
        };
        match &thread.assistant_id {
            Some(a) if a != &req.assistant_id => {
                return Err(ApiError::conflict(format!(
                    "thread `{tid}` belongs to assistant `{a}`"
                )))
            }
            Some(_) => {}
            None => {
                thread.assistant_id = Some(req.assistant_id.clone());
                app.store.update_thread(&thread).await?;
            }
        }
    }
    let now = now_ms();
    let entry = RunEntry::new(Run {
        run_id: uuid::Uuid::new_v4().to_string(),
        thread_id,
        assistant_id: req.assistant_id.clone(),
        status: RunStatus::Pending,
        created_at: now,
        updated_at: now,
        multitask_strategy: req.multitask_strategy,
        metadata: req.metadata.clone().unwrap_or_else(|| json!({})),
        output: None,
        error: None,
        interrupted_before: Vec::new(),
    });
    let rx = subscribe.then(|| entry.subscribe());
    let guard = app
        .runs
        .admit(&entry, req.multitask_strategy)
        .map_err(|r| match r {
            Refused::Busy => ApiError::conflict(
                "thread has a run in flight (multitask_strategy=reject); use enqueue or interrupt",
            ),
            Refused::Closing => ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "shutting down"),
        })?;
    tokio::spawn(drive(app.clone(), entry.clone(), graph, req, guard));
    Ok((entry, rx))
}

async fn drive(
    app: App,
    entry: Arc<RunEntry>,
    graph: Arc<dyn ErasedGraph>,
    req: RunCreate,
    _active: ActiveGuard,
) {
    let webhook = req.webhook.clone();
    // Hold the thread's gate until the outcome is recorded.
    let (fin, _gate) = match app.runs.acquire(&entry).await {
        None => (Final::cancelled(None), None),
        Some(gate) => {
            entry.set_running();
            (execute(&app, &entry, graph, req).await, gate)
        }
    };
    let run = entry.finish(fin);
    if let Some(tid) = &entry.thread_id {
        if let Ok(Some(mut t)) = app.store.get_thread(tid).await {
            t.status = match run.status {
                RunStatus::Interrupted => ThreadStatus::Interrupted,
                RunStatus::Error => ThreadStatus::Error,
                _ => ThreadStatus::Idle,
            };
            t.updated_at = now_ms();
            let _ = app.store.update_thread(&t).await;
        }
    }
    if let Some(url) = webhook {
        let http = app.http.clone();
        tokio::spawn(async move {
            let _ = http.post(url).json(&run).send().await;
        });
    }
    app.runs.release(&entry);
}

async fn execute(
    app: &App,
    entry: &RunEntry,
    graph: Arc<dyn ErasedGraph>,
    req: RunCreate,
) -> Final {
    let mut prev = None;
    if let Some(tid) = &entry.thread_id {
        let latest = match app.cp.latest(tid).await {
            Ok(c) => c,
            Err(e) => return Final::error(e),
        };
        let update = req.command.as_ref().and_then(|c| {
            let resume = c.resume.clone().filter(Value::is_object);
            match (c.update.clone(), resume) {
                (Some(u), Some(r)) => Some(merge(u, r)),
                (u, r) => u.or(r),
            }
        });
        prev = match (latest, update) {
            (Some(mut cp), Some(u)) => {
                cp.state = merge(cp.state, u);
                if let Err(e) = graph.validate(&cp.state) {
                    return Final::error(e);
                }
                let state = cp.state.clone();
                if let Err(e) = app.cp.put(cp).await {
                    return Final::error(e);
                }
                Some(state)
            }
            (None, Some(_)) => return Final::error("thread has no state to update"),
            (cp, None) => cp.map(|c| c.state),
        };
        if req.input.is_none() && prev.is_none() {
            return Final::error("thread has no checkpoint to resume; pass `input`");
        }
    }
    let spec = StartSpec {
        thread_id: entry.thread_id.clone(),
        input: req.input,
        prev,
        modes: Modes::from_list(&req.stream_mode),
        recursion_limit: req.recursion_limit,
        interrupt_before: req.interrupt_before,
    };
    let mut stream = match graph.start(spec) {
        Ok(s) => s,
        Err(e) => return Final::error(e),
    };
    let cancelled = entry.cancel.cancelled();
    tokio::pin!(cancelled);
    loop {
        tokio::select! {
            biased;
            _ = &mut cancelled => return Final::cancelled(None),
            step = stream.next() => match step {
                Some(Step::Frame(b)) => entry.broadcast(b),
                Some(Step::Done(Outcome::Completed(v))) => {
                    return Final { status: RunStatus::Success, output: Some(v), error: None, next: Vec::new() }
                }
                Some(Step::Done(Outcome::Interrupted { next, values })) => {
                    return Final { status: RunStatus::Interrupted, output: Some(values), error: None, next }
                }
                Some(Step::Done(Outcome::Error(e))) => return Final::error(e),
                None => return Final::error("graph stream ended without a result"),
            }
        }
    }
}

fn sse(
    app: &App,
    rx: tokio::sync::mpsc::UnboundedReceiver<Bytes>,
    cancel_on_drop: Option<&RunEntry>,
) -> Response {
    let on_drop: Option<Box<dyn FnOnce() + Send + Sync>> = cancel_on_drop.map(|e| {
        let token = e.cancel.clone();
        Box::new(move || token.cancel()) as Box<dyn FnOnce() + Send + Sync>
    });
    let body = SseBody::new(rx, app.config.keep_alive, on_drop).counted(app.streams.clone());
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(body))
        .unwrap()
}

async fn run_background(app: &App, tid: Option<String>, b: &Bytes) -> ApiResult<Run> {
    let req: RunCreate = body(b)?;
    let (entry, _) = start_run(app, tid, req, false).await?;
    Ok(Json(entry.snapshot()))
}

async fn run_waiting(app: &App, tid: Option<String>, b: &Bytes) -> ApiResult<Run> {
    let req: RunCreate = body(b)?;
    let (entry, _) = start_run(app, tid, req, false).await?;
    Ok(Json(entry.wait().await))
}

async fn run_streaming(app: &App, tid: Option<String>, b: &Bytes) -> Result<Response, ApiError> {
    let req: RunCreate = body(b)?;
    let cancel = req.on_disconnect == OnDisconnect::Cancel;
    let (entry, rx) = start_run(app, tid, req, true).await?;
    let rx = rx.expect("subscribed");
    Ok(sse(app, rx, cancel.then_some(&*entry)))
}

async fn thread_run_bg(
    State(app): State<App>,
    Path(tid): Path<String>,
    b: Bytes,
) -> ApiResult<Run> {
    run_background(&app, Some(tid), &b).await
}
async fn thread_run_wait(
    State(app): State<App>,
    Path(tid): Path<String>,
    b: Bytes,
) -> ApiResult<Run> {
    run_waiting(&app, Some(tid), &b).await
}
async fn thread_run_stream(
    State(app): State<App>,
    Path(tid): Path<String>,
    b: Bytes,
) -> Result<Response, ApiError> {
    run_streaming(&app, Some(tid), &b).await
}
async fn run_bg(State(app): State<App>, b: Bytes) -> ApiResult<Run> {
    run_background(&app, None, &b).await
}
async fn run_wait(State(app): State<App>, b: Bytes) -> ApiResult<Run> {
    run_waiting(&app, None, &b).await
}
async fn run_stream(State(app): State<App>, b: Bytes) -> Result<Response, ApiError> {
    run_streaming(&app, None, &b).await
}

async fn list_runs(State(app): State<App>, Path(tid): Path<String>) -> ApiResult<Vec<Run>> {
    app.thread(&tid).await?;
    Ok(Json(app.runs.of_thread(&tid)))
}

fn find_run(app: &App, tid: Option<&str>, rid: &str) -> Result<Arc<RunEntry>, ApiError> {
    app.runs
        .get(rid)
        .filter(|e| tid.is_none() || e.thread_id.as_deref() == tid)
        .ok_or_else(|| ApiError::not_found(format_args!("run `{rid}`")))
}

#[derive(Deserialize)]
struct CancelQuery {
    #[serde(default)]
    wait: bool,
}

async fn cancel(app: &App, tid: Option<&str>, rid: &str, wait: bool) -> ApiResult<Run> {
    let e = find_run(app, tid, rid)?;
    e.cancel.cancel();
    Ok(Json(if wait { e.wait().await } else { e.snapshot() }))
}

async fn thread_get_run(
    State(app): State<App>,
    Path((tid, rid)): Path<(String, String)>,
) -> ApiResult<Run> {
    Ok(Json(find_run(&app, Some(&tid), &rid)?.snapshot()))
}
async fn thread_cancel_run(
    State(app): State<App>,
    Path((tid, rid)): Path<(String, String)>,
    Query(q): Query<CancelQuery>,
) -> ApiResult<Run> {
    cancel(&app, Some(&tid), &rid, q.wait).await
}
async fn thread_join_run(
    State(app): State<App>,
    Path((tid, rid)): Path<(String, String)>,
) -> ApiResult<Run> {
    Ok(Json(find_run(&app, Some(&tid), &rid)?.wait().await))
}
async fn thread_join_stream(
    State(app): State<App>,
    Path((tid, rid)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let e = find_run(&app, Some(&tid), &rid)?;
    Ok(sse(&app, e.subscribe(), None))
}
async fn get_run(State(app): State<App>, Path(rid): Path<String>) -> ApiResult<Run> {
    Ok(Json(find_run(&app, None, &rid)?.snapshot()))
}
async fn cancel_run(
    State(app): State<App>,
    Path(rid): Path<String>,
    Query(q): Query<CancelQuery>,
) -> ApiResult<Run> {
    cancel(&app, None, &rid, q.wait).await
}
async fn join_run(State(app): State<App>, Path(rid): Path<String>) -> ApiResult<Run> {
    Ok(Json(find_run(&app, None, &rid)?.wait().await))
}
async fn join_stream(
    State(app): State<App>,
    Path(rid): Path<String>,
) -> Result<Response, ApiError> {
    let e = find_run(&app, None, &rid)?;
    Ok(sse(&app, e.subscribe(), None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webhook_policy() {
        assert!(!WebhookPolicy::Disabled.allows("http://x"));
        assert!(WebhookPolicy::Any.allows("https://x/y"));
        assert!(!WebhookPolicy::Any.allows("file:///etc/passwd"));
        let p = WebhookPolicy::Prefixes(vec!["http://hooks.local/".into()]);
        assert!(p.allows("http://hooks.local/run"));
        assert!(!p.allows("http://evil/"));
    }

    #[test]
    fn secrets_compare() {
        assert!(same_secret(b"abc", b"abc"));
        assert!(!same_secret(b"abc", b"abd"));
        assert!(!same_secret(b"abc", b"ab"));
    }
}
