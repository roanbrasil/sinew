//! HTTP server: ingest, query API, datasets/experiments, prompt hub,
//! feedback, annotation queues, monitoring, rules, alerts and the embedded UI.

use std::{collections::HashMap, net::SocketAddr};

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Path, Query, Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sinew_eval::{Dataset, ExperimentReport};
use sinew_trace::Run;

use crate::{
    api,
    engine::{Engine, EngineConfig},
    otlp,
    store::{Store, StoreError, TraceQuery},
};

/// Header that selects the project on ingest.
pub const PROJECT_HEADER: &str = "x-sinew-project";

const INDEX_HTML: &str = include_str!("ui/index.html");
const APP_JS: &str = include_str!("ui/app.js");
const PAGES_JS: &str = include_str!("ui/pages.js");
const STYLE_CSS: &str = include_str!("ui/style.css");
const LOGO_PNG: &[u8] = include_bytes!("ui/logo.png");

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) store: Store,
    pub(crate) token: Option<String>,
    pub(crate) engine: Option<Engine>,
}

/// Options for [`router_with`].
#[derive(Clone, Default)]
pub struct ServerConfig {
    /// Bearer token for `/api/*` and `/v1/*`.
    pub token: Option<String>,
    /// Background engine settings (rules, alerts, retention); `None` uses
    /// the defaults.
    pub engine: Option<EngineConfig>,
    /// Do not start the background engine.
    pub disable_engine: bool,
}

/// JSON error response: `{"error": "..."}` with a status code.
#[derive(Debug)]
pub struct ApiError(pub(crate) StatusCode, pub(crate) String);

impl ApiError {
    pub(crate) fn bad_request(msg: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, msg.into())
    }
    pub(crate) fn not_found(what: &str) -> Self {
        Self(StatusCode::NOT_FOUND, format!("{what} not found"))
    }
}

impl From<StoreError> for ApiError {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::Invalid(m) => Self(StatusCode::BAD_REQUEST, m),
            StoreError::Db(e) => Self(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

pub(crate) type ApiResult<T> = Result<Json<T>, ApiError>;

/// Builds the full router. `token` enables bearer auth on `/api/*` and
/// `/v1/*` (the UI assets and `/api/health` stay public). When called inside
/// a Tokio runtime, the background engine (online rules, alerts, retention)
/// is started with default settings; use [`router_with`] to configure it.
pub fn router(store: Store, token: Option<String>) -> Router {
    router_with(
        store,
        ServerConfig {
            token,
            ..Default::default()
        },
    )
}

/// Builds the router with explicit [`ServerConfig`].
pub fn router_with(store: Store, config: ServerConfig) -> Router {
    let engine = (!config.disable_engine && tokio::runtime::Handle::try_current().is_ok())
        .then(|| Engine::start(store.clone(), config.engine.unwrap_or_default()));
    let state = AppState {
        store,
        token: config.token.filter(|t| !t.is_empty()),
        engine,
    };
    let protected = Router::new()
        .route("/api/runs", post(ingest_runs).get(api::search_runs))
        .route("/api/runs/{id}", get(get_run))
        .route("/api/runs/{id}/feedback", get(api::run_feedback))
        .route("/api/metrics", get(api::metrics))
        .route(
            "/api/feedback",
            get(api::list_feedback).post(api::create_feedback),
        )
        .route("/api/feedback/stats", get(api::feedback_stats))
        .route("/api/feedback/{id}", delete(api::delete_feedback))
        .route(
            "/api/prompts",
            get(api::list_prompts).post(api::push_prompt),
        )
        .route(
            "/api/prompts/{name}",
            get(api::get_prompt).delete(api::delete_prompt),
        )
        .route("/api/prompts/{name}/diff", get(api::diff_prompt))
        .route("/api/prompts/{name}/{reference}", get(api::resolve_prompt))
        .route(
            "/api/prompts/{name}/tags/{tag}",
            put(api::set_prompt_tag).delete(api::delete_prompt_tag),
        )
        .route("/api/queues", get(api::list_queues).post(api::create_queue))
        .route(
            "/api/queues/{id}",
            get(api::get_queue).delete(api::delete_queue),
        )
        .route("/api/queues/{id}/items", post(api::enqueue))
        .route("/api/queues/{id}/next", get(api::queue_next))
        .route(
            "/api/queues/{id}/items/{run_id}",
            delete(api::remove_queue_item),
        )
        .route(
            "/api/queues/{id}/items/{run_id}/done",
            post(api::complete_queue_item),
        )
        .route("/api/datasets/{name}/examples", post(api::add_example))
        .route(
            "/api/pairwise",
            get(api::list_pairwise).post(api::put_pairwise),
        )
        .route("/api/pairwise/{id}", get(api::get_pairwise))
        .route(
            "/api/pricing",
            get(api::get_pricing)
                .put(api::put_pricing)
                .delete(api::reset_pricing),
        )
        .route("/api/pricing/recompute", post(api::recompute_costs))
        .route(
            "/api/settings",
            get(api::get_settings).put(api::put_settings),
        )
        .route("/api/retention/run", post(api::run_retention))
        .route("/api/rules", get(api::list_rules).post(api::create_rule))
        .route(
            "/api/rules/{id}",
            get(api::get_rule)
                .put(api::update_rule)
                .delete(api::delete_rule),
        )
        .route("/api/rules/{id}/run", post(api::run_rule))
        .route("/api/alerts", get(api::list_alerts).post(api::create_alert))
        .route(
            "/api/alerts/{id}",
            get(api::get_alert)
                .put(api::update_alert)
                .delete(api::delete_alert),
        )
        .route("/api/alerts/{id}/check", post(api::check_alert))
        .route("/api/traces", get(list_traces))
        .route("/api/traces/{trace_id}", get(get_trace))
        .route("/api/projects", get(list_projects))
        .route("/api/datasets", get(list_datasets).post(put_dataset))
        .route("/api/datasets/{name}", get(get_dataset))
        .route(
            "/api/experiments",
            get(list_experiments).post(put_experiment),
        )
        .route("/api/experiments/{id}", get(get_experiment))
        .route("/api/compare", get(compare_experiments))
        .route("/v1/traces", post(ingest_otlp))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth));
    Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/pages.js", get(pages_js))
        .route("/logo.png", get(logo_png))
        .route("/style.css", get(style_css))
        .route("/api/health", get(health))
        .merge(protected)
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}

/// Serves `router` on an already-bound listener until `shutdown` resolves.
pub async fn serve_with_shutdown(
    listener: tokio::net::TcpListener,
    app: Router,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
}

/// Binds `addr` and serves until Ctrl-C.
pub async fn serve(addr: SocketAddr, store: Store, token: Option<String>) -> std::io::Result<()> {
    serve_with_config(
        addr,
        store,
        ServerConfig {
            token,
            ..Default::default()
        },
    )
    .await
}

/// Binds `addr` and serves with `config` until Ctrl-C.
pub async fn serve_with_config(
    addr: SocketAddr,
    store: Store,
    config: ServerConfig,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!(
        "sinew-scope listening on http://{}{}",
        listener.local_addr()?,
        if config.token.as_deref().is_some_and(|t| !t.is_empty()) {
            "  (bearer auth on)"
        } else {
            ""
        }
    );
    serve_with_shutdown(listener, router_with(store, config), async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await
}

/// Compares in time independent of where the first difference is.
fn same_secret(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn auth(State(state): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(expected) = &state.token {
        let given = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| {
                v.strip_prefix("Bearer ")
                    .or_else(|| v.strip_prefix("bearer "))
            })
            .map(str::trim);
        if !given.is_some_and(|g| same_secret(g.as_bytes(), expected.as_bytes())) {
            let mut resp = ApiError(
                StatusCode::UNAUTHORIZED,
                "missing or invalid bearer token".into(),
            )
            .into_response();
            resp.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                HeaderValue::from_static("Bearer realm=\"sinew-scope\""),
            );
            return resp;
        }
    }
    next.run(req).await
}

fn asset(content_type: &'static str, body: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response()
}

async fn index() -> Response {
    asset("text/html; charset=utf-8", INDEX_HTML)
}

async fn app_js() -> Response {
    asset("text/javascript; charset=utf-8", APP_JS)
}

async fn pages_js() -> Response {
    asset("text/javascript; charset=utf-8", PAGES_JS)
}

async fn style_css() -> Response {
    asset("text/css; charset=utf-8", STYLE_CSS)
}

async fn logo_png() -> Response {
    (
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "max-age=86400"),
        ],
        LOGO_PNG,
    )
        .into_response()
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "auth": state.token.is_some(),
        "fts": state.store.has_fts(),
        "engine": state.engine.is_some(),
        "otlp_protobuf": cfg!(feature = "otlp-proto"),
    }))
}

pub(crate) fn header_project(headers: &HeaderMap) -> Option<String> {
    headers
        .get(PROJECT_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
}

pub(crate) fn parse_json(body: &Bytes) -> Result<Value, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::bad_request(format!("invalid JSON: {e}")))
}

/// `POST /api/runs`: either a JSON array of runs, or
/// `{"project": "...", "runs": [...]}`. Project precedence: body field,
/// then the `X-Sinew-Project` header, then `default`.
async fn ingest_runs(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<Value> {
    let body = parse_json(&body)?;
    let (body_project, items) = match body {
        Value::Array(items) => (None, items),
        Value::Object(mut o) => {
            let project = o
                .get("project")
                .and_then(Value::as_str)
                .filter(|p| !p.is_empty())
                .map(str::to_string);
            match o.remove("runs") {
                Some(Value::Array(items)) => (project, items),
                _ => return Err(ApiError::bad_request("expected `runs` to be an array")),
            }
        }
        _ => return Err(ApiError::bad_request("expected an array of runs")),
    };
    let project = body_project
        .or_else(|| header_project(&headers))
        .unwrap_or_else(|| "default".into());
    let mut runs = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        let run: Run = serde_json::from_value(item)
            .map_err(|e| ApiError::bad_request(format!("run {i}: {e}")))?;
        runs.push((project.clone(), run));
    }
    let result = state.store.ingest(&runs).await?;
    if let Some(engine) = &state.engine {
        engine.submit(runs);
    }
    Ok(Json(json!(result)))
}

/// `POST /v1/traces`: OTLP/HTTP with JSON encoding.
async fn ingest_otlp(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let ct = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let body = if ct.contains("protobuf") {
        decode_protobuf(&body)?
    } else {
        parse_json(&body)?
    };
    let runs = otlp::otlp_to_runs(&body, header_project(&headers).as_deref())
        .map_err(ApiError::bad_request)?;
    state.store.ingest(&runs).await?;
    if let Some(engine) = &state.engine {
        engine.submit(runs);
    }
    // ExportTraceServiceResponse: an empty partialSuccess means all accepted.
    Ok(Json(json!({ "partialSuccess": {} })))
}

#[cfg(feature = "otlp-proto")]
fn decode_protobuf(body: &Bytes) -> Result<Value, ApiError> {
    sinew_trace::otlp::proto::decode_request(body).map_err(|e| ApiError::bad_request(e.to_string()))
}

#[cfg(not(feature = "otlp-proto"))]
fn decode_protobuf(_body: &Bytes) -> Result<Value, ApiError> {
    Err(ApiError(
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "this server was built without OTLP protobuf support (feature `otlp-proto`); \
         set the exporter protocol to http/json"
            .into(),
    ))
}

async fn get_run(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Value> {
    let run = state
        .store
        .run(&id)
        .await?
        .ok_or_else(|| ApiError::not_found("run"))?;
    Ok(Json(json!(run)))
}

async fn list_traces(State(state): State<AppState>, uri: axum::http::Uri) -> ApiResult<Value> {
    let mut q: TraceQuery = api::query_from(&uri)?;
    q.metadata = api::meta_filters(&uri);
    Ok(Json(json!(state.store.list_traces(&q).await?)))
}

async fn get_trace(
    State(state): State<AppState>,
    Path(trace_id): Path<String>,
) -> ApiResult<Value> {
    let t = state
        .store
        .trace(&trace_id)
        .await?
        .ok_or_else(|| ApiError::not_found("trace"))?;
    Ok(Json(json!(t)))
}

async fn list_projects(State(state): State<AppState>) -> ApiResult<Value> {
    Ok(Json(json!(state.store.projects().await?)))
}

async fn list_datasets(State(state): State<AppState>) -> ApiResult<Value> {
    Ok(Json(json!(state.store.datasets().await?)))
}

async fn put_dataset(State(state): State<AppState>, body: Bytes) -> ApiResult<Value> {
    let ds: Dataset = serde_json::from_value(parse_json(&body)?)
        .map_err(|e| ApiError::bad_request(format!("invalid dataset: {e}")))?;
    state.store.put_dataset(&ds).await?;
    Ok(Json(
        json!({ "name": ds.name, "examples": ds.examples.len() }),
    ))
}

async fn get_dataset(State(state): State<AppState>, Path(name): Path<String>) -> ApiResult<Value> {
    let ds = state
        .store
        .dataset(&name)
        .await?
        .ok_or_else(|| ApiError::not_found("dataset"))?;
    let experiments = state.store.experiments(Some(&name)).await?;
    Ok(Json(json!({ "dataset": ds, "experiments": experiments })))
}

#[derive(Deserialize)]
struct ExperimentFilter {
    dataset: Option<String>,
}

async fn list_experiments(
    State(state): State<AppState>,
    Query(f): Query<ExperimentFilter>,
) -> ApiResult<Value> {
    Ok(Json(json!(
        state.store.experiments(f.dataset.as_deref()).await?
    )))
}

async fn put_experiment(State(state): State<AppState>, body: Bytes) -> ApiResult<Value> {
    let r: ExperimentReport = serde_json::from_value(parse_json(&body)?)
        .map_err(|e| ApiError::bad_request(format!("invalid experiment report: {e}")))?;
    state.store.put_experiment(&r).await?;
    Ok(Json(json!({ "id": r.id })))
}

async fn get_experiment(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Value> {
    let r = state
        .store
        .experiment(&id)
        .await?
        .ok_or_else(|| ApiError::not_found("experiment"))?;
    Ok(Json(json!(r)))
}

#[derive(Deserialize)]
struct CompareQuery {
    a: String,
    b: String,
}

/// `GET /api/compare?a=<id>&b=<id>`: both reports plus rows aligned by
/// example id (falling back to position), with per-key score deltas.
async fn compare_experiments(
    State(state): State<AppState>,
    Query(q): Query<CompareQuery>,
) -> ApiResult<Value> {
    let a = state
        .store
        .experiment(&q.a)
        .await?
        .ok_or_else(|| ApiError::not_found("experiment a"))?;
    let b = state
        .store
        .experiment(&q.b)
        .await?
        .ok_or_else(|| ApiError::not_found("experiment b"))?;
    Ok(Json(compare(&a, &b)))
}

/// Aligns two reports by example id (and repetition) and computes score
/// deltas (b - a).
pub fn compare(a: &ExperimentReport, b: &ExperimentReport) -> Value {
    let b_by_id: HashMap<(&str, usize), usize> = b
        .results
        .iter()
        .enumerate()
        .map(|(i, r)| ((r.example_id.as_str(), r.repetition), i))
        .collect();
    let mut used = vec![false; b.results.len()];
    let mut rows = Vec::new();
    for ra in &a.results {
        let ib = b_by_id
            .get(&(ra.example_id.as_str(), ra.repetition))
            .copied();
        if let Some(i) = ib {
            used[i] = true;
        }
        let rb = ib.map(|i| &b.results[i]);
        let mut deltas = serde_json::Map::new();
        if let Some(rb) = rb {
            for s in &ra.scores {
                if let Some(sb) = rb.score(&s.key) {
                    deltas.insert(s.key.clone(), json!(sb - s.score));
                }
            }
        }
        rows.push(json!({
            "example_id": ra.example_id,
            "repetition": ra.repetition,
            "inputs": ra.inputs,
            "reference": ra.reference,
            "a": ra,
            "b": rb,
            "deltas": deltas,
        }));
    }
    for (i, rb) in b.results.iter().enumerate() {
        if !used[i] {
            rows.push(json!({
                "example_id": rb.example_id,
                "inputs": rb.inputs,
                "reference": rb.reference,
                "a": null,
                "b": rb,
                "deltas": {},
            }));
        }
    }
    let mut keys: Vec<&String> = a
        .aggregates
        .scores
        .keys()
        .chain(b.aggregates.scores.keys())
        .collect();
    keys.sort();
    keys.dedup();
    let summary: Vec<Value> = keys
        .iter()
        .map(|k| {
            let ma = a.aggregates.scores.get(*k).map(|s| s.mean);
            let mb = b.aggregates.scores.get(*k).map(|s| s.mean);
            json!({
                "key": k,
                "a": ma,
                "b": mb,
                "delta": ma.zip(mb).map(|(x, y)| y - x),
            })
        })
        .collect();
    json!({
        "a": { "id": a.id, "name": a.name, "dataset": a.dataset, "created_ms": a.created_ms, "aggregates": a.aggregates },
        "b": { "id": b.id, "name": b.name, "dataset": b.dataset, "created_ms": b.created_ms, "aggregates": b.aggregates },
        "keys": keys,
        "summary": summary,
        "rows": rows,
    })
}
