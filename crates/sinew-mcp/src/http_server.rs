//! Streamable HTTP server transport (feature `http-server`).
//!
//! Serves an [`McpServer`] at one endpoint (default `/mcp`) following the MCP
//! Streamable HTTP transport:
//!
//! - `POST` carries one JSON-RPC message (or a batch from older clients).
//!   Requests get a JSON response; notifications and responses get `202`.
//! - `initialize` opens a session and returns it in `Mcp-Session-Id`; later
//!   requests must send it back (`400` without it, `404` once it expired).
//!   [`McpHttpConfig::stateless`] turns sessions off for horizontally scaled
//!   deployments behind a load balancer.
//! - `DELETE` ends the session. `GET` answers `405`: this server never pushes
//!   messages on its own, so it offers no standalone SSE stream.
//! - Optional bearer-token auth, an `Origin` allow-list (DNS-rebinding
//!   protection) and an unauthenticated `GET /health`.
//!
//! The result is a plain [`axum::Router`], so a microservice can `merge` it
//! next to its own routes:
//!
//! ```no_run
//! # async fn demo(tools: sinew_core::ToolBox) -> std::io::Result<()> {
//! use sinew_mcp::{McpHttpConfig, McpServer};
//!
//! let mcp = McpServer::new("orders", env!("CARGO_PKG_VERSION"))
//!     .with_tools(tools)
//!     .http_router(McpHttpConfig::default().with_bearer_token("s3cret"));
//! let app = axum::Router::new()
//!     .route("/orders", axum::routing::get(|| async { "[]" }))
//!     .merge(mcp);
//! let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
//! axum::serve(listener, app).await
//! # }
//! ```

use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use serde_json::{json, Value};

use crate::{protocol::SUPPORTED_PROTOCOL_VERSIONS, McpServer};

const SESSION_HEADER: &str = "mcp-session-id";
const VERSION_HEADER: &str = "mcp-protocol-version";

/// Settings for [`McpServer::http_router`].
#[derive(Debug, Clone)]
pub struct McpHttpConfig {
    path: String,
    tokens: Vec<String>,
    origins: Option<Vec<String>>,
    stateless: bool,
    session_ttl: Duration,
    max_sessions: usize,
}

impl Default for McpHttpConfig {
    fn default() -> Self {
        Self {
            path: "/mcp".into(),
            tokens: Vec::new(),
            origins: None,
            stateless: false,
            session_ttl: Duration::from_secs(3600),
            max_sessions: 10_000,
        }
    }
}

impl McpHttpConfig {
    /// Endpoint path (default `/mcp`).
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }

    /// Requires `Authorization: Bearer <token>`. Call again to accept several
    /// tokens (e.g. during rotation).
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.tokens.push(token.into());
        self
    }

    /// Rejects requests whose `Origin` header is present and not listed
    /// (exact match, e.g. `https://app.example.com`). Requests without an
    /// `Origin` (server-to-server calls) are allowed.
    pub fn with_allowed_origins<I, S>(mut self, origins: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.origins = Some(origins.into_iter().map(Into::into).collect());
        self
    }

    /// No sessions: every request stands alone, so any replica can answer it.
    pub fn stateless(mut self) -> Self {
        self.stateless = true;
        self
    }

    /// Idle time after which a session expires (default one hour).
    pub fn with_session_ttl(mut self, ttl: Duration) -> Self {
        self.session_ttl = ttl;
        self
    }

    /// Upper bound on open sessions; the least recently used is evicted.
    pub fn with_max_sessions(mut self, n: usize) -> Self {
        self.max_sessions = n.max(1);
        self
    }
}

struct Shared {
    server: McpServer,
    config: McpHttpConfig,
    sessions: Mutex<HashMap<String, Instant>>,
}

impl Shared {
    fn open_session(&self) -> String {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let mut s = self.sessions.lock().unwrap();
        let now = Instant::now();
        s.retain(|_, seen| now.duration_since(*seen) < self.config.session_ttl);
        if s.len() >= self.config.max_sessions {
            if let Some(oldest) = s.iter().min_by_key(|(_, t)| **t).map(|(k, _)| k.clone()) {
                s.remove(&oldest);
            }
        }
        s.insert(id.clone(), now);
        id
    }

    /// Refreshes a session; false if it is unknown or expired.
    fn touch(&self, id: &str) -> bool {
        let mut s = self.sessions.lock().unwrap();
        let now = Instant::now();
        match s.get_mut(id) {
            Some(seen) if now.duration_since(*seen) < self.config.session_ttl => {
                *seen = now;
                true
            }
            Some(_) => {
                s.remove(id);
                false
            }
            None => false,
        }
    }
}

impl McpServer {
    /// The MCP endpoint as an axum [`Router`], ready to `merge` into a service.
    pub fn http_router(self, config: McpHttpConfig) -> Router {
        let path = config.path.clone();
        let shared = Arc::new(Shared {
            server: self,
            config,
            sessions: Mutex::new(HashMap::new()),
        });
        Router::new()
            .route(&path, get(get_endpoint).post(post).delete(delete))
            .route("/health", get(|| async { "ok" }))
            .with_state(shared)
    }

    /// Binds `addr` and serves `/mcp` with the default config until the
    /// process stops.
    pub async fn serve_http(self, addr: &str) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        self.serve_http_with(listener, McpHttpConfig::default(), std::future::pending())
            .await
    }

    /// Serves on `listener` until `shutdown` resolves (graceful shutdown).
    pub async fn serve_http_with(
        self,
        listener: tokio::net::TcpListener,
        config: McpHttpConfig,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        axum::serve(listener, self.http_router(config))
            .with_graceful_shutdown(shutdown)
            .await
    }
}

fn rpc_error(status: StatusCode, code: i64, message: &str) -> Response {
    let body = json!({"jsonrpc": "2.0", "id": null, "error": {"code": code, "message": message}});
    (status, axum::Json(body)).into_response()
}

/// Auth and origin checks shared by every method.
fn guard(shared: &Shared, headers: &HeaderMap) -> Option<Response> {
    if let Some(allowed) = &shared.config.origins {
        if let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
            if !allowed.iter().any(|o| o == origin) {
                return Some(rpc_error(
                    StatusCode::FORBIDDEN,
                    -32000,
                    "origin not allowed",
                ));
            }
        }
    }
    if !shared.config.tokens.is_empty() {
        let presented = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        let ok = presented.is_some_and(|t| {
            shared
                .config
                .tokens
                .iter()
                .any(|k| constant_time_eq(k.as_bytes(), t.as_bytes()))
        });
        if !ok {
            let mut r = rpc_error(StatusCode::UNAUTHORIZED, -32001, "unauthorized");
            r.headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
            return Some(r);
        }
    }
    if let Some(v) = headers.get(VERSION_HEADER).and_then(|v| v.to_str().ok()) {
        if !SUPPORTED_PROTOCOL_VERSIONS.contains(&v) {
            return Some(rpc_error(
                StatusCode::BAD_REQUEST,
                -32600,
                "unsupported MCP-Protocol-Version",
            ));
        }
    }
    None
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn session_of(headers: &HeaderMap) -> Option<&str> {
    headers.get(SESSION_HEADER).and_then(|v| v.to_str().ok())
}

async fn post(State(shared): State<Arc<Shared>>, headers: HeaderMap, body: Bytes) -> Response {
    if let Some(r) = guard(&shared, &headers) {
        return r;
    }
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return rpc_error(StatusCode::BAD_REQUEST, -32700, "parse error"),
    };
    let (messages, batch) = match parsed {
        Value::Array(items) if !items.is_empty() => (items, true),
        Value::Array(_) => return rpc_error(StatusCode::BAD_REQUEST, -32600, "empty batch"),
        v => (vec![v], false),
    };
    let initializing = messages.iter().any(|m| m["method"] == "initialize");

    let mut new_session = None;
    if !shared.config.stateless {
        if initializing {
            new_session = Some(shared.open_session());
        } else {
            match session_of(&headers) {
                None => {
                    return rpc_error(StatusCode::BAD_REQUEST, -32600, "missing Mcp-Session-Id")
                }
                Some(id) if !shared.touch(id) => {
                    return rpc_error(StatusCode::NOT_FOUND, -32001, "session not found")
                }
                Some(_) => {}
            }
        }
    }

    let mut replies = Vec::new();
    for m in messages {
        if let Some(r) = shared.server.handle_message(m).await {
            replies.push(r);
        }
    }

    let mut resp = if replies.is_empty() {
        StatusCode::ACCEPTED.into_response()
    } else if batch {
        axum::Json(Value::Array(replies)).into_response()
    } else {
        axum::Json(replies.pop().unwrap_or(Value::Null)).into_response()
    };
    if let Some(id) = new_session {
        if let Ok(v) = HeaderValue::from_str(&id) {
            resp.headers_mut().insert(SESSION_HEADER, v);
        }
    }
    resp
}

async fn get_endpoint(State(shared): State<Arc<Shared>>, headers: HeaderMap) -> Response {
    if let Some(r) = guard(&shared, &headers) {
        return r;
    }
    let mut r = StatusCode::METHOD_NOT_ALLOWED.into_response();
    r.headers_mut()
        .insert(header::ALLOW, HeaderValue::from_static("POST, DELETE"));
    r
}

async fn delete(State(shared): State<Arc<Shared>>, headers: HeaderMap) -> Response {
    if let Some(r) = guard(&shared, &headers) {
        return r;
    }
    if shared.config.stateless {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    match session_of(&headers) {
        Some(id) if shared.sessions.lock().unwrap().remove(id).is_some() => {
            StatusCode::OK.into_response()
        }
        Some(_) => StatusCode::NOT_FOUND.into_response(),
        None => StatusCode::BAD_REQUEST.into_response(),
    }
}
