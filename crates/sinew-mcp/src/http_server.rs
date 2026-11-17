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
//! - `GET` (with the session id and `Accept: text/event-stream`) opens the
//!   session's server-to-client SSE stream, which carries
//!   `notifications/resources/updated` for the session's subscriptions and
//!   the `list_changed` notifications. One stream per session; a new `GET`
//!   replaces the old one. Notifications sent while no stream is open are
//!   dropped. In stateless mode `GET` answers `405`.
//! - `DELETE` ends the session (and closes its stream).
//! - Optional bearer-token auth, an `Origin` allow-list (DNS-rebinding
//!   protection) and an unauthenticated `GET /health`.
//! - With the `oauth-server` feature, OAuth 2.1 per the MCP authorization
//!   specification (`McpHttpConfig::with_oauth`): protected resource
//!   metadata, JWT access tokens checked against the issuer's JWKS, and
//!   scopes per tool.
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
    convert::Infallible,
    future::Future,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
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
    #[cfg(feature = "oauth-server")]
    oauth: Option<crate::oauth::OAuthConfig>,
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
            #[cfg(feature = "oauth-server")]
            oauth: None,
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

    /// Protects the endpoint with OAuth 2.1 (feature `oauth-server`), as the
    /// MCP authorization specification describes:
    ///
    /// - `GET /.well-known/oauth-protected-resource` (and the same path with
    ///   the resource's path appended, per RFC 9728) serves the protected
    ///   resource metadata, without authentication;
    /// - requests without a valid access token get `401` with
    ///   `WWW-Authenticate: Bearer resource_metadata="..."` (plus
    ///   `error="invalid_token"` when a token was sent);
    /// - tokens are JWTs validated by a [`JwtValidator`](crate::oauth::JwtValidator)
    ///   (signature against the JWKS, issuer, audience, expiry);
    /// - missing scopes give `403` with `error="insufficient_scope"`;
    /// - when the JWKS cannot be fetched, requests get `503`.
    ///
    /// Static tokens from [`with_bearer_token`](Self::with_bearer_token) are
    /// not accepted on an OAuth endpoint: the specification forbids accepting
    /// tokens other than those issued for the resource.
    #[cfg(feature = "oauth-server")]
    pub fn with_oauth(mut self, oauth: crate::oauth::OAuthConfig) -> Self {
        self.oauth = Some(oauth);
        self
    }
}

struct Shared {
    server: McpServer,
    config: McpHttpConfig,
    sessions: Mutex<HashMap<String, Instant>>,
    #[cfg(feature = "oauth-server")]
    validator: Option<crate::oauth::JwtValidator>,
}

impl Shared {
    /// Opens a session and registers it as a notification peer (without a
    /// stream until the client sends `GET`).
    fn open_session(&self) -> String {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let mut s = self.sessions.lock().unwrap();
        let now = Instant::now();
        let mut gone = Vec::new();
        s.retain(|k, seen| {
            let alive = now.duration_since(*seen) < self.config.session_ttl;
            if !alive {
                gone.push(k.clone());
            }
            alive
        });
        if s.len() >= self.config.max_sessions {
            if let Some(oldest) = s.iter().min_by_key(|(_, t)| **t).map(|(k, _)| k.clone()) {
                s.remove(&oldest);
                gone.push(oldest);
            }
        }
        s.insert(id.clone(), now);
        drop(s);
        let hub = self.server.hub();
        for g in gone {
            hub.remove_peer(&g);
        }
        hub.add_peer(&id, None);
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
                drop(s);
                self.server.hub().remove_peer(id);
                false
            }
            None => false,
        }
    }

    /// Ends a session; false if it was unknown.
    fn close_session(&self, id: &str) -> bool {
        let existed = self.sessions.lock().unwrap().remove(id).is_some();
        self.server.hub().remove_peer(id);
        existed
    }
}

impl McpServer {
    /// The MCP endpoint as an axum [`Router`], ready to `merge` into a service.
    pub fn http_router(self, config: McpHttpConfig) -> Router {
        let path = config.path.clone();
        #[cfg(feature = "oauth-server")]
        let oauth = config.oauth.clone();
        let shared = Arc::new(Shared {
            server: self,
            #[cfg(feature = "oauth-server")]
            validator: config.oauth.clone().map(crate::oauth::JwtValidator::new),
            config,
            sessions: Mutex::new(HashMap::new()),
        });
        #[allow(unused_mut)]
        let mut router = Router::new()
            .route(&path, get(get_endpoint).post(post).delete(delete))
            .route("/health", get(|| async { "ok" }));
        #[cfg(feature = "oauth-server")]
        if let Some(oauth) = oauth {
            let doc = Arc::new(serde_json::to_value(oauth.metadata()).unwrap_or(Value::Null));
            for p in oauth.metadata_paths() {
                let doc = doc.clone();
                router = router.route(
                    &p,
                    get(move || {
                        let doc = doc.clone();
                        async move {
                            let mut r = axum::Json((*doc).clone()).into_response();
                            // Public document; browser-based clients read it cross-origin.
                            r.headers_mut().insert(
                                header::ACCESS_CONTROL_ALLOW_ORIGIN,
                                HeaderValue::from_static("*"),
                            );
                            r
                        }
                    }),
                );
            }
        }
        router.with_state(shared)
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

/// Who the request was authorized as.
#[derive(Default)]
struct Auth {
    #[cfg(feature = "oauth-server")]
    claims: Option<crate::oauth::AccessClaims>,
}

/// Auth and origin checks shared by every method.
async fn guard(shared: &Shared, headers: &HeaderMap) -> Result<Auth, Box<Response>> {
    if let Some(allowed) = &shared.config.origins {
        if let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
            if !allowed.iter().any(|o| o == origin) {
                return Err(Box::new(rpc_error(
                    StatusCode::FORBIDDEN,
                    -32000,
                    "origin not allowed",
                )));
            }
        }
    }
    #[allow(unused_mut)]
    let mut auth = Auth::default();
    #[cfg(feature = "oauth-server")]
    let oauth = shared.validator.is_some();
    #[cfg(not(feature = "oauth-server"))]
    let oauth = false;
    #[cfg(feature = "oauth-server")]
    if let Some(validator) = &shared.validator {
        auth.claims = Some(oauth_check(validator, headers).await?);
    }
    if !oauth && !shared.config.tokens.is_empty() {
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
            return Err(Box::new(r));
        }
    }
    if let Some(v) = headers.get(VERSION_HEADER).and_then(|v| v.to_str().ok()) {
        if !SUPPORTED_PROTOCOL_VERSIONS.contains(&v) {
            return Err(Box::new(rpc_error(
                StatusCode::BAD_REQUEST,
                -32600,
                "unsupported MCP-Protocol-Version",
            )));
        }
    }
    Ok(auth)
}

/// `WWW-Authenticate` value for an OAuth endpoint.
#[cfg(feature = "oauth-server")]
fn challenge(
    config: &crate::oauth::OAuthConfig,
    error: Option<(&str, String)>,
    scope: Option<&str>,
) -> HeaderValue {
    use crate::oauth::metadata::quote;
    let mut parts = Vec::new();
    if let Some((code, description)) = error {
        parts.push(format!("error={}", quote(code)));
        parts.push(format!("error_description={}", quote(&description)));
    }
    if let Some(scope) = scope {
        parts.push(format!("scope={}", quote(scope)));
    }
    parts.push(format!(
        "resource_metadata={}",
        quote(&config.metadata_url())
    ));
    HeaderValue::from_str(&format!("Bearer {}", parts.join(", ")))
        .unwrap_or_else(|_| HeaderValue::from_static("Bearer"))
}

/// Validates the bearer token of an OAuth endpoint and the scopes every
/// request needs.
#[cfg(feature = "oauth-server")]
async fn oauth_check(
    validator: &crate::oauth::JwtValidator,
    headers: &HeaderMap,
) -> Result<crate::oauth::AccessClaims, Box<Response>> {
    use crate::oauth::TokenError;
    let config = validator.config();
    let value = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    let token = value.and_then(|v| {
        let (scheme, rest) = v.split_once(' ')?;
        scheme.eq_ignore_ascii_case("bearer").then(|| rest.trim())
    });
    let Some(token) = token.filter(|t| !t.is_empty()) else {
        let mut r = rpc_error(StatusCode::UNAUTHORIZED, -32001, "unauthorized");
        r.headers_mut()
            .insert(header::WWW_AUTHENTICATE, challenge(config, None, None));
        return Err(Box::new(r));
    };
    let claims = match validator.validate(token).await {
        Ok(c) => c,
        Err(TokenError::KeysUnavailable(_)) => {
            return Err(Box::new(rpc_error(
                StatusCode::SERVICE_UNAVAILABLE,
                -32000,
                "authorization server keys unavailable",
            )));
        }
        Err(e) => {
            let mut r = rpc_error(StatusCode::UNAUTHORIZED, -32001, "invalid token");
            r.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                challenge(config, Some(("invalid_token", e.to_string())), None),
            );
            return Err(Box::new(r));
        }
    };
    if !claims.has_scopes(&config.required_scopes) {
        return Err(Box::new(insufficient_scope(
            config,
            &config.required_scopes,
        )));
    }
    Ok(claims)
}

#[cfg(feature = "oauth-server")]
fn insufficient_scope(config: &crate::oauth::OAuthConfig, needed: &[String]) -> Response {
    let mut r = rpc_error(StatusCode::FORBIDDEN, -32001, "insufficient scope");
    r.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        challenge(
            config,
            Some((
                "insufficient_scope",
                "the token lacks a required scope".into(),
            )),
            Some(&needed.join(" ")),
        ),
    );
    r
}

/// `403` when a `tools/call` in `messages` names a tool whose scopes the
/// token lacks.
#[cfg(feature = "oauth-server")]
fn check_tool_scopes(shared: &Shared, auth: &Auth, messages: &[Value]) -> Option<Response> {
    let (validator, claims) = (shared.validator.as_ref()?, auth.claims.as_ref()?);
    let config = validator.config();
    if config.tool_scopes.is_empty() {
        return None;
    }
    for m in messages {
        if m["method"] != "tools/call" {
            continue;
        }
        let tool = m["params"]["name"].as_str().unwrap_or_default();
        let needed = config.scopes_for_tool(tool);
        if !claims.has_scopes(&needed) {
            return Some(insufficient_scope(config, &needed));
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
    #[allow(unused_variables)]
    let auth = match guard(&shared, &headers).await {
        Ok(a) => a,
        Err(r) => return *r,
    };
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return rpc_error(StatusCode::BAD_REQUEST, -32700, "parse error"),
    };
    let (messages, batch) = match parsed {
        Value::Array(items) if !items.is_empty() => (items, true),
        Value::Array(_) => return rpc_error(StatusCode::BAD_REQUEST, -32600, "empty batch"),
        v => (vec![v], false),
    };
    #[cfg(feature = "oauth-server")]
    if let Some(r) = check_tool_scopes(&shared, &auth, &messages) {
        return r;
    }
    let initializing = messages.iter().any(|m| m["method"] == "initialize");

    let mut new_session = None;
    // The session the messages belong to, used for resource subscriptions.
    let mut peer = None;
    if !shared.config.stateless {
        if initializing {
            let id = shared.open_session();
            peer = Some(id.clone());
            new_session = Some(id);
        } else {
            match session_of(&headers) {
                None => {
                    return rpc_error(StatusCode::BAD_REQUEST, -32600, "missing Mcp-Session-Id")
                }
                Some(id) if !shared.touch(id) => {
                    return rpc_error(StatusCode::NOT_FOUND, -32001, "session not found")
                }
                Some(id) => peer = Some(id.to_string()),
            }
        }
    }

    let mut replies = Vec::new();
    for m in messages {
        if let Some(r) = shared.server.handle_message_from(peer.as_deref(), m).await {
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

/// `GET`: the session's server-to-client SSE stream. Notifications for the
/// session (resource updates, list changes) are sent here. A new `GET`
/// replaces the previous stream of the same session.
async fn get_endpoint(State(shared): State<Arc<Shared>>, headers: HeaderMap) -> Response {
    if let Err(r) = guard(&shared, &headers).await {
        return *r;
    }
    if shared.config.stateless {
        // No sessions, so nothing to address notifications to.
        let mut r = StatusCode::METHOD_NOT_ALLOWED.into_response();
        r.headers_mut()
            .insert(header::ALLOW, HeaderValue::from_static("POST"));
        return r;
    }
    let accepts_sse = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|a| a.contains("text/event-stream") || a.contains("*/*"));
    if !accepts_sse {
        return rpc_error(
            StatusCode::NOT_ACCEPTABLE,
            -32600,
            "GET must accept text/event-stream",
        );
    }
    let id = match session_of(&headers) {
        None => return rpc_error(StatusCode::BAD_REQUEST, -32600, "missing Mcp-Session-Id"),
        Some(id) if !shared.touch(id) => {
            return rpc_error(StatusCode::NOT_FOUND, -32001, "session not found")
        }
        Some(id) => id,
    };
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    if !shared.server.hub().attach(id, tx) {
        return rpc_error(StatusCode::NOT_FOUND, -32001, "session not found");
    }
    // The stream ends when the session is closed, evicted or replaced (the
    // sender is dropped), or when the client disconnects.
    let events = futures_util::stream::unfold(rx, |mut rx| async move {
        let msg = rx.recv().await?;
        let event = Event::default().event("message").data(msg.to_string());
        Some((Ok::<_, Infallible>(event), rx))
    });
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn delete(State(shared): State<Arc<Shared>>, headers: HeaderMap) -> Response {
    if let Err(r) = guard(&shared, &headers).await {
        return *r;
    }
    if shared.config.stateless {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    match session_of(&headers) {
        Some(id) if shared.close_session(id) => StatusCode::OK.into_response(),
        Some(_) => StatusCode::NOT_FOUND.into_response(),
        None => StatusCode::BAD_REQUEST.into_response(),
    }
}
