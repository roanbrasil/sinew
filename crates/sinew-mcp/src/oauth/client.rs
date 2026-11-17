//! Client side (feature `oauth-client`): the authorization code flow with
//! PKCE for MCP HTTP clients.
//!
//! [`OAuthClient::discover`] asks the MCP server for its protected resource
//! metadata (from the `401` challenge, or the RFC 9728 well-known URL), reads
//! the authorization server's metadata, and registers a client (RFC 7591)
//! unless a client id is configured or stored. Redirect handling stays with
//! the application:
//!
//! ```no_run
//! # async fn demo() -> sinew_mcp::Result<()> {
//! use sinew_mcp::{oauth::{OAuthClient, OAuthClientConfig}, McpClient};
//!
//! let url = "https://mcp.example.com/mcp";
//! let oauth = OAuthClient::discover(
//!     url,
//!     OAuthClientConfig::new("http://127.0.0.1:8765/callback").with_client_name("my agent"),
//! )
//! .await?;
//! let (authorize_url, state) = oauth.begin()?;
//! // Open `authorize_url` in a browser; the authorization server redirects
//! // to the callback with `?code=...&state=...`.
//! # let (code, returned_state) = (String::new(), String::new());
//! oauth.complete(&code, &returned_state).await?;
//! let client = McpClient::connect_http_with_oauth(url, oauth.clone()).await?;
//! # Ok(()) }
//! ```
//!
//! The access token is refreshed before it expires and after a `401`, with
//! the refresh token. Tokens and the client registration live in a
//! [`TokenStore`] ([`InMemoryTokenStore`] by default); implement it to keep
//! them in a keychain or an encrypted file. Use one store per MCP server.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use reqwest::{header, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{
    constant_time_eq,
    metadata::{
        check_secure, fetch_authorization_server_metadata, is_loopback, parse_url,
        protected_resource_metadata_url, AuthorizationServerMetadata, BearerChallenge,
        ProtectedResourceMetadata, PROTECTED_RESOURCE_WELL_KNOWN,
    },
};
use crate::{error::Result, McpError};

/// Pending authorizations older than this are discarded.
const PENDING_TTL: Duration = Duration::from_secs(600);
/// At most this many authorizations wait for `complete` at once.
const MAX_PENDING: usize = 16;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn proto(msg: impl Into<String>) -> McpError {
    McpError::Protocol(msg.into())
}

/// `n` random bytes from the operating system, base64url without padding.
fn random_token(n: usize) -> Result<String> {
    let mut buf = vec![0u8; n];
    ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut buf)
        .map_err(|_| proto("the system random number generator failed"))?;
    Ok(URL_SAFE_NO_PAD.encode(buf))
}

/// A PKCE pair (RFC 7636) with the `S256` method.
#[derive(Clone)]
pub struct Pkce {
    verifier: String,
    challenge: String,
}

impl std::fmt::Debug for Pkce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pkce")
            .field("verifier", &"<redacted>")
            .field("challenge", &self.challenge)
            .finish()
    }
}

impl Pkce {
    /// A new pair with a 256-bit random verifier (43 characters).
    pub fn generate() -> Result<Self> {
        Ok(Self::from_verifier(random_token(32)?))
    }

    /// The pair for a given verifier: `challenge = BASE64URL(SHA256(verifier))`.
    pub fn from_verifier(verifier: impl Into<String>) -> Self {
        let verifier = verifier.into();
        let digest = ring::digest::digest(&ring::digest::SHA256, verifier.as_bytes());
        Self {
            challenge: URL_SAFE_NO_PAD.encode(digest.as_ref()),
            verifier,
        }
    }

    /// The secret sent with the code exchange.
    pub fn verifier(&self) -> &str {
        &self.verifier
    }

    /// The value sent in the authorization request.
    pub fn challenge(&self) -> &str {
        &self.challenge
    }

    /// Always `S256`.
    pub fn method(&self) -> &'static str {
        "S256"
    }
}

/// Tokens from the token endpoint. `Debug` redacts them.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenSet {
    /// The access token sent as `Authorization: Bearer`.
    pub access_token: String,
    /// `Bearer`.
    pub token_type: String,
    /// The refresh token, when the server issued one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Expiry of the access token (seconds since the Unix epoch), from
    /// `expires_in`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    /// Granted scopes (space-separated), when the server said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

impl std::fmt::Debug for TokenSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenSet")
            .field("access_token", &"<redacted>")
            .field("token_type", &self.token_type)
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field("expires_at", &self.expires_at)
            .field("scope", &self.scope)
            .finish()
    }
}

impl TokenSet {
    /// True when the access token expires within `skew` (false when the
    /// expiry is unknown).
    pub fn expires_within(&self, skew: Duration) -> bool {
        self.expires_at
            .is_some_and(|at| now_secs().saturating_add(skew.as_secs()) >= at)
    }
}

/// The client's identity at the authorization server, configured or
/// obtained through dynamic registration. `Debug` redacts the secret.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientRegistration {
    /// Issuer the client is registered with.
    pub issuer: String,
    /// `client_id`.
    pub client_id: String,
    /// `client_secret` (confidential clients only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    /// `none`, `client_secret_basic` or `client_secret_post`.
    pub token_endpoint_auth_method: String,
    /// Redirect URI registered with the client.
    pub redirect_uri: String,
}

impl std::fmt::Debug for ClientRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientRegistration")
            .field("issuer", &self.issuer)
            .field("client_id", &self.client_id)
            .field(
                "client_secret",
                &self.client_secret.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "token_endpoint_auth_method",
                &self.token_endpoint_auth_method,
            )
            .field("redirect_uri", &self.redirect_uri)
            .finish()
    }
}

/// Where an [`OAuthClient`] keeps its tokens and client registration.
/// Use one store per MCP server.
#[async_trait]
pub trait TokenStore: Send + Sync {
    /// The stored tokens.
    async fn load(&self) -> Result<Option<TokenSet>>;
    /// Replaces the stored tokens.
    async fn save(&self, tokens: &TokenSet) -> Result<()>;
    /// Forgets the tokens.
    async fn clear(&self) -> Result<()>;
    /// The stored client registration (default: none, so the client
    /// registers again on every [`OAuthClient::discover`]).
    async fn load_client(&self) -> Result<Option<ClientRegistration>> {
        Ok(None)
    }
    /// Stores a client registration (default: discarded).
    async fn save_client(&self, _registration: &ClientRegistration) -> Result<()> {
        Ok(())
    }
}

/// A [`TokenStore`] in process memory; everything is lost on exit.
#[derive(Debug, Default)]
pub struct InMemoryTokenStore {
    tokens: Mutex<Option<TokenSet>>,
    client: Mutex<Option<ClientRegistration>>,
}

impl InMemoryTokenStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl TokenStore for InMemoryTokenStore {
    async fn load(&self) -> Result<Option<TokenSet>> {
        Ok(self.tokens.lock().unwrap().clone())
    }
    async fn save(&self, tokens: &TokenSet) -> Result<()> {
        *self.tokens.lock().unwrap() = Some(tokens.clone());
        Ok(())
    }
    async fn clear(&self) -> Result<()> {
        *self.tokens.lock().unwrap() = None;
        Ok(())
    }
    async fn load_client(&self) -> Result<Option<ClientRegistration>> {
        Ok(self.client.lock().unwrap().clone())
    }
    async fn save_client(&self, registration: &ClientRegistration) -> Result<()> {
        *self.client.lock().unwrap() = Some(registration.clone());
        Ok(())
    }
}

/// Supplies access tokens to [`McpClient::connect_http_with_oauth`](crate::McpClient::connect_http_with_oauth).
/// [`OAuthClient`] implements it; implement it yourself for other grant
/// types (client credentials, token exchange).
#[async_trait]
pub trait TokenProvider: Send + Sync {
    /// The access token for the next request.
    async fn access_token(&self) -> Result<String>;

    /// Called when the server answered `401` to `rejected`, with the
    /// `WWW-Authenticate` value. Return `true` when a different token is now
    /// available; the request is then retried once.
    async fn on_unauthorized(&self, _rejected: &str, _challenge: Option<&str>) -> Result<bool> {
        Ok(false)
    }
}

#[async_trait]
impl<T: TokenProvider + ?Sized> TokenProvider for Arc<T> {
    async fn access_token(&self) -> Result<String> {
        (**self).access_token().await
    }
    async fn on_unauthorized(&self, rejected: &str, challenge: Option<&str>) -> Result<bool> {
        (**self).on_unauthorized(rejected, challenge).await
    }
}

/// Settings for [`OAuthClient::discover`].
#[derive(Clone)]
pub struct OAuthClientConfig {
    redirect_uri: String,
    client_name: String,
    client_id: Option<String>,
    client_secret: Option<String>,
    scopes: Option<Vec<String>>,
    store: Arc<dyn TokenStore>,
    http: Option<reqwest::Client>,
    allow_http: bool,
    refresh_skew: Duration,
}

impl std::fmt::Debug for OAuthClientConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthClientConfig")
            .field("redirect_uri", &self.redirect_uri)
            .field("client_name", &self.client_name)
            .field("client_id", &self.client_id)
            .field(
                "client_secret",
                &self.client_secret.as_ref().map(|_| "<redacted>"),
            )
            .field("scopes", &self.scopes)
            .field("allow_http", &self.allow_http)
            .field("refresh_skew", &self.refresh_skew)
            .finish_non_exhaustive()
    }
}

impl OAuthClientConfig {
    /// `redirect_uri` is where the authorization server sends the user back
    /// with the code. It must be HTTPS or a loopback `http` URL.
    pub fn new(redirect_uri: impl Into<String>) -> Self {
        Self {
            redirect_uri: redirect_uri.into(),
            client_name: "Sinew MCP client".into(),
            client_id: None,
            client_secret: None,
            scopes: None,
            store: Arc::new(InMemoryTokenStore::new()),
            http: None,
            allow_http: false,
            refresh_skew: Duration::from_secs(30),
        }
    }

    /// `client_name` sent with dynamic registration.
    pub fn with_client_name(mut self, name: impl Into<String>) -> Self {
        self.client_name = name.into();
        self
    }

    /// A pre-registered client id; dynamic registration is skipped.
    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = Some(client_id.into());
        self
    }

    /// The secret of a pre-registered confidential client (sent with HTTP
    /// Basic authentication).
    pub fn with_client_secret(mut self, secret: impl Into<String>) -> Self {
        self.client_secret = Some(secret.into());
        self
    }

    /// Scopes to request. Default: the `scope` of the server's challenge,
    /// else the resource's `scopes_supported`, else none.
    pub fn with_scopes<I, S>(mut self, scopes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.scopes = Some(scopes.into_iter().map(Into::into).collect());
        self
    }

    /// Where tokens and the client registration are kept (default
    /// [`InMemoryTokenStore`]).
    pub fn with_store(mut self, store: Arc<dyn TokenStore>) -> Self {
        self.store = store;
        self
    }

    /// HTTP client for the discovery, registration and token requests
    /// (default: Sinew's shared client).
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = Some(client);
        self
    }

    /// Accepts plain-HTTP authorization server URLs on non-loopback hosts
    /// (development networks only).
    pub fn allow_insecure_http(mut self) -> Self {
        self.allow_http = true;
        self
    }

    /// Refresh the access token this long before it expires (default 30
    /// seconds).
    pub fn with_refresh_skew(mut self, skew: Duration) -> Self {
        self.refresh_skew = skew;
        self
    }
}

struct Pending {
    state: String,
    pkce: Pkce,
    created: Instant,
}

struct Inner {
    config: OAuthClientConfig,
    http: reqwest::Client,
    resource: String,
    resource_metadata: ProtectedResourceMetadata,
    server: AuthorizationServerMetadata,
    authorization_endpoint: String,
    token_endpoint: String,
    registration: ClientRegistration,
    scopes: Vec<String>,
    pending: Mutex<Vec<Pending>>,
    refresh_lock: tokio::sync::Mutex<()>,
}

/// The OAuth 2.1 authorization code flow with PKCE against one MCP server.
/// Cheap to clone; clones share tokens and pending authorizations.
#[derive(Clone)]
pub struct OAuthClient {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for OAuthClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthClient")
            .field("resource", &self.inner.resource)
            .field("issuer", &self.inner.server.issuer)
            .field("client_id", &self.inner.registration.client_id)
            .finish_non_exhaustive()
    }
}

/// True when `url` is `resource` or below it (same origin, path prefix on a
/// segment boundary).
fn is_within(resource: &Url, url: &Url) -> bool {
    if resource.scheme() != url.scheme()
        || !resource
            .host_str()
            .zip(url.host_str())
            .is_some_and(|(a, b)| a.eq_ignore_ascii_case(b))
        || resource.port_or_known_default() != url.port_or_known_default()
    {
        return false;
    }
    let rp = resource.path().trim_end_matches('/');
    let up = url.path().trim_end_matches('/');
    up == rp || up.starts_with(&format!("{rp}/"))
}

/// Minimal `application/x-www-form-urlencoded` encoding of one component.
fn form_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'*' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Error text of a failed OAuth endpoint response (`error`,
/// `error_description`), never the body otherwise.
async fn endpoint_error(what: &str, resp: reqwest::Response) -> McpError {
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or(Value::Null);
    let code = body.get("error").and_then(Value::as_str);
    let desc = body.get("error_description").and_then(Value::as_str);
    match (code, desc) {
        (Some(c), Some(d)) => proto(format!("{what} failed (HTTP {status}): {c}: {d}")),
        (Some(c), None) => proto(format!("{what} failed (HTTP {status}): {c}")),
        _ => proto(format!("{what} failed (HTTP {status})")),
    }
}

impl OAuthClient {
    /// Discovers how `mcp_url` is protected and prepares the flow: protected
    /// resource metadata (from the `WWW-Authenticate` challenge of an
    /// unauthenticated request, else the RFC 9728 well-known URLs), the first
    /// authorization server's metadata (RFC 8414 or OpenID Connect
    /// discovery), and the client registration (configured, stored, or
    /// registered dynamically per RFC 7591).
    pub async fn discover(mcp_url: &str, config: OAuthClientConfig) -> Result<Self> {
        let http = config
            .http
            .clone()
            .unwrap_or_else(sinew_core::providers::shared_client);
        let mcp = parse_url("MCP server", mcp_url)?;
        let redirect = parse_url("redirect", &config.redirect_uri)?;
        if redirect.scheme() != "https" && !is_loopback(&redirect) {
            return Err(proto(format!(
                "redirect URI `{}` must use https or a loopback host",
                config.redirect_uri
            )));
        }

        // 1. The challenge of an unauthenticated request.
        let probe = http
            .post(mcp_url)
            .header(header::ACCEPT, "application/json, text/event-stream")
            .header(header::CONTENT_TYPE, "application/json")
            .body(json!({"jsonrpc": "2.0", "id": 0, "method": "ping"}).to_string())
            .send()
            .await?;
        let challenge = if probe.status() == StatusCode::UNAUTHORIZED {
            probe
                .headers()
                .get_all(header::WWW_AUTHENTICATE)
                .iter()
                .filter_map(|v| v.to_str().ok())
                .find_map(BearerChallenge::parse)
        } else {
            None
        };

        // 2. Protected resource metadata.
        let mut candidates = Vec::new();
        if let Some(url) = challenge.as_ref().and_then(|c| c.resource_metadata()) {
            candidates.push(url.to_string());
        } else {
            candidates.push(protected_resource_metadata_url(mcp_url)?);
            let mut root = mcp.clone();
            root.set_path(PROTECTED_RESOURCE_WELL_KNOWN);
            root.set_query(None);
            let root = root.to_string();
            if !candidates.contains(&root) {
                candidates.push(root);
            }
        }
        let mut resource_metadata = None;
        let mut tried = Vec::new();
        for url in &candidates {
            parse_url("resource metadata", url)?;
            match http
                .get(url)
                .header(header::ACCEPT, "application/json")
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => {
                    match r.json::<ProtectedResourceMetadata>().await {
                        Ok(m) => {
                            resource_metadata = Some(m);
                            break;
                        }
                        Err(e) => tried.push(format!("{url}: {e}")),
                    }
                }
                Ok(r) => tried.push(format!("{url}: HTTP {}", r.status())),
                Err(e) => tried.push(format!("{url}: {e}")),
            }
        }
        let resource_metadata = resource_metadata.ok_or_else(|| {
            proto(format!(
                "no protected resource metadata for {mcp_url} ({})",
                tried.join("; ")
            ))
        })?;
        // The metadata must describe the server we talk to (RFC 9728 7.3).
        let resource_url = parse_url("resource", &resource_metadata.resource)?;
        if !is_within(&resource_url, &mcp) {
            return Err(proto(format!(
                "protected resource metadata names resource `{}`, which does not cover {mcp_url}",
                resource_metadata.resource
            )));
        }
        let resource = resource_metadata.resource.clone();

        // 3. Authorization server metadata.
        let issuer = resource_metadata
            .authorization_servers
            .first()
            .ok_or_else(|| proto("protected resource metadata lists no authorization server"))?
            .clone();
        let server = fetch_authorization_server_metadata(&http, &issuer, config.allow_http).await?;
        let endpoint = |what: &str, v: &Option<String>| -> Result<String> {
            let v = v
                .as_deref()
                .ok_or_else(|| proto(format!("authorization server metadata has no {what}")))?;
            check_secure(what, v, config.allow_http)?;
            Ok(v.to_string())
        };
        let authorization_endpoint =
            endpoint("authorization_endpoint", &server.authorization_endpoint)?;
        let token_endpoint = endpoint("token_endpoint", &server.token_endpoint)?;
        if !server.code_challenge_methods_supported.is_empty()
            && !server
                .code_challenge_methods_supported
                .iter()
                .any(|m| m == "S256")
        {
            return Err(proto(
                "the authorization server does not support PKCE with S256",
            ));
        }

        let scopes = config
            .scopes
            .clone()
            .or_else(|| {
                challenge
                    .as_ref()
                    .and_then(|c| c.scope())
                    .map(|s| s.split_whitespace().map(str::to_string).collect())
            })
            .unwrap_or_else(|| resource_metadata.scopes_supported.clone());

        // 4. Client registration.
        let registration = match &config.client_id {
            Some(id) => ClientRegistration {
                issuer: server.issuer.clone(),
                client_id: id.clone(),
                client_secret: config.client_secret.clone(),
                token_endpoint_auth_method: if config.client_secret.is_some() {
                    "client_secret_basic".into()
                } else {
                    "none".into()
                },
                redirect_uri: config.redirect_uri.clone(),
            },
            None => match config.store.load_client().await? {
                Some(r) if r.issuer == server.issuer && r.redirect_uri == config.redirect_uri => r,
                _ => {
                    let r = register(&http, &server, &config, &scopes).await?;
                    config.store.save_client(&r).await?;
                    r
                }
            },
        };

        Ok(Self {
            inner: Arc::new(Inner {
                http,
                resource,
                resource_metadata,
                server,
                authorization_endpoint,
                token_endpoint,
                registration,
                scopes,
                pending: Mutex::new(Vec::new()),
                refresh_lock: tokio::sync::Mutex::new(()),
                config,
            }),
        })
    }

    /// The resource the tokens are requested for (RFC 8707 `resource`).
    pub fn resource(&self) -> &str {
        &self.inner.resource
    }

    /// The MCP server's protected resource metadata.
    pub fn resource_metadata(&self) -> &ProtectedResourceMetadata {
        &self.inner.resource_metadata
    }

    /// The authorization server's metadata.
    pub fn authorization_server(&self) -> &AuthorizationServerMetadata {
        &self.inner.server
    }

    /// The client registration in use.
    pub fn registration(&self) -> &ClientRegistration {
        &self.inner.registration
    }

    /// Scopes requested in the authorization request.
    pub fn scopes(&self) -> &[String] {
        &self.inner.scopes
    }

    /// Starts an authorization: returns the URL to open in the user's
    /// browser and the `state` it carries. Keep nothing else; the PKCE
    /// verifier stays inside the client until [`complete`](Self::complete)
    /// (pending authorizations expire after ten minutes).
    pub fn begin(&self) -> Result<(String, String)> {
        let pkce = Pkce::generate()?;
        let state = random_token(32)?;
        let mut url = Url::parse(&self.inner.authorization_endpoint)
            .map_err(|e| proto(format!("invalid authorization_endpoint: {e}")))?;
        {
            let mut q = url.query_pairs_mut();
            q.append_pair("response_type", "code")
                .append_pair("client_id", &self.inner.registration.client_id)
                .append_pair("redirect_uri", &self.inner.registration.redirect_uri)
                .append_pair("code_challenge", pkce.challenge())
                .append_pair("code_challenge_method", pkce.method())
                .append_pair("state", &state)
                .append_pair("resource", &self.inner.resource);
            if !self.inner.scopes.is_empty() {
                q.append_pair("scope", &self.inner.scopes.join(" "));
            }
        }
        let mut pending = self.inner.pending.lock().unwrap();
        pending.retain(|p| p.created.elapsed() < PENDING_TTL);
        if pending.len() >= MAX_PENDING {
            pending.remove(0);
        }
        pending.push(Pending {
            state: state.clone(),
            pkce,
            created: Instant::now(),
        });
        Ok((url.to_string(), state))
    }

    /// Exchanges the authorization `code` for tokens and stores them.
    /// `state` must be the one returned by [`begin`](Self::begin) for this
    /// authorization; each state is accepted once.
    pub async fn complete(&self, code: &str, state: &str) -> Result<TokenSet> {
        let pkce = {
            let mut pending = self.inner.pending.lock().unwrap();
            pending.retain(|p| p.created.elapsed() < PENDING_TTL);
            // Compare every entry in constant time; no early exit on a match.
            let mut found = None;
            for (i, p) in pending.iter().enumerate() {
                if constant_time_eq(p.state.as_bytes(), state.as_bytes()) {
                    found = Some(i);
                }
            }
            found.map(|i| pending.remove(i).pkce)
        }
        .ok_or_else(|| proto("unknown or expired authorization state"))?;
        let reg = &self.inner.registration;
        let tokens = self
            .token_request(
                vec![
                    ("grant_type", "authorization_code".to_string()),
                    ("code", code.to_string()),
                    ("redirect_uri", reg.redirect_uri.clone()),
                    ("code_verifier", pkce.verifier().to_string()),
                ],
                false,
            )
            .await?;
        self.inner.config.store.save(&tokens).await?;
        Ok(tokens)
    }

    /// Like [`complete`](Self::complete), from the full redirect URL the
    /// authorization server sent the browser to (`...?code=...&state=...`).
    /// An `error` parameter becomes an error.
    pub async fn complete_redirect(&self, redirect_url: &str) -> Result<TokenSet> {
        let url =
            Url::parse(redirect_url).map_err(|e| proto(format!("invalid redirect URL: {e}")))?;
        let param = |k: &str| {
            url.query_pairs()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.into_owned())
        };
        if let Some(err) = param("error") {
            let desc = param("error_description").unwrap_or_default();
            return Err(proto(
                format!("authorization failed: {err} {desc}").trim_end(),
            ));
        }
        let code = param("code").ok_or_else(|| proto("redirect URL has no `code`"))?;
        let state = param("state").ok_or_else(|| proto("redirect URL has no `state`"))?;
        self.complete(&code, &state).await
    }

    /// The stored tokens.
    pub async fn tokens(&self) -> Result<Option<TokenSet>> {
        self.inner.config.store.load().await
    }

    /// Forgets the stored tokens.
    pub async fn sign_out(&self) -> Result<()> {
        self.inner.config.store.clear().await
    }

    /// Uses the refresh token now and stores the result. A rejected refresh
    /// token (`invalid_grant`) clears the stored tokens, so the next
    /// authorization starts clean.
    pub async fn refresh(&self) -> Result<TokenSet> {
        let _one_at_a_time = self.inner.refresh_lock.lock().await;
        self.refresh_locked().await
    }

    async fn refresh_locked(&self) -> Result<TokenSet> {
        let store = &self.inner.config.store;
        let current = store
            .load()
            .await?
            .ok_or_else(|| proto("no tokens to refresh: run the authorization flow"))?;
        let refresh_token = current
            .refresh_token
            .clone()
            .ok_or_else(|| proto("no refresh token: run the authorization flow again"))?;
        let mut tokens = match self
            .token_request(
                vec![
                    ("grant_type", "refresh_token".to_string()),
                    ("refresh_token", refresh_token.clone()),
                ],
                true,
            )
            .await
        {
            Ok(t) => t,
            Err(e) => {
                if e.to_string().contains("): invalid_grant") {
                    store.clear().await?;
                }
                return Err(e);
            }
        };
        if tokens.refresh_token.is_none() {
            tokens.refresh_token = Some(refresh_token);
        }
        store.save(&tokens).await?;
        Ok(tokens)
    }

    /// POSTs to the token endpoint with client authentication and the
    /// `resource` parameter. `refreshing` marks a refresh request, whose
    /// failure is reported as such.
    async fn token_request(
        &self,
        mut form: Vec<(&'static str, String)>,
        refreshing: bool,
    ) -> Result<TokenSet> {
        let reg = &self.inner.registration;
        form.push(("resource", self.inner.resource.clone()));
        let mut req = self
            .inner
            .http
            .post(&self.inner.token_endpoint)
            .header(header::ACCEPT, "application/json");
        match (reg.token_endpoint_auth_method.as_str(), &reg.client_secret) {
            ("client_secret_post", Some(secret)) => {
                form.push(("client_id", reg.client_id.clone()));
                form.push(("client_secret", secret.clone()));
            }
            ("client_secret_basic", Some(secret)) => {
                // RFC 6749 2.3.1: both parts form-encoded before Basic.
                req = req.basic_auth(form_encode(&reg.client_id), Some(form_encode(secret)));
            }
            _ => form.push(("client_id", reg.client_id.clone())),
        }
        let body = form
            .iter()
            .map(|(k, v)| format!("{k}={}", form_encode(v)))
            .collect::<Vec<_>>()
            .join("&");
        let resp = req
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await?;
        let what = if refreshing {
            "token refresh"
        } else {
            "code exchange"
        };
        if !resp.status().is_success() {
            return Err(endpoint_error(what, resp).await);
        }
        #[derive(Deserialize)]
        struct Raw {
            access_token: String,
            token_type: String,
            #[serde(default)]
            expires_in: Option<u64>,
            #[serde(default)]
            refresh_token: Option<String>,
            #[serde(default)]
            scope: Option<String>,
        }
        let raw: Raw = resp
            .json()
            .await
            .map_err(|_| proto(format!("{what}: malformed token response")))?;
        if !raw.token_type.eq_ignore_ascii_case("bearer") {
            return Err(proto(format!(
                "{what}: unsupported token type `{}`",
                raw.token_type
            )));
        }
        Ok(TokenSet {
            access_token: raw.access_token,
            token_type: "Bearer".into(),
            refresh_token: raw.refresh_token,
            expires_at: raw.expires_in.map(|s| now_secs().saturating_add(s)),
            scope: raw.scope,
        })
    }
}

/// Dynamic client registration (RFC 7591) as a public client.
async fn register(
    http: &reqwest::Client,
    server: &AuthorizationServerMetadata,
    config: &OAuthClientConfig,
    scopes: &[String],
) -> Result<ClientRegistration> {
    let endpoint = server.registration_endpoint.as_deref().ok_or_else(|| {
        proto(
            "the authorization server does not support dynamic client registration; \
             configure a client id with OAuthClientConfig::with_client_id",
        )
    })?;
    check_secure("registration_endpoint", endpoint, config.allow_http)?;
    let mut body = json!({
        "client_name": config.client_name,
        "redirect_uris": [config.redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    });
    if !scopes.is_empty() {
        body["scope"] = json!(scopes.join(" "));
    }
    let resp = http
        .post(endpoint)
        .header(header::ACCEPT, "application/json")
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(endpoint_error("client registration", resp).await);
    }
    let v: Value = resp
        .json()
        .await
        .map_err(|_| proto("client registration: malformed response"))?;
    let client_id = v
        .get("client_id")
        .and_then(Value::as_str)
        .ok_or_else(|| proto("client registration: no client_id in the response"))?
        .to_string();
    let client_secret = v
        .get("client_secret")
        .and_then(Value::as_str)
        .map(str::to_string);
    let method = v
        .get("token_endpoint_auth_method")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| {
            if client_secret.is_some() {
                "client_secret_basic".into()
            } else {
                "none".into()
            }
        });
    Ok(ClientRegistration {
        issuer: server.issuer.clone(),
        client_id,
        client_secret,
        token_endpoint_auth_method: method,
        redirect_uri: config.redirect_uri.clone(),
    })
}

#[async_trait]
impl TokenProvider for OAuthClient {
    /// The stored access token, refreshed first when it expires within the
    /// configured skew and a refresh token is available.
    async fn access_token(&self) -> Result<String> {
        let store = &self.inner.config.store;
        let tokens = store.load().await?.ok_or_else(|| {
            proto("no access token: run the authorization flow (OAuthClient::begin / complete)")
        })?;
        if !tokens.expires_within(self.inner.config.refresh_skew) || tokens.refresh_token.is_none()
        {
            return Ok(tokens.access_token);
        }
        let _one_at_a_time = self.inner.refresh_lock.lock().await;
        // Another task may have refreshed while we waited.
        if let Some(t) = store.load().await? {
            if !t.expires_within(self.inner.config.refresh_skew) {
                return Ok(t.access_token);
            }
        }
        Ok(self.refresh_locked().await?.access_token)
    }

    /// Refreshes once when the rejected token is still the stored one and a
    /// refresh token exists; reports `true` when another task already
    /// replaced it.
    async fn on_unauthorized(&self, rejected: &str, _challenge: Option<&str>) -> Result<bool> {
        let _one_at_a_time = self.inner.refresh_lock.lock().await;
        let Some(current) = self.inner.config.store.load().await? else {
            return Ok(false);
        };
        if !constant_time_eq(current.access_token.as_bytes(), rejected.as_bytes()) {
            return Ok(true);
        }
        if current.refresh_token.is_none() {
            return Ok(false);
        }
        self.refresh_locked().await?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_matches_rfc_7636_appendix_b() {
        let p = Pkce::from_verifier("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk");
        assert_eq!(p.challenge(), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
        assert_eq!(p.method(), "S256");
    }

    #[test]
    fn generated_pkce_is_valid() {
        let a = Pkce::generate().unwrap();
        let b = Pkce::generate().unwrap();
        // 32 random bytes: 43 base64url characters, within RFC 7636's 43..=128.
        assert_eq!(a.verifier().len(), 43);
        assert!(a
            .verifier()
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'));
        assert_ne!(a.verifier(), b.verifier());
        assert_eq!(Pkce::from_verifier(a.verifier()).challenge(), a.challenge());
        assert!(!format!("{a:?}").contains(a.verifier()));
    }

    #[test]
    fn debug_redacts_secrets() {
        let t = TokenSet {
            access_token: "at-secret".into(),
            token_type: "Bearer".into(),
            refresh_token: Some("rt-secret".into()),
            expires_at: None,
            scope: None,
        };
        let s = format!("{t:?}");
        assert!(!s.contains("at-secret") && !s.contains("rt-secret"));
        let r = ClientRegistration {
            issuer: "i".into(),
            client_id: "c".into(),
            client_secret: Some("cs-secret".into()),
            token_endpoint_auth_method: "client_secret_basic".into(),
            redirect_uri: "http://127.0.0.1/cb".into(),
        };
        assert!(!format!("{r:?}").contains("cs-secret"));
    }

    #[test]
    fn resource_covers_url() {
        let r = Url::parse("https://mcp.example.com/mcp").unwrap();
        assert!(is_within(
            &r,
            &Url::parse("https://mcp.example.com/mcp").unwrap()
        ));
        assert!(is_within(
            &r,
            &Url::parse("https://MCP.example.com:443/mcp/").unwrap()
        ));
        assert!(is_within(
            &r,
            &Url::parse("https://mcp.example.com/mcp/x").unwrap()
        ));
        assert!(!is_within(
            &r,
            &Url::parse("https://mcp.example.com/mcpx").unwrap()
        ));
        assert!(!is_within(
            &r,
            &Url::parse("http://mcp.example.com/mcp").unwrap()
        ));
        assert!(!is_within(
            &r,
            &Url::parse("https://evil.example.com/mcp").unwrap()
        ));
        let root = Url::parse("https://mcp.example.com").unwrap();
        assert!(is_within(
            &root,
            &Url::parse("https://mcp.example.com/mcp").unwrap()
        ));
    }

    #[test]
    fn form_encoding() {
        assert_eq!(form_encode("a b&c=d/é"), "a+b%26c%3Dd%2F%C3%A9");
    }

    #[test]
    fn expiry_with_skew() {
        let mut t = TokenSet {
            access_token: String::new(),
            token_type: "Bearer".into(),
            refresh_token: None,
            expires_at: Some(now_secs() + 10),
            scope: None,
        };
        assert!(t.expires_within(Duration::from_secs(30)));
        assert!(!t.expires_within(Duration::from_secs(0)));
        t.expires_at = None;
        assert!(!t.expires_within(Duration::from_secs(30)));
    }
}
