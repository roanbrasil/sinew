//! Resource-server side (feature `oauth-server`): [`OAuthConfig`] for
//! [`McpHttpConfig::with_oauth`](crate::McpHttpConfig::with_oauth) and the
//! [`JwtValidator`] behind it.
//!
//! Access tokens must be JWTs signed with an asymmetric key published in the
//! authorization server's JWKS (RS256/384/512, PS256/384/512, ES256/384,
//! EdDSA). A token is accepted when all of these hold:
//!
//! - the header's `alg` is in the allowed list (never `none` or an HMAC
//!   algorithm) and matches the key (`kty`, `crv`, and the JWK's `alg` when
//!   set);
//! - the signature verifies with a key from the JWKS, selected by `kid`;
//! - `iss` equals the configured issuer;
//! - `aud` (a string or an array) contains the resource URL or another
//!   configured audience;
//! - `exp` is a number in the future and `nbf`, when present, a number in
//!   the past, both with a small leeway (60 seconds by default).
//!
//! The JWKS is fetched on first use (from `jwks_uri`, or from the issuer's
//! metadata), cached for an hour, and refetched when a token names an unknown
//! `kid` (at most once per refresh interval, 30 seconds by default), so key
//! rotation works without a restart.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use jsonwebtoken::{
    jwk::{AlgorithmParameters, EllipticCurve, Jwk, PublicKeyUse},
    Algorithm, DecodingKey, Validation,
};
use serde_json::Value;

use super::metadata::{
    check_secure, fetch_authorization_server_metadata, parse_url, protected_resource_metadata_url,
    ProtectedResourceMetadata, PROTECTED_RESOURCE_WELL_KNOWN,
};

/// Asymmetric algorithms accepted by default.
const DEFAULT_ALGORITHMS: &[Algorithm] = &[
    Algorithm::RS256,
    Algorithm::RS384,
    Algorithm::RS512,
    Algorithm::PS256,
    Algorithm::PS384,
    Algorithm::PS512,
    Algorithm::ES256,
    Algorithm::ES384,
    Algorithm::EdDSA,
];

fn is_hmac(alg: Algorithm) -> bool {
    matches!(alg, Algorithm::HS256 | Algorithm::HS384 | Algorithm::HS512)
}

/// OAuth settings of an MCP endpoint: who issues its tokens, which
/// audience they must carry, and which scopes each tool needs.
///
/// ```
/// use sinew_mcp::{oauth::OAuthConfig, McpHttpConfig};
///
/// let config = McpHttpConfig::default().with_oauth(
///     OAuthConfig::new("https://mcp.example.com/mcp", "https://auth.example.com/realms/mcp")
///         .with_scopes_supported(["mcp:tools", "mcp:admin"])
///         .with_required_scopes(["mcp:tools"])
///         .with_tool_scopes("delete_order", ["mcp:admin"]),
/// );
/// ```
#[derive(Debug, Clone)]
pub struct OAuthConfig {
    pub(crate) resource: String,
    issuer: String,
    authorization_servers: Option<Vec<String>>,
    jwks_uri: Option<String>,
    audiences: Vec<String>,
    scopes_supported: Vec<String>,
    pub(crate) required_scopes: Vec<String>,
    pub(crate) tool_scopes: HashMap<String, Vec<String>>,
    algorithms: Vec<Algorithm>,
    leeway: Duration,
    jwks_ttl: Duration,
    jwks_refresh_interval: Duration,
    resource_name: Option<String>,
    resource_documentation: Option<String>,
    allow_http: bool,
    http: Option<reqwest::Client>,
}

impl OAuthConfig {
    /// `resource` is the canonical URL clients use to reach the endpoint
    /// (e.g. `https://mcp.example.com/mcp`); tokens must name it in `aud`.
    /// `issuer` is the authorization server's issuer identifier (e.g.
    /// `https://auth.example.com/realms/mcp`); tokens must carry it in `iss`.
    pub fn new(resource: impl Into<String>, issuer: impl Into<String>) -> Self {
        let resource: String = resource.into();
        Self {
            audiences: vec![resource.clone()],
            resource,
            issuer: issuer.into(),
            authorization_servers: None,
            jwks_uri: None,
            scopes_supported: Vec::new(),
            required_scopes: Vec::new(),
            tool_scopes: HashMap::new(),
            algorithms: DEFAULT_ALGORITHMS.to_vec(),
            leeway: Duration::from_secs(60),
            jwks_ttl: Duration::from_secs(3600),
            jwks_refresh_interval: Duration::from_secs(30),
            resource_name: None,
            resource_documentation: None,
            allow_http: false,
            http: None,
        }
    }

    /// Fetches keys from this URL instead of the `jwks_uri` in the issuer's
    /// metadata.
    pub fn with_jwks_uri(mut self, uri: impl Into<String>) -> Self {
        self.jwks_uri = Some(uri.into());
        self
    }

    /// Authorization servers advertised in the protected resource metadata
    /// (default: the issuer).
    pub fn with_authorization_servers<I, S>(mut self, servers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.authorization_servers = Some(servers.into_iter().map(Into::into).collect());
        self
    }

    /// Accepts tokens whose `aud` holds `audience` as well as the resource
    /// URL. Use it for providers that cannot put the resource URL in `aud`
    /// (some put a client id there); prefer the resource URL when possible.
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.audiences.push(audience.into());
        self
    }

    /// Scopes listed in the protected resource metadata (`scopes_supported`).
    /// Clients request them by default.
    pub fn with_scopes_supported<I, S>(mut self, scopes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.scopes_supported = scopes.into_iter().map(Into::into).collect();
        self
    }

    /// Scopes every request needs; `403 insufficient_scope` otherwise.
    pub fn with_required_scopes<I, S>(mut self, scopes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.required_scopes = scopes.into_iter().map(Into::into).collect();
        self
    }

    /// Scopes a `tools/call` of `tool` needs, on top of the required ones;
    /// `403 insufficient_scope` otherwise. Listing tools is not restricted.
    pub fn with_tool_scopes<I, S>(mut self, tool: impl Into<String>, scopes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tool_scopes
            .insert(tool.into(), scopes.into_iter().map(Into::into).collect());
        self
    }

    /// Restricts the accepted signature algorithms. HMAC algorithms are
    /// dropped: a JWKS publishes public keys, never shared secrets.
    pub fn with_algorithms(mut self, algorithms: impl IntoIterator<Item = Algorithm>) -> Self {
        self.algorithms = algorithms.into_iter().filter(|a| !is_hmac(*a)).collect();
        self
    }

    /// Clock skew tolerated on `exp` and `nbf` (default 60 seconds).
    pub fn with_leeway(mut self, leeway: Duration) -> Self {
        self.leeway = leeway;
        self
    }

    /// How long fetched keys are used before they are fetched again
    /// (default one hour).
    pub fn with_jwks_cache_ttl(mut self, ttl: Duration) -> Self {
        self.jwks_ttl = ttl;
        self
    }

    /// Minimum time between two JWKS fetches triggered by unknown `kid`s
    /// (default 30 seconds). Bounds the load forged tokens can cause.
    pub fn with_jwks_refresh_interval(mut self, interval: Duration) -> Self {
        self.jwks_refresh_interval = interval;
        self
    }

    /// `resource_name` in the protected resource metadata.
    pub fn with_resource_name(mut self, name: impl Into<String>) -> Self {
        self.resource_name = Some(name.into());
        self
    }

    /// `resource_documentation` in the protected resource metadata.
    pub fn with_resource_documentation(mut self, url: impl Into<String>) -> Self {
        self.resource_documentation = Some(url.into());
        self
    }

    /// Accepts plain-HTTP issuer and JWKS URLs on non-loopback hosts (for
    /// a development network such as docker compose). Loopback hosts are
    /// always accepted.
    pub fn allow_insecure_http(mut self) -> Self {
        self.allow_http = true;
        self
    }

    /// HTTP client for metadata and JWKS requests (default: Sinew's shared
    /// client).
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = Some(client);
        self
    }

    /// The resource URL.
    pub fn resource(&self) -> &str {
        &self.resource
    }

    /// The issuer identifier.
    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    /// The protected resource metadata document this endpoint serves.
    pub fn metadata(&self) -> ProtectedResourceMetadata {
        ProtectedResourceMetadata {
            resource: self.resource.clone(),
            authorization_servers: self
                .authorization_servers
                .clone()
                .unwrap_or_else(|| vec![self.issuer.clone()]),
            scopes_supported: self.scopes_supported.clone(),
            bearer_methods_supported: vec!["header".into()],
            resource_name: self.resource_name.clone(),
            resource_documentation: self.resource_documentation.clone(),
        }
    }

    /// Absolute URL of the metadata, sent in `WWW-Authenticate`.
    pub fn metadata_url(&self) -> String {
        protected_resource_metadata_url(&self.resource).unwrap_or_else(|_| {
            format!(
                "{}{PROTECTED_RESOURCE_WELL_KNOWN}",
                self.resource.trim_end_matches('/')
            )
        })
    }

    /// Router paths the metadata is served at: the RFC 9728 path for the
    /// resource (`/.well-known/oauth-protected-resource/mcp`) and the bare
    /// well-known path.
    pub(crate) fn metadata_paths(&self) -> Vec<String> {
        let mut paths = vec![PROTECTED_RESOURCE_WELL_KNOWN.to_string()];
        if let Ok(url) = parse_url("resource", &self.resource) {
            let p = url.path().trim_end_matches('/');
            if !p.is_empty() {
                paths.push(format!("{PROTECTED_RESOURCE_WELL_KNOWN}{p}"));
            }
        }
        paths
    }

    /// Scopes a `tools/call` of `tool` needs (required plus per-tool).
    pub(crate) fn scopes_for_tool(&self, tool: &str) -> Vec<String> {
        let mut s = self.required_scopes.clone();
        if let Some(extra) = self.tool_scopes.get(tool) {
            for x in extra {
                if !s.contains(x) {
                    s.push(x.clone());
                }
            }
        }
        s
    }
}

/// Why a token was rejected. The `Display` text is safe to return to the
/// client as `error_description`: it never contains the token or its claims.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TokenError {
    /// Not a JWT, an unparsable header, or a claim with the wrong type.
    #[error("malformed token")]
    Malformed,
    /// The header's `alg` is not accepted (`none`, HMAC, or not configured).
    #[error("unsupported signing algorithm")]
    UnsupportedAlgorithm,
    /// No key in the JWKS matches the token's `kid` and `alg`.
    #[error("unknown signing key")]
    UnknownKey,
    /// The signature does not verify.
    #[error("invalid signature")]
    InvalidSignature,
    /// `exp` is in the past (beyond the leeway), or missing.
    #[error("token expired")]
    Expired,
    /// `nbf` is in the future (beyond the leeway).
    #[error("token not yet valid")]
    NotYetValid,
    /// `iss` is not the configured issuer.
    #[error("wrong issuer")]
    WrongIssuer,
    /// `aud` does not name this resource.
    #[error("wrong audience")]
    WrongAudience,
    /// The JWKS could not be fetched; the token was not checked.
    #[error("signing keys unavailable: {0}")]
    KeysUnavailable(String),
}

/// What a validated access token says.
#[derive(Debug, Clone)]
pub struct AccessClaims {
    /// `sub`.
    pub subject: Option<String>,
    /// `client_id`, or `azp` when absent.
    pub client_id: Option<String>,
    /// `scope` (space-separated) or `scp` (string or array).
    pub scopes: Vec<String>,
    /// `exp` (seconds since the Unix epoch).
    pub expires_at: u64,
    /// Every claim, as sent.
    pub claims: Value,
}

impl AccessClaims {
    fn from_claims(claims: Value) -> Self {
        let str_claim = |k: &str| claims.get(k).and_then(Value::as_str).map(str::to_string);
        let mut scopes: Vec<String> = Vec::new();
        for key in ["scope", "scp"] {
            match claims.get(key) {
                Some(Value::String(s)) => scopes.extend(s.split_whitespace().map(str::to_string)),
                Some(Value::Array(items)) => {
                    scopes.extend(items.iter().filter_map(Value::as_str).map(str::to_string))
                }
                _ => {}
            }
        }
        scopes.dedup();
        Self {
            subject: str_claim("sub"),
            client_id: str_claim("client_id").or_else(|| str_claim("azp")),
            scopes,
            expires_at: claims.get("exp").and_then(Value::as_u64).unwrap_or(0),
            claims,
        }
    }

    /// True when every scope in `required` was granted.
    pub fn has_scopes<S: AsRef<str>>(&self, required: &[S]) -> bool {
        required
            .iter()
            .all(|r| self.scopes.iter().any(|s| s == r.as_ref()))
    }
}

#[derive(Default)]
struct JwksCache {
    keys: Vec<Jwk>,
    fetched: Option<Instant>,
    last_attempt: Option<Instant>,
    /// `jwks_uri` discovered from the issuer's metadata.
    discovered_uri: Option<String>,
}

/// Validates JWT access tokens for one resource against the issuer's JWKS.
/// [`McpHttpConfig::with_oauth`](crate::McpHttpConfig::with_oauth) builds
/// one per router; build your own to protect other routes with the same
/// rules.
pub struct JwtValidator {
    config: OAuthConfig,
    http: reqwest::Client,
    cache: tokio::sync::RwLock<JwksCache>,
    refresh: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for JwtValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JwtValidator")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Which key family and curve an algorithm needs.
fn key_matches(jwk: &Jwk, alg: Algorithm) -> bool {
    if matches!(jwk.common.public_key_use, Some(PublicKeyUse::Encryption)) {
        return false;
    }
    if let Some(key_alg) = jwk.common.key_algorithm {
        // Both enums serialize to the JOSE name (`RS256`, ...).
        let a = serde_json::to_value(key_alg).ok();
        let b = serde_json::to_value(alg).ok();
        if a.is_none() || a != b {
            return false;
        }
    }
    match (&jwk.algorithm, alg) {
        (
            AlgorithmParameters::RSA(_),
            Algorithm::RS256
            | Algorithm::RS384
            | Algorithm::RS512
            | Algorithm::PS256
            | Algorithm::PS384
            | Algorithm::PS512,
        ) => true,
        (AlgorithmParameters::EllipticCurve(p), Algorithm::ES256) => p.curve == EllipticCurve::P256,
        (AlgorithmParameters::EllipticCurve(p), Algorithm::ES384) => p.curve == EllipticCurve::P384,
        (AlgorithmParameters::OctetKeyPair(p), Algorithm::EdDSA) => {
            p.curve == EllipticCurve::Ed25519
        }
        _ => false,
    }
}

impl JwtValidator {
    /// A validator for `config`. Keys are fetched on the first
    /// [`validate`](Self::validate).
    pub fn new(config: OAuthConfig) -> Self {
        let http = config
            .http
            .clone()
            .unwrap_or_else(sinew_core::providers::shared_client);
        Self {
            config,
            http,
            cache: tokio::sync::RwLock::new(JwksCache::default()),
            refresh: tokio::sync::Mutex::new(()),
        }
    }

    /// The configuration.
    pub fn config(&self) -> &OAuthConfig {
        &self.config
    }

    /// Checks `token` (without the `Bearer ` prefix) and returns its claims.
    pub async fn validate(&self, token: &str) -> Result<AccessClaims, TokenError> {
        let header = jsonwebtoken::decode_header(token).map_err(|_| TokenError::Malformed)?;
        let alg = header.alg;
        if is_hmac(alg) || !self.config.algorithms.contains(&alg) {
            return Err(TokenError::UnsupportedAlgorithm);
        }
        let kid = header.kid.as_deref();
        let mut candidates = self.candidates(kid, alg).await?;
        if candidates.is_empty() && kid.is_some() {
            // Unknown `kid`: the issuer may have rotated its keys.
            self.refresh_keys(false).await?;
            candidates = self.candidates(kid, alg).await?;
        }
        if candidates.is_empty() {
            return Err(TokenError::UnknownKey);
        }

        let mut validation = Validation::new(alg);
        validation.algorithms = vec![alg];
        validation.leeway = self.config.leeway.as_secs();
        validation.validate_exp = true;
        validation.validate_nbf = true;
        // `iss` and `aud` are checked below, on the raw claims.
        validation.validate_aud = false;
        validation.set_required_spec_claims(&["exp"]);

        let mut last = TokenError::InvalidSignature;
        for jwk in &candidates {
            let key = match DecodingKey::from_jwk(jwk) {
                Ok(k) => k,
                Err(_) => continue,
            };
            match jsonwebtoken::decode::<Value>(token, &key, &validation) {
                Ok(data) => return self.check_claims(data.claims),
                Err(e) => {
                    use jsonwebtoken::errors::ErrorKind as K;
                    last = match e.kind() {
                        K::InvalidSignature | K::InvalidEcdsaKey | K::InvalidRsaKey(_) => {
                            TokenError::InvalidSignature
                        }
                        K::ExpiredSignature | K::MissingRequiredClaim(_) => TokenError::Expired,
                        K::ImmatureSignature => TokenError::NotYetValid,
                        K::InvalidAlgorithm => TokenError::UnsupportedAlgorithm,
                        _ => TokenError::Malformed,
                    };
                    if last != TokenError::InvalidSignature {
                        // The signature verified; the claims are wrong.
                        return Err(last);
                    }
                }
            }
        }
        Err(last)
    }

    /// Checks that `jsonwebtoken` leaves to the caller or treats leniently:
    /// claim types, issuer and audience.
    fn check_claims(&self, claims: Value) -> Result<AccessClaims, TokenError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let leeway = self.config.leeway.as_secs();
        // Numeric date claims must be numbers; a string would be skipped by
        // the generic validation.
        let numeric = |k: &str| match claims.get(k) {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .filter(|n| n.is_finite())
                .map(Some)
                .ok_or(TokenError::Malformed),
        };
        let exp = numeric("exp")?.ok_or(TokenError::Expired)?;
        if exp + (leeway as f64) < now as f64 {
            return Err(TokenError::Expired);
        }
        if let Some(nbf) = numeric("nbf")? {
            if nbf > (now + leeway) as f64 {
                return Err(TokenError::NotYetValid);
            }
        }
        numeric("iat")?;
        match claims.get("iss") {
            Some(Value::String(iss)) if *iss == self.config.issuer => {}
            _ => return Err(TokenError::WrongIssuer),
        }
        let accepted = |a: &Value| {
            a.as_str()
                .is_some_and(|a| self.config.audiences.iter().any(|x| x == a))
        };
        let aud_ok = match claims.get("aud") {
            Some(Value::String(_)) => claims.get("aud").is_some_and(accepted),
            Some(Value::Array(items)) => items.iter().any(accepted),
            _ => false,
        };
        if !aud_ok {
            return Err(TokenError::WrongAudience);
        }
        Ok(AccessClaims::from_claims(claims))
    }

    /// Keys that may have signed a token with this `kid` and `alg`. Loads
    /// the JWKS when nothing is cached yet or the cache is stale.
    async fn candidates(&self, kid: Option<&str>, alg: Algorithm) -> Result<Vec<Jwk>, TokenError> {
        let stale = {
            let c = self.cache.read().await;
            c.fetched
                .is_none_or(|t| t.elapsed() >= self.config.jwks_ttl)
        };
        if stale {
            // A failed periodic refresh keeps using the keys already cached.
            let has_keys = self.cache.read().await.fetched.is_some();
            if let Err(e) = self.refresh_keys(true).await {
                if !has_keys {
                    return Err(e);
                }
            }
        }
        let c = self.cache.read().await;
        Ok(c.keys
            .iter()
            .filter(|k| match kid {
                Some(kid) => k.common.key_id.as_deref() == Some(kid),
                None => true,
            })
            .filter(|k| key_matches(k, alg))
            .cloned()
            .collect())
    }

    /// Fetches the JWKS again. `due` is a TTL refresh; otherwise (unknown
    /// `kid`) at most one fetch happens per refresh interval.
    async fn refresh_keys(&self, due: bool) -> Result<(), TokenError> {
        let _one_at_a_time = self.refresh.lock().await;
        {
            let c = self.cache.read().await;
            let recent = c
                .last_attempt
                .is_some_and(|t| t.elapsed() < self.config.jwks_refresh_interval);
            let fresh = c
                .fetched
                .is_some_and(|t| t.elapsed() < self.config.jwks_ttl);
            if (due && fresh) || recent {
                // Another task refreshed meanwhile, or we tried too recently.
                return if c.fetched.is_some() {
                    Ok(())
                } else {
                    Err(TokenError::KeysUnavailable(
                        "previous fetch failed; retrying after the refresh interval".into(),
                    ))
                };
            }
        }
        self.cache.write().await.last_attempt = Some(Instant::now());
        let keys = self.fetch_keys().await?;
        let mut c = self.cache.write().await;
        c.keys = keys;
        c.fetched = Some(Instant::now());
        Ok(())
    }

    async fn jwks_uri(&self) -> Result<String, TokenError> {
        if let Some(uri) = &self.config.jwks_uri {
            return Ok(uri.clone());
        }
        if let Some(uri) = self.cache.read().await.discovered_uri.clone() {
            return Ok(uri);
        }
        let meta = fetch_authorization_server_metadata(
            &self.http,
            &self.config.issuer,
            self.config.allow_http,
        )
        .await
        .map_err(|e| TokenError::KeysUnavailable(e.to_string()))?;
        let uri = meta.jwks_uri.ok_or_else(|| {
            TokenError::KeysUnavailable("the issuer's metadata has no jwks_uri".into())
        })?;
        self.cache.write().await.discovered_uri = Some(uri.clone());
        Ok(uri)
    }

    async fn fetch_keys(&self) -> Result<Vec<Jwk>, TokenError> {
        let uri = self.jwks_uri().await?;
        check_secure("jwks", &uri, self.config.allow_http)
            .map_err(|e| TokenError::KeysUnavailable(e.to_string()))?;
        let unavailable = |e: String| TokenError::KeysUnavailable(format!("{uri}: {e}"));
        let resp = self
            .http
            .get(&uri)
            .header(reqwest::header::ACCEPT, "application/json")
            .send()
            .await
            .map_err(|e| unavailable(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(unavailable(format!("HTTP {}", resp.status())));
        }
        let body: Value = resp.json().await.map_err(|e| unavailable(e.to_string()))?;
        let items = body
            .get("keys")
            .and_then(Value::as_array)
            .ok_or_else(|| unavailable("no `keys` array".into()))?;
        // Keys of unsupported types are skipped rather than failing the set.
        Ok(items
            .iter()
            .filter_map(|k| serde_json::from_value::<Jwk>(k.clone()).ok())
            .filter(|k| !matches!(k.algorithm, AlgorithmParameters::OctetKey(_)))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scopes_from_scope_and_scp() {
        let c = AccessClaims::from_claims(json!({"scope": "a b", "exp": 1}));
        assert_eq!(c.scopes, ["a", "b"]);
        assert!(c.has_scopes(&["b"]));
        assert!(!c.has_scopes(&["c"]));
        let c = AccessClaims::from_claims(json!({"scp": ["x", "y"], "azp": "app"}));
        assert_eq!(c.scopes, ["x", "y"]);
        assert_eq!(c.client_id.as_deref(), Some("app"));
    }

    #[test]
    fn metadata_and_paths() {
        let c = OAuthConfig::new("https://mcp.example.com/mcp", "https://auth.example.com")
            .with_scopes_supported(["mcp:tools"]);
        let m = c.metadata();
        assert_eq!(m.authorization_servers, ["https://auth.example.com"]);
        assert_eq!(m.bearer_methods_supported, ["header"]);
        assert_eq!(
            c.metadata_url(),
            "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        );
        assert_eq!(
            c.metadata_paths(),
            [
                "/.well-known/oauth-protected-resource",
                "/.well-known/oauth-protected-resource/mcp"
            ]
        );
    }

    #[test]
    fn hmac_is_never_allowed() {
        let c = OAuthConfig::new("https://r", "https://i")
            .with_algorithms([Algorithm::HS256, Algorithm::ES256]);
        assert_eq!(c.algorithms, [Algorithm::ES256]);
    }

    #[test]
    fn tool_scopes_add_to_required() {
        let c = OAuthConfig::new("https://r", "https://i")
            .with_required_scopes(["a"])
            .with_tool_scopes("t", ["a", "b"]);
        assert_eq!(c.scopes_for_tool("t"), ["a", "b"]);
        assert_eq!(c.scopes_for_tool("u"), ["a"]);
    }
}
