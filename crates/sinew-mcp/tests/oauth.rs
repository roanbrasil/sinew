//! OAuth 2.1 end to end, offline: an in-process fake authorization server
//! (axum) signs JWTs with the test keys in `fixtures/oauth/` (generated for
//! these tests only), and an MCP server on 127.0.0.1:0 validates them.
#![cfg(all(feature = "oauth-client", feature = "oauth-server"))]

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicI64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{Form, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use jsonwebtoken::{Algorithm, EncodingKey, Header};
use serde_json::{json, Value};
use sinew_core::{FnTool, ToolBox};
use sinew_mcp::{
    oauth::{
        BearerChallenge, InMemoryTokenStore, JwtValidator, OAuthClient, OAuthClientConfig,
        OAuthConfig, Pkce, TokenError, TokenProvider, TokenStore,
    },
    McpClient, McpHttpConfig, McpServer,
};

const RSA1: &str = include_str!("fixtures/oauth/test-rsa1.pem");
const RSA2: &str = include_str!("fixtures/oauth/test-rsa2.pem");
const EC1: &str = include_str!("fixtures/oauth/test-ec1.pem");
const JWK: &str = include_str!("fixtures/oauth/test-jwk.json");
const REDIRECT: &str = "http://127.0.0.1:9/callback";

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// One signing key of the fake authorization server.
struct Key {
    kid: &'static str,
    alg: Algorithm,
    encoding: EncodingKey,
    jwk: Value,
}

fn keys() -> Vec<Key> {
    let comps: Value = serde_json::from_str(JWK).unwrap();
    let rsa = |kid: &'static str, pem: &str, name: &str| Key {
        kid,
        alg: Algorithm::RS256,
        encoding: EncodingKey::from_rsa_pem(pem.as_bytes()).unwrap(),
        jwk: json!({"kty": "RSA", "kid": kid, "use": "sig", "alg": "RS256",
                    "n": comps[name]["n"], "e": comps[name]["e"]}),
    };
    vec![
        rsa("rsa-1", RSA1, "rsa1"),
        rsa("rsa-2", RSA2, "rsa2"),
        Key {
            kid: "ec-1",
            alg: Algorithm::ES256,
            encoding: EncodingKey::from_ec_pem(EC1.as_bytes()).unwrap(),
            jwk: json!({"kty": "EC", "crv": "P-256", "kid": "ec-1", "use": "sig", "alg": "ES256",
                        "x": comps["ec1"]["x"], "y": comps["ec1"]["y"]}),
        },
    ]
}

struct Grant {
    client_id: String,
    redirect_uri: String,
    challenge: String,
    resource: String,
    scope: String,
}

/// A minimal authorization server: metadata, JWKS, dynamic registration,
/// authorization endpoint (consent granted immediately), token endpoint with
/// PKCE and refresh-token rotation.
struct FakeAs {
    issuer: String,
    keys: Vec<Key>,
    /// Kids published in the JWKS.
    published: Mutex<Vec<&'static str>>,
    /// Kid used to sign new tokens.
    signing: Mutex<&'static str>,
    dcr: bool,
    jwks_fetches: AtomicUsize,
    refreshes: AtomicUsize,
    /// Lifetime written into new JWTs (negative: already expired).
    jwt_ttl: AtomicI64,
    /// `expires_in` in token responses.
    advertised_ttl: AtomicI64,
    clients: Mutex<HashMap<String, Vec<String>>>,
    codes: Mutex<HashMap<String, Grant>>,
    refresh_tokens: Mutex<HashMap<String, (String, String, String)>>,
    counter: AtomicUsize,
}

type Shared = Arc<FakeAs>;

impl FakeAs {
    fn key(&self, kid: &str) -> &Key {
        self.keys.iter().find(|k| k.kid == kid).unwrap()
    }

    /// Signs `claims` with key `kid`.
    fn sign(&self, kid: &str, claims: &Value) -> String {
        let key = self.key(kid);
        let mut h = Header::new(key.alg);
        h.kid = Some(kid.to_string());
        jsonwebtoken::encode(&h, claims, &key.encoding).unwrap()
    }

    fn next(&self, prefix: &str) -> String {
        format!("{prefix}-{}", self.counter.fetch_add(1, Ordering::SeqCst))
    }

    fn issue(&self, client_id: &str, resource: &str, scope: &str) -> Value {
        let claims = json!({
            "iss": self.issuer, "aud": resource, "sub": "user-1", "client_id": client_id,
            "scope": scope, "iat": now(), "exp": now() + self.jwt_ttl.load(Ordering::SeqCst),
            "jti": self.next("jti"),
        });
        let kid = *self.signing.lock().unwrap();
        let refresh = self.next("rt");
        self.refresh_tokens.lock().unwrap().insert(
            refresh.clone(),
            (client_id.into(), resource.into(), scope.into()),
        );
        json!({
            "access_token": self.sign(kid, &claims),
            "token_type": "Bearer",
            "expires_in": self.advertised_ttl.load(Ordering::SeqCst),
            "refresh_token": refresh,
            "scope": scope,
        })
    }
}

fn oauth_error(code: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": code}))).into_response()
}

async fn as_metadata(State(s): State<Shared>) -> Json<Value> {
    let mut m = json!({
        "issuer": s.issuer,
        "authorization_endpoint": format!("{}/authorize", s.issuer),
        "token_endpoint": format!("{}/token", s.issuer),
        "jwks_uri": format!("{}/jwks", s.issuer),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
    });
    if s.dcr {
        m["registration_endpoint"] = json!(format!("{}/register", s.issuer));
    }
    Json(m)
}

async fn jwks(State(s): State<Shared>) -> Json<Value> {
    s.jwks_fetches.fetch_add(1, Ordering::SeqCst);
    let published = s.published.lock().unwrap().clone();
    let mut keys: Vec<Value> = published.iter().map(|k| s.key(k).jwk.clone()).collect();
    // Keys the validator must skip: an encryption key and an unknown type.
    keys.push(json!({"kty": "RSA", "kid": "enc", "use": "enc", "n": "AQAB", "e": "AQAB"}));
    keys.push(json!({"kty": "weird", "kid": "w"}));
    Json(json!({ "keys": keys }))
}

async fn register(State(s): State<Shared>, Json(body): Json<Value>) -> Response {
    let uris: Vec<String> = body["redirect_uris"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if uris.is_empty() || body["token_endpoint_auth_method"] != "none" {
        return oauth_error("invalid_client_metadata");
    }
    let id = s.next("client");
    s.clients.lock().unwrap().insert(id.clone(), uris.clone());
    (
        StatusCode::CREATED,
        Json(json!({"client_id": id, "redirect_uris": uris, "token_endpoint_auth_method": "none"})),
    )
        .into_response()
}

async fn authorize(State(s): State<Shared>, Query(q): Query<HashMap<String, String>>) -> Response {
    let p = |k: &str| q.get(k).cloned().unwrap_or_default();
    let registered = s
        .clients
        .lock()
        .unwrap()
        .get(&p("client_id"))
        .is_some_and(|uris| uris.contains(&p("redirect_uri")));
    if !registered
        || p("response_type") != "code"
        || p("code_challenge_method") != "S256"
        || p("code_challenge").is_empty()
        || p("resource").is_empty()
    {
        return oauth_error("invalid_request");
    }
    let code = s.next("code");
    s.codes.lock().unwrap().insert(
        code.clone(),
        Grant {
            client_id: p("client_id"),
            redirect_uri: p("redirect_uri"),
            challenge: p("code_challenge"),
            resource: p("resource"),
            scope: p("scope"),
        },
    );
    let location = format!("{}?code={code}&state={}", p("redirect_uri"), p("state"));
    (StatusCode::FOUND, [(header::LOCATION, location)]).into_response()
}

async fn token(State(s): State<Shared>, Form(f): Form<HashMap<String, String>>) -> Response {
    let p = |k: &str| f.get(k).cloned().unwrap_or_default();
    match p("grant_type").as_str() {
        "authorization_code" => {
            let Some(g) = s.codes.lock().unwrap().remove(&p("code")) else {
                return oauth_error("invalid_grant");
            };
            let pkce_ok = Pkce::from_verifier(p("code_verifier")).challenge() == g.challenge;
            if !pkce_ok
                || g.client_id != p("client_id")
                || g.redirect_uri != p("redirect_uri")
                || g.resource != p("resource")
            {
                return oauth_error("invalid_grant");
            }
            Json(s.issue(&g.client_id, &g.resource, &g.scope)).into_response()
        }
        "refresh_token" => {
            // Rotation: a refresh token works once.
            let Some((client, resource, scope)) =
                s.refresh_tokens.lock().unwrap().remove(&p("refresh_token"))
            else {
                return oauth_error("invalid_grant");
            };
            if client != p("client_id") || resource != p("resource") {
                return oauth_error("invalid_grant");
            }
            s.refreshes.fetch_add(1, Ordering::SeqCst);
            Json(s.issue(&client, &resource, &scope)).into_response()
        }
        _ => oauth_error("unsupported_grant_type"),
    }
}

async fn spawn_as(dcr: bool) -> Shared {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let issuer = format!("http://{}", listener.local_addr().unwrap());
    let state = Arc::new(FakeAs {
        issuer,
        keys: keys(),
        published: Mutex::new(vec!["rsa-1", "ec-1"]),
        signing: Mutex::new("rsa-1"),
        dcr,
        jwks_fetches: AtomicUsize::new(0),
        refreshes: AtomicUsize::new(0),
        jwt_ttl: AtomicI64::new(3600),
        advertised_ttl: AtomicI64::new(3600),
        clients: Mutex::new(HashMap::new()),
        codes: Mutex::new(HashMap::new()),
        refresh_tokens: Mutex::new(HashMap::new()),
        counter: AtomicUsize::new(0),
    });
    let app = Router::new()
        .route("/.well-known/oauth-authorization-server", get(as_metadata))
        .route("/jwks", get(jwks))
        .route("/register", post(register))
        .route("/authorize", get(authorize))
        .route("/token", post(token))
        .with_state(state.clone());
    tokio::spawn(async move { axum::serve(listener, app).await });
    state
}

fn tools() -> ToolBox {
    let upper = |a: Value| Ok(json!(a["text"].as_str().unwrap_or("").to_uppercase()));
    let schema = json!({"type": "object", "properties": {"text": {"type": "string"}}});
    ToolBox::new()
        .with(FnTool::new(
            "upper",
            "Uppercase `text`",
            schema.clone(),
            upper,
        ))
        .with(FnTool::new(
            "lower",
            "Lowercase `text`",
            schema,
            |a: Value| Ok(json!(a["text"].as_str().unwrap_or("").to_lowercase())),
        ))
}

/// Binds the MCP server first so its resource URL is known, then applies
/// `configure` to the OAuth settings. Returns the MCP URL.
async fn spawn_mcp(
    issuer: &str,
    stateless: bool,
    configure: impl FnOnce(OAuthConfig) -> OAuthConfig,
) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let oauth =
        configure(OAuthConfig::new(&url, issuer).with_jwks_refresh_interval(Duration::ZERO));
    let mut config = McpHttpConfig::default().with_oauth(oauth);
    if stateless {
        config = config.stateless();
    }
    let server = McpServer::new("svc", "1.0").with_tools(tools());
    tokio::spawn(server.serve_http_with(listener, config, std::future::pending()));
    url
}

fn claims(fa: &FakeAs, aud: &str) -> Value {
    json!({"iss": fa.issuer, "aud": aud, "sub": "u", "exp": now() + 300, "iat": now()})
}

fn rpc(id: u64, method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

fn initialize() -> Value {
    rpc(
        1,
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "t", "version": "1"}}),
    )
}

async fn send(url: &str, token: Option<&str>, body: &Value) -> reqwest::Response {
    let mut req = reqwest::Client::new()
        .post(url)
        .header("accept", "application/json, text/event-stream")
        .json(body);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    req.send().await.unwrap()
}

fn challenge_of(r: &reqwest::Response) -> BearerChallenge {
    BearerChallenge::parse(r.headers()["www-authenticate"].to_str().unwrap()).unwrap()
}

/// `header.payload.signature` with a hand-written header.
fn forged(header: Value, claims: &Value, signature: &str) -> String {
    let enc = |v: &Value| URL_SAFE_NO_PAD.encode(v.to_string());
    format!("{}.{}.{signature}", enc(&header), enc(claims))
}

#[tokio::test]
async fn serves_metadata_and_challenges_with_401() {
    let fa = spawn_as(true).await;
    let url = spawn_mcp(&fa.issuer, false, |c| {
        c.with_scopes_supported(["mcp:tools"])
            .with_resource_name("Test MCP")
    })
    .await;
    let base = url.trim_end_matches("/mcp");
    let expected_meta = format!("{base}/.well-known/oauth-protected-resource/mcp");

    for path in [
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-protected-resource",
    ] {
        let r = reqwest::get(format!("{base}{path}")).await.unwrap();
        assert_eq!(r.status(), 200, "{path}");
        let m: Value = r.json().await.unwrap();
        assert_eq!(m["resource"], url);
        assert_eq!(m["authorization_servers"], json!([fa.issuer]));
        assert_eq!(m["scopes_supported"], json!(["mcp:tools"]));
        assert_eq!(m["bearer_methods_supported"], json!(["header"]));
        assert_eq!(m["resource_name"], "Test MCP");
    }

    // No token: 401 pointing at the metadata, without an error code.
    let r = send(&url, None, &initialize()).await;
    assert_eq!(r.status(), 401);
    let c = challenge_of(&r);
    assert_eq!(c.resource_metadata(), Some(expected_meta.as_str()));
    assert_eq!(c.error(), None);

    // A token that is not a JWT: invalid_token.
    let r = send(&url, Some("not-a-jwt"), &initialize()).await;
    assert_eq!(r.status(), 401);
    let c = challenge_of(&r);
    assert_eq!(c.error(), Some("invalid_token"));
    assert_eq!(c.resource_metadata(), Some(expected_meta.as_str()));

    // A different scheme is not a bearer token.
    let r = reqwest::Client::new()
        .post(&url)
        .header("authorization", "Basic dXNlcjpwYXNz")
        .json(&initialize())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);

    // A valid token opens a session; health stays public.
    let token = fa.sign("rsa-1", &claims(&fa, &url));
    let r = send(&url, Some(&token), &initialize()).await;
    assert_eq!(r.status(), 200);
    assert!(r.headers().contains_key("mcp-session-id"));
    assert_eq!(
        reqwest::get(format!("{base}/health"))
            .await
            .unwrap()
            .status(),
        200
    );
}

#[tokio::test]
async fn static_bearer_tokens_are_not_accepted_on_an_oauth_endpoint() {
    let fa = spawn_as(true).await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let config = McpHttpConfig::default()
        .with_bearer_token("static-secret")
        .with_oauth(OAuthConfig::new(&url, &fa.issuer));
    let server = McpServer::new("svc", "1.0").with_tools(tools());
    tokio::spawn(server.serve_http_with(listener, config, std::future::pending()));
    let r = send(&url, Some("static-secret"), &initialize()).await;
    assert_eq!(r.status(), 401);
    let token = fa.sign("rsa-1", &claims(&fa, &url));
    assert_eq!(send(&url, Some(&token), &initialize()).await.status(), 200);
}

#[tokio::test]
async fn validator_checks_signature_issuer_audience_and_times() {
    let fa = spawn_as(true).await;
    let resource = "https://mcp.example.com/mcp";
    let v = JwtValidator::new(
        OAuthConfig::new(resource, &fa.issuer)
            .with_leeway(Duration::from_secs(30))
            .with_jwks_refresh_interval(Duration::ZERO),
    );
    let base = claims(&fa, resource);
    let with = |patch: Value| {
        let mut c = base.clone();
        for (k, val) in patch.as_object().unwrap() {
            if val.is_null() {
                c.as_object_mut().unwrap().remove(k);
            } else {
                c[k] = val.clone();
            }
        }
        c
    };

    // Valid RS256 and ES256 tokens; claims are exposed.
    let ok = v
        .validate(&fa.sign("rsa-1", &with(json!({"scope": "a b", "client_id": "app"}))))
        .await
        .unwrap();
    assert_eq!(ok.subject.as_deref(), Some("u"));
    assert_eq!(ok.client_id.as_deref(), Some("app"));
    assert_eq!(ok.scopes, ["a", "b"]);
    v.validate(&fa.sign("ec-1", &base)).await.unwrap();
    // `aud` as an array containing the resource.
    v.validate(&fa.sign("rsa-1", &with(json!({"aud": ["other", resource]}))))
        .await
        .unwrap();
    // Within the leeway.
    v.validate(&fa.sign("rsa-1", &with(json!({"exp": now() - 10}))))
        .await
        .unwrap();

    let err = |token: String| {
        let v = &v;
        async move { v.validate(&token).await.unwrap_err() }
    };
    assert_eq!(
        err(fa.sign("rsa-1", &with(json!({"exp": now() - 120})))).await,
        TokenError::Expired
    );
    assert_eq!(
        err(fa.sign("rsa-1", &with(json!({"exp": null})))).await,
        TokenError::Expired
    );
    assert_eq!(
        err(fa.sign("rsa-1", &with(json!({"nbf": now() + 600})))).await,
        TokenError::NotYetValid
    );
    // Claims with the wrong type are not skipped.
    assert_eq!(
        err(fa.sign("rsa-1", &with(json!({"nbf": "later"})))).await,
        TokenError::Malformed
    );
    assert_eq!(
        err(fa.sign("rsa-1", &with(json!({"exp": "never"})))).await,
        TokenError::Expired
    );
    assert_eq!(
        err(fa.sign(
            "rsa-1",
            &with(json!({"aud": "https://other.example.com/mcp"}))
        ))
        .await,
        TokenError::WrongAudience
    );
    assert_eq!(
        err(fa.sign("rsa-1", &with(json!({"aud": null})))).await,
        TokenError::WrongAudience
    );
    assert_eq!(
        err(fa.sign("rsa-1", &with(json!({"iss": "https://evil.example.com"})))).await,
        TokenError::WrongIssuer
    );
    assert_eq!(
        err(fa.sign("rsa-1", &with(json!({"iss": [fa.issuer.clone()]})))).await,
        TokenError::WrongIssuer
    );

    // Signed by rsa-2 but claiming kid rsa-1.
    let real = fa.sign("rsa-2", &base);
    let sig = real.rsplit('.').next().unwrap();
    let swapped = forged(
        json!({"alg": "RS256", "kid": "rsa-1", "typ": "JWT"}),
        &base,
        sig,
    );
    assert_eq!(err(swapped).await, TokenError::InvalidSignature);
    // Tampered payload.
    let good = fa.sign("rsa-1", &base);
    let sig = good.rsplit('.').next().unwrap();
    let tampered = forged(
        json!({"alg": "RS256", "kid": "rsa-1", "typ": "JWT"}),
        &with(json!({"sub": "admin"})),
        sig,
    );
    assert_eq!(err(tampered).await, TokenError::InvalidSignature);

    // `alg: none` and HMAC are refused before any key lookup.
    let none = forged(json!({"alg": "none", "typ": "JWT"}), &base, "");
    assert!(matches!(
        err(none).await,
        TokenError::Malformed | TokenError::UnsupportedAlgorithm
    ));
    let mut h = Header::new(Algorithm::HS256);
    h.kid = Some("rsa-1".into());
    let hmac = jsonwebtoken::encode(&h, &base, &EncodingKey::from_secret(b"guess")).unwrap();
    assert_eq!(err(hmac).await, TokenError::UnsupportedAlgorithm);
    // RS256 header naming the EC key: no compatible key.
    let wrong_family = forged(json!({"alg": "RS256", "kid": "ec-1"}), &base, "AAAA");
    assert_eq!(err(wrong_family).await, TokenError::UnknownKey);
    assert_eq!(err("a.b".into()).await, TokenError::Malformed);

    // Restricting algorithms rejects the others.
    let es_only = JwtValidator::new(
        OAuthConfig::new(resource, &fa.issuer).with_algorithms([Algorithm::ES256]),
    );
    assert_eq!(
        es_only
            .validate(&fa.sign("rsa-1", &base))
            .await
            .unwrap_err(),
        TokenError::UnsupportedAlgorithm
    );

    // Extra audience (providers that put a client id in `aud`).
    let extra = JwtValidator::new(OAuthConfig::new(resource, &fa.issuer).with_audience("mcp-api"));
    extra
        .validate(&fa.sign("rsa-1", &with(json!({"aud": "mcp-api"}))))
        .await
        .unwrap();
}

#[tokio::test]
async fn jwks_is_cached_and_refetched_for_unknown_kids() {
    let fa = spawn_as(true).await;
    let resource = "https://mcp.example.com/mcp";
    let v = JwtValidator::new(
        OAuthConfig::new(resource, &fa.issuer).with_jwks_refresh_interval(Duration::ZERO),
    );
    let c = claims(&fa, resource);
    for _ in 0..3 {
        v.validate(&fa.sign("rsa-1", &c)).await.unwrap();
    }
    assert_eq!(fa.jwks_fetches.load(Ordering::SeqCst), 1, "cached");

    // The issuer rotates to rsa-2: unknown until it is published.
    let rotated = fa.sign("rsa-2", &c);
    assert_eq!(
        v.validate(&rotated).await.unwrap_err(),
        TokenError::UnknownKey
    );
    fa.published.lock().unwrap().push("rsa-2");
    v.validate(&rotated).await.unwrap();
    let fetches = fa.jwks_fetches.load(Ordering::SeqCst);
    v.validate(&rotated).await.unwrap();
    assert_eq!(fa.jwks_fetches.load(Ordering::SeqCst), fetches);

    // With a refresh interval, unknown kids cannot force a fetch per token.
    let throttled = JwtValidator::new(OAuthConfig::new(resource, &fa.issuer));
    throttled.validate(&fa.sign("rsa-1", &c)).await.unwrap();
    let before = fa.jwks_fetches.load(Ordering::SeqCst);
    for i in 0..5 {
        let t = forged(
            json!({"alg": "RS256", "kid": format!("bogus-{i}")}),
            &c,
            "AAAA",
        );
        assert_eq!(
            throttled.validate(&t).await.unwrap_err(),
            TokenError::UnknownKey
        );
    }
    assert!(fa.jwks_fetches.load(Ordering::SeqCst) <= before + 1);

    // Keys come from `jwks_uri` when configured; an unreachable JWKS is
    // reported as unavailable (503 at the endpoint), not as a bad token.
    let dead = JwtValidator::new(
        OAuthConfig::new(resource, &fa.issuer).with_jwks_uri("http://127.0.0.1:9/jwks"),
    );
    assert!(matches!(
        dead.validate(&fa.sign("rsa-1", &c)).await.unwrap_err(),
        TokenError::KeysUnavailable(_)
    ));
    let direct = JwtValidator::new(
        OAuthConfig::new(resource, &fa.issuer).with_jwks_uri(format!("{}/jwks", fa.issuer)),
    );
    direct.validate(&fa.sign("ec-1", &c)).await.unwrap();
    // Plain http to a non-loopback issuer is refused.
    let insecure = JwtValidator::new(OAuthConfig::new(resource, "http://auth.example.com"));
    assert!(matches!(
        insecure.validate(&fa.sign("rsa-1", &c)).await.unwrap_err(),
        TokenError::KeysUnavailable(_)
    ));
}

#[tokio::test]
async fn unreachable_jwks_answers_503() {
    let fa = spawn_as(true).await;
    let url = spawn_mcp(&fa.issuer, true, |c| {
        c.with_jwks_uri("http://127.0.0.1:9/jwks")
    })
    .await;
    let token = fa.sign("rsa-1", &claims(&fa, &url));
    assert_eq!(send(&url, Some(&token), &initialize()).await.status(), 503);
}

#[tokio::test]
async fn scopes_are_enforced_per_request_and_per_tool() {
    let fa = spawn_as(true).await;
    let url = spawn_mcp(&fa.issuer, true, |c| {
        c.with_required_scopes(["mcp:tools"])
            .with_tool_scopes("upper", ["mcp:write"])
    })
    .await;
    let token = |scope: &str| {
        let mut c = claims(&fa, &url);
        c["scope"] = json!(scope);
        fa.sign("rsa-1", &c)
    };
    let call = |name: &str| {
        rpc(
            3,
            "tools/call",
            json!({"name": name, "arguments": {"text": "Hi"}}),
        )
    };

    // Missing a required scope: 403 for every request.
    let r = send(&url, Some(&token("other")), &initialize()).await;
    assert_eq!(r.status(), 403);
    let c = challenge_of(&r);
    assert_eq!(c.error(), Some("insufficient_scope"));
    assert_eq!(c.scope(), Some("mcp:tools"));
    assert!(c.resource_metadata().is_some());

    let read = token("mcp:tools");
    assert_eq!(send(&url, Some(&read), &initialize()).await.status(), 200);
    let r = send(&url, Some(&read), &rpc(2, "tools/list", json!({}))).await;
    assert_eq!(r.status(), 200);
    assert_eq!(
        r.json::<Value>().await.unwrap()["result"]["tools"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    // `lower` needs only the required scope; `upper` also needs mcp:write.
    assert_eq!(send(&url, Some(&read), &call("lower")).await.status(), 200);
    let r = send(&url, Some(&read), &call("upper")).await;
    assert_eq!(r.status(), 403);
    assert_eq!(challenge_of(&r).scope(), Some("mcp:tools mcp:write"));
    // In a batch too.
    let batch = json!([call("lower"), call("upper")]);
    assert_eq!(send(&url, Some(&read), &batch).await.status(), 403);

    let write = token("mcp:tools mcp:write");
    let r = send(&url, Some(&write), &call("upper")).await;
    assert_eq!(r.status(), 200);
    assert!(r.text().await.unwrap().contains("HI"));
}

/// Opens the authorization URL like a browser whose user consents at once,
/// and returns the redirect the authorization server answered with.
async fn consent(authorize_url: &str) -> String {
    let http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let r = http.get(authorize_url).send().await.unwrap();
    assert_eq!(r.status(), 302, "authorization request rejected");
    r.headers()["location"].to_str().unwrap().to_string()
}

fn query_param(url: &str, name: &str) -> Option<String> {
    reqwest::Url::parse(url)
        .unwrap()
        .query_pairs()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.into_owned())
}

#[tokio::test]
async fn client_discovers_registers_authorizes_and_calls_tools() {
    let fa = spawn_as(true).await;
    let url = spawn_mcp(&fa.issuer, false, |c| {
        c.with_scopes_supported(["mcp:tools"])
            .with_required_scopes(["mcp:tools"])
    })
    .await;
    let store = Arc::new(InMemoryTokenStore::new());
    let config = OAuthClientConfig::new(REDIRECT)
        .with_client_name("sinew test")
        .with_store(store.clone());
    let oauth = OAuthClient::discover(&url, config.clone()).await.unwrap();
    assert_eq!(oauth.resource(), url);
    assert_eq!(oauth.authorization_server().issuer, fa.issuer);
    assert_eq!(oauth.scopes(), ["mcp:tools"]);
    let client_id = oauth.registration().client_id.clone();
    assert!(client_id.starts_with("client-"));
    // The registration is stored and reused.
    let again = OAuthClient::discover(&url, config).await.unwrap();
    assert_eq!(again.registration().client_id, client_id);
    assert_eq!(fa.clients.lock().unwrap().len(), 1);

    // Not authorized yet: connecting fails without contacting the AS.
    assert!(McpClient::connect_http_with_oauth(&url, oauth.clone())
        .await
        .is_err());

    let (authorize_url, state) = oauth.begin().unwrap();
    assert_eq!(
        query_param(&authorize_url, "state").as_deref(),
        Some(state.as_str())
    );
    assert_eq!(
        query_param(&authorize_url, "code_challenge_method").as_deref(),
        Some("S256")
    );
    assert_eq!(
        query_param(&authorize_url, "resource").as_deref(),
        Some(url.as_str())
    );
    assert_eq!(
        query_param(&authorize_url, "scope").as_deref(),
        Some("mcp:tools")
    );
    assert_eq!(query_param(&authorize_url, "client_id"), Some(client_id));
    assert!(query_param(&authorize_url, "code_verifier").is_none());

    let redirect = consent(&authorize_url).await;
    let tokens = oauth.complete_redirect(&redirect).await.unwrap();
    assert!(tokens.refresh_token.is_some());
    assert_eq!(store.load().await.unwrap(), Some(tokens.clone()));
    // A state is accepted once.
    assert!(oauth.complete_redirect(&redirect).await.is_err());

    let client = McpClient::connect_http_with_oauth(&url, oauth.clone())
        .await
        .unwrap();
    let r = client
        .call_tool("upper", json!({"text": "oauth"}))
        .await
        .unwrap();
    assert!(r.to_json().to_string().contains("OAUTH"));
    client.close().await;
}

#[tokio::test]
async fn state_and_pkce_mismatches_are_rejected() {
    let fa = spawn_as(true).await;
    let url = spawn_mcp(&fa.issuer, true, |c| c).await;
    let oauth = OAuthClient::discover(&url, OAuthClientConfig::new(REDIRECT))
        .await
        .unwrap();
    let (first, _s1) = oauth.begin().unwrap();
    let (_second, s2) = oauth.begin().unwrap();
    let code = query_param(&consent(&first).await, "code").unwrap();
    // The code of the first authorization with the second's state (and so
    // the second's PKCE verifier): the AS refuses the verifier.
    let e = oauth.complete(&code, &s2).await.unwrap_err().to_string();
    assert!(e.contains("invalid_grant"), "{e}");
    let e = oauth
        .complete("x", "bogus-state")
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("unknown or expired authorization state"), "{e}");
    // An error redirect is reported.
    let e = oauth
        .complete_redirect("http://127.0.0.1:9/callback?error=access_denied&state=x")
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("access_denied"), "{e}");
}

/// Discovers, authorizes and returns the client plus its store.
async fn authorized(url: &str) -> (OAuthClient, Arc<InMemoryTokenStore>) {
    let store = Arc::new(InMemoryTokenStore::new());
    let oauth = OAuthClient::discover(
        url,
        OAuthClientConfig::new(REDIRECT).with_store(store.clone()),
    )
    .await
    .unwrap();
    let (authorize_url, _) = oauth.begin().unwrap();
    oauth
        .complete_redirect(&consent(&authorize_url).await)
        .await
        .unwrap();
    (oauth, store)
}

#[tokio::test]
async fn expired_token_is_refreshed_after_a_401_and_the_request_retried() {
    let fa = spawn_as(true).await;
    let url = spawn_mcp(&fa.issuer, false, |c| c).await;
    // The first access token is already expired although the response
    // claims an hour, so only the server's 401 reveals it.
    fa.jwt_ttl.store(-3600, Ordering::SeqCst);
    let (oauth, store) = authorized(&url).await;
    let first = store.load().await.unwrap().unwrap();
    fa.jwt_ttl.store(3600, Ordering::SeqCst);

    let client = McpClient::connect_http_with_oauth(&url, oauth.clone())
        .await
        .unwrap();
    assert_eq!(fa.refreshes.load(Ordering::SeqCst), 1);
    let second = store.load().await.unwrap().unwrap();
    assert_ne!(second.access_token, first.access_token);
    assert_ne!(second.refresh_token, first.refresh_token, "rotated");
    let r = client
        .call_tool("lower", json!({"text": "ABC"}))
        .await
        .unwrap();
    assert!(r.to_json().to_string().contains("abc"));
    assert_eq!(fa.refreshes.load(Ordering::SeqCst), 1);

    // A 401 for a token that was already replaced does not refresh again.
    assert!(oauth
        .on_unauthorized(&first.access_token, None)
        .await
        .unwrap());
    assert_eq!(fa.refreshes.load(Ordering::SeqCst), 1);
    client.close().await;
}

#[tokio::test]
async fn tokens_close_to_expiry_are_refreshed_before_use() {
    let fa = spawn_as(true).await;
    let url = spawn_mcp(&fa.issuer, true, |c| c).await;
    fa.advertised_ttl.store(5, Ordering::SeqCst);
    let (oauth, store) = authorized(&url).await;
    let before = store.load().await.unwrap().unwrap();
    fa.advertised_ttl.store(3600, Ordering::SeqCst);
    // Within the default 30-second skew: refreshed.
    let token = oauth.access_token().await.unwrap();
    assert_ne!(token, before.access_token);
    assert_eq!(fa.refreshes.load(Ordering::SeqCst), 1);
    // Now valid for an hour: reused.
    assert_eq!(oauth.access_token().await.unwrap(), token);
    assert_eq!(fa.refreshes.load(Ordering::SeqCst), 1);

    // A revoked refresh token clears the stored tokens.
    fa.refresh_tokens.lock().unwrap().clear();
    let e = oauth.refresh().await.unwrap_err().to_string();
    assert!(e.contains("invalid_grant"), "{e}");
    assert!(oauth.tokens().await.unwrap().is_none());
}

#[tokio::test]
async fn discovery_without_registration_needs_a_client_id() {
    let fa = spawn_as(false).await;
    let url = spawn_mcp(&fa.issuer, true, |c| c).await;
    let e = OAuthClient::discover(&url, OAuthClientConfig::new(REDIRECT))
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("with_client_id"), "{e}");

    // A pre-registered public client.
    fa.clients
        .lock()
        .unwrap()
        .insert("static-client".into(), vec![REDIRECT.into()]);
    let oauth = OAuthClient::discover(
        &url,
        OAuthClientConfig::new(REDIRECT).with_client_id("static-client"),
    )
    .await
    .unwrap();
    let (authorize_url, _) = oauth.begin().unwrap();
    oauth
        .complete_redirect(&consent(&authorize_url).await)
        .await
        .unwrap();
    let client = McpClient::connect_http_with_oauth(&url, oauth)
        .await
        .unwrap();
    assert_eq!(client.list_tools().await.unwrap().len(), 2);

    // Redirect URIs must be HTTPS or loopback.
    let e = OAuthClient::discover(&url, OAuthClientConfig::new("http://app.example.com/cb"))
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("redirect URI"), "{e}");
}

#[tokio::test]
async fn discovery_falls_back_to_the_well_known_url() {
    // A server that answers without a challenge still publishes metadata
    // at the RFC 9728 location.
    let fa = spawn_as(true).await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let url = format!("{base}/mcp");
    let issuer = fa.issuer.clone();
    let resource = url.clone();
    let app = Router::new()
        .route("/mcp", post(|| async { StatusCode::BAD_REQUEST }))
        .route(
            "/.well-known/oauth-protected-resource/mcp",
            get(move || async move {
                Json(json!({"resource": resource, "authorization_servers": [issuer]}))
            }),
        );
    tokio::spawn(async move { axum::serve(listener, app).await });
    let oauth = OAuthClient::discover(&url, OAuthClientConfig::new(REDIRECT))
        .await
        .unwrap();
    assert_eq!(oauth.resource(), url);

    // Metadata for a different resource is refused.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let other = format!("http://{}/mcp", listener.local_addr().unwrap());
    let issuer = fa.issuer.clone();
    let app = Router::new().route(
        "/.well-known/oauth-protected-resource/mcp",
        get(move || async move {
            Json(json!({"resource": "http://127.0.0.1:1/mcp", "authorization_servers": [issuer]}))
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await });
    let e = OAuthClient::discover(&other, OAuthClientConfig::new(REDIRECT))
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("does not cover"), "{e}");
}
