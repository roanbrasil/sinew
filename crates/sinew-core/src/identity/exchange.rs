//! On-behalf-of credentials: RFC 8693 token exchange and the tool wrapper
//! that hands each tool a user-scoped, short-lived token.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use serde_json::Value;

use super::{
    current_context, now_ms, pattern_match, with_user_credential, IdentityError, Secret,
    UserCredential,
};
use crate::{Error, Result, Tool, ToolBox, ToolSource, ToolSpec};

/// RFC 8693 `grant_type` of a token exchange.
pub const GRANT_TYPE_TOKEN_EXCHANGE: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
/// RFC 8693 token type: an OAuth 2.0 access token.
pub const TOKEN_TYPE_ACCESS_TOKEN: &str = "urn:ietf:params:oauth:token-type:access_token";
/// RFC 8693 token type: an OpenID Connect ID token.
pub const TOKEN_TYPE_ID_TOKEN: &str = "urn:ietf:params:oauth:token-type:id_token";
/// RFC 8693 token type: a JWT.
pub const TOKEN_TYPE_JWT: &str = "urn:ietf:params:oauth:token-type:jwt";

/// An RFC 8693 token exchange request: trade the user's token (the
/// subject) for one meant for `audience`, narrowed to `scope`, naming the
/// agent as the actor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenExchangeRequest {
    pub subject_token: Secret,
    pub subject_token_type: String,
    /// A token for the acting party (the agent), when the STS wants one.
    pub actor_token: Option<Secret>,
    pub actor_token_type: Option<String>,
    /// Logical names of the target services.
    pub audience: Vec<String>,
    /// URIs of the target resources.
    pub resource: Vec<String>,
    pub scope: Vec<String>,
    pub requested_token_type: Option<String>,
    /// The acting agent's id (not an RFC 8693 parameter; for exchangers
    /// that record or check it).
    pub actor: Option<String>,
    /// The end user's id, when known (not an RFC 8693 parameter).
    pub user: Option<String>,
}

impl TokenExchangeRequest {
    /// A request for `subject_token` (an access token) to be exchanged for
    /// an access token.
    pub fn new(subject_token: impl Into<Secret>) -> Self {
        Self {
            subject_token: subject_token.into(),
            subject_token_type: TOKEN_TYPE_ACCESS_TOKEN.into(),
            actor_token: None,
            actor_token_type: None,
            audience: Vec::new(),
            resource: Vec::new(),
            scope: Vec::new(),
            requested_token_type: Some(TOKEN_TYPE_ACCESS_TOKEN.into()),
            actor: None,
            user: None,
        }
    }

    pub fn audience(mut self, audience: impl Into<String>) -> Self {
        self.audience.push(audience.into());
        self
    }

    pub fn resource(mut self, resource: impl Into<String>) -> Self {
        self.resource.push(resource.into());
        self
    }

    pub fn scope<I: IntoIterator<Item = S>, S: Into<String>>(mut self, scope: I) -> Self {
        self.scope.extend(scope.into_iter().map(Into::into));
        self
    }

    pub fn actor(mut self, agent: impl Into<String>) -> Self {
        self.actor = Some(agent.into());
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub fn actor_token(mut self, token: impl Into<Secret>, token_type: impl Into<String>) -> Self {
        self.actor_token = Some(token.into());
        self.actor_token_type = Some(token_type.into());
        self
    }

    /// The `application/x-www-form-urlencoded` parameters of the request
    /// (RFC 8693 section 2.1), for an exchanger that POSTs to a security
    /// token service.
    ///
    /// ```
    /// use sinew_core::identity::TokenExchangeRequest;
    /// let req = TokenExchangeRequest::new("user-token").audience("payments").scope(["refunds:write"]);
    /// let form = req.form_params();
    /// assert!(form.contains(&("grant_type", "urn:ietf:params:oauth:grant-type:token-exchange".to_string())));
    /// assert!(form.contains(&("scope", "refunds:write".to_string())));
    /// ```
    pub fn form_params(&self) -> Vec<(&'static str, String)> {
        let mut p = vec![
            ("grant_type", GRANT_TYPE_TOKEN_EXCHANGE.to_string()),
            ("subject_token", self.subject_token.expose().to_string()),
            ("subject_token_type", self.subject_token_type.clone()),
        ];
        if let (Some(t), Some(ty)) = (&self.actor_token, &self.actor_token_type) {
            p.push(("actor_token", t.expose().to_string()));
            p.push(("actor_token_type", ty.clone()));
        }
        p.extend(self.audience.iter().map(|a| ("audience", a.clone())));
        p.extend(self.resource.iter().map(|r| ("resource", r.clone())));
        if !self.scope.is_empty() {
            p.push(("scope", self.scope.join(" ")));
        }
        if let Some(t) = &self.requested_token_type {
            p.push(("requested_token_type", t.clone()));
        }
        p
    }
}

/// An RFC 8693 token exchange response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenExchangeResponse {
    pub access_token: Secret,
    pub issued_token_type: String,
    /// `Bearer`, or `N_A` for tokens that are not access tokens.
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub scope: Vec<String>,
}

impl TokenExchangeResponse {
    /// Parses a token endpoint's JSON answer: a success (RFC 8693 section
    /// 2.2.1) or an error (`{"error": ..., "error_description": ...}`).
    ///
    /// ```
    /// use serde_json::json;
    /// use sinew_core::identity::TokenExchangeResponse;
    /// let r = TokenExchangeResponse::from_json(&json!({
    ///     "access_token": "t", "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
    ///     "token_type": "Bearer", "expires_in": 300, "scope": "a b"
    /// })).unwrap();
    /// assert_eq!(r.scope, ["a", "b"]);
    /// assert!(TokenExchangeResponse::from_json(&json!({"error": "invalid_grant"})).is_err());
    /// ```
    pub fn from_json(body: &Value) -> std::result::Result<Self, IdentityError> {
        if let Some(e) = body.get("error").and_then(Value::as_str) {
            return Err(IdentityError::Exchange {
                error: e.into(),
                description: body
                    .get("error_description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            });
        }
        let s = |k: &str| body.get(k).and_then(Value::as_str);
        let access_token = s("access_token")
            .ok_or_else(|| IdentityError::Invalid("token response: no `access_token`".into()))?;
        Ok(Self {
            access_token: Secret::new(access_token),
            issued_token_type: s("issued_token_type")
                .unwrap_or(TOKEN_TYPE_ACCESS_TOKEN)
                .into(),
            token_type: s("token_type").unwrap_or("Bearer").into(),
            expires_in: body.get("expires_in").and_then(Value::as_u64),
            scope: s("scope")
                .map(|x| x.split_whitespace().map(str::to_owned).collect())
                .unwrap_or_default(),
        })
    }
}

/// Exchanges a user's token for a narrower, short-lived one (RFC 8693).
/// Implement it over your security token service (Keycloak, Okta, Auth0,
/// Entra ID, a cloud STS): POST [`TokenExchangeRequest::form_params`] to the
/// token endpoint and parse the answer with
/// [`TokenExchangeResponse::from_json`]. [`InMemoryTokenExchanger`] (feature
/// `identity`) is an in-process implementation for tests and local runs.
#[async_trait]
pub trait TokenExchanger: Send + Sync {
    async fn exchange(
        &self,
        request: TokenExchangeRequest,
    ) -> std::result::Result<TokenExchangeResponse, IdentityError>;
}

#[derive(Debug, Clone)]
struct Grant {
    pattern: String,
    audience: String,
    scope: Vec<String>,
}

struct OboInner {
    exchanger: Arc<dyn TokenExchanger>,
    grants: Vec<Grant>,
    refresh_margin_ms: u64,
    cache: Mutex<HashMap<(String, String, String), UserCredential>>,
}

/// Gives tools user-scoped credentials: before a tool with a grant runs,
/// the user's token from the [`CallContext`](super::CallContext) is
/// exchanged for a token for that tool's audience and scopes, and the tool
/// reads it with [`user_credential`](super::user_credential). Tokens are
/// cached per user, audience and scope until shortly before they expire.
///
/// A tool with a grant fails (closed) when there is no context, no user or
/// no subject token; tools without a grant run unchanged. MCP tools pass the
/// credential on to the server in `_meta`, and the A2A agent tool to the
/// remote agent, which uses it as its own subject token.
///
/// ```
/// # #[cfg(feature = "identity")]
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use std::sync::Arc;
/// use serde_json::{json, Value};
/// use sinew_core::{AsyncFnTool, ToolBox, ToolCall};
/// use sinew_core::identity::{user_credential, with_context, CallContext, InMemoryTokenExchanger, OnBehalfOf};
///
/// let sts = Arc::new(InMemoryTokenExchanger::new());
/// sts.register_subject("ana-session", "ana");
/// sts.allow("ana", "https://payments.example", ["refunds:write"]);
///
/// let refund = AsyncFnTool::new("refund", "Refund an order", json!({}), |_args: Value| async {
///     let cred = user_credential().expect("set by OnBehalfOf");
///     Ok(json!({"user": cred.user, "scope": cred.scope}))
/// });
/// let tools = OnBehalfOf::new(sts.clone())
///     .grant("refund", "https://payments.example", ["refunds:write"])
///     .wrap(&ToolBox::new().with(refund));
///
/// let ctx = CallContext::new("support").user("ana").subject_token("ana-session");
/// let call = ToolCall { id: "1".into(), name: "refund".into(), arguments: json!({}) };
/// let out = with_context(ctx, tools.call(&call)).await.unwrap();
/// assert_eq!(out, json!({"user": "ana", "scope": ["refunds:write"]}));
/// // Without a user context the call fails closed.
/// assert!(tools.call(&call).await.is_err());
/// # });
/// ```
#[derive(Clone)]
pub struct OnBehalfOf {
    inner: Arc<OboInner>,
}

impl OnBehalfOf {
    pub fn new(exchanger: Arc<dyn TokenExchanger>) -> Self {
        Self {
            inner: Arc::new(OboInner {
                exchanger,
                grants: Vec::new(),
                refresh_margin_ms: 30_000,
                cache: Mutex::new(HashMap::new()),
            }),
        }
    }

    fn inner_mut(&mut self) -> &mut OboInner {
        Arc::get_mut(&mut self.inner).expect("configure OnBehalfOf before cloning it")
    }

    /// Tools whose name matches `tool_pattern` get a token for `audience`
    /// with `scope`. The first matching grant wins.
    pub fn grant<I: IntoIterator<Item = S>, S: Into<String>>(
        mut self,
        tool_pattern: impl Into<String>,
        audience: impl Into<String>,
        scope: I,
    ) -> Self {
        self.inner_mut().grants.push(Grant {
            pattern: tool_pattern.into(),
            audience: audience.into(),
            scope: scope.into_iter().map(Into::into).collect(),
        });
        self
    }

    /// Exchange again when a cached token expires within `margin` (default
    /// 30 s).
    pub fn refresh_margin(mut self, margin: Duration) -> Self {
        self.inner_mut().refresh_margin_ms = margin.as_millis() as u64;
        self
    }

    fn grant_for(&self, tool: &str) -> Option<&Grant> {
        self.inner
            .grants
            .iter()
            .find(|g| pattern_match(&g.pattern, tool))
    }

    /// Whether `tool` gets on-behalf-of credentials.
    pub fn covers(&self, tool: &str) -> bool {
        self.grant_for(tool).is_some()
    }

    /// The credential `tool` gets in the current context (exchanged, or
    /// from the cache); `Ok(None)` when the tool has no grant.
    pub async fn credential_for(
        &self,
        tool: &str,
    ) -> std::result::Result<Option<UserCredential>, IdentityError> {
        let Some(grant) = self.grant_for(tool) else {
            return Ok(None);
        };
        let ctx = current_context().ok_or_else(|| {
            IdentityError::Missing(
                "call context (run the agent inside sinew_core::identity::with_context)".into(),
            )
        })?;
        let user = ctx
            .user
            .clone()
            .ok_or_else(|| IdentityError::Missing("user in the call context".into()))?;
        let subject = ctx
            .subject_token
            .clone()
            .ok_or_else(|| IdentityError::Missing("subject token in the call context".into()))?;
        let key = (user.clone(), grant.audience.clone(), grant.scope.join(" "));
        if let Some(c) = self
            .inner
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .filter(|c| !c.expires_within(self.inner.refresh_margin_ms))
        {
            return Ok(Some(c.clone()));
        }
        let req = TokenExchangeRequest::new(subject)
            .audience(grant.audience.clone())
            .scope(grant.scope.clone())
            .actor(ctx.agent.clone())
            .user(user.clone());
        let resp = self.inner.exchanger.exchange(req).await?;
        let cred = UserCredential {
            token: resp.access_token,
            token_type: resp.issued_token_type,
            user: Some(user),
            audience: Some(grant.audience.clone()),
            scope: if resp.scope.is_empty() {
                grant.scope.clone()
            } else {
                resp.scope
            },
            expires_at_ms: resp
                .expires_in
                .map(|s| now_ms().saturating_add(s.saturating_mul(1000))),
        };
        self.inner
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, cred.clone());
        Ok(Some(cred))
    }

    /// Wraps one tool (unchanged when it has no grant).
    pub fn wrap_tool(&self, tool: Arc<dyn Tool>) -> Arc<dyn Tool> {
        if self.covers(&tool.spec().name) {
            Arc::new(OnBehalfOfTool {
                inner: tool,
                obo: self.clone(),
            })
        } else {
            tool
        }
    }

    /// `toolbox` with every tool that has a grant wrapped in an
    /// [`OnBehalfOfTool`].
    pub fn wrap(&self, toolbox: &ToolBox) -> ToolBox {
        let mut out = ToolBox::new();
        for t in toolbox.tools() {
            out.register_arc(self.wrap_tool(t.clone()));
        }
        out
    }
}

/// A tool that runs with an on-behalf-of [`UserCredential`]; built by
/// [`OnBehalfOf::wrap`].
pub struct OnBehalfOfTool {
    inner: Arc<dyn Tool>,
    obo: OnBehalfOf,
}

#[async_trait]
impl Tool for OnBehalfOfTool {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }

    fn side_effects(&self) -> bool {
        self.inner.side_effects()
    }

    fn source(&self) -> Option<ToolSource> {
        self.inner.source()
    }

    async fn call(&self, arguments: Value) -> Result<Value> {
        let name = self.inner.spec().name;
        match self.obo.credential_for(&name).await {
            Ok(Some(cred)) => with_user_credential(cred, self.inner.call(arguments)).await,
            Ok(None) => self.inner.call(arguments).await,
            Err(e) => Err(Error::Tool {
                name,
                message: format!("no user credential: {e}"),
            }),
        }
    }
}

#[cfg(feature = "identity")]
pub use memory::{InMemoryTokenExchanger, IssuedTokenInfo};

#[cfg(feature = "identity")]
mod memory {
    use std::collections::{BTreeSet, HashMap};

    use super::*;

    /// What [`InMemoryTokenExchanger::introspect`] knows about a token it
    /// issued (like RFC 7662 introspection).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct IssuedTokenInfo {
        pub active: bool,
        pub user: String,
        /// The agent that asked for it.
        pub actor: Option<String>,
        pub audience: String,
        pub scope: Vec<String>,
        pub expires_at_ms: u64,
    }

    #[derive(Default)]
    struct State {
        subjects: HashMap<String, String>,
        allowed: HashMap<(String, String), BTreeSet<String>>,
        issued: HashMap<String, IssuedTokenInfo>,
    }

    /// An in-process security token service for tests and local runs.
    /// Knows user tokens ([`register_subject`](Self::register_subject)) and
    /// what each user may get per audience ([`allow`](Self::allow)); issues
    /// random opaque tokens with the requested scopes narrowed to the
    /// allowed ones, and answers [`introspect`](Self::introspect). Tokens
    /// it issued are themselves valid subject tokens, so a chain of agents
    /// can exchange them again (RFC 8693 delegation).
    pub struct InMemoryTokenExchanger {
        state: Mutex<State>,
        ttl: Duration,
    }

    impl Default for InMemoryTokenExchanger {
        fn default() -> Self {
            Self::new()
        }
    }

    impl InMemoryTokenExchanger {
        /// Issues tokens valid for 5 minutes.
        pub fn new() -> Self {
            Self {
                state: Mutex::default(),
                ttl: Duration::from_secs(300),
            }
        }

        /// Lifetime of issued tokens.
        pub fn with_ttl(mut self, ttl: Duration) -> Self {
            self.ttl = ttl;
            self
        }

        fn state(&self) -> std::sync::MutexGuard<'_, State> {
            self.state.lock().unwrap_or_else(|e| e.into_inner())
        }

        /// `token` is a valid subject token for `user`.
        pub fn register_subject(&self, token: impl Into<String>, user: impl Into<String>) {
            self.state().subjects.insert(token.into(), user.into());
        }

        /// `user` may get tokens for `audience` with these scopes.
        pub fn allow<I: IntoIterator<Item = S>, S: Into<String>>(
            &self,
            user: impl Into<String>,
            audience: impl Into<String>,
            scope: I,
        ) {
            self.state()
                .allowed
                .entry((user.into(), audience.into()))
                .or_default()
                .extend(scope.into_iter().map(Into::into));
        }

        /// What the exchanger knows about `token`, if it issued it.
        pub fn introspect(&self, token: &str) -> Option<IssuedTokenInfo> {
            self.state().issued.get(token).cloned().map(|mut i| {
                i.active = i.expires_at_ms > now_ms();
                i
            })
        }

        /// How many tokens it issued.
        pub fn issued_count(&self) -> usize {
            self.state().issued.len()
        }
    }

    fn denied(error: &str, description: String) -> IdentityError {
        IdentityError::Exchange {
            error: error.into(),
            description: Some(description),
        }
    }

    #[async_trait]
    impl TokenExchanger for InMemoryTokenExchanger {
        async fn exchange(
            &self,
            request: TokenExchangeRequest,
        ) -> std::result::Result<TokenExchangeResponse, IdentityError> {
            let mut st = self.state();
            let subject = request.subject_token.expose();
            let user = match st.subjects.get(subject) {
                Some(u) => u.clone(),
                None => match st.issued.get(subject) {
                    Some(i) if i.expires_at_ms > now_ms() => i.user.clone(),
                    _ => {
                        return Err(denied(
                            "invalid_grant",
                            "unknown or expired subject token".into(),
                        ))
                    }
                },
            };
            let [audience] = request.audience.as_slice() else {
                return Err(denied(
                    "invalid_target",
                    "exactly one audience is supported".into(),
                ));
            };
            let allowed = st
                .allowed
                .get(&(user.clone(), audience.clone()))
                .ok_or_else(|| {
                    denied(
                        "invalid_target",
                        format!("`{user}` may not get tokens for `{audience}`"),
                    )
                })?;
            let scope: Vec<String> = if request.scope.is_empty() {
                allowed.iter().cloned().collect()
            } else {
                request
                    .scope
                    .iter()
                    .filter(|s| allowed.contains(*s))
                    .cloned()
                    .collect()
            };
            if scope.is_empty() {
                return Err(denied(
                    "invalid_scope",
                    format!("none of the requested scopes is allowed for `{user}`"),
                ));
            }
            let mut raw = [0u8; 24];
            getrandom::getrandom(&mut raw).map_err(|e| IdentityError::Invalid(e.to_string()))?;
            let token = format!("obo-{}", super::super::keys::hex(&raw));
            let expires_in = self.ttl.as_secs();
            st.issued.insert(
                token.clone(),
                IssuedTokenInfo {
                    active: true,
                    user,
                    actor: request.actor.clone(),
                    audience: audience.clone(),
                    scope: scope.clone(),
                    expires_at_ms: now_ms().saturating_add(self.ttl.as_millis() as u64),
                },
            );
            Ok(TokenExchangeResponse {
                access_token: Secret::new(token),
                issued_token_type: TOKEN_TYPE_ACCESS_TOKEN.into(),
                token_type: "Bearer".into(),
                expires_in: Some(expires_in),
                scope,
            })
        }
    }
}

#[cfg(all(test, feature = "identity"))]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        identity::{user_credential, with_context, CallContext},
        AsyncFnTool, FnTool, ToolCall,
    };

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: "1".into(),
            name: name.into(),
            arguments: json!({}),
        }
    }

    fn sts() -> Arc<InMemoryTokenExchanger> {
        let s = Arc::new(InMemoryTokenExchanger::new());
        s.register_subject("ana-session", "ana");
        s.allow("ana", "payments", ["refunds:write", "refunds:read"]);
        s
    }

    #[tokio::test]
    async fn exchanges_narrows_and_caches_tokens() {
        let sts = sts();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        let tool = AsyncFnTool::new("refund", "", json!({}), move |_| {
            let seen = s2.clone();
            async move {
                let c = user_credential().unwrap();
                seen.lock().unwrap().push(c.bearer().to_string());
                Ok(json!(c.scope))
            }
        });
        let plain = FnTool::new("echo", "", json!({}), |_| {
            Ok(json!(user_credential().is_some()))
        });
        let obo = OnBehalfOf::new(sts.clone())
            .grant("refund*", "payments", ["refunds:write", "admin"])
            .refresh_margin(Duration::from_secs(1));
        assert!(obo.covers("refund") && !obo.covers("echo"));
        let tools = obo.wrap(&ToolBox::new().with(tool).with(plain));
        let t = tools.get("refund").unwrap();
        assert_eq!(t.spec().name, "refund");
        assert!(!t.side_effects() && t.source().is_none());

        let ctx = CallContext::new("support")
            .user("ana")
            .subject_token("ana-session");
        let (a, b, e) = with_context(ctx, async {
            (
                tools.call(&call("refund")).await.unwrap(),
                tools.call(&call("refund")).await.unwrap(),
                tools.call(&call("echo")).await.unwrap(),
            )
        })
        .await;
        // `admin` is not allowed for ana: narrowed away.
        assert_eq!(a, json!(["refunds:write"]));
        assert_eq!(a, b);
        assert_eq!(e, json!(false));
        let tokens = seen.lock().unwrap().clone();
        assert_eq!(tokens[0], tokens[1], "cached");
        assert_eq!(sts.issued_count(), 1);
        let info = sts.introspect(&tokens[0]).unwrap();
        assert!(info.active);
        assert_eq!(info.user, "ana");
        assert_eq!(info.actor.as_deref(), Some("support"));
        assert_eq!(info.audience, "payments");
        assert!(sts.introspect("nope").is_none());
    }

    #[tokio::test]
    async fn fails_closed_without_user_or_grant() {
        let sts = sts();
        let tool = FnTool::new("refund", "", json!({}), |_| Ok(json!("ran")));
        let tools = OnBehalfOf::new(sts.clone())
            .grant("refund", "payments", ["refunds:write"])
            .wrap(&ToolBox::new().with(tool));
        let msg = |r: Result<Value>| r.unwrap_err().to_string();
        assert!(msg(tools.call(&call("refund")).await).contains("call context"));
        let no_user = with_context(CallContext::new("a"), tools.call(&call("refund"))).await;
        assert!(msg(no_user).contains("user"));
        let no_token = with_context(
            CallContext::new("a").user("ana"),
            tools.call(&call("refund")),
        )
        .await;
        assert!(msg(no_token).contains("subject token"));
        let bad = with_context(
            CallContext::new("a").user("ana").subject_token("forged"),
            tools.call(&call("refund")),
        )
        .await;
        assert!(msg(bad).contains("invalid_grant"));
    }

    #[tokio::test]
    async fn exchanger_enforces_audience_and_scope_and_chains() {
        let sts = sts();
        let ex = |aud: &str, scope: &[&str], subject: &str| {
            let mut r = TokenExchangeRequest::new(subject).scope(scope.iter().copied());
            if !aud.is_empty() {
                r = r.audience(aud);
            }
            sts.exchange(r)
        };
        let e = ex("", &[], "ana-session").await.unwrap_err();
        assert!(e.to_string().contains("invalid_target"));
        let e = ex("hr", &[], "ana-session").await.unwrap_err();
        assert!(e.to_string().contains("may not"));
        let e = ex("payments", &["admin"], "ana-session").await.unwrap_err();
        assert!(e.to_string().contains("invalid_scope"));
        let all = ex("payments", &[], "ana-session").await.unwrap();
        assert_eq!(all.scope.len(), 2);
        assert_eq!(all.token_type, "Bearer");
        // An issued token is a valid subject for a further exchange.
        let again = ex("payments", &["refunds:read"], all.access_token.expose())
            .await
            .unwrap();
        assert_eq!(again.scope, ["refunds:read"]);
        // Expired tokens are not.
        let short = InMemoryTokenExchanger::default().with_ttl(Duration::ZERO);
        short.register_subject("s", "u");
        short.allow("u", "x", ["r"]);
        let t = short
            .exchange(TokenExchangeRequest::new("s").audience("x"))
            .await
            .unwrap();
        assert!(!short.introspect(t.access_token.expose()).unwrap().active);
        assert!(short
            .exchange(TokenExchangeRequest::new(t.access_token.expose()).audience("x"))
            .await
            .is_err());
    }

    #[test]
    fn form_params_and_responses() {
        let req = TokenExchangeRequest::new("s")
            .audience("a")
            .resource("https://r")
            .scope(["x", "y"])
            .actor_token("act", TOKEN_TYPE_JWT)
            .actor("agent")
            .user("u");
        let f = req.form_params();
        assert!(f.contains(&("actor_token", "act".into())));
        assert!(f.contains(&("actor_token_type", TOKEN_TYPE_JWT.into())));
        assert!(f.contains(&("resource", "https://r".into())));
        assert!(f.contains(&("scope", "x y".into())));
        assert!(TokenExchangeResponse::from_json(&json!({})).is_err());
        let r = TokenExchangeResponse::from_json(&json!({"access_token": "t"})).unwrap();
        assert_eq!(r.issued_token_type, TOKEN_TYPE_ACCESS_TOKEN);
        assert!(r.scope.is_empty() && r.expires_in.is_none());
        let e = TokenExchangeResponse::from_json(
            &json!({"error": "invalid_scope", "error_description": "no"}),
        )
        .unwrap_err();
        assert_eq!(e.to_string(), "token exchange failed: invalid_scope: no");
    }
}
