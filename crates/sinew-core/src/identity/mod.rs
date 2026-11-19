//! Agent identity: who is calling, for which user, through which agents.
//!
//! Three layers, from the always-available context types to the signed
//! documents behind the `identity` feature:
//!
//! - **Call context** ([`CallContext`]): the acting agent, the end user and
//!   tenant it acts for, the trace it belongs to and the delegation chain
//!   (every agent the request went through, with how it got there). Set it
//!   once where a request enters the program with [`with_context`]; tools,
//!   tracers, the control plane, `sinew-a2a` and `sinew-mcp` read it with
//!   [`current_context`]. Tracers record it on every tool run, so the chain
//!   of custody is visible in a trace.
//! - **On-behalf-of credentials** ([`OnBehalfOf`], [`TokenExchanger`]): a
//!   tool receives a short-lived token scoped to the end user and to that
//!   tool's audience (an [RFC 8693](https://www.rfc-editor.org/rfc/rfc8693)
//!   token exchange) instead of a shared service key. The tool reads it with
//!   [`user_credential`].
//! - **Signed identity** (feature `identity`): an [`AgentIdentity`]
//!   document (id, owner, version, capabilities, allowed tools, the agent's
//!   own Ed25519 public key) signed by an issuer key ([`AgentKey`]),
//!   verified against a [`TrustStore`]. The agent signs a short-lived
//!   [`CallAssertion`] of its [`CallContext`] for every remote call
//!   ([`AgentCredentials::prove`]); the receiver checks it with an
//!   [`IdentityVerifier`] (trusted issuer, signature, audience, expiry,
//!   replay). [`Signed`] is the generic signed envelope.
//! - **Authenticated messages** (OWASP ASI07): an agent signs what it sends
//!   to a peer ([`AgentCredentials::sign_message`], a [`MessageEnvelope`]
//!   with sender, recipient, thread, timestamp, nonce and the text's
//!   SHA-256); the receiver checks it with a [`MessageVerifier`] (trusted
//!   identity, signature, sender, recipient, freshness, replay) and keeps
//!   the verified sender as metadata ([`PeerRecord`], [`PeerLedger`]),
//!   apart from the text, which models see fenced as data
//!   ([`peer_data_block`]). The record types need no feature.
//!
//! ```
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! use sinew_core::identity::{current_context, with_context, CallContext, Via};
//!
//! let ctx = CallContext::new("orchestrator").user("ana").tenant("acme");
//! let chain = with_context(ctx, async {
//!     // Deeper in the program, e.g. inside a tool or before a remote call:
//!     let here = current_context().unwrap().delegate_to("billing", Via::A2a);
//!     here.chain_agents().join(" -> ")
//! })
//! .await;
//! assert_eq!(chain, "orchestrator -> billing");
//! # });
//! ```

#[cfg(feature = "identity")]
mod document;
mod exchange;
#[cfg(feature = "identity")]
mod keys;
#[cfg(feature = "identity")]
mod message;
mod peer;

use std::{
    fmt,
    future::Future,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[cfg(feature = "identity")]
pub use document::{
    AgentCredentials, AgentIdentity, CallAssertion, IdentityProof, IdentityVerifier,
    SignedAgentIdentity, TrustStore, VerifiedCaller, VerifiedIdentity, DOMAIN_CALL_ASSERTION,
    DOMAIN_IDENTITY,
};
#[cfg(feature = "identity")]
pub use exchange::{InMemoryTokenExchanger, IssuedTokenInfo};
pub use exchange::{
    OnBehalfOf, OnBehalfOfTool, TokenExchangeRequest, TokenExchangeResponse, TokenExchanger,
    GRANT_TYPE_TOKEN_EXCHANGE, TOKEN_TYPE_ACCESS_TOKEN, TOKEN_TYPE_ID_TOKEN, TOKEN_TYPE_JWT,
};
#[cfg(feature = "identity")]
pub use keys::{AgentKey, AgentPublicKey, Signed};
#[cfg(feature = "identity")]
pub use message::{
    MessageEnvelope, MessageVerifier, SignedMessage, VerifiedMessage, DOMAIN_MESSAGE,
};
pub use peer::{
    content_sha256, peer_data_block, peer_note, PeerLedger, PeerRecord, PeerStatus,
    PEER_NOTE_PREFIX,
};

/// Key under which `sinew-a2a` and `sinew-mcp` carry an [`IdentityProof`]
/// in request metadata (A2A `metadata`, MCP `_meta`).
pub const META_IDENTITY: &str = "sinew.dev/identity";

/// Key under which `sinew-a2a` carries a signed inter-agent message
/// ([`SignedMessage`], feature `identity`) in message and task `metadata`.
pub const META_MESSAGE: &str = "sinew.dev/message";

/// Key under which `sinew-a2a` and `sinew-mcp` carry a [`UserCredential`]
/// (the on-behalf-of token for the receiver) in request metadata.
pub const META_CREDENTIAL: &str = "sinew.dev/credential";

/// Errors from identity verification and token exchange.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IdentityError {
    /// Something required is absent (no proof, no user, no subject token).
    #[error("missing {0}")]
    Missing(String),
    /// Malformed input: bad hex, bad JSON, a signature that does not verify.
    #[error("invalid {0}")]
    Invalid(String),
    /// Signed by a key the trust store does not know, or revoked.
    #[error("untrusted: {0}")]
    Untrusted(String),
    /// Past its expiry, or issued in the future.
    #[error("expired: {0}")]
    Expired(String),
    /// A call assertion whose nonce was already seen.
    #[error("replayed: {0}")]
    Replay(String),
    /// Two things that must agree do not (agent id and key, audience).
    #[error("mismatch: {0}")]
    Mismatch(String),
    /// Verified, but not allowed to do this (a tool outside `allowed_tools`).
    #[error("not allowed: {0}")]
    NotAllowed(String),
    /// The token exchange failed (`error` is the RFC 6749 error code, e.g.
    /// `invalid_grant`, `invalid_target`).
    #[error("token exchange failed: {error}{}", .description.as_deref().map(|d| format!(": {d}")).unwrap_or_default())]
    Exchange {
        error: String,
        description: Option<String>,
    },
}

/// A secret string (a token): `Debug` and `Display` print `[REDACTED]`, it
/// is never serialized as part of a [`CallContext`], and you read it with
/// [`expose`](Self::expose).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Secret(String);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The secret value.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl From<&str> for Secret {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Secret {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// How a request reached an agent in the delegation chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Via {
    /// Where the request entered (the first hop).
    Origin,
    /// In process: a subagent, a supervisor handoff, an agent as a tool.
    Local,
    /// An A2A call.
    A2a,
    /// An MCP call.
    Mcp,
}

/// One agent in a [`CallContext`]'s delegation chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationHop {
    /// The agent that acted at this hop.
    pub agent: String,
    pub via: Via,
    /// When the request reached this agent (milliseconds since the epoch).
    pub at_ms: u64,
    /// Whether the call into this agent carried a verified identity proof
    /// (checked by an [`IdentityVerifier`] on receipt).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub verified: bool,
    /// The key id of the caller that signed the proof, when verified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
}

/// Who acts, for whom, and through which agents: the identity that travels
/// with a request through tools, A2A hops and MCP calls.
///
/// The user's own token ([`subject_token`](Self::subject_token)) stays in
/// process: it is never serialized, so it does not reach traces or remote
/// agents. Remote agents get an exchanged token for their audience instead
/// (see [`OnBehalfOf`]).
///
/// ```
/// use sinew_core::identity::{CallContext, Via};
///
/// let ctx = CallContext::new("orchestrator")
///     .user("ana")
///     .tenant("acme")
///     .subject_token("user-session-token");
/// let remote = ctx.delegate_to("billing", Via::A2a);
/// assert_eq!(remote.agent, "billing");
/// assert_eq!(remote.user.as_deref(), Some("ana"));
/// assert_eq!(remote.chain_agents(), ["orchestrator", "billing"]);
/// // The user's token is never serialized.
/// assert!(!serde_json::to_string(&ctx).unwrap().contains("user-session-token"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CallContext {
    /// The agent acting now (the last hop of the chain).
    pub agent: String,
    /// The end user the request is for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// The tenant (customer, organization) the request belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// The trace the request belongs to; a tracer starting a root run joins
    /// this trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// The run new root runs attach to (the tool run that made a remote
    /// call, for the receiving side).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<String>,
    /// Every agent the request went through, first to last.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chain: Vec<DelegationHop>,
    /// The end user's token, the subject of on-behalf-of token exchanges.
    /// Never serialized.
    #[serde(skip)]
    pub subject_token: Option<Secret>,
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl CallContext {
    /// A context where `agent` is the origin of the request.
    pub fn new(agent: impl Into<String>) -> Self {
        let agent = agent.into();
        Self {
            chain: vec![DelegationHop {
                agent: agent.clone(),
                via: Via::Origin,
                at_ms: now_ms(),
                verified: false,
                key_id: None,
            }],
            agent,
            ..Self::default()
        }
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub fn tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    /// Sets the trace id.
    pub fn trace(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Sets the trace and the run that new root runs attach to.
    pub fn run(mut self, trace_id: impl Into<String>, run_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self.parent_run_id = Some(run_id.into());
        self
    }

    /// Sets the end user's token (the subject of token exchanges).
    pub fn subject_token(mut self, token: impl Into<Secret>) -> Self {
        self.subject_token = Some(token.into());
        self
    }

    /// The context as seen by `agent` after the request is handed to it:
    /// same user, tenant and trace, `agent` appended to the chain.
    pub fn delegate_to(&self, agent: impl Into<String>, via: Via) -> Self {
        self.delegate(agent, via, None)
    }

    /// Like [`delegate_to`](Self::delegate_to), for a hop whose caller was
    /// verified with the key `key_id`.
    pub fn delegate_verified(
        &self,
        agent: impl Into<String>,
        via: Via,
        key_id: impl Into<String>,
    ) -> Self {
        self.delegate(agent, via, Some(key_id.into()))
    }

    fn delegate(&self, agent: impl Into<String>, via: Via, key_id: Option<String>) -> Self {
        let agent = agent.into();
        let mut next = self.clone();
        next.chain.push(DelegationHop {
            agent: agent.clone(),
            via,
            at_ms: now_ms(),
            verified: key_id.is_some(),
            key_id,
        });
        next.agent = agent;
        next
    }

    /// The agent ids of the chain, first to last.
    pub fn chain_agents(&self) -> Vec<&str> {
        self.chain.iter().map(|h| h.agent.as_str()).collect()
    }

    /// The agent where the request entered (the first hop), if any.
    pub fn origin(&self) -> Option<&str> {
        self.chain.first().map(|h| h.agent.as_str())
    }

    /// The context as trace metadata: agent, user, tenant and chain (never
    /// the subject token).
    pub fn to_trace_value(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Some(o) = v.as_object_mut() {
            // The trace already knows its ids.
            o.remove("trace_id");
            o.remove("parent_run_id");
        }
        v
    }
}

/// A short-lived token for one audience, scoped to the end user: what an
/// on-behalf-of tool receives instead of a shared service key. `Debug`
/// never prints the token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserCredential {
    /// The token. Serialized (it travels to the receiving tool or agent),
    /// but `Debug` prints `[REDACTED]`.
    #[serde(with = "secret_serde")]
    pub token: Secret,
    /// RFC 8693 `issued_token_type`, e.g. [`TOKEN_TYPE_ACCESS_TOKEN`].
    pub token_type: String,
    /// The user the token is for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// The service the token is meant for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audience: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scope: Vec<String>,
    /// Expiry (milliseconds since the epoch), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_ms: Option<u64>,
}

impl UserCredential {
    /// The token, to send as `Authorization: Bearer <token>`.
    pub fn bearer(&self) -> &str {
        self.token.expose()
    }

    /// Whether the token expires within `margin_ms` from now.
    pub fn expires_within(&self, margin_ms: u64) -> bool {
        self.expires_at_ms
            .is_some_and(|e| e <= now_ms().saturating_add(margin_ms))
    }

    /// Whether the token is past its expiry.
    pub fn is_expired(&self) -> bool {
        self.expires_within(0)
    }

    /// Whether the token carries `scope`.
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scope.iter().any(|s| s == scope)
    }
}

mod secret_serde {
    use super::Secret;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(s: &Secret, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(s.expose())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Secret, D::Error> {
        String::deserialize(de).map(Secret)
    }
}

tokio::task_local! {
    static CALL_CONTEXT: CallContext;
    static USER_CREDENTIAL: UserCredential;
}

/// The [`CallContext`] of the request being handled, if one was set with
/// [`with_context`] (or by `sinew-a2a` / `sinew-mcp` for a verified remote
/// caller).
pub fn current_context() -> Option<CallContext> {
    CALL_CONTEXT.try_with(Clone::clone).ok()
}

/// Runs `f` with [`current_context`] returning `ctx`.
pub async fn with_context<F: Future>(ctx: CallContext, f: F) -> F::Output {
    CALL_CONTEXT.scope(ctx, f).await
}

/// The on-behalf-of credential for the tool call being executed, set by
/// [`OnBehalfOf`] (or by `sinew-a2a` / `sinew-mcp` when the caller sent
/// one).
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use sinew_core::identity::{user_credential, with_user_credential, UserCredential, TOKEN_TYPE_ACCESS_TOKEN};
///
/// assert!(user_credential().is_none());
/// let cred = UserCredential {
///     token: "t-123".into(),
///     token_type: TOKEN_TYPE_ACCESS_TOKEN.into(),
///     user: Some("ana".into()),
///     audience: Some("https://payments.example".into()),
///     scope: vec!["refunds:write".into()],
///     expires_at_ms: None,
/// };
/// let bearer = with_user_credential(cred, async { user_credential().unwrap().bearer().to_string() }).await;
/// assert_eq!(bearer, "t-123");
/// # });
/// ```
pub fn user_credential() -> Option<UserCredential> {
    USER_CREDENTIAL.try_with(Clone::clone).ok()
}

/// Runs `f` with [`user_credential`] returning `credential`.
pub async fn with_user_credential<F: Future>(credential: UserCredential, f: F) -> F::Output {
    USER_CREDENTIAL.scope(credential, f).await
}

/// Runs `f` with an optional context and credential (each scoped only when
/// present).
pub async fn with_identity_scope<F: Future>(
    ctx: Option<CallContext>,
    credential: Option<UserCredential>,
    f: F,
) -> F::Output {
    match (ctx, credential) {
        (Some(c), Some(cr)) => CALL_CONTEXT.scope(c, USER_CREDENTIAL.scope(cr, f)).await,
        (Some(c), None) => CALL_CONTEXT.scope(c, f).await,
        (None, Some(cr)) => USER_CREDENTIAL.scope(cr, f).await,
        (None, None) => f.await,
    }
}

/// JSON with object keys sorted recursively and no whitespace: the bytes
/// that signatures cover (the JSON Canonicalization Scheme, RFC 8785, for
/// documents without floating-point numbers).
///
/// ```
/// use serde_json::json;
/// use sinew_core::identity::canonical_json;
/// assert_eq!(canonical_json(&json!({"b": [1, {"z": 1, "a": 2}], "a": "x"})), r#"{"a":"x","b":[1,{"a":2,"z":1}]}"#);
/// ```
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            // RFC 8785 sorts by UTF-16 code units.
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(k.clone()).to_string());
                out.push(':');
                write_canonical(&map[k], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(v, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// Whether `name` matches `pattern` (`*` any run of characters, `?` one
/// character), as in `allowed_tools`.
///
/// ```
/// use sinew_core::identity::pattern_match;
/// assert!(pattern_match("get_*", "get_invoice"));
/// assert!(!pattern_match("get_*", "delete_invoice"));
/// ```
pub fn pattern_match(pattern: &str, name: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = name.chars().collect();
    let (mut pi, mut ti) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some((pi, ti));
            pi += 1;
        } else if let Some((sp, st)) = star {
            pi = sp + 1;
            ti = st + 1;
            star = Some((sp, st + 1));
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|c| *c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn context_delegation_keeps_user_and_appends_hops() {
        let ctx = CallContext::new("a")
            .user("ana")
            .tenant("acme")
            .run("t1", "r1")
            .subject_token("secret");
        assert_eq!(ctx.origin(), Some("a"));
        let b = ctx.delegate_verified("b", Via::A2a, "k1");
        let c = b.delegate_to("c", Via::Mcp);
        assert_eq!(c.chain_agents(), ["a", "b", "c"]);
        assert!(c.chain[1].verified);
        assert_eq!(c.chain[1].key_id.as_deref(), Some("k1"));
        assert!(!c.chain[2].verified);
        assert_eq!(c.tenant.as_deref(), Some("acme"));
        assert_eq!(c.subject_token.as_ref().unwrap().expose(), "secret");

        let wire = serde_json::to_string(&c).unwrap();
        assert!(!wire.contains("secret"), "{wire}");
        let back: CallContext = serde_json::from_str(&wire).unwrap();
        assert_eq!(back.chain, c.chain);
        assert!(back.subject_token.is_none());

        let tv = c.to_trace_value();
        assert_eq!(tv["user"], "ana");
        assert!(tv.get("trace_id").is_none());
        assert_eq!(tv["chain"][2]["via"], "mcp");
    }

    #[test]
    fn secrets_and_credentials_never_print() {
        let s = Secret::new("hunter2");
        assert_eq!(format!("{s:?} {s}"), "[REDACTED] [REDACTED]");
        let c = UserCredential {
            token: "tok".into(),
            token_type: TOKEN_TYPE_ACCESS_TOKEN.into(),
            user: None,
            audience: None,
            scope: vec!["a".into()],
            expires_at_ms: Some(1),
        };
        assert!(!format!("{c:?}").contains("tok\""));
        assert!(c.is_expired());
        assert!(c.has_scope("a") && !c.has_scope("b"));
        assert_eq!(c.bearer(), "tok");
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["token"], "tok");
        let back: UserCredential = serde_json::from_value(v).unwrap();
        assert_eq!(back, c);
        let fresh = UserCredential {
            expires_at_ms: None,
            ..c
        };
        assert!(!fresh.is_expired());
    }

    #[tokio::test]
    async fn scopes_context_and_credential() {
        assert!(current_context().is_none());
        let cred = UserCredential {
            token: "t".into(),
            token_type: TOKEN_TYPE_ACCESS_TOKEN.into(),
            user: None,
            audience: None,
            scope: vec![],
            expires_at_ms: None,
        };
        let (a, b) = with_identity_scope(Some(CallContext::new("x")), Some(cred.clone()), async {
            (current_context(), user_credential())
        })
        .await;
        assert_eq!(a.unwrap().agent, "x");
        assert!(b.is_some());
        let (a, b) = with_identity_scope(None, Some(cred), async {
            (current_context(), user_credential())
        })
        .await;
        assert!(a.is_none() && b.is_some());
        let (a, b) = with_identity_scope(Some(CallContext::new("y")), None, async {
            (current_context(), user_credential())
        })
        .await;
        assert!(a.is_some() && b.is_none());
        let (a, b) =
            with_identity_scope(None, None, async { (current_context(), user_credential()) }).await;
        assert!(a.is_none() && b.is_none());
    }

    #[test]
    fn canonical_json_and_patterns() {
        assert_eq!(
            canonical_json(&json!({"é": 1, "z": [true, null], "a": {"y": "q", "b": 2}})),
            r#"{"a":{"b":2,"y":"q"},"z":[true,null],"é":1}"#
        );
        assert!(pattern_match("*", "anything"));
        assert!(pattern_match("send_?mail", "send_email"));
        assert!(pattern_match("a*c*", "abxcyz"));
        assert!(!pattern_match("a*c", "abxcyz"));
        assert!(!pattern_match("", "x"));
    }

    #[test]
    fn errors_display() {
        let e = IdentityError::Exchange {
            error: "invalid_grant".into(),
            description: Some("unknown subject".into()),
        };
        assert_eq!(
            e.to_string(),
            "token exchange failed: invalid_grant: unknown subject"
        );
        let e = IdentityError::Exchange {
            error: "invalid_target".into(),
            description: None,
        };
        assert_eq!(e.to_string(), "token exchange failed: invalid_target");
    }
}
