//! Signed agent identity documents, the trust store, call assertions and
//! their verification.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    keys::{hex, AgentKey, AgentPublicKey, Signed},
    now_ms, pattern_match, CallContext, IdentityError, Via,
};

/// Domain of [`AgentIdentity`] signatures.
pub const DOMAIN_IDENTITY: &str = "sinew-agent-identity-v1";
/// Domain of [`CallAssertion`] signatures.
pub const DOMAIN_CALL_ASSERTION: &str = "sinew-call-assertion-v1";

/// An agent's identity document: who it is, who answers for it, what it may
/// do, and the public key it signs its calls with. An issuer (the platform
/// team, the control plane) signs it with [`sign`](Self::sign); receivers
/// trust the issuer's key in a [`TrustStore`].
///
/// ```
/// use std::time::Duration;
/// use sinew_core::identity::{AgentIdentity, AgentKey, TrustStore};
///
/// let issuer = AgentKey::from_bytes(&[1; 32]);
/// let agent_key = AgentKey::from_bytes(&[2; 32]);
/// let signed = AgentIdentity::new("billing-bot", agent_key.public_key())
///     .owner("payments-team")
///     .version("1.4.0")
///     .tenant("acme")
///     .capability("refunds")
///     .allow_tools(["get_*", "refund"])
///     .expires_in(Duration::from_secs(86_400))
///     .sign(&issuer)
///     .unwrap();
///
/// let trust = TrustStore::new().trust(issuer.public_key());
/// let verified = trust.verify(&signed).unwrap();
/// assert_eq!(verified.identity.id, "billing-bot");
/// assert!(verified.identity.tool_allowed("get_invoice"));
/// assert!(!verified.identity.tool_allowed("delete_invoice"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentIdentity {
    /// The agent's id (the `ControlPlane` registry id, the A2A audience).
    pub id: String,
    /// Team or person accountable for the agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// The agent's version.
    pub version: String,
    /// The tenant the agent belongs to, if it is tenant-specific.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// What the agent does (free-form labels, e.g. `refunds`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// Tool name patterns (`*`, `?`) the agent may call. Empty: none.
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    /// The agent's own key: it signs the agent's call assertions.
    pub public_key: AgentPublicKey,
    /// Who issued the document (informational; trust comes from the key).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub issuer: Option<String>,
    /// When it was issued (milliseconds since the epoch).
    pub issued_at_ms: u64,
    /// When it stops being valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_ms: Option<u64>,
}

/// An [`AgentIdentity`] signed by its issuer.
pub type SignedAgentIdentity = Signed<AgentIdentity>;

impl AgentIdentity {
    /// A document for agent `id` holding `public_key`, version `0.0.0`, no
    /// tools, issued now and without expiry.
    pub fn new(id: impl Into<String>, public_key: AgentPublicKey) -> Self {
        Self {
            id: id.into(),
            owner: None,
            version: "0.0.0".into(),
            tenant: None,
            capabilities: Vec::new(),
            allowed_tools: Vec::new(),
            public_key,
            issuer: None,
            issued_at_ms: now_ms(),
            expires_at_ms: None,
        }
    }

    pub fn owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    pub fn tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    pub fn issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = Some(issuer.into());
        self
    }

    pub fn capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    /// Adds tool name patterns to `allowed_tools`.
    pub fn allow_tools<I: IntoIterator<Item = S>, S: Into<String>>(mut self, p: I) -> Self {
        self.allowed_tools.extend(p.into_iter().map(Into::into));
        self
    }

    /// Expires `ttl` after `issued_at_ms`.
    pub fn expires_in(mut self, ttl: Duration) -> Self {
        self.expires_at_ms = Some(
            self.issued_at_ms
                .saturating_add(ttl.as_millis().min(u64::MAX as u128) as u64),
        );
        self
    }

    /// Whether `allowed_tools` lets the agent call `tool`.
    pub fn tool_allowed(&self, tool: &str) -> bool {
        self.allowed_tools.iter().any(|p| pattern_match(p, tool))
    }

    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }

    /// Signs the document with the issuer's key.
    pub fn sign(self, issuer: &AgentKey) -> Result<SignedAgentIdentity, IdentityError> {
        Signed::sign(self, issuer, DOMAIN_IDENTITY)
    }

    /// Signs the document with the agent's own key (trust it by adding that
    /// key to the [`TrustStore`]). Fails if `key` is not the document's key.
    pub fn self_signed(self, key: &AgentKey) -> Result<SignedAgentIdentity, IdentityError> {
        if key.public_key() != self.public_key {
            return Err(IdentityError::Mismatch(
                "self-signing key is not the document's public key".into(),
            ));
        }
        self.sign(key)
    }
}

#[derive(Default)]
struct TrustInner {
    issuers: BTreeMap<String, (AgentPublicKey, Option<String>)>,
    revoked_agents: BTreeSet<String>,
    revoked_keys: BTreeSet<String>,
}

/// The issuer keys a receiver trusts, plus revocations. Clones share state,
/// so a revocation applies at once to every verifier built from it.
#[derive(Clone)]
pub struct TrustStore {
    inner: Arc<RwLock<TrustInner>>,
    pub(super) skew_ms: u64,
}

impl Default for TrustStore {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TrustStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let i = self.inner.read().unwrap_or_else(|e| e.into_inner());
        f.debug_struct("TrustStore")
            .field("issuers", &i.issuers.keys().collect::<Vec<_>>())
            .field("revoked_agents", &i.revoked_agents)
            .field("revoked_keys", &i.revoked_keys)
            .finish()
    }
}

/// An identity document whose signature and validity were checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedIdentity {
    pub identity: AgentIdentity,
    /// Key id of the issuer key that signed it.
    pub issuer_key_id: String,
    /// The issuer's name in the trust store, if given.
    pub issuer_name: Option<String>,
}

impl TrustStore {
    /// An empty store: nothing verifies until an issuer is trusted.
    pub fn new() -> Self {
        Self {
            inner: Arc::default(),
            skew_ms: 30_000,
        }
    }

    /// Trusts documents signed by `issuer`.
    pub fn trust(self, issuer: AgentPublicKey) -> Self {
        self.add_issuer(issuer, None);
        self
    }

    /// Trusts documents signed by `issuer`, named `name` in verification
    /// results.
    pub fn trust_named(self, name: impl Into<String>, issuer: AgentPublicKey) -> Self {
        self.add_issuer(issuer, Some(name.into()));
        self
    }

    /// Clock skew tolerated on `issued_at` / expiry checks (default 30 s).
    pub fn clock_skew(mut self, skew: Duration) -> Self {
        self.skew_ms = skew.as_millis() as u64;
        self
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, TrustInner> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds an issuer key at runtime.
    pub fn add_issuer(&self, issuer: AgentPublicKey, name: Option<String>) {
        self.write().issuers.insert(issuer.key_id(), (issuer, name));
    }

    /// Stops trusting an issuer key; returns whether it was trusted.
    pub fn remove_issuer(&self, key_id: &str) -> bool {
        self.write().issuers.remove(key_id).is_some()
    }

    /// Rejects every document for agent `id` from now on.
    pub fn revoke_agent(&self, id: impl Into<String>) {
        self.write().revoked_agents.insert(id.into());
    }

    /// Rejects documents and calls involving the key `key_id` (an issuer or
    /// an agent key) from now on.
    pub fn revoke_key(&self, key_id: impl Into<String>) {
        self.write().revoked_keys.insert(key_id.into());
    }

    /// Whether `key_id` was revoked.
    pub fn is_key_revoked(&self, key_id: &str) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .revoked_keys
            .contains(key_id)
    }

    /// Checks the document's issuer signature, revocations and validity
    /// period, now.
    pub fn verify(&self, signed: &SignedAgentIdentity) -> Result<VerifiedIdentity, IdentityError> {
        self.verify_at(signed, now_ms())
    }

    /// Like [`verify`](Self::verify), at `now_ms`.
    pub fn verify_at(
        &self,
        signed: &SignedAgentIdentity,
        now_ms: u64,
    ) -> Result<VerifiedIdentity, IdentityError> {
        let (issuer, name) = {
            let i = self.inner.read().unwrap_or_else(|e| e.into_inner());
            let doc = &signed.payload;
            if i.revoked_agents.contains(&doc.id) {
                return Err(IdentityError::Untrusted(format!(
                    "agent `{}` is revoked",
                    doc.id
                )));
            }
            for k in [&signed.key_id, &doc.public_key.key_id()] {
                if i.revoked_keys.contains(k) {
                    return Err(IdentityError::Untrusted(format!("key {k} is revoked")));
                }
            }
            i.issuers.get(&signed.key_id).cloned().ok_or_else(|| {
                IdentityError::Untrusted(format!(
                    "identity of `{}` is signed by unknown key {}",
                    doc.id, signed.key_id
                ))
            })?
        };
        let doc = signed.verify(&issuer, DOMAIN_IDENTITY)?;
        if doc.issued_at_ms > now_ms.saturating_add(self.skew_ms) {
            return Err(IdentityError::Expired(format!(
                "identity of `{}` is issued in the future",
                doc.id
            )));
        }
        if doc
            .expires_at_ms
            .is_some_and(|e| e.saturating_add(self.skew_ms) < now_ms)
        {
            return Err(IdentityError::Expired(format!(
                "identity of `{}` has expired",
                doc.id
            )));
        }
        Ok(VerifiedIdentity {
            identity: doc.clone(),
            issuer_key_id: signed.key_id.clone(),
            issuer_name: name,
        })
    }
}

/// What an agent signs for one remote call: its [`CallContext`] (with the
/// delegation chain so far), the receiver it is meant for, a validity window
/// and a nonce against replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallAssertion {
    pub context: CallContext,
    /// The receiving agent's id (or MCP server name).
    pub audience: String,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    /// Random, unique per call.
    pub nonce: String,
}

/// The caller's signed identity plus its signed call assertion: what
/// `sinew-a2a` and `sinew-mcp` send under
/// [`META_IDENTITY`](super::META_IDENTITY).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityProof {
    pub identity: SignedAgentIdentity,
    pub assertion: Signed<CallAssertion>,
}

impl IdentityProof {
    /// The proof as JSON.
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// An agent's private key with its signed identity: what it needs to prove
/// who it is on outgoing calls.
///
/// ```
/// use sinew_core::identity::{AgentCredentials, AgentIdentity, AgentKey, CallContext, IdentityVerifier, TrustStore};
///
/// let issuer = AgentKey::from_bytes(&[1; 32]);
/// let key = AgentKey::from_bytes(&[2; 32]);
/// let identity = AgentIdentity::new("orchestrator", key.public_key()).sign(&issuer).unwrap();
/// let creds = AgentCredentials::new(key, identity).unwrap();
///
/// let ctx = CallContext::new("orchestrator").user("ana");
/// let proof = creds.prove(Some(&ctx), "billing").unwrap();
///
/// let verifier = IdentityVerifier::new(TrustStore::new().trust(issuer.public_key())).audience("billing");
/// let caller = verifier.verify(&proof).unwrap();
/// assert_eq!(caller.context.user.as_deref(), Some("ana"));
/// assert!(verifier.verify(&proof).is_err()); // replayed
/// ```
#[derive(Debug, Clone)]
pub struct AgentCredentials {
    key: AgentKey,
    identity: SignedAgentIdentity,
    ttl_ms: u64,
}

pub(super) fn nonce() -> Result<String, IdentityError> {
    let mut b = [0u8; 16];
    getrandom::getrandom(&mut b).map_err(|e| IdentityError::Invalid(e.to_string()))?;
    Ok(hex(&b))
}

impl AgentCredentials {
    /// Fails if `key` is not the key named in `identity`.
    pub fn new(key: AgentKey, identity: SignedAgentIdentity) -> Result<Self, IdentityError> {
        if key.public_key() != identity.payload.public_key {
            return Err(IdentityError::Mismatch(format!(
                "key {} is not the key of identity `{}`",
                key.public_key().key_id(),
                identity.payload.id
            )));
        }
        Ok(Self {
            key,
            identity,
            ttl_ms: 60_000,
        })
    }

    /// How long a call assertion stays valid (default 60 s).
    pub fn assertion_ttl(mut self, ttl: Duration) -> Self {
        self.ttl_ms = ttl.as_millis() as u64;
        self
    }

    /// The agent's id.
    pub fn agent_id(&self) -> &str {
        &self.identity.payload.id
    }

    pub fn identity(&self) -> &SignedAgentIdentity {
        &self.identity
    }

    pub fn key(&self) -> &AgentKey {
        &self.key
    }

    /// Signs a call assertion of `ctx` for `audience`. Without a context,
    /// asserts a fresh one with this agent as the origin. A context acting
    /// as another agent is refused (a confused deputy).
    pub fn prove(
        &self,
        ctx: Option<&CallContext>,
        audience: &str,
    ) -> Result<IdentityProof, IdentityError> {
        let context = match ctx {
            Some(c) if c.agent == self.agent_id() => c.clone(),
            Some(c) => {
                return Err(IdentityError::Mismatch(format!(
                    "the call context acts as `{}`, these credentials are `{}`",
                    c.agent,
                    self.agent_id()
                )))
            }
            None => CallContext::new(self.agent_id()),
        };
        let now = now_ms();
        let assertion = CallAssertion {
            context,
            audience: audience.into(),
            issued_at_ms: now,
            expires_at_ms: now.saturating_add(self.ttl_ms),
            nonce: nonce()?,
        };
        Ok(IdentityProof {
            identity: self.identity.clone(),
            assertion: Signed::sign(assertion, &self.key, DOMAIN_CALL_ASSERTION)?,
        })
    }
}

/// A remote caller whose proof verified: its identity and the context it
/// asserted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCaller {
    pub identity: VerifiedIdentity,
    /// The caller's asserted context (its agent is the caller).
    pub context: CallContext,
}

impl VerifiedCaller {
    /// The context for the receiving agent `me`: the caller's context with
    /// `me` appended as a verified hop.
    pub fn context_for(&self, me: &str, via: Via) -> CallContext {
        self.context
            .delegate_verified(me, via, self.identity.identity.public_key.key_id())
    }

    /// The caller's agent id.
    pub fn agent(&self) -> &str {
        &self.identity.identity.id
    }
}

struct VerifierInner {
    trust: TrustStore,
    audience: Option<String>,
    required: bool,
    max_ttl_ms: u64,
    seen: Mutex<HashMap<String, u64>>,
}

/// Checks [`IdentityProof`]s on receipt: trusted identity, assertion signed
/// by the identity's key, matching agent and audience, inside its validity
/// window, nonce never seen before. Clones share the replay cache.
#[derive(Clone)]
pub struct IdentityVerifier {
    inner: Arc<VerifierInner>,
}

impl std::fmt::Debug for IdentityVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdentityVerifier")
            .field("audience", &self.inner.audience)
            .field("required", &self.inner.required)
            .finish_non_exhaustive()
    }
}

impl IdentityVerifier {
    /// A verifier over `trust` that requires a proof on every call and
    /// accepts assertions valid for at most 5 minutes.
    pub fn new(trust: TrustStore) -> Self {
        Self {
            inner: Arc::new(VerifierInner {
                trust,
                audience: None,
                required: true,
                max_ttl_ms: 300_000,
                seen: Mutex::new(HashMap::new()),
            }),
        }
    }

    fn inner_mut(&mut self) -> &mut VerifierInner {
        Arc::get_mut(&mut self.inner).expect("configure the verifier before cloning it")
    }

    /// The audience assertions must name (this agent's id). `sinew-a2a`
    /// and `sinew-mcp` fill it in from the server when unset.
    pub fn audience(mut self, audience: impl Into<String>) -> Self {
        self.inner_mut().audience = Some(audience.into());
        self
    }

    /// `false`: calls without a proof are accepted (as unverified); a proof
    /// that is present must still verify. Default `true`.
    pub fn required(mut self, required: bool) -> Self {
        self.inner_mut().required = required;
        self
    }

    /// The longest validity window accepted on an assertion (default 5 min).
    pub fn max_assertion_ttl(mut self, ttl: Duration) -> Self {
        self.inner_mut().max_ttl_ms = ttl.as_millis() as u64;
        self
    }

    pub fn is_required(&self) -> bool {
        self.inner.required
    }

    pub fn configured_audience(&self) -> Option<&str> {
        self.inner.audience.as_deref()
    }

    pub fn trust_store(&self) -> &TrustStore {
        &self.inner.trust
    }

    /// Verifies `proof` against the configured audience.
    pub fn verify(&self, proof: &IdentityProof) -> Result<VerifiedCaller, IdentityError> {
        self.verify_for(proof, self.inner.audience.as_deref())
    }

    /// Verifies `proof`, expecting `audience` (`None`: any audience).
    pub fn verify_for(
        &self,
        proof: &IdentityProof,
        audience: Option<&str>,
    ) -> Result<VerifiedCaller, IdentityError> {
        let now = now_ms();
        let skew = self.inner.trust.skew_ms;
        let verified = self.inner.trust.verify_at(&proof.identity, now)?;
        let agent_key = verified.identity.public_key;
        if self.inner.trust.is_key_revoked(&proof.assertion.key_id) {
            return Err(IdentityError::Untrusted(format!(
                "key {} is revoked",
                proof.assertion.key_id
            )));
        }
        let a = proof.assertion.verify(&agent_key, DOMAIN_CALL_ASSERTION)?;
        if a.context.agent != verified.identity.id {
            return Err(IdentityError::Mismatch(format!(
                "assertion acts as `{}`, identity is `{}`",
                a.context.agent, verified.identity.id
            )));
        }
        if a.context
            .chain
            .last()
            .is_some_and(|h| h.agent != a.context.agent)
        {
            return Err(IdentityError::Mismatch(
                "the delegation chain does not end at the caller".into(),
            ));
        }
        if let Some(want) = audience {
            if a.audience != want {
                return Err(IdentityError::Mismatch(format!(
                    "assertion is for `{}`, this is `{want}`",
                    a.audience
                )));
            }
        }
        if a.issued_at_ms > now.saturating_add(skew) {
            return Err(IdentityError::Expired(
                "assertion issued in the future".into(),
            ));
        }
        if a.expires_at_ms.saturating_add(skew) < now {
            return Err(IdentityError::Expired("assertion has expired".into()));
        }
        if a.expires_at_ms.saturating_sub(a.issued_at_ms) > self.inner.max_ttl_ms {
            return Err(IdentityError::Invalid(
                "assertion validity window is too long".into(),
            ));
        }
        {
            let mut seen = self.inner.seen.lock().unwrap_or_else(|e| e.into_inner());
            seen.retain(|_, exp| exp.saturating_add(skew) >= now);
            let key = format!("{}/{}", proof.assertion.key_id, a.nonce);
            if seen.contains_key(&key) {
                return Err(IdentityError::Replay(format!(
                    "nonce {} from `{}` was already used",
                    a.nonce, a.context.agent
                )));
            }
            seen.insert(key, a.expires_at_ms);
        }
        Ok(VerifiedCaller {
            context: a.context.clone(),
            identity: verified,
        })
    }

    /// Verifies the proof in a metadata value (the JSON under
    /// [`META_IDENTITY`](super::META_IDENTITY)), or none: `Ok(None)` when
    /// there is no proof and one is not required.
    pub fn verify_value(
        &self,
        proof: Option<&Value>,
        audience: Option<&str>,
    ) -> Result<Option<VerifiedCaller>, IdentityError> {
        let audience = self.inner.audience.as_deref().or(audience);
        match proof {
            None | Some(Value::Null) if self.inner.required => {
                Err(IdentityError::Missing("identity proof".into()))
            }
            None | Some(Value::Null) => Ok(None),
            Some(v) => {
                let proof: IdentityProof = serde_json::from_value(v.clone())
                    .map_err(|e| IdentityError::Invalid(format!("identity proof: {e}")))?;
                self.verify_for(&proof, audience).map(Some)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn setup() -> (AgentKey, AgentKey, SignedAgentIdentity, TrustStore) {
        let issuer = AgentKey::from_bytes(&[1; 32]);
        let key = AgentKey::from_bytes(&[2; 32]);
        let id = AgentIdentity::new("a", key.public_key())
            .allow_tools(["read_*"])
            .capability("search")
            .issuer("platform")
            .sign(&issuer)
            .unwrap();
        let trust = TrustStore::new().trust_named("platform", issuer.public_key());
        (issuer, key, id, trust)
    }

    #[test]
    fn trust_store_checks_issuer_validity_and_revocation() {
        let (issuer, key, id, trust) = setup();
        let v = trust.verify(&id).unwrap();
        assert_eq!(v.issuer_name.as_deref(), Some("platform"));
        assert!(v.identity.has_capability("search") && !v.identity.has_capability("x"));
        assert!(format!("{trust:?}").contains(&issuer.public_key().key_id()));

        // Unknown issuer.
        assert!(matches!(
            TrustStore::default().verify(&id),
            Err(IdentityError::Untrusted(_))
        ));
        // Tampered document.
        let mut forged = id.clone();
        forged.payload.allowed_tools.push("*".into());
        assert!(matches!(
            trust.verify(&forged),
            Err(IdentityError::Invalid(_))
        ));
        // Expired and future documents.
        let mut doc = AgentIdentity::new("a", key.public_key()).expires_in(Duration::from_secs(1));
        let late = doc.issued_at_ms + 120_000;
        let signed = doc.clone().sign(&issuer).unwrap();
        assert!(matches!(
            trust.verify_at(&signed, late),
            Err(IdentityError::Expired(_))
        ));
        doc.issued_at_ms = now_ms() + 3_600_000;
        assert!(matches!(
            trust.verify(&doc.sign(&issuer).unwrap()),
            Err(IdentityError::Expired(_))
        ));
        // Revocations.
        let t2 = trust.clone().clock_skew(Duration::ZERO);
        t2.revoke_agent("a");
        assert!(matches!(
            trust.verify(&id),
            Err(IdentityError::Untrusted(_))
        ));
        let t3 = TrustStore::new().trust(issuer.public_key());
        t3.revoke_key(key.public_key().key_id());
        assert!(t3.verify(&id).is_err());
        let t4 = TrustStore::new().trust(issuer.public_key());
        assert!(t4.remove_issuer(&issuer.public_key().key_id()));
        assert!(!t4.remove_issuer("nope"));
        assert!(t4.verify(&id).is_err());
    }

    #[test]
    fn self_signed_needs_the_matching_key() {
        let key = AgentKey::from_bytes(&[7; 32]);
        let other = AgentKey::from_bytes(&[8; 32]);
        let doc = AgentIdentity::new("solo", key.public_key());
        assert!(doc.clone().self_signed(&other).is_err());
        let s = doc.self_signed(&key).unwrap();
        assert!(TrustStore::new().trust(key.public_key()).verify(&s).is_ok());
        assert!(AgentCredentials::new(other, s).is_err());
    }

    #[test]
    fn verifier_rejects_forged_replayed_and_misdirected_assertions() {
        let (_issuer, key, id, trust) = setup();
        let creds = AgentCredentials::new(key.clone(), id.clone()).unwrap();
        assert_eq!(creds.agent_id(), "a");
        assert_eq!(creds.key().public_key(), key.public_key());
        assert_eq!(creds.identity(), &id);
        let v = IdentityVerifier::new(trust.clone()).audience("b");
        assert!(v.is_required());
        assert_eq!(v.configured_audience(), Some("b"));
        assert!(format!("{v:?}").contains("\"b\""));

        let ctx = CallContext::new("a").user("ana");
        let p = creds.prove(Some(&ctx), "b").unwrap();
        let caller = v.verify(&p).unwrap();
        assert_eq!(caller.agent(), "a");
        let next = caller.context_for("b", Via::A2a);
        assert_eq!(next.chain_agents(), ["a", "b"]);
        assert!(next.chain[1].verified);
        // Replay.
        assert!(matches!(v.verify(&p), Err(IdentityError::Replay(_))));
        // Wrong audience.
        let p = creds.prove(None, "c").unwrap();
        assert!(matches!(v.verify(&p), Err(IdentityError::Mismatch(_))));
        assert!(v.verify_for(&p, None).is_ok());
        // Confused deputy: context acting as another agent.
        assert!(creds.prove(Some(&CallContext::new("z")), "b").is_err());
        // An assertion signed by another key.
        let mut p = creds.prove(None, "b").unwrap();
        let other = AgentKey::from_bytes(&[9; 32]);
        p.assertion =
            Signed::sign(p.assertion.payload.clone(), &other, DOMAIN_CALL_ASSERTION).unwrap();
        assert!(matches!(v.verify(&p), Err(IdentityError::Mismatch(_))));
        // A forged context (re-signed by nobody).
        let mut p = creds.prove(None, "b").unwrap();
        p.assertion.payload.context.user = Some("admin".into());
        assert!(matches!(v.verify(&p), Err(IdentityError::Invalid(_))));
        // Agent mismatch and broken chain, validly signed.
        let mut bad = creds.prove(None, "b").unwrap().assertion.payload;
        bad.context.agent = "x".into();
        let p = IdentityProof {
            identity: id.clone(),
            assertion: Signed::sign(bad.clone(), &key, DOMAIN_CALL_ASSERTION).unwrap(),
        };
        assert!(matches!(v.verify(&p), Err(IdentityError::Mismatch(_))));
        bad.context.agent = "a".into();
        bad.context.chain = CallContext::new("x").chain;
        let p = IdentityProof {
            identity: id.clone(),
            assertion: Signed::sign(bad, &key, DOMAIN_CALL_ASSERTION).unwrap(),
        };
        assert!(matches!(v.verify(&p), Err(IdentityError::Mismatch(_))));
        // Expired, future and too-long assertions.
        for (iat, exp, short) in [
            (0u64, 1u64, false),
            (now_ms() + 3_600_000, now_ms() + 3_600_001, false),
            (now_ms(), now_ms() + 3_600_000, true),
        ] {
            let mut a = creds.prove(None, "b").unwrap().assertion.payload;
            a.issued_at_ms = iat;
            a.expires_at_ms = exp;
            let p = IdentityProof {
                identity: id.clone(),
                assertion: Signed::sign(a, &key, DOMAIN_CALL_ASSERTION).unwrap(),
            };
            let e = v.verify(&p).unwrap_err();
            if short {
                assert!(matches!(e, IdentityError::Invalid(_)), "{e}");
            } else {
                assert!(matches!(e, IdentityError::Expired(_)), "{e}");
            }
        }
        // A revoked agent key.
        trust.revoke_key(key.public_key().key_id());
        let p = creds.prove(None, "b").unwrap();
        assert!(matches!(v.verify(&p), Err(IdentityError::Untrusted(_))));
    }

    #[test]
    fn verify_value_handles_missing_and_malformed_proofs() {
        let (_i, key, id, trust) = setup();
        let creds = AgentCredentials::new(key, id)
            .unwrap()
            .assertion_ttl(Duration::from_secs(10));
        let req = IdentityVerifier::new(trust.clone());
        assert!(matches!(
            req.verify_value(None, Some("b")),
            Err(IdentityError::Missing(_))
        ));
        assert!(matches!(
            req.verify_value(Some(&json!({"x": 1})), Some("b")),
            Err(IdentityError::Invalid(_))
        ));
        let p = creds.prove(None, "b").unwrap().to_value();
        assert_eq!(
            req.verify_value(Some(&p), Some("b"))
                .unwrap()
                .unwrap()
                .agent(),
            "a"
        );
        let opt = IdentityVerifier::new(trust)
            .required(false)
            .max_assertion_ttl(Duration::from_secs(5));
        assert!(opt
            .verify_value(Some(&Value::Null), None)
            .unwrap()
            .is_none());
        // 10 s window is above the 5 s maximum.
        let p = creds.prove(None, "b").unwrap().to_value();
        assert!(opt.verify_value(Some(&p), None).is_err());
        assert!(!opt.trust_store().is_key_revoked("x"));
    }
}
