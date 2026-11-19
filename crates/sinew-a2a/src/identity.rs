//! Signed Agent Cards and verified callers.
//!
//! - **Card signatures** (A2A `AgentCard.signatures`): a JWS (RFC 7515)
//!   with `alg: EdDSA` over the card's canonical JSON (RFC 8785) without its
//!   `signatures`, made by the agent's own key ([`AgentCard::sign`]).
//! - **Identity extension**: the agent's [`SignedAgentIdentity`] (id, owner,
//!   version, capabilities, allowed tools, public key, signed by an issuer)
//!   in `capabilities.extensions` under [`IDENTITY_EXTENSION_URI`]
//!   ([`AgentCard::with_identity`]).
//! - [`AgentCard::verify`] checks both against a [`TrustStore`]: the
//!   identity is signed by a trusted issuer and valid, and the card is
//!   signed by the key that identity names. [`A2aClient::discover_verified`](crate::A2aClient::discover_verified)
//!   refuses agents whose card does not verify.
//! - **Calls**: a client with [`AgentCredentials`](sinew_core::identity::AgentCredentials)
//!   ([`A2aClient::with_credentials`](crate::A2aClient::with_credentials))
//!   signs a call assertion of the current
//!   [`CallContext`](sinew_core::identity::CallContext) on every request;
//!   a server with an [`IdentityVerifier`](sinew_core::identity::IdentityVerifier)
//!   ([`A2aServer::with_identity_verifier`](crate::A2aServer::with_identity_verifier))
//!   checks it and runs the executor in the caller's context, with this
//!   agent appended to the delegation chain.
//!
//! A verified card proves who issued the document and what it allows; it
//! does not prove that the server answering is the agent (anyone can serve
//! a copy of a public card). Use TLS for the transport.

use serde_json::{json, Map, Value};
use sinew_core::identity::{
    canonical_json, AgentKey, IdentityError, SignedAgentIdentity, TrustStore, VerifiedIdentity,
};

use crate::types::{AgentCard, AgentCardSignature, AgentExtension};

/// The extension URI under which a card carries the agent's signed identity
/// (in `params.identity`).
pub const IDENTITY_EXTENSION_URI: &str =
    "https://github.com/roanbrasil/sinew/blob/main/docs/governance.md#agent-identity";

const B64URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Base64url without padding (RFC 4648 section 5), as JWS uses.
pub(crate) fn b64url_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        let chars = chunk.len() + 1;
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            if i < chars {
                out.push(B64URL[(n >> shift) as usize & 63] as char);
            }
        }
    }
    out
}

pub(crate) fn b64url_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim_end_matches('=');
    let val = |c: u8| B64URL.iter().position(|x| *x == c).map(|p| p as u32);
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    for chunk in s.as_bytes().chunks(4) {
        if chunk.len() == 1 {
            return None;
        }
        let mut n = 0u32;
        for (i, c) in chunk.iter().enumerate() {
            n |= val(*c)? << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    Some(out)
}

/// The JWS payload of a card: its canonical JSON without `signatures`.
fn card_payload(card: &AgentCard) -> Result<String, IdentityError> {
    let mut v =
        serde_json::to_value(card).map_err(|e| IdentityError::Invalid(format!("card: {e}")))?;
    if let Some(o) = v.as_object_mut() {
        o.remove("signatures");
    }
    Ok(canonical_json(&v))
}

impl AgentCard {
    /// Adds the agent's signed identity to the card (the extension
    /// [`IDENTITY_EXTENSION_URI`]), replacing a previous one.
    pub fn with_identity(mut self, identity: &SignedAgentIdentity) -> Self {
        self.capabilities
            .extensions
            .retain(|e| e.uri != IDENTITY_EXTENSION_URI);
        let mut params = Map::new();
        params.insert(
            "identity".into(),
            serde_json::to_value(identity).unwrap_or(Value::Null),
        );
        self.capabilities.extensions.push(AgentExtension {
            uri: IDENTITY_EXTENSION_URI.into(),
            description: Some("Signed Sinew agent identity".into()),
            required: Some(false),
            params: Some(params),
        });
        self
    }

    /// The signed identity the card carries, if any.
    pub fn identity(&self) -> Option<SignedAgentIdentity> {
        self.capabilities
            .extensions
            .iter()
            .find(|e| e.uri == IDENTITY_EXTENSION_URI)
            .and_then(|e| e.params.as_ref()?.get("identity").cloned())
            .and_then(|v| serde_json::from_value(v).ok())
    }

    /// The agent id other agents should address (the identity's id, else
    /// the card name): the audience of call assertions.
    pub fn agent_id(&self) -> String {
        self.identity()
            .map(|i| i.payload.id)
            .unwrap_or_else(|| self.name.clone())
    }

    /// Signs the card with `key` (a JWS with `alg: EdDSA` and `kid` = the
    /// key id over the card without its signatures), replacing earlier
    /// signatures by the same key. Sign last: any later change to the card
    /// breaks the signature.
    ///
    /// ```
    /// use sinew_a2a::AgentCard;
    /// use sinew_core::identity::{AgentIdentity, AgentKey, TrustStore};
    ///
    /// let issuer = AgentKey::from_bytes(&[1; 32]);
    /// let agent_key = AgentKey::from_bytes(&[2; 32]);
    /// let identity = AgentIdentity::new("billing", agent_key.public_key())
    ///     .owner("payments-team")
    ///     .allow_tools(["refund"])
    ///     .sign(&issuer)
    ///     .unwrap();
    /// let card = AgentCard::new("billing", "Refunds and invoices", "https://billing.example/a2a")
    ///     .with_identity(&identity)
    ///     .sign(&agent_key)
    ///     .unwrap();
    ///
    /// let trust = TrustStore::new().trust(issuer.public_key());
    /// assert_eq!(card.verify(&trust).unwrap().identity.id, "billing");
    ///
    /// let mut tampered = card.clone();
    /// tampered.url = "https://evil.example/a2a".into();
    /// assert!(tampered.verify(&trust).is_err());
    /// ```
    pub fn sign(mut self, key: &AgentKey) -> Result<Self, IdentityError> {
        let kid = key.public_key().key_id();
        self.signatures
            .retain(|s| protected_kid(s).as_deref() != Some(&kid));
        let protected = b64url_encode(
            json!({"alg": "EdDSA", "kid": kid, "typ": "JOSE"})
                .to_string()
                .as_bytes(),
        );
        let input = format!(
            "{protected}.{}",
            b64url_encode(card_payload(&self)?.as_bytes())
        );
        let signature = b64url_encode(&key.sign(input.as_bytes()));
        self.signatures.push(AgentCardSignature {
            protected,
            signature,
            header: None,
        });
        Ok(self)
    }

    /// Verifies the card: its identity extension against `trust`, and a
    /// card signature by the key that identity names.
    pub fn verify(&self, trust: &TrustStore) -> Result<VerifiedIdentity, IdentityError> {
        let signed = self.identity().ok_or_else(|| {
            IdentityError::Missing(format!("signed identity in the card of `{}`", self.name))
        })?;
        let verified = trust.verify(&signed)?;
        let key = verified.identity.public_key;
        let kid = key.key_id();
        let sig = self
            .signatures
            .iter()
            .find(|s| protected_kid(s).as_deref() == Some(&kid))
            .ok_or_else(|| {
                IdentityError::Missing(format!(
                    "card signature by the key of `{}` ({kid})",
                    verified.identity.id
                ))
            })?;
        let header = protected_header(sig)
            .ok_or_else(|| IdentityError::Invalid("card signature header".into()))?;
        if header.get("alg").and_then(Value::as_str) != Some("EdDSA") {
            return Err(IdentityError::Invalid(
                "card signature: alg must be EdDSA".into(),
            ));
        }
        let input = format!(
            "{}.{}",
            sig.protected,
            b64url_encode(card_payload(self)?.as_bytes())
        );
        let bytes = b64url_decode(&sig.signature)
            .ok_or_else(|| IdentityError::Invalid("card signature encoding".into()))?;
        if !key.verify(input.as_bytes(), &bytes) {
            return Err(IdentityError::Invalid(format!(
                "the card of `{}` does not match its signature",
                self.name
            )));
        }
        Ok(verified)
    }
}

fn protected_header(sig: &AgentCardSignature) -> Option<Value> {
    serde_json::from_slice(&b64url_decode(&sig.protected)?).ok()
}

fn protected_kid(sig: &AgentCardSignature) -> Option<String> {
    protected_header(sig)?
        .get("kid")?
        .as_str()
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sinew_core::identity::AgentIdentity;

    #[test]
    fn base64url_round_trips() {
        for (raw, enc) in [
            (&b""[..], ""),
            (b"f", "Zg"),
            (b"fo", "Zm8"),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg"),
            (&[0xfb, 0xff][..], "-_8"),
        ] {
            assert_eq!(b64url_encode(raw), enc);
            assert_eq!(b64url_decode(enc).unwrap(), raw);
        }
        assert_eq!(b64url_decode("Zg==").unwrap(), b"f");
        assert!(b64url_decode("Z").is_none());
        assert!(b64url_decode("Z*").is_none());
    }

    fn signed_card() -> (AgentKey, AgentKey, AgentCard) {
        let issuer = AgentKey::from_bytes(&[1; 32]);
        let key = AgentKey::from_bytes(&[2; 32]);
        let id = AgentIdentity::new("billing-agent", key.public_key())
            .sign(&issuer)
            .unwrap();
        let card = AgentCard::new("Billing", "d", "http://x/")
            .with_identity(&id)
            .with_identity(&id)
            .sign(&key)
            .unwrap()
            .sign(&key)
            .unwrap();
        (issuer, key, card)
    }

    #[test]
    fn verifies_cards_and_rejects_bad_ones() {
        let (issuer, key, card) = signed_card();
        assert_eq!(card.capabilities.extensions.len(), 1);
        assert_eq!(card.signatures.len(), 1, "re-signing replaces");
        assert_eq!(card.agent_id(), "billing-agent");
        let trust = TrustStore::new().trust(issuer.public_key());
        assert!(card.verify(&trust).is_ok());
        // Survives a JSON round trip.
        let back: AgentCard = serde_json::from_str(&serde_json::to_string(&card).unwrap()).unwrap();
        assert!(back.verify(&trust).is_ok());

        // No identity, untrusted issuer, unsigned, wrong key, bad alg/encoding.
        let plain = AgentCard::new("p", "d", "u");
        assert_eq!(plain.agent_id(), "p");
        assert!(matches!(
            plain.verify(&trust),
            Err(IdentityError::Missing(_))
        ));
        assert!(matches!(
            card.verify(&TrustStore::new()),
            Err(IdentityError::Untrusted(_))
        ));
        let mut unsigned = card.clone();
        unsigned.signatures.clear();
        assert!(matches!(
            unsigned.verify(&trust),
            Err(IdentityError::Missing(_))
        ));
        let other = AgentCard {
            signatures: Vec::new(),
            ..card.clone()
        }
        .sign(&AgentKey::from_bytes(&[3; 32]))
        .unwrap();
        assert!(other.verify(&trust).is_err());
        let mut bad = card.clone();
        bad.signatures[0].protected = b64url_encode(
            json!({"alg": "none", "kid": key.public_key().key_id()})
                .to_string()
                .as_bytes(),
        );
        assert!(matches!(bad.verify(&trust), Err(IdentityError::Invalid(_))));
        let mut bad = card.clone();
        bad.signatures[0].signature = "***".into();
        assert!(matches!(bad.verify(&trust), Err(IdentityError::Invalid(_))));
        let mut bad = card;
        bad.skills.push(crate::types::AgentSkill::new(
            "steal",
            "steal",
            "exfiltrate",
        ));
        assert!(matches!(bad.verify(&trust), Err(IdentityError::Invalid(_))));
    }
}
