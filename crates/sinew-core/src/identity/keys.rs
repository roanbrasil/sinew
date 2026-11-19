//! Ed25519 keys and the generic signed envelope [`Signed`].

use std::fmt;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use super::{canonical_json, IdentityError};

pub(crate) fn hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(H[(b >> 4) as usize] as char);
        s.push(H[(b & 15) as usize] as char);
    }
    s
}

pub(crate) fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

/// An agent's (or an issuer's) Ed25519 private key.
///
/// It signs [`AgentIdentity`](super::AgentIdentity) documents (as an
/// issuer), call assertions and any [`Signed`] payload (as an agent). Keep
/// it in a secret manager; `Debug` prints only the key id.
///
/// ```
/// use sinew_core::identity::{AgentKey, AgentPublicKey};
///
/// let key = AgentKey::generate().unwrap();
/// let sig = key.sign(b"hello");
/// assert!(key.public_key().verify(b"hello", &sig));
///
/// let again = AgentKey::from_hex(&key.to_hex()).unwrap();
/// assert_eq!(again.public_key(), key.public_key());
/// let public = AgentPublicKey::from_hex(&key.public_key().to_hex()).unwrap();
/// assert_eq!(public.key_id().len(), 16);
/// ```
#[derive(Clone)]
pub struct AgentKey {
    key: SigningKey,
}

impl fmt::Debug for AgentKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentKey")
            .field("key_id", &self.public_key().key_id())
            .finish_non_exhaustive()
    }
}

impl AgentKey {
    /// A key from a 32-byte Ed25519 secret key (seed).
    pub fn from_bytes(secret: &[u8; 32]) -> Self {
        Self {
            key: SigningKey::from_bytes(secret),
        }
    }

    /// A key from the secret key in hex (64 characters).
    pub fn from_hex(secret: &str) -> Result<Self, IdentityError> {
        let bytes: [u8; 32] = unhex(secret)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| IdentityError::Invalid("signing key: need 32 bytes of hex".into()))?;
        Ok(Self::from_bytes(&bytes))
    }

    /// A new random key from the operating system's generator.
    pub fn generate() -> Result<Self, IdentityError> {
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed).map_err(|e| IdentityError::Invalid(e.to_string()))?;
        Ok(Self::from_bytes(&seed))
    }

    /// The secret key in hex.
    pub fn to_hex(&self) -> String {
        hex(self.key.as_bytes())
    }

    pub fn public_key(&self) -> AgentPublicKey {
        AgentPublicKey {
            key: self.key.verifying_key(),
        }
    }

    /// Signs raw bytes (a 64-byte Ed25519 signature).
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        self.key.sign(message).to_bytes()
    }
}

/// An Ed25519 public key. Serialized as hex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentPublicKey {
    key: VerifyingKey,
}

impl AgentPublicKey {
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, IdentityError> {
        VerifyingKey::from_bytes(bytes)
            .map(|key| Self { key })
            .map_err(|e| IdentityError::Invalid(format!("public key: {e}")))
    }

    /// A public key in hex (64 characters).
    pub fn from_hex(s: &str) -> Result<Self, IdentityError> {
        let bytes: [u8; 32] = unhex(s)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| IdentityError::Invalid("public key: need 32 bytes of hex".into()))?;
        Self::from_bytes(&bytes)
    }

    pub fn to_hex(&self) -> String {
        hex(self.key.as_bytes())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        self.key.as_bytes()
    }

    /// The first 16 hex characters of the SHA-256 of the key: tells keys
    /// apart (and is the JWS `kid` in signed A2A cards).
    pub fn key_id(&self) -> String {
        hex(&Sha256::digest(self.key.as_bytes())[..8])
    }

    /// Whether `signature` is this key's signature of `message`.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> bool {
        let Ok(bytes) = <[u8; 64]>::try_from(signature) else {
            return false;
        };
        self.key
            .verify(message, &Signature::from_bytes(&bytes))
            .is_ok()
    }
}

impl Serialize for AgentPublicKey {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for AgentPublicKey {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// A payload with an Ed25519 signature over its canonical JSON, prefixed by
/// a domain string so a signature made for one purpose (an identity
/// document, a call assertion, an inter-agent message) never verifies for
/// another.
///
/// ```
/// use serde_json::json;
/// use sinew_core::identity::{AgentKey, Signed};
///
/// let key = AgentKey::from_bytes(&[1; 32]);
/// let signed = Signed::sign(json!({"approve": "refund-42"}), &key, "my-app-approval-v1").unwrap();
/// assert!(signed.verify(&key.public_key(), "my-app-approval-v1").is_ok());
/// assert!(signed.verify(&key.public_key(), "another-purpose").is_err());
///
/// let mut forged = signed.clone();
/// forged.payload["approve"] = json!("refund-43");
/// assert!(forged.verify(&key.public_key(), "my-app-approval-v1").is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signed<T> {
    pub payload: T,
    /// [`AgentPublicKey::key_id`] of the signing key.
    pub key_id: String,
    /// The Ed25519 signature in hex.
    pub signature: String,
}

fn signing_input<T: Serialize>(payload: &T, domain: &str) -> Result<Vec<u8>, IdentityError> {
    let value = serde_json::to_value(payload)
        .map_err(|e| IdentityError::Invalid(format!("payload: {e}")))?;
    Ok(format!("{domain}\n{}", canonical_json(&value)).into_bytes())
}

impl<T: Serialize> Signed<T> {
    /// Signs `payload` with `key` for `domain`.
    pub fn sign(payload: T, key: &AgentKey, domain: &str) -> Result<Self, IdentityError> {
        let sig = key.sign(&signing_input(&payload, domain)?);
        Ok(Self {
            payload,
            key_id: key.public_key().key_id(),
            signature: hex(&sig),
        })
    }

    /// The payload, if `key` signed it for `domain`.
    pub fn verify(&self, key: &AgentPublicKey, domain: &str) -> Result<&T, IdentityError> {
        if self.key_id != key.key_id() {
            return Err(IdentityError::Mismatch(format!(
                "signed with key {}, expected {}",
                self.key_id,
                key.key_id()
            )));
        }
        let sig = unhex(&self.signature)
            .ok_or_else(|| IdentityError::Invalid("signature: not hex".into()))?;
        if key.verify(&signing_input(&self.payload, domain)?, &sig) {
            Ok(&self.payload)
        } else {
            Err(IdentityError::Invalid("signature does not verify".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keys_round_trip_and_reject_garbage() {
        let k = AgentKey::from_bytes(&[3; 32]);
        assert!(!format!("{k:?}").contains(&k.to_hex()));
        assert!(AgentKey::from_hex("abc").is_err());
        assert!(AgentKey::from_hex("zz").is_err());
        assert!(AgentPublicKey::from_hex("xyz").is_err());
        let p = k.public_key();
        assert_eq!(AgentPublicKey::from_hex(&p.to_hex()).unwrap(), p);
        assert_eq!(AgentPublicKey::from_bytes(p.as_bytes()).unwrap(), p);
        let v = serde_json::to_value(p).unwrap();
        assert_eq!(v, json!(p.to_hex()));
        assert_eq!(serde_json::from_value::<AgentPublicKey>(v).unwrap(), p);
        assert!(serde_json::from_value::<AgentPublicKey>(json!("00")).is_err());
        assert!(!p.verify(b"x", &[0; 10]));
        assert_ne!(AgentKey::generate().unwrap().to_hex(), k.to_hex());
    }

    #[test]
    fn signed_envelopes_detect_tampering_and_wrong_keys() {
        let k = AgentKey::from_bytes(&[4; 32]);
        let other = AgentKey::from_bytes(&[5; 32]);
        let s = Signed::sign(json!({"b": 1, "a": [1, 2]}), &k, "d").unwrap();
        assert_eq!(s.verify(&k.public_key(), "d").unwrap()["b"], 1);
        assert!(matches!(
            s.verify(&other.public_key(), "d"),
            Err(IdentityError::Mismatch(_))
        ));
        let mut bad = s.clone();
        bad.signature = "nothex".into();
        assert!(matches!(
            bad.verify(&k.public_key(), "d"),
            Err(IdentityError::Invalid(_))
        ));
        let mut bad = s.clone();
        bad.signature = hex(&[0; 64]);
        assert!(bad.verify(&k.public_key(), "d").is_err());
        // Key order does not matter: the signature covers canonical JSON.
        let reordered: Signed<serde_json::Value> = serde_json::from_str(&format!(
            r#"{{"payload":{{"a":[1,2],"b":1}},"key_id":"{}","signature":"{}"}}"#,
            s.key_id, s.signature
        ))
        .unwrap();
        assert!(reordered.verify(&k.public_key(), "d").is_ok());
        assert_eq!(unhex("0"), None);
    }
}
