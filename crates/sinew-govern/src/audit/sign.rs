//! Ed25519 keys for audit checkpoints.

use std::fmt;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};

use super::{hex, unhex, AuditError};

/// The bytes a checkpoint signs: the checkpoint's own sequence number, the
/// hash of the entry before it and its time.
pub(crate) fn checkpoint_message(seq: u64, prev_hash: &str, time_ms: u64) -> Vec<u8> {
    format!("sinew-audit-checkpoint-v1\n{seq}\n{prev_hash}\n{time_ms}").into_bytes()
}

/// Signs audit checkpoints (an Ed25519 private key).
///
/// Keep the key out of the store the log is written to: whoever holds it
/// can rewrite the log undetected. `Debug` never prints it.
///
/// ```
/// use sinew_govern::audit::{AuditPublicKey, AuditSigner};
///
/// let signer = AuditSigner::generate().unwrap();
/// let again = AuditSigner::from_hex(&signer.to_hex()).unwrap();
/// assert_eq!(again.public_key(), signer.public_key());
///
/// let public = AuditPublicKey::from_hex(&signer.public_key().to_hex()).unwrap();
/// assert_eq!(public.key_id().len(), 16);
/// ```
#[derive(Clone)]
pub struct AuditSigner {
    key: SigningKey,
}

impl fmt::Debug for AuditSigner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuditSigner")
            .field("key_id", &self.public_key().key_id())
            .finish_non_exhaustive()
    }
}

impl AuditSigner {
    /// A signer from a 32-byte Ed25519 secret key (seed).
    pub fn from_bytes(secret: &[u8; 32]) -> Self {
        Self {
            key: SigningKey::from_bytes(secret),
        }
    }

    /// A signer from the secret key in hex (64 characters).
    pub fn from_hex(secret: &str) -> Result<Self, AuditError> {
        let bytes: [u8; 32] = unhex(secret)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| AuditError::Invalid("signing key must be 32 bytes of hex".into()))?;
        Ok(Self::from_bytes(&bytes))
    }

    /// A new random key from the operating system's generator.
    pub fn generate() -> Result<Self, AuditError> {
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed).map_err(|e| AuditError::Invalid(e.to_string()))?;
        Ok(Self::from_bytes(&seed))
    }

    /// The secret key in hex, to store in a secret manager.
    pub fn to_hex(&self) -> String {
        hex(self.key.as_bytes())
    }

    /// The matching public key, for verification.
    pub fn public_key(&self) -> AuditPublicKey {
        AuditPublicKey {
            key: self.key.verifying_key(),
        }
    }

    pub(crate) fn sign_checkpoint(&self, seq: u64, prev_hash: &str, time_ms: u64) -> String {
        let sig = self.key.sign(&checkpoint_message(seq, prev_hash, time_ms));
        hex(&sig.to_bytes())
    }
}

/// Verifies audit checkpoints (an Ed25519 public key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditPublicKey {
    key: VerifyingKey,
}

impl AuditPublicKey {
    /// A public key from its 32 bytes.
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, AuditError> {
        VerifyingKey::from_bytes(bytes)
            .map(|key| Self { key })
            .map_err(|e| AuditError::Invalid(format!("public key: {e}")))
    }

    /// A public key in hex (64 characters).
    pub fn from_hex(s: &str) -> Result<Self, AuditError> {
        let bytes: [u8; 32] = unhex(s)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| AuditError::Invalid("public key must be 32 bytes of hex".into()))?;
        Self::from_bytes(&bytes)
    }

    pub fn to_hex(&self) -> String {
        hex(self.key.as_bytes())
    }

    /// The first 16 hex characters of the SHA-256 of the key, written in
    /// each checkpoint to tell keys apart after a rotation.
    pub fn key_id(&self) -> String {
        hex(&Sha256::digest(self.key.as_bytes())[..8])
    }

    /// Whether `signature` (hex) signs the checkpoint `seq` written after
    /// `prev_hash` at `time_ms`.
    pub(crate) fn verify_checkpoint(
        &self,
        seq: u64,
        prev_hash: &str,
        time_ms: u64,
        signature: &str,
    ) -> bool {
        let Some(bytes) = unhex(signature).and_then(|b| <[u8; 64]>::try_from(b).ok()) else {
            return false;
        };
        let sig = Signature::from_bytes(&bytes);
        self.key
            .verify(&checkpoint_message(seq, prev_hash, time_ms), &sig)
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signs_and_verifies_checkpoints() {
        let s = AuditSigner::from_bytes(&[1; 32]);
        let sig = s.sign_checkpoint(5, "ab", 10);
        let k = s.public_key();
        assert!(k.verify_checkpoint(5, "ab", 10, &sig));
        assert!(!k.verify_checkpoint(6, "ab", 10, &sig));
        assert!(!k.verify_checkpoint(5, "ac", 10, &sig));
        assert!(!k.verify_checkpoint(5, "ab", 11, &sig));
        assert!(!k.verify_checkpoint(5, "ab", 10, "zz"));
        let other = AuditSigner::from_bytes(&[2; 32]).public_key();
        assert!(!other.verify_checkpoint(5, "ab", 10, &sig));
        assert_ne!(other.key_id(), k.key_id());
        assert!(!format!("{s:?}").contains(&s.to_hex()));
    }

    #[test]
    fn rejects_bad_keys() {
        assert!(AuditSigner::from_hex("abcd").is_err());
        assert!(AuditPublicKey::from_hex("xyz").is_err());
        let k = AuditSigner::from_bytes(&[3; 32]).public_key();
        assert_eq!(AuditPublicKey::from_hex(&k.to_hex()).unwrap(), k);
    }
}
