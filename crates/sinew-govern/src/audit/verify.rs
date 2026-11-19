//! Checking an audit log: hashes, links, sequence and checkpoint
//! signatures.

use std::fmt;

use serde::Serialize;

use super::{AuditEntry, AuditError, AuditKind, AuditPublicKey, AuditStore, GENESIS_HASH};

/// What [`ChainVerifier`] checks besides the chain itself.
#[derive(Debug, Clone, Default)]
pub struct VerifyOptions {
    pub(crate) public_keys: Vec<AuditPublicKey>,
    pub(crate) expected_head: Option<(u64, String)>,
}

impl VerifyOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// A key checkpoints must be signed with. Add every key used over the
    /// life of the log after a rotation. With no key, checkpoint signatures
    /// are not checked (and a forged log with self-signed checkpoints
    /// passes), so always pass the key when auditing.
    pub fn public_key(mut self, key: AuditPublicKey) -> Self {
        self.public_keys.push(key);
        self
    }

    /// The log must contain entry `seq` with this `hash`: a head recorded
    /// elsewhere earlier (see [`AuditLog::head`](super::AuditLog::head)).
    /// Detects entries removed from the end of the log, which the chain
    /// alone cannot.
    pub fn expect_head(mut self, seq: u64, hash: impl Into<String>) -> Self {
        self.expected_head = Some((seq, hash.into()));
        self
    }
}

/// One inconsistency found in an audit log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "problem", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AuditProblem {
    /// The entry's hash does not match its content: it was edited.
    HashMismatch { seq: u64 },
    /// The entry does not point at the previous entry's hash: an entry
    /// before it was edited, removed or moved.
    BrokenLink { seq: u64 },
    /// Entries are missing or out of order.
    OutOfSequence { expected: u64, found: u64 },
    /// A checkpoint's signature does not verify.
    BadSignature { seq: u64 },
    /// A checkpoint is signed with a key that is not among the expected
    /// public keys.
    UnknownKey { seq: u64, key_id: String },
    /// A checkpoint entry without a `key_id` or `signature`.
    MalformedCheckpoint { seq: u64 },
    /// The expected head ([`VerifyOptions::expect_head`]) is missing or has
    /// another hash: entries were removed from the end or rewritten.
    HeadMismatch {
        seq: u64,
        expected: String,
        found: Option<String>,
    },
}

impl fmt::Display for AuditProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuditProblem::HashMismatch { seq } => {
                write!(f, "entry {seq}: hash does not match its content")
            }
            AuditProblem::BrokenLink { seq } => {
                write!(
                    f,
                    "entry {seq}: prev_hash does not match the previous entry"
                )
            }
            AuditProblem::OutOfSequence { expected, found } => {
                write!(f, "expected entry {expected}, found entry {found}")
            }
            AuditProblem::BadSignature { seq } => {
                write!(f, "checkpoint {seq}: invalid signature")
            }
            AuditProblem::UnknownKey { seq, key_id } => {
                write!(f, "checkpoint {seq}: signed with unknown key {key_id}")
            }
            AuditProblem::MalformedCheckpoint { seq } => {
                write!(f, "checkpoint {seq}: missing key_id or signature")
            }
            AuditProblem::HeadMismatch {
                seq,
                expected,
                found: None,
            } => write!(f, "expected head entry {seq} ({expected}) is missing"),
            AuditProblem::HeadMismatch {
                seq,
                expected,
                found: Some(found),
            } => write!(f, "entry {seq} has hash {found}, expected {expected}"),
        }
    }
}

/// The outcome of a verification.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct AuditReport {
    /// Entries read.
    pub entries: u64,
    /// Checkpoint entries read.
    pub checkpoints: u64,
    /// Checkpoints whose signature verified with one of the public keys.
    pub verified_checkpoints: u64,
    /// `seq` and `hash` of the last entry read.
    pub head: Option<(u64, String)>,
    /// Entries after the last verified checkpoint (all of them when none
    /// verified): protected by the chain against partial edits, but not
    /// against a full rewrite of that tail.
    pub unsigned_tail: u64,
    /// Every inconsistency, in log order. Empty when the log is intact.
    pub problems: Vec<AuditProblem>,
}

impl AuditReport {
    /// No problem found.
    pub fn is_ok(&self) -> bool {
        self.problems.is_empty()
    }
}

impl fmt::Display for AuditReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.is_ok() { "OK" } else { "TAMPERED" };
        write!(
            f,
            "{status}: {} entries, {} checkpoints ({} verified), {} entries after the last verified checkpoint",
            self.entries, self.checkpoints, self.verified_checkpoints, self.unsigned_tail
        )?;
        if let Some((seq, hash)) = &self.head {
            write!(f, "\nhead: {seq} {hash}")?;
        }
        for p in &self.problems {
            write!(f, "\n- {p}")?;
        }
        Ok(())
    }
}

/// Checks entries one at a time, in log order, so logs of any size can be
/// verified while streaming them.
///
/// ```
/// use serde_json::json;
/// use sinew_govern::audit::{AuditEntry, AuditKind, AuditProblem, ChainVerifier, VerifyOptions, GENESIS_HASH};
///
/// let a = AuditEntry::new(0, 1, AuditKind::Other, json!("a"), GENESIS_HASH.into());
/// let b = AuditEntry::new(1, 2, AuditKind::Other, json!("b"), a.hash.clone());
/// let c = AuditEntry::new(2, 3, AuditKind::Other, json!("c"), b.hash.clone());
///
/// let mut v = ChainVerifier::new(VerifyOptions::new());
/// for e in [&a, &c] { v.push(e); } // `b` deleted
/// let report = v.finish();
/// assert!(report.problems.contains(&AuditProblem::OutOfSequence { expected: 1, found: 2 }));
/// assert!(report.problems.contains(&AuditProblem::BrokenLink { seq: 2 }));
/// ```
#[derive(Debug)]
pub struct ChainVerifier {
    options: VerifyOptions,
    next_seq: u64,
    prev_hash: String,
    head_seen: Option<String>,
    report: AuditReport,
}

impl ChainVerifier {
    pub fn new(options: VerifyOptions) -> Self {
        Self {
            options,
            next_seq: 0,
            prev_hash: GENESIS_HASH.into(),
            head_seen: None,
            report: AuditReport::default(),
        }
    }

    /// Checks the next entry.
    pub fn push(&mut self, entry: &AuditEntry) {
        let r = &mut self.report;
        r.entries += 1;
        r.unsigned_tail += 1;
        if entry.compute_hash() != entry.hash {
            r.problems
                .push(AuditProblem::HashMismatch { seq: entry.seq });
        }
        if entry.seq != self.next_seq {
            r.problems.push(AuditProblem::OutOfSequence {
                expected: self.next_seq,
                found: entry.seq,
            });
        }
        if entry.prev_hash != self.prev_hash {
            r.problems.push(AuditProblem::BrokenLink { seq: entry.seq });
        }
        if entry.kind == AuditKind::Checkpoint {
            r.checkpoints += 1;
            self.check_checkpoint(entry);
        }
        if let Some((seq, _)) = &self.options.expected_head {
            if *seq == entry.seq {
                self.head_seen = Some(entry.hash.clone());
            }
        }
        self.next_seq = entry.seq.saturating_add(1);
        self.prev_hash.clone_from(&entry.hash);
        self.report.head = Some((entry.seq, entry.hash.clone()));
    }

    fn check_checkpoint(&mut self, entry: &AuditEntry) {
        let r = &mut self.report;
        let (Some(key_id), Some(signature)) = (
            entry.payload.get("key_id").and_then(|v| v.as_str()),
            entry.payload.get("signature").and_then(|v| v.as_str()),
        ) else {
            r.problems
                .push(AuditProblem::MalformedCheckpoint { seq: entry.seq });
            return;
        };
        if self.options.public_keys.is_empty() {
            return;
        }
        let Some(key) = self
            .options
            .public_keys
            .iter()
            .find(|k| k.key_id() == key_id)
        else {
            r.problems.push(AuditProblem::UnknownKey {
                seq: entry.seq,
                key_id: key_id.to_string(),
            });
            return;
        };
        if key.verify_checkpoint(entry.seq, &entry.prev_hash, entry.time_ms, signature) {
            r.verified_checkpoints += 1;
            r.unsigned_tail = 0;
        } else {
            r.problems
                .push(AuditProblem::BadSignature { seq: entry.seq });
        }
    }

    /// The report, after the expected head check.
    pub fn finish(mut self) -> AuditReport {
        if let Some((seq, expected)) = self.options.expected_head.take() {
            if self.head_seen.as_deref() != Some(expected.as_str()) {
                self.report.problems.push(AuditProblem::HeadMismatch {
                    seq,
                    expected,
                    found: self.head_seen,
                });
            }
        }
        self.report
    }
}

/// Reads every entry of `store`, by `seq`, and verifies the chain.
pub async fn verify_store(
    store: &dyn AuditStore,
    options: &VerifyOptions,
) -> Result<AuditReport, AuditError> {
    const PAGE: usize = 1000;
    let mut v = ChainVerifier::new(options.clone());
    let mut from = 0;
    loop {
        let page = store.entries(from, PAGE).await?;
        for e in &page {
            v.push(e);
        }
        match page.last() {
            Some(last) if page.len() == PAGE => from = last.seq.saturating_add(1),
            _ => break,
        }
    }
    Ok(v.finish())
}
