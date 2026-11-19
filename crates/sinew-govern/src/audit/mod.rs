//! A tamper-evident audit log: an append-only, hash-chained record of
//! governance decisions, operator actions, policy changes, approvals and
//! tool calls with side effects, with periodic Ed25519-signed checkpoints.
//!
//! - Every [`AuditEntry`] carries a sequence number, a time, an
//!   [`AuditKind`], a JSON payload, the hash of the previous entry and its
//!   own SHA-256 hash (over a canonical JSON form of everything else). Edit,
//!   delete or reorder an entry and the chain no longer links up.
//! - A [`AuditKind::Checkpoint`] entry, written every
//!   [`checkpoint_every`](AuditLogBuilder::checkpoint_every) entries or on
//!   [`AuditLog::checkpoint`], signs the head of the chain with an
//!   [`AuditSigner`] (Ed25519). Someone with write access to the store but
//!   without the signing key can rewrite the chain, but not the signatures:
//!   anything before the last checkpoint is protected even against a full
//!   recomputation of the hashes.
//! - Payloads pass through a [`sinew_trace::Redactor`] (secrets,
//!   e-mails, card numbers, credential fields) before they are hashed and
//!   written.
//! - Stores: [`MemoryAuditStore`], `SqliteAuditStore` (feature
//!   `audit-sqlite`) and `PostgresAuditStore` (feature `audit-postgres`),
//!   or your own [`AuditStore`]. [`export_jsonl`] writes one entry per line;
//!   [`verify_store`], [`verify_jsonl`] and the [`ChainVerifier`] check a log,
//!   and so does the `sinew-govern verify-audit` command (feature `cli`).
//! - [`AuditLog`] is a [`DecisionSink`]: add it to a
//!   [`ControlPlane`](crate::ControlPlane) and every decision, operator
//!   action and policy change is appended.
//!
//! ```
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! use std::sync::Arc;
//! use serde_json::json;
//! use sinew_govern::audit::{AuditLog, AuditSigner, MemoryAuditStore, VerifyOptions};
//! use sinew_govern::{AgentProfile, ControlPlane, Request};
//!
//! let signer = AuditSigner::from_bytes(&[7; 32]);
//! let public_key = signer.public_key();
//! let store = Arc::new(MemoryAuditStore::new());
//! let audit = AuditLog::builder(store.clone()).signer(signer).checkpoint_every(100).build();
//!
//! let cp = ControlPlane::builder()
//!     .agent(AgentProfile::new("billing-bot"))
//!     .sink(Arc::new(audit.clone()))
//!     .build();
//! cp.check(Request::tool("billing-bot", "get_invoice", json!({"id": 7}))).await;
//! cp.suspend("billing-bot", "incident 42");
//! audit.flush().await;
//! audit.checkpoint().await.unwrap();
//!
//! let report = audit.verify(&VerifyOptions::new().public_key(public_key)).await.unwrap();
//! assert!(report.is_ok(), "{report}");
//! assert_eq!(report.entries, 3); // decision, suspension, checkpoint
//! assert_eq!(report.verified_checkpoints, 1);
//! # });
//! ```

mod jsonl;
mod sign;
mod verify;

#[cfg(feature = "cli")]
#[doc(hidden)]
pub mod cli;
#[cfg(feature = "audit-postgres")]
mod postgres;
#[cfg(feature = "audit-sqlite")]
mod sqlite;

/// The `sqlx` version the SQL stores use, for `from_pool` and test queries.
#[cfg(any(feature = "audit-sqlite", feature = "audit-postgres"))]
pub use sqlx;

/// Row conversions shared by the SQL stores.
#[cfg(any(feature = "audit-sqlite", feature = "audit-postgres"))]
mod db {
    use super::{AuditEntry, AuditError, AuditKind};

    pub(super) fn int(v: u64) -> Result<i64, AuditError> {
        i64::try_from(v).map_err(|_| AuditError::Invalid(format!("{v} does not fit in BIGINT")))
    }

    pub(super) fn entry(
        seq: i64,
        time_ms: i64,
        kind: String,
        payload: String,
        prev_hash: String,
        hash: String,
    ) -> Result<AuditEntry, AuditError> {
        let bad = |what: &str| AuditError::Invalid(format!("entry {seq}: {what}"));
        Ok(AuditEntry {
            seq: u64::try_from(seq).map_err(|_| bad("negative seq"))?,
            time_ms: u64::try_from(time_ms).map_err(|_| bad("negative time"))?,
            kind: AuditKind::parse(&kind).ok_or_else(|| bad(&format!("unknown kind `{kind}`")))?,
            payload: serde_json::from_str(&payload).map_err(|e| bad(&e.to_string()))?,
            prev_hash,
            hash,
        })
    }

    /// Maps a unique violation on `seq` to [`AuditError::Conflict`].
    pub(super) fn insert_result<R>(
        res: Result<R, sqlx::Error>,
        seq: u64,
    ) -> Result<(), AuditError> {
        match res {
            Ok(_) => Ok(()),
            Err(sqlx::Error::Database(e)) if e.is_unique_violation() => {
                Err(AuditError::Conflict(seq))
            }
            Err(e) => Err(e.into()),
        }
    }
}

use std::{
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock, Weak,
    },
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sinew_trace::Redactor;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

pub use jsonl::{export_jsonl, parse_jsonl_line, read_jsonl, verify_jsonl};
#[cfg(feature = "audit-postgres")]
pub use postgres::PostgresAuditStore;
pub use sign::{AuditPublicKey, AuditSigner};
#[cfg(feature = "audit-sqlite")]
pub use sqlite::SqliteAuditStore;
pub use verify::{verify_store, AuditProblem, AuditReport, ChainVerifier, VerifyOptions};

use crate::{
    request::now_ms,
    sink::{ControlAction, ControlEvent, DecisionEvent, DecisionSink},
};

/// `prev_hash` of the first entry: 64 zeros.
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Version of the hashing scheme, part of every entry hash.
pub const HASH_VERSION: u32 = 1;

/// Errors from audit stores, signing and verification input.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AuditError {
    /// The store failed (database error, I/O).
    #[error("audit store: {0}")]
    Store(String),
    /// Another writer appended an entry with the same sequence number.
    #[error("audit store: entry {0} already exists")]
    Conflict(u64),
    /// [`AuditLog::checkpoint`] without an [`AuditSigner`].
    #[error("audit log has no signer")]
    NoSigner,
    /// A key, signature or record could not be parsed.
    #[error("invalid audit data: {0}")]
    Invalid(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(any(feature = "audit-sqlite", feature = "audit-postgres"))]
impl From<sqlx::Error> for AuditError {
    fn from(e: sqlx::Error) -> Self {
        AuditError::Store(e.to_string())
    }
}

/// What an [`AuditEntry`] records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AuditKind {
    /// A [`DecisionEvent`] (payload: the event).
    Decision,
    /// An operator action, a [`ControlEvent`] (payload: the event).
    Control,
    /// A change to the policies (payload: the [`ControlEvent`]).
    PolicyChange,
    /// A human approval or rejection (payload: an [`ApprovalRecord`]).
    Approval,
    /// A tool call with side effects and its outcome (payload: a
    /// [`ToolCallRecord`]).
    ToolCall,
    /// A signed checkpoint of the chain (payload: `key_id`, `signature`).
    Checkpoint,
    /// Anything else (payload: free-form).
    Other,
}

impl AuditKind {
    /// The snake_case name, as stored.
    pub fn as_str(self) -> &'static str {
        match self {
            AuditKind::Decision => "decision",
            AuditKind::Control => "control",
            AuditKind::PolicyChange => "policy_change",
            AuditKind::Approval => "approval",
            AuditKind::ToolCall => "tool_call",
            AuditKind::Checkpoint => "checkpoint",
            AuditKind::Other => "other",
        }
    }

    /// Parses [`as_str`](Self::as_str).
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "decision" => AuditKind::Decision,
            "control" => AuditKind::Control,
            "policy_change" => AuditKind::PolicyChange,
            "approval" => AuditKind::Approval,
            "tool_call" => AuditKind::ToolCall,
            "checkpoint" => AuditKind::Checkpoint,
            "other" => AuditKind::Other,
            _ => return None,
        })
    }
}

impl fmt::Display for AuditKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One record of the audit log.
///
/// `hash` is the SHA-256 (hex) of the canonical JSON (object keys sorted,
/// no whitespace) of `{"v", "seq", "time_ms", "kind", "prev_hash",
/// "payload"}`, `v` being [`HASH_VERSION`]. See [`AuditEntry::compute_hash`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Position in the log, from 0, without gaps.
    pub seq: u64,
    /// When the entry was written (milliseconds since the Unix epoch).
    pub time_ms: u64,
    pub kind: AuditKind,
    /// What happened, already redacted.
    pub payload: Value,
    /// `hash` of the previous entry ([`GENESIS_HASH`] for the first).
    pub prev_hash: String,
    /// This entry's hash.
    pub hash: String,
}

impl AuditEntry {
    /// Builds an entry and computes its hash.
    pub fn new(seq: u64, time_ms: u64, kind: AuditKind, payload: Value, prev_hash: String) -> Self {
        let mut e = Self {
            seq,
            time_ms,
            kind,
            payload,
            prev_hash,
            hash: String::new(),
        };
        e.hash = e.compute_hash();
        e
    }

    /// The hash this entry should have, from its other fields.
    ///
    /// ```
    /// use serde_json::json;
    /// use sinew_govern::audit::{AuditEntry, AuditKind, GENESIS_HASH};
    ///
    /// let mut e = AuditEntry::new(0, 1, AuditKind::Other, json!({"a": 1}), GENESIS_HASH.into());
    /// assert_eq!(e.hash, e.compute_hash());
    /// e.payload = json!({"a": 2});
    /// assert_ne!(e.hash, e.compute_hash());
    /// ```
    pub fn compute_hash(&self) -> String {
        let doc = json!({
            "v": HASH_VERSION,
            "seq": self.seq,
            "time_ms": self.time_ms,
            "kind": self.kind.as_str(),
            "prev_hash": self.prev_hash,
            "payload": self.payload,
        });
        let mut s = String::new();
        canonical_json(&doc, &mut s);
        hex(&Sha256::digest(s.as_bytes()))
    }
}

/// Writes `value` as JSON with object keys sorted and no whitespace, so the
/// same value always hashes the same whatever map order it was built or
/// stored with.
pub fn canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(k.clone()).to_string());
                out.push(':');
                canonical_json(&map[k], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonical_json(v, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 0xf) as usize] as char);
    }
    s
}

pub(crate) fn unhex(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

/// A human's answer to an `Ask` decision, for [`AuditLog::record_approval`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalRecord {
    /// Id of the [`DecisionEvent`] that asked, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_id: Option<String>,
    pub agent: String,
    /// The tool or model the approval is for.
    pub target: String,
    /// Who answered.
    pub approver: String,
    pub approved: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// A tool call that changed something, for [`AuditLog::record_tool_call`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub agent: String,
    pub tool: String,
    pub arguments: Value,
    /// Id of the [`DecisionEvent`] that allowed it, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_id: Option<String>,
    /// The result, or `null`.
    #[serde(default)]
    pub output: Value,
    /// The error, when the call failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Where audit entries are kept. Append-only: there is no update or delete.
///
/// `append` must fail with [`AuditError::Conflict`] when an entry with the
/// same `seq` exists, so that two writers never fork the chain.
#[async_trait]
pub trait AuditStore: Send + Sync {
    async fn append(&self, entry: &AuditEntry) -> Result<(), AuditError>;

    /// The entry with the highest `seq`.
    async fn last(&self) -> Result<Option<AuditEntry>, AuditError>;

    /// Up to `limit` entries with `seq >= from`, by `seq`.
    async fn entries(&self, from: u64, limit: usize) -> Result<Vec<AuditEntry>, AuditError>;
}

/// Keeps entries in memory (tests, short-lived processes).
#[derive(Debug, Default)]
pub struct MemoryAuditStore {
    entries: Mutex<Vec<AuditEntry>>,
}

impl MemoryAuditStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// A copy of every entry.
    pub fn snapshot(&self) -> Vec<AuditEntry> {
        self.lock().clone()
    }

    /// Direct access to the entries, to simulate tampering in tests.
    #[doc(hidden)]
    pub fn tamper<R>(&self, f: impl FnOnce(&mut Vec<AuditEntry>) -> R) -> R {
        f(&mut self.lock())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<AuditEntry>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[async_trait]
impl AuditStore for MemoryAuditStore {
    async fn append(&self, entry: &AuditEntry) -> Result<(), AuditError> {
        let mut v = self.lock();
        if v.iter().any(|e| e.seq == entry.seq) {
            return Err(AuditError::Conflict(entry.seq));
        }
        v.push(entry.clone());
        Ok(())
    }

    async fn last(&self) -> Result<Option<AuditEntry>, AuditError> {
        Ok(self.lock().iter().max_by_key(|e| e.seq).cloned())
    }

    async fn entries(&self, from: u64, limit: usize) -> Result<Vec<AuditEntry>, AuditError> {
        let mut v: Vec<AuditEntry> = self
            .lock()
            .iter()
            .filter(|e| e.seq >= from)
            .cloned()
            .collect();
        v.sort_by_key(|e| e.seq);
        v.truncate(limit);
        Ok(v)
    }
}

/// The head of the chain as this process knows it.
#[derive(Debug, Clone)]
struct Head {
    next_seq: u64,
    prev_hash: String,
    since_checkpoint: u64,
}

enum Msg {
    Append(AuditKind, Value),
    Flush(oneshot::Sender<()>),
}

struct LogInner {
    store: Arc<dyn AuditStore>,
    head: tokio::sync::Mutex<Option<Head>>,
    redactor: Option<Redactor>,
    signer: Option<AuditSigner>,
    checkpoint_every: Option<u64>,
    failures: AtomicU64,
    last_error: Mutex<Option<String>>,
    queue: OnceLock<mpsc::UnboundedSender<Msg>>,
}

/// Builds an [`AuditLog`].
pub struct AuditLogBuilder {
    store: Arc<dyn AuditStore>,
    redactor: Option<Redactor>,
    signer: Option<AuditSigner>,
    checkpoint_every: Option<u64>,
}

impl AuditLogBuilder {
    /// The redactor applied to every payload before it is hashed and
    /// written (default: [`Redactor::new`], i.e. secrets, e-mails, card
    /// numbers and credential fields).
    pub fn redactor(mut self, redactor: Redactor) -> Self {
        self.redactor = Some(redactor);
        self
    }

    /// Writes payloads as they are. Only for payloads known to hold no
    /// secrets or personal data.
    pub fn no_redaction(mut self) -> Self {
        self.redactor = None;
        self
    }

    /// Signs checkpoints with `signer`.
    pub fn signer(mut self, signer: AuditSigner) -> Self {
        self.signer = Some(signer);
        self
    }

    /// Writes a signed checkpoint after every `n` entries (needs a
    /// [`signer`](Self::signer)). Entries are counted from the start of the
    /// process.
    pub fn checkpoint_every(mut self, n: u64) -> Self {
        self.checkpoint_every = (n > 0).then_some(n);
        self
    }

    pub fn build(self) -> AuditLog {
        AuditLog {
            inner: Arc::new(LogInner {
                store: self.store,
                head: tokio::sync::Mutex::new(None),
                redactor: self.redactor,
                signer: self.signer,
                checkpoint_every: self.checkpoint_every,
                failures: AtomicU64::new(0),
                last_error: Mutex::new(None),
                queue: OnceLock::new(),
            }),
        }
    }
}

/// Writes the audit log: hashes, chains, redacts and signs entries, and
/// appends them to an [`AuditStore`].
///
/// Cheap to clone; clones share the store and the head of the chain. One
/// `AuditLog` (or clones of it) should write a given store: a second writer
/// in another process is detected (the store reports a conflict) and this
/// log then reloads the head and retries.
///
/// As a [`DecisionSink`], events are queued and appended by a background
/// task (the control plane's calls are synchronous);
/// [`flush`](Self::flush) waits until the queue is written and
/// [`failures`](Self::failures) counts the events that could not be.
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use std::sync::Arc;
/// use serde_json::json;
/// use sinew_govern::audit::{ApprovalRecord, AuditKind, AuditLog, MemoryAuditStore, VerifyOptions};
///
/// let store = Arc::new(MemoryAuditStore::new());
/// let audit = AuditLog::new(store.clone());
/// audit.append(AuditKind::Other, json!({"note": "contact ana@example.com"})).await.unwrap();
/// audit.record_approval(&ApprovalRecord {
///     decision_id: None, agent: "support".into(), target: "refund".into(),
///     approver: "supervisor-1".into(), approved: true, reason: None,
/// }).await.unwrap();
///
/// let entries = store.snapshot();
/// assert_eq!(entries[1].prev_hash, entries[0].hash);
/// assert_eq!(entries[0].payload["note"], "contact [REDACTED:email]");
/// assert!(audit.verify(&VerifyOptions::new()).await.unwrap().is_ok());
/// # });
/// ```
#[derive(Clone)]
pub struct AuditLog {
    inner: Arc<LogInner>,
}

impl fmt::Debug for AuditLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuditLog")
            .field("signed", &self.inner.signer.is_some())
            .field("checkpoint_every", &self.inner.checkpoint_every)
            .field("failures", &self.failures())
            .finish_non_exhaustive()
    }
}

impl AuditLog {
    /// A log over `store` with the default redactor, no signer.
    pub fn new(store: Arc<dyn AuditStore>) -> Self {
        Self::builder(store).build()
    }

    pub fn builder(store: Arc<dyn AuditStore>) -> AuditLogBuilder {
        AuditLogBuilder {
            store,
            redactor: Some(Redactor::new()),
            signer: None,
            checkpoint_every: None,
        }
    }

    /// The store this log writes to.
    pub fn store(&self) -> &Arc<dyn AuditStore> {
        &self.inner.store
    }

    /// Redacts `payload`, chains it after the current head and appends it.
    /// Returns the written entry. May also write a checkpoint after it
    /// (see [`AuditLogBuilder::checkpoint_every`]).
    pub async fn append(
        &self,
        kind: AuditKind,
        mut payload: Value,
    ) -> Result<AuditEntry, AuditError> {
        if let Some(r) = &self.inner.redactor {
            r.redact_value(&mut payload);
        }
        let mut head = self.inner.head.lock().await;
        let entry = self.write(&mut head, kind, payload).await?;
        let due = matches!(
            (&self.inner.signer, self.inner.checkpoint_every, head.as_ref()),
            (Some(_), Some(n), Some(h)) if h.since_checkpoint >= n
        );
        if due {
            // The entry is written: a failed checkpoint must not make the
            // caller retry it. It is counted, and tried again next time.
            if let Err(e) = self.write_checkpoint(&mut head).await {
                self.fail(format!("checkpoint: {e}"));
            }
        }
        Ok(entry)
    }

    /// Records a human's answer to an `Ask` decision.
    pub async fn record_approval(
        &self,
        approval: &ApprovalRecord,
    ) -> Result<AuditEntry, AuditError> {
        self.append(AuditKind::Approval, json!(approval)).await
    }

    /// Records a tool call with side effects and its outcome.
    pub async fn record_tool_call(&self, call: &ToolCallRecord) -> Result<AuditEntry, AuditError> {
        self.append(AuditKind::ToolCall, json!(call)).await
    }

    /// Writes a signed checkpoint of the current head now.
    pub async fn checkpoint(&self) -> Result<AuditEntry, AuditError> {
        if self.inner.signer.is_none() {
            return Err(AuditError::NoSigner);
        }
        let mut head = self.inner.head.lock().await;
        self.write_checkpoint(&mut head).await
    }

    /// `(seq, hash)` of the last entry this log wrote or loaded, `None`
    /// before the first write. Keep it somewhere else (a ticket, another
    /// system) to detect later truncation with
    /// [`VerifyOptions::expect_head`].
    pub async fn head(&self) -> Result<Option<(u64, String)>, AuditError> {
        let mut head = self.inner.head.lock().await;
        let h = self.load_head(&mut head).await?;
        Ok((h.next_seq > 0).then(|| (h.next_seq - 1, h.prev_hash.clone())))
    }

    /// Verifies the whole store. Unless `options` already expects a head,
    /// the head this log knows is expected, so entries removed from the end
    /// of the store while this log runs are detected too.
    pub async fn verify(&self, options: &VerifyOptions) -> Result<AuditReport, AuditError> {
        let mut options = options.clone();
        if options.expected_head.is_none() {
            let known = self.inner.head.lock().await.clone();
            if let Some(h) = known.filter(|h| h.next_seq > 0) {
                options.expected_head = Some((h.next_seq - 1, h.prev_hash));
            }
        }
        verify_store(self.inner.store.as_ref(), &options).await
    }

    /// Waits until every event queued by the [`DecisionSink`] methods is
    /// written (or failed).
    pub async fn flush(&self) {
        let Some(q) = self.inner.queue.get() else {
            return;
        };
        let (tx, rx) = oneshot::channel();
        if q.send(Msg::Flush(tx)).is_ok() {
            let _ = rx.await;
        }
    }

    /// Queued events that could not be written, plus periodic checkpoints
    /// that failed.
    pub fn failures(&self) -> u64 {
        self.inner.failures.load(Ordering::Relaxed)
    }

    /// The last write error of a queued event.
    pub fn last_error(&self) -> Option<String> {
        self.inner
            .last_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    async fn load_head<'a>(&self, head: &'a mut Option<Head>) -> Result<&'a mut Head, AuditError> {
        if head.is_none() {
            let last = self.inner.store.last().await?;
            *head = Some(match last {
                Some(e) => Head {
                    next_seq: e.seq + 1,
                    prev_hash: e.hash,
                    since_checkpoint: 0,
                },
                None => Head {
                    next_seq: 0,
                    prev_hash: GENESIS_HASH.into(),
                    since_checkpoint: 0,
                },
            });
        }
        Ok(head.as_mut().expect("head loaded"))
    }

    /// Appends one entry after the head, reloading the head and retrying
    /// when another writer got there first.
    async fn write(
        &self,
        head: &mut Option<Head>,
        kind: AuditKind,
        payload: Value,
    ) -> Result<AuditEntry, AuditError> {
        let mut attempts = 0;
        loop {
            let h = self.load_head(head).await?;
            let entry = if kind == AuditKind::Checkpoint {
                // Checkpoints sign the head they are written after.
                let signer = self.inner.signer.as_ref().ok_or(AuditError::NoSigner)?;
                let time_ms = now_ms();
                let signature = signer.sign_checkpoint(h.next_seq, &h.prev_hash, time_ms);
                let payload =
                    json!({"key_id": signer.public_key().key_id(), "signature": signature});
                AuditEntry::new(h.next_seq, time_ms, kind, payload, h.prev_hash.clone())
            } else {
                AuditEntry::new(
                    h.next_seq,
                    now_ms(),
                    kind,
                    payload.clone(),
                    h.prev_hash.clone(),
                )
            };
            match self.inner.store.append(&entry).await {
                Ok(()) => {
                    h.next_seq += 1;
                    h.prev_hash.clone_from(&entry.hash);
                    if kind == AuditKind::Checkpoint {
                        h.since_checkpoint = 0;
                    } else {
                        h.since_checkpoint += 1;
                    }
                    return Ok(entry);
                }
                Err(AuditError::Conflict(_)) if attempts < 3 => {
                    attempts += 1;
                    *head = None;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn write_checkpoint(&self, head: &mut Option<Head>) -> Result<AuditEntry, AuditError> {
        self.write(head, AuditKind::Checkpoint, Value::Null).await
    }

    /// Queues an event for the background writer, starting it on first use
    /// (on the current Tokio runtime, or on a thread of its own outside
    /// one).
    fn enqueue(&self, kind: AuditKind, payload: Value) {
        let q = self.inner.queue.get_or_init(|| {
            let (tx, rx) = mpsc::unbounded_channel();
            spawn_writer(Arc::downgrade(&self.inner), rx);
            tx
        });
        if q.send(Msg::Append(kind, payload)).is_err() {
            self.fail("audit writer stopped".into());
        }
    }

    fn fail(&self, error: String) {
        self.inner.failures.fetch_add(1, Ordering::Relaxed);
        *self
            .inner
            .last_error
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(error);
    }
}

fn spawn_writer(log: Weak<LogInner>, rx: mpsc::UnboundedReceiver<Msg>) {
    let task = writer(log, rx);
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            handle.spawn(task);
        }
        Err(_) => {
            std::thread::spawn(move || {
                if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    rt.block_on(task);
                }
            });
        }
    }
}

async fn writer(log: Weak<LogInner>, mut rx: mpsc::UnboundedReceiver<Msg>) {
    while let Some(msg) = rx.recv().await {
        match msg {
            Msg::Append(kind, payload) => {
                let Some(inner) = log.upgrade() else { break };
                let log = AuditLog { inner };
                if let Err(e) = log.append(kind, payload).await {
                    log.fail(e.to_string());
                }
            }
            Msg::Flush(done) => {
                let _ = done.send(());
            }
        }
    }
}

impl DecisionSink for AuditLog {
    fn record(&self, event: &DecisionEvent) {
        self.enqueue(AuditKind::Decision, json!(event));
    }

    fn record_control(&self, event: &ControlEvent) {
        let kind = match event.action {
            ControlAction::UpdatePolicies => AuditKind::PolicyChange,
            _ => AuditKind::Control,
        };
        self.enqueue(kind, json!(event));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_keys_recursively() {
        let v = json!({"b": [1, {"z": null, "a": "x"}], "a": 1.5});
        let mut s = String::new();
        canonical_json(&v, &mut s);
        assert_eq!(s, r#"{"a":1.5,"b":[1,{"a":"x","z":null}]}"#);
    }

    #[test]
    fn hex_round_trips() {
        assert_eq!(hex(&[0, 255, 16]), "00ff10");
        assert_eq!(unhex("00ff10"), Some(vec![0, 255, 16]));
        assert_eq!(unhex("0"), None);
        assert_eq!(unhex("zz"), None);
    }

    #[test]
    fn kinds_round_trip() {
        for k in [
            AuditKind::Decision,
            AuditKind::Control,
            AuditKind::PolicyChange,
            AuditKind::Approval,
            AuditKind::ToolCall,
            AuditKind::Checkpoint,
            AuditKind::Other,
        ] {
            assert_eq!(AuditKind::parse(k.as_str()), Some(k));
            assert_eq!(serde_json::to_value(k).unwrap(), json!(k.to_string()));
        }
        assert_eq!(AuditKind::parse("nope"), None);
    }
}
