//! Tamper-evident audit log with the in-memory store: chaining, control
//! plane integration, redaction, checkpoints and tamper detection.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_govern::{
    audit::{
        export_jsonl, read_jsonl, verify_jsonl, ApprovalRecord, AuditEntry, AuditError, AuditKind,
        AuditLog, AuditProblem, AuditPublicKey, AuditSigner, AuditStore, ChainVerifier,
        MemoryAuditStore, ToolCallRecord, VerifyOptions, GENESIS_HASH,
    },
    AgentProfile, ControlPlane, DecisionEvent, DecisionSink, PolicySet, Request, Rule, Verdict,
};

fn signer() -> AuditSigner {
    AuditSigner::from_bytes(&[42; 32])
}

fn key() -> AuditPublicKey {
    signer().public_key()
}

fn opts() -> VerifyOptions {
    VerifyOptions::new().public_key(key())
}

/// A log of `n` `Other` entries with a checkpoint at the end.
async fn log_with(n: usize) -> (Arc<MemoryAuditStore>, AuditLog) {
    let store = Arc::new(MemoryAuditStore::new());
    let log = AuditLog::builder(store.clone()).signer(signer()).build();
    for i in 0..n {
        log.append(AuditKind::Other, json!({"i": i, "amount": 1.1 * i as f64}))
            .await
            .unwrap();
    }
    log.checkpoint().await.unwrap();
    (store, log)
}

async fn verify(store: &MemoryAuditStore) -> sinew_govern::audit::AuditReport {
    sinew_govern::audit::verify_store(store, &opts())
        .await
        .unwrap()
}

#[tokio::test]
async fn entries_are_chained_and_verify() {
    let (store, log) = log_with(5).await;
    let entries = store.snapshot();
    assert_eq!(entries.len(), 6);
    assert_eq!(entries[0].prev_hash, GENESIS_HASH);
    for w in entries.windows(2) {
        assert_eq!(w[1].prev_hash, w[0].hash);
        assert_eq!(w[1].seq, w[0].seq + 1);
    }
    let report = log.verify(&opts()).await.unwrap();
    assert!(report.is_ok(), "{report}");
    assert_eq!(report.entries, 6);
    assert_eq!(report.checkpoints, 1);
    assert_eq!(report.verified_checkpoints, 1);
    assert_eq!(report.unsigned_tail, 0);
    assert_eq!(report.head, Some((5, entries[5].hash.clone())));
    assert_eq!(log.head().await.unwrap(), report.head);
    assert!(report.to_string().starts_with("OK: 6 entries"));

    // Without a key, checkpoints are read but not verified.
    let r = log.verify(&VerifyOptions::new()).await.unwrap();
    assert!(r.is_ok());
    assert_eq!(
        (r.checkpoints, r.verified_checkpoints, r.unsigned_tail),
        (1, 0, 6)
    );
}

#[tokio::test]
async fn edited_entry_is_detected() {
    let (store, _) = log_with(5).await;
    store.tamper(|v| v[2].payload["amount"] = json!(9999));
    let r = verify(&store).await;
    assert_eq!(r.problems, [AuditProblem::HashMismatch { seq: 2 }]);
    assert!(r.to_string().contains("TAMPERED"));
    assert!(r.to_string().contains("entry 2: hash does not match"));
}

#[tokio::test]
async fn edited_entry_with_recomputed_hash_breaks_the_next_link() {
    let (store, _) = log_with(5).await;
    store.tamper(|v| {
        v[2].payload["amount"] = json!(0);
        v[2].hash = v[2].compute_hash();
    });
    assert_eq!(
        verify(&store).await.problems,
        [AuditProblem::BrokenLink { seq: 3 }]
    );
}

#[tokio::test]
async fn rewritten_chain_fails_the_checkpoint_signature() {
    let (store, _) = log_with(5).await;
    // An attacker edits entry 1 and recomputes every later hash and link,
    // the checkpoint included, but cannot re-sign it.
    store.tamper(|v| {
        v[1].payload = json!({"i": 1, "amount": 0});
        let mut prev = v[0].hash.clone();
        for e in v.iter_mut().skip(1) {
            e.prev_hash = prev;
            e.hash = e.compute_hash();
            prev = e.hash.clone();
        }
    });
    assert_eq!(
        verify(&store).await.problems,
        [AuditProblem::BadSignature { seq: 5 }]
    );

    // Re-signed with the attacker's own key: unknown key.
    let forged = AuditSigner::from_bytes(&[9; 32]);
    let fake = Arc::new(MemoryAuditStore::new());
    let log = AuditLog::builder(fake.clone())
        .signer(forged.clone())
        .build();
    log.append(AuditKind::Other, json!("forged")).await.unwrap();
    log.checkpoint().await.unwrap();
    let r = verify(&fake).await;
    assert_eq!(
        r.problems,
        [AuditProblem::UnknownKey {
            seq: 1,
            key_id: forged.public_key().key_id()
        }]
    );
    assert!(r.problems[0].to_string().contains("unknown key"));
}

#[tokio::test]
async fn deleted_entry_is_detected() {
    let (store, _) = log_with(5).await;
    store.tamper(|v| v.remove(2));
    let r = verify(&store).await;
    assert_eq!(
        r.problems,
        [
            AuditProblem::OutOfSequence {
                expected: 2,
                found: 3
            },
            AuditProblem::BrokenLink { seq: 3 }
        ]
    );
    assert!(r.problems[0].to_string().contains("expected entry 2"));
}

#[tokio::test]
async fn deleted_checkpoint_is_detected() {
    let (store, log) = log_with(3).await;
    log.append(AuditKind::Other, json!("after")).await.unwrap();
    store.tamper(|v| v.remove(3));
    assert!(!verify(&store).await.is_ok());
}

#[tokio::test]
async fn reordered_entries_are_detected() {
    let (store, _) = log_with(5).await;
    // Swap the contents of two rows, keeping the sequence numbers.
    store.tamper(|v| {
        let (a, b) = (v[1].clone(), v[2].clone());
        v[1] = AuditEntry { seq: 1, ..b };
        v[2] = AuditEntry { seq: 2, ..a };
    });
    let r = verify(&store).await;
    assert!(r.problems.contains(&AuditProblem::HashMismatch { seq: 1 }));
    assert!(r.problems.contains(&AuditProblem::BrokenLink { seq: 1 }));

    // Swap whole entries in an export.
    let (store, _) = log_with(5).await;
    let mut out = Vec::new();
    export_jsonl(store.as_ref(), &mut out).await.unwrap();
    let text = String::from_utf8(out).unwrap();
    let mut lines: Vec<&str> = text.lines().collect();
    lines.swap(1, 2);
    let r = verify_jsonl(lines.join("\n").as_bytes(), &opts()).unwrap();
    assert!(r.problems.contains(&AuditProblem::OutOfSequence {
        expected: 1,
        found: 2
    }));
}

#[tokio::test]
async fn truncated_tail_is_detected_against_a_known_head() {
    let (store, log) = log_with(5).await;
    let (seq, hash) = log.head().await.unwrap().unwrap();
    store.tamper(|v| v.truncate(4));

    // The chain alone cannot tell...
    assert!(verify(&store).await.is_ok());
    // ...a head recorded elsewhere can,
    let r = sinew_govern::audit::verify_store(store.as_ref(), &opts().expect_head(seq, &hash))
        .await
        .unwrap();
    assert_eq!(
        r.problems,
        [AuditProblem::HeadMismatch {
            seq,
            expected: hash.clone(),
            found: None
        }]
    );
    assert!(r.problems[0].to_string().contains("is missing"));
    // and so does the log that wrote it.
    assert!(!log.verify(&opts()).await.unwrap().is_ok());

    let r = sinew_govern::audit::verify_store(store.as_ref(), &opts().expect_head(2, "beef"))
        .await
        .unwrap();
    assert!(r.problems[0].to_string().contains("expected beef"));
}

#[tokio::test]
async fn malformed_and_bad_signatures_are_reported() {
    let (store, _) = log_with(1).await;
    store.tamper(|v| {
        v[1].payload["signature"] = json!("00");
        v[1].hash = v[1].compute_hash();
    });
    assert_eq!(
        verify(&store).await.problems,
        [AuditProblem::BadSignature { seq: 1 }]
    );
    store.tamper(|v| {
        v[1].payload = json!({});
        v[1].hash = v[1].compute_hash();
    });
    let r = verify(&store).await;
    assert_eq!(r.problems, [AuditProblem::MalformedCheckpoint { seq: 1 }]);
    assert!(r.problems[0].to_string().contains("missing key_id"));
}

#[tokio::test]
async fn periodic_checkpoints() {
    let store = Arc::new(MemoryAuditStore::new());
    let log = AuditLog::builder(store.clone())
        .signer(signer())
        .checkpoint_every(3)
        .build();
    for i in 0..7 {
        log.append(AuditKind::Other, json!(i)).await.unwrap();
    }
    let kinds: Vec<AuditKind> = store.snapshot().iter().map(|e| e.kind).collect();
    let cp = AuditKind::Checkpoint;
    let o = AuditKind::Other;
    assert_eq!(kinds, [o, o, o, cp, o, o, o, cp, o]);
    let r = log.verify(&opts()).await.unwrap();
    assert!(r.is_ok());
    assert_eq!((r.verified_checkpoints, r.unsigned_tail), (2, 1));
    assert!(format!("{log:?}").contains("checkpoint_every: Some(3)"));
}

#[tokio::test]
async fn checkpoint_needs_a_signer() {
    let log = AuditLog::new(Arc::new(MemoryAuditStore::new()));
    assert!(matches!(log.checkpoint().await, Err(AuditError::NoSigner)));
    assert_eq!(log.head().await.unwrap(), None);
}

#[tokio::test]
async fn control_plane_decisions_actions_and_policy_changes_are_logged() {
    let store = Arc::new(MemoryAuditStore::new());
    let audit = AuditLog::builder(store.clone()).signer(signer()).build();
    let cp = ControlPlane::builder()
        .agent(AgentProfile::new("billing-bot"))
        .policy(Rule::deny("no-deletes").tools(["delete_*"]))
        .sink(Arc::new(audit.clone()))
        .build();

    let secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789";
    cp.check(Request::tool(
        "billing-bot",
        "send_invoice",
        json!({"to": "ana@example.com", "api_key": secret, "amount": 10}),
    ))
    .await;
    cp.check(Request::tool(
        "billing-bot",
        "delete_invoice",
        json!({"id": 1}),
    ))
    .await;
    cp.add_policy(Arc::new(Rule::allow("reads").tools(["get_*"])));
    cp.set_policies(PolicySet::new().with(Rule::deny("all").tools(["*"])));
    cp.suspend("billing-bot", "incident 42");
    audit.flush().await;
    audit
        .record_approval(&ApprovalRecord {
            decision_id: Some("d1".into()),
            agent: "billing-bot".into(),
            target: "refund".into(),
            approver: "supervisor".into(),
            approved: false,
            reason: Some("too large".into()),
            details: Value::Null,
        })
        .await
        .unwrap();
    audit
        .record_tool_call(&ToolCallRecord {
            agent: "billing-bot".into(),
            tool: "refund".into(),
            arguments: json!({"amount": 5, "password": "hunter2"}),
            decision_id: None,
            output: json!("ok"),
            error: None,
        })
        .await
        .unwrap();
    audit.checkpoint().await.unwrap();

    let entries = store.snapshot();
    let kinds: Vec<AuditKind> = entries.iter().map(|e| e.kind).collect();
    assert_eq!(
        kinds,
        [
            AuditKind::Decision,
            AuditKind::Decision,
            AuditKind::PolicyChange,
            AuditKind::PolicyChange,
            AuditKind::Control,
            AuditKind::Approval,
            AuditKind::ToolCall,
            AuditKind::Checkpoint,
        ]
    );
    // Secrets and PII never reach the store.
    let all = serde_json::to_string(&entries).unwrap();
    assert!(!all.contains(secret));
    assert!(!all.contains("ana@example.com"));
    assert!(!all.contains("hunter2"));
    let args = &entries[0].payload["request"]["arguments"];
    assert_eq!(args["to"], "[REDACTED:email]");
    assert_eq!(args["amount"], 10);
    assert_eq!(entries[1].payload["verdict"]["policy_id"], "no-deletes");
    assert_eq!(entries[2].payload["target"], "reads");
    assert_eq!(entries[3].payload["action"], "update_policies");
    assert_eq!(entries[3].payload["target"], "all");
    assert_eq!(entries[4].payload["action"], "suspend");

    let r = audit.verify(&opts()).await.unwrap();
    assert!(r.is_ok(), "{r}");
    assert_eq!(audit.failures(), 0);
}

#[tokio::test]
async fn redaction_can_be_customized_or_disabled() {
    let store = Arc::new(MemoryAuditStore::new());
    let log = AuditLog::builder(store.clone())
        .redactor(sinew_trace::Redactor::empty().field("ssn"))
        .build();
    log.append(AuditKind::Other, json!({"ssn": "123", "mail": "a@b.co"}))
        .await
        .unwrap();
    let raw = AuditLog::builder(store.clone()).no_redaction().build();
    raw.append(AuditKind::Other, json!({"ssn": "456"}))
        .await
        .unwrap();
    let e = store.snapshot();
    assert_eq!(
        e[0].payload,
        json!({"ssn": "[REDACTED:field]", "mail": "a@b.co"})
    );
    assert_eq!(e[1].payload, json!({"ssn": "456"}));
    assert_eq!(e[1].prev_hash, e[0].hash);
}

#[tokio::test]
async fn two_writers_do_not_fork_the_chain() {
    let store = Arc::new(MemoryAuditStore::new());
    let a = AuditLog::new(store.clone());
    let b = AuditLog::new(store.clone());
    a.append(AuditKind::Other, json!("a1")).await.unwrap();
    b.append(AuditKind::Other, json!("b1")).await.unwrap(); // loads head after a1
    a.append(AuditKind::Other, json!("a2")).await.unwrap(); // conflict, reload, retry
    let seqs: Vec<u64> = store.snapshot().iter().map(|e| e.seq).collect();
    assert_eq!(seqs, [0, 1, 2]);
    assert!(a.verify(&VerifyOptions::new()).await.unwrap().is_ok());
}

/// A store whose appends always fail.
struct BrokenStore;

#[async_trait]
impl AuditStore for BrokenStore {
    async fn append(&self, _: &AuditEntry) -> Result<(), AuditError> {
        Err(AuditError::Store("disk full".into()))
    }
    async fn last(&self) -> Result<Option<AuditEntry>, AuditError> {
        Ok(None)
    }
    async fn entries(&self, _: u64, _: usize) -> Result<Vec<AuditEntry>, AuditError> {
        Ok(vec![])
    }
}

fn event() -> DecisionEvent {
    DecisionEvent {
        id: "e1".into(),
        time_ms: 1,
        request: Request::tool("bot", "t", json!({})),
        verdict: Verdict::new(sinew_govern::Decision::Allow, None),
    }
}

#[tokio::test]
async fn sink_failures_are_counted() {
    let log = AuditLog::new(Arc::new(BrokenStore));
    log.record(&event());
    log.flush().await;
    assert_eq!(log.failures(), 1);
    assert_eq!(log.last_error().as_deref(), Some("audit store: disk full"));
    assert!(log.store().entries(0, 10).await.unwrap().is_empty());
}

#[test]
fn sink_works_outside_a_runtime() {
    let store = Arc::new(MemoryAuditStore::new());
    let log = AuditLog::new(store.clone());
    log.record(&event()); // starts its own writer thread
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(log.flush());
    assert_eq!(store.snapshot().len(), 1);
    assert_eq!(store.snapshot()[0].kind, AuditKind::Decision);
}

#[tokio::test]
async fn jsonl_round_trip_and_errors() {
    let (store, _) = log_with(3).await;
    let mut out = Vec::new();
    assert_eq!(export_jsonl(store.as_ref(), &mut out).await.unwrap(), 4);
    let text = format!("{}\n\n", String::from_utf8(out).unwrap());
    let back = read_jsonl(text.as_bytes()).unwrap();
    assert_eq!(back, store.snapshot());

    let mut v = ChainVerifier::new(opts());
    for e in &back {
        v.push(e);
    }
    assert!(v.finish().is_ok());

    let err = read_jsonl("{}\n".as_bytes()).unwrap_err();
    assert!(err.to_string().contains("line 1"), "{err}");
    let err = verify_jsonl("\nnot json\n".as_bytes(), &opts()).unwrap_err();
    assert!(err.to_string().contains("line 2"), "{err}");
}

#[tokio::test]
async fn floats_survive_export_exactly() {
    let store = Arc::new(MemoryAuditStore::new());
    let log = AuditLog::new(store.clone());
    let tricky: Vec<Value> = [
        0.1,
        1.0 / 3.0,
        2.2250738585072014e-308,
        1e300,
        123456.789012345,
    ]
    .iter()
    .map(|f| json!(f))
    .collect();
    log.append(AuditKind::Other, json!(tricky)).await.unwrap();
    let mut out = Vec::new();
    export_jsonl(store.as_ref(), &mut out).await.unwrap();
    assert!(verify_jsonl(out.as_slice(), &VerifyOptions::new())
        .unwrap()
        .is_ok());
}

/// A store that refuses checkpoint entries.
#[derive(Default)]
struct NoCheckpoints(MemoryAuditStore);

#[async_trait]
impl AuditStore for NoCheckpoints {
    async fn append(&self, e: &AuditEntry) -> Result<(), AuditError> {
        if e.kind == AuditKind::Checkpoint {
            return Err(AuditError::Store("read-only".into()));
        }
        self.0.append(e).await
    }
    async fn last(&self) -> Result<Option<AuditEntry>, AuditError> {
        self.0.last().await
    }
    async fn entries(&self, from: u64, limit: usize) -> Result<Vec<AuditEntry>, AuditError> {
        self.0.entries(from, limit).await
    }
}

#[tokio::test]
async fn failed_periodic_checkpoint_keeps_the_entry() {
    let log = AuditLog::builder(Arc::new(NoCheckpoints::default()))
        .signer(signer())
        .checkpoint_every(1)
        .build();
    let e = log.append(AuditKind::Other, json!(1)).await.unwrap();
    assert_eq!(e.seq, 0);
    assert_eq!(log.failures(), 1);
    assert_eq!(
        log.last_error().as_deref(),
        Some("checkpoint: audit store: read-only")
    );
    assert_eq!(log.append(AuditKind::Other, json!(2)).await.unwrap().seq, 1);
}
