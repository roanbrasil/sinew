//! The audit log in SQLite: append-only triggers, reopening, and tamper
//! detection after someone with database access drops the triggers.

use std::sync::Arc;

use serde_json::json;
use sinew_govern::{
    audit::{
        export_jsonl, sqlx, verify_jsonl, verify_store, AuditKind, AuditLog, AuditProblem,
        AuditSigner, SqliteAuditStore, VerifyOptions,
    },
    AgentProfile, ControlPlane, Request,
};

fn signer() -> AuditSigner {
    AuditSigner::from_bytes(&[5; 32])
}

fn opts() -> VerifyOptions {
    VerifyOptions::new().public_key(signer().public_key())
}

async fn filled(url: &str) -> (Arc<SqliteAuditStore>, AuditLog) {
    let store = Arc::new(SqliteAuditStore::connect(url).await.unwrap());
    let log = AuditLog::builder(store.clone())
        .signer(signer())
        .checkpoint_every(4)
        .build();
    let cp = ControlPlane::builder()
        .agent(AgentProfile::new("bot"))
        .sink(Arc::new(log.clone()))
        .build();
    for i in 0..6 {
        cp.check(Request::tool(
            "bot",
            "lookup",
            json!({"i": i, "price": 0.1 * i as f64}),
        ))
        .await;
    }
    log.flush().await;
    log.checkpoint().await.unwrap();
    (store, log)
}

async fn exec(store: &SqliteAuditStore, sql: &str) {
    sqlx::query(sql).execute(store.pool()).await.unwrap();
}

async fn drop_triggers(store: &SqliteAuditStore) {
    exec(store, "DROP TRIGGER sinew_audit_log_no_update").await;
    exec(store, "DROP TRIGGER sinew_audit_log_no_delete").await;
}

#[tokio::test]
async fn writes_reopens_and_exports() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}", dir.path().join("audit.db").display());
    let (store, log) = filled(&url).await;
    let r = log.verify(&opts()).await.unwrap();
    assert!(r.is_ok(), "{r}");
    // 6 decisions, a periodic checkpoint after 4, a final checkpoint.
    assert_eq!((r.entries, r.verified_checkpoints), (8, 2));
    drop(log);

    // A new process continues the same chain.
    let store2 = Arc::new(SqliteAuditStore::connect(&url).await.unwrap());
    let log2 = AuditLog::new(store2.clone());
    let e = log2
        .append(AuditKind::Other, json!("restart"))
        .await
        .unwrap();
    assert_eq!(e.seq, 8);
    assert!(verify_store(store2.as_ref(), &opts())
        .await
        .unwrap()
        .is_ok());

    let mut out = Vec::new();
    assert_eq!(export_jsonl(store.as_ref(), &mut out).await.unwrap(), 9);
    assert!(verify_jsonl(out.as_slice(), &opts()).unwrap().is_ok());
}

#[tokio::test]
async fn triggers_refuse_update_and_delete() {
    let (store, _) = filled("sqlite::memory:").await;
    for sql in [
        "UPDATE sinew_audit_log SET payload = '{}' WHERE seq = 1",
        "DELETE FROM sinew_audit_log WHERE seq = 1",
    ] {
        let err = sqlx::query(sql).execute(store.pool()).await.unwrap_err();
        assert!(err.to_string().contains("append-only"), "{err}");
    }
}

#[tokio::test]
async fn edit_is_detected() {
    let (store, _) = filled("sqlite::memory:").await;
    drop_triggers(&store).await;
    exec(
        &store,
        "UPDATE sinew_audit_log SET payload = replace(payload, '\"lookup\"', '\"wire_money\"') WHERE seq = 2",
    )
    .await;
    let r = verify_store(store.as_ref(), &opts()).await.unwrap();
    assert_eq!(r.problems, [AuditProblem::HashMismatch { seq: 2 }]);
}

#[tokio::test]
async fn delete_is_detected() {
    let (store, _) = filled("sqlite::memory:").await;
    drop_triggers(&store).await;
    exec(&store, "DELETE FROM sinew_audit_log WHERE seq = 3").await;
    let r = verify_store(store.as_ref(), &opts()).await.unwrap();
    assert!(r.problems.contains(&AuditProblem::OutOfSequence {
        expected: 3,
        found: 4
    }));
    assert!(r.problems.contains(&AuditProblem::BrokenLink { seq: 4 }));
}

#[tokio::test]
async fn reorder_is_detected() {
    let (store, _) = filled("sqlite::memory:").await;
    drop_triggers(&store).await;
    // Swap the sequence numbers of rows 1 and 2.
    exec(&store, "UPDATE sinew_audit_log SET seq = -1 WHERE seq = 1").await;
    exec(&store, "UPDATE sinew_audit_log SET seq = 1 WHERE seq = 2").await;
    exec(&store, "UPDATE sinew_audit_log SET seq = 2 WHERE seq = -1").await;
    let r = verify_store(store.as_ref(), &opts()).await.unwrap();
    assert!(r.problems.contains(&AuditProblem::HashMismatch { seq: 1 }));
    assert!(r.problems.contains(&AuditProblem::BrokenLink { seq: 1 }));
}

#[tokio::test]
async fn truncation_is_detected_by_the_running_log() {
    let (store, log) = filled("sqlite::memory:").await;
    drop_triggers(&store).await;
    exec(&store, "DELETE FROM sinew_audit_log WHERE seq >= 6").await;
    let r = log.verify(&opts()).await.unwrap();
    assert!(matches!(
        r.problems[..],
        [AuditProblem::HeadMismatch { seq: 7, .. }]
    ));
}

#[tokio::test]
async fn bad_rows_are_errors() {
    let (store, _) = filled("sqlite::memory:").await;
    exec(
        &store,
        "INSERT INTO sinew_audit_log VALUES (100, 1, 'mystery', '{}', 'x', 'y')",
    )
    .await;
    let err = verify_store(store.as_ref(), &opts()).await.unwrap_err();
    assert!(err.to_string().contains("unknown kind"), "{err}");
}
