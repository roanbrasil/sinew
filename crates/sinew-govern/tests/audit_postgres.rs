//! The audit log in Postgres. Runs only when `SINEW_TEST_PG_URL` is set,
//! e.g. `postgres://postgres:sinew@127.0.0.1:55432/postgres`; otherwise each
//! test returns early. Every test works in its own throwaway schema.

use std::sync::Arc;

use serde_json::json;
use sinew_govern::audit::{
    export_jsonl,
    sqlx::{
        postgres::{PgConnectOptions, PgPoolOptions},
        Executor, PgPool,
    },
    verify_jsonl, verify_store, AuditKind, AuditLog, AuditProblem, AuditSigner, PostgresAuditStore,
    VerifyOptions,
};

struct Fixture {
    store: Arc<PostgresAuditStore>,
    admin: PgPool,
    schema: String,
}

impl Fixture {
    async fn new() -> Option<Self> {
        let url = std::env::var("SINEW_TEST_PG_URL").ok()?;
        let schema = format!("sinew_audit_{}", uuid::Uuid::new_v4().simple());
        let admin = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("connect to SINEW_TEST_PG_URL");
        admin
            .execute(format!("CREATE SCHEMA {schema}").as_str())
            .await
            .unwrap();
        let opts: PgConnectOptions = url.parse().unwrap();
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .connect_with(opts.options([("search_path", schema.as_str())]))
            .await
            .unwrap();
        let store = Arc::new(PostgresAuditStore::from_pool(pool).await.unwrap());
        Some(Self {
            store,
            admin,
            schema,
        })
    }

    async fn exec(&self, sql: &str) {
        self.admin
            .execute(format!("SET search_path TO {}; {sql}", self.schema).as_str())
            .await
            .unwrap();
    }

    async fn cleanup(self) {
        self.admin
            .execute(format!("DROP SCHEMA {} CASCADE", self.schema).as_str())
            .await
            .unwrap();
    }
}

fn signer() -> AuditSigner {
    AuditSigner::from_bytes(&[11; 32])
}

fn opts() -> VerifyOptions {
    VerifyOptions::new().public_key(signer().public_key())
}

async fn fill(store: Arc<PostgresAuditStore>) -> AuditLog {
    let log = AuditLog::builder(store).signer(signer()).build();
    for i in 0..5 {
        log.append(AuditKind::Other, json!({"i": i, "x": 0.1 * i as f64}))
            .await
            .unwrap();
    }
    log.checkpoint().await.unwrap();
    log
}

#[tokio::test]
async fn postgres_round_trip_and_append_only() {
    let Some(fx) = Fixture::new().await else {
        return;
    };
    let log = fill(fx.store.clone()).await;
    assert!(log.verify(&opts()).await.unwrap().is_ok());

    // Schema creation is idempotent; a second writer continues the chain.
    let again = Arc::new(
        PostgresAuditStore::from_pool(fx.store.pool().clone())
            .await
            .unwrap(),
    );
    let other = AuditLog::new(again);
    assert_eq!(
        other
            .append(AuditKind::Other, json!("b"))
            .await
            .unwrap()
            .seq,
        6
    );
    // The first log is behind: it gets a conflict, reloads and retries.
    assert_eq!(
        log.append(AuditKind::Other, json!("a")).await.unwrap().seq,
        7
    );

    let mut out = Vec::new();
    assert_eq!(export_jsonl(fx.store.as_ref(), &mut out).await.unwrap(), 8);
    assert!(verify_jsonl(out.as_slice(), &opts()).unwrap().is_ok());

    for sql in [
        "UPDATE sinew_audit_log SET payload = '{}'",
        "DELETE FROM sinew_audit_log",
        "TRUNCATE sinew_audit_log",
    ] {
        let err = fx.store.pool().execute(sql).await.expect_err("append-only");
        assert!(err.to_string().contains("append-only"), "{err}");
    }
    fx.cleanup().await;
}

#[tokio::test]
async fn postgres_tampering_is_detected() {
    let Some(fx) = Fixture::new().await else {
        return;
    };
    fill(fx.store.clone()).await;
    fx.exec("DROP TRIGGER sinew_audit_log_append_only ON sinew_audit_log")
        .await;

    fx.exec("UPDATE sinew_audit_log SET payload = '{\"i\":9}' WHERE seq = 1")
        .await;
    let r = verify_store(fx.store.as_ref(), &opts()).await.unwrap();
    assert_eq!(r.problems, [AuditProblem::HashMismatch { seq: 1 }]);

    fx.exec("DELETE FROM sinew_audit_log WHERE seq = 3").await;
    let r = verify_store(fx.store.as_ref(), &opts()).await.unwrap();
    assert!(r.problems.contains(&AuditProblem::BrokenLink { seq: 4 }));

    fx.exec("UPDATE sinew_audit_log SET seq = 3 WHERE seq = 4")
        .await;
    let r = verify_store(fx.store.as_ref(), &opts()).await.unwrap();
    assert!(r.problems.contains(&AuditProblem::HashMismatch { seq: 3 }));
    fx.cleanup().await;
}
