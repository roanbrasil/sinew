//! `sinew-govern verify-audit`, `export-audit` and `keygen`.

use std::sync::Arc;

use serde_json::json;
use sinew_govern::audit::{cli, sqlx, AuditKind, AuditLog, AuditSigner, SqliteAuditStore};

async fn run(args: &[&str]) -> (i32, String, String) {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = cli::run(&args, &mut out, &mut err).await;
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

#[tokio::test]
async fn verify_export_and_detect_tampering() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}", dir.path().join("audit.db").display());
    let signer = AuditSigner::from_bytes(&[3; 32]);
    let pk = signer.public_key().to_hex();
    let store = Arc::new(SqliteAuditStore::connect(&url).await.unwrap());
    let log = AuditLog::builder(store.clone()).signer(signer).build();
    for i in 0..4 {
        log.append(AuditKind::Other, json!({"i": i})).await.unwrap();
    }
    log.checkpoint().await.unwrap();
    let (seq, hash) = log.head().await.unwrap().unwrap();
    let head = format!("{seq}:{hash}");

    let (code, out, _) = run(&["verify-audit", "--sqlite", &url, "--public-key", &pk]).await;
    assert_eq!(code, 0, "{out}");
    assert!(
        out.starts_with("OK: 5 entries, 1 checkpoints (1 verified)"),
        "{out}"
    );

    // Export to a file and verify it with a key file and the known head.
    let file = dir.path().join("audit.jsonl");
    let file = file.to_str().unwrap();
    let (code, _, err) = run(&["export-audit", "--sqlite", &url, "--out", file]).await;
    assert_eq!(code, 0);
    assert!(err.contains("exported 5 entries"));
    let keys = dir.path().join("keys.txt");
    std::fs::write(&keys, format!("{pk}\n\n")).unwrap();
    let (code, out, _) = run(&[
        "verify-audit",
        "--jsonl",
        file,
        "--public-key-file",
        keys.to_str().unwrap(),
        "--expect-head",
        &head,
        "--json",
    ])
    .await;
    assert_eq!(code, 0, "{out}");
    let report: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(report["verified_checkpoints"], 1);

    // Export to stdout.
    let (code, out, _) = run(&["export-audit", "--sqlite", &url]).await;
    assert_eq!(code, 0);
    assert_eq!(out.lines().count(), 5);

    // Tamper with the database.
    sqlx::query("DROP TRIGGER sinew_audit_log_no_update")
        .execute(store.pool())
        .await
        .unwrap();
    sqlx::query("UPDATE sinew_audit_log SET payload = '{\"i\":7}' WHERE seq = 1")
        .execute(store.pool())
        .await
        .unwrap();
    let (code, out, _) = run(&["verify-audit", "--sqlite", &url, "--public-key", &pk]).await;
    assert_eq!(code, 1);
    assert!(
        out.contains("TAMPERED") && out.contains("entry 1: hash does not match"),
        "{out}"
    );

    // A truncated export, against the known head.
    let text = std::fs::read_to_string(file).unwrap();
    let short: Vec<&str> = text.lines().take(3).collect();
    std::fs::write(file, short.join("\n")).unwrap();
    let (code, out, _) = run(&["verify-audit", "--jsonl", file, "--expect-head", &head]).await;
    assert_eq!(code, 1);
    assert!(out.contains("is missing"), "{out}");
}

#[tokio::test]
async fn usage_errors() {
    for (args, msg) in [
        (vec![], "usage"),
        (vec!["frobnicate"], "unknown command"),
        (vec!["verify-audit"], "needs a SOURCE"),
        (vec!["verify-audit", "--jsonl"], "needs a value"),
        (vec!["verify-audit", "--bogus"], "unknown argument"),
        (vec!["verify-audit", "--public-key", "zz"], "public key"),
        (vec!["verify-audit", "--expect-head", "x"], "SEQ:HASH"),
        (
            vec!["verify-audit", "--jsonl", "/nonexistent/a.jsonl"],
            "nonexistent",
        ),
        (
            vec!["verify-audit", "--public-key-file", "/nonexistent/k"],
            "nonexistent",
        ),
        (vec!["export-audit"], "needs a SOURCE"),
        (vec!["export-audit", "--jsonl", "x"], "reads --sqlite"),
        (
            vec![
                "export-audit",
                "--sqlite",
                "sqlite::memory:",
                "--out",
                "/nonexistent/o",
            ],
            "nonexistent",
        ),
        (
            vec!["verify-audit", "--sqlite", "sqlite:///nonexistent/dir/a.db"],
            "error",
        ),
    ] {
        let (code, _, err) = run(&args).await;
        assert_eq!(code, 2, "{args:?}");
        assert!(err.contains(msg), "{args:?}: {err}");
    }
    #[cfg(not(feature = "audit-postgres"))]
    {
        let (code, _, err) = run(&["verify-audit", "--postgres", "postgres://x"]).await;
        assert_eq!(code, 2);
        assert!(err.contains("audit-postgres"));
    }
}

#[tokio::test]
async fn help_and_keygen() {
    let (code, out, _) = run(&["--help"]).await;
    assert_eq!(code, 0);
    assert!(out.contains("verify-audit"));
    let (code, out, _) = run(&["keygen"]).await;
    assert_eq!(code, 0);
    let secret = out
        .lines()
        .find_map(|l| l.strip_prefix("signing key (keep secret): "))
        .unwrap();
    let public = out
        .lines()
        .find_map(|l| l.strip_prefix("public key: "))
        .unwrap();
    assert_eq!(
        AuditSigner::from_hex(secret).unwrap().public_key().to_hex(),
        public
    );
}
