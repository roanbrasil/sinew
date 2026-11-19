//! The `sinew-govern` command line (feature `cli`): `verify-audit`,
//! `export-audit` and `keygen`. Not a stable API; the binary calls [`run`].

use std::{
    fs::File,
    io::{BufReader, Write},
    sync::Arc,
};

use super::{
    export_jsonl, verify_jsonl, verify_store, AuditError, AuditPublicKey, AuditReport, AuditSigner,
    AuditStore, VerifyOptions,
};

pub const USAGE: &str = "\
usage:
  sinew-govern verify-audit SOURCE [--public-key HEX]... [--public-key-file FILE]
                            [--expect-head SEQ:HASH] [--json]
  sinew-govern export-audit SOURCE [--out FILE]
  sinew-govern keygen

SOURCE is one of:
  --jsonl FILE     an export (`-` for standard input)
  --sqlite URL     a SQLite store, e.g. sqlite://audit.db (feature audit-sqlite)
  --postgres URL   a Postgres store (feature audit-postgres)

verify-audit exits with 0 when the log is intact, 1 when tampering is
detected and 2 on usage or read errors.";

enum Source {
    Jsonl(String),
    Sqlite(String),
    Postgres(String),
}

#[derive(Default)]
struct Args {
    source: Option<Source>,
    options: VerifyOptions,
    json: bool,
    out: Option<String>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--jsonl" => a.source = Some(Source::Jsonl(value()?)),
            "--sqlite" => a.source = Some(Source::Sqlite(value()?)),
            "--postgres" => a.source = Some(Source::Postgres(value()?)),
            "--public-key" => {
                let k = AuditPublicKey::from_hex(&value()?).map_err(|e| e.to_string())?;
                a.options = a.options.public_key(k);
            }
            "--public-key-file" => {
                let path = value()?;
                let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
                for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
                    let k = AuditPublicKey::from_hex(line).map_err(|e| format!("{path}: {e}"))?;
                    a.options = a.options.public_key(k);
                }
            }
            "--expect-head" => {
                let v = value()?;
                let (seq, hash) = v
                    .split_once(':')
                    .and_then(|(s, h)| Some((s.parse::<u64>().ok()?, h)))
                    .ok_or_else(|| format!("--expect-head wants SEQ:HASH, got `{v}`"))?;
                a.options = a.options.expect_head(seq, hash);
            }
            "--json" => a.json = true,
            "--out" => a.out = Some(value()?),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(a)
}

async fn open(source: &Source) -> Result<Arc<dyn AuditStore>, String> {
    match source {
        #[cfg(feature = "audit-sqlite")]
        Source::Sqlite(url) => Ok(Arc::new(
            super::SqliteAuditStore::connect(url)
                .await
                .map_err(|e| e.to_string())?,
        )),
        #[cfg(feature = "audit-postgres")]
        Source::Postgres(url) => Ok(Arc::new(
            super::PostgresAuditStore::connect(url)
                .await
                .map_err(|e| e.to_string())?,
        )),
        #[cfg(not(feature = "audit-sqlite"))]
        Source::Sqlite(url) => Err(format!(
            "cannot open {url}: built without the `audit-sqlite` feature"
        )),
        #[cfg(not(feature = "audit-postgres"))]
        Source::Postgres(url) => Err(format!(
            "cannot open {url}: built without the `audit-postgres` feature"
        )),
        Source::Jsonl(_) => Err("not a store".into()),
    }
}

async fn verify(a: &Args) -> Result<AuditReport, String> {
    let source = a.source.as_ref().ok_or("verify-audit needs a SOURCE")?;
    let report: Result<AuditReport, AuditError> = match source {
        Source::Jsonl(path) if path == "-" => verify_jsonl(std::io::stdin().lock(), &a.options),
        Source::Jsonl(path) => {
            let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
            verify_jsonl(BufReader::new(f), &a.options)
        }
        store => verify_store(open(store).await?.as_ref(), &a.options).await,
    };
    report.map_err(|e| e.to_string())
}

async fn export(a: &Args, out: &mut dyn Write) -> Result<u64, String> {
    let source = a.source.as_ref().ok_or("export-audit needs a SOURCE")?;
    if matches!(source, Source::Jsonl(_)) {
        return Err("export-audit reads --sqlite or --postgres".into());
    }
    let store = open(source).await?;
    let n = match &a.out {
        Some(path) => {
            let mut f = File::create(path).map_err(|e| format!("{path}: {e}"))?;
            export_jsonl(store.as_ref(), &mut f).await
        }
        None => export_jsonl(store.as_ref(), out).await,
    };
    n.map_err(|e| e.to_string())
}

/// Runs the command line with `args` (without the program name). Returns
/// the exit code.
pub async fn run(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let Some((cmd, rest)) = args.split_first() else {
        let _ = writeln!(err, "{USAGE}");
        return 2;
    };
    match cmd.as_str() {
        "-h" | "--help" | "help" => {
            let _ = writeln!(out, "{USAGE}");
            0
        }
        "keygen" => match AuditSigner::generate() {
            Ok(s) => {
                let _ = writeln!(out, "signing key (keep secret): {}", s.to_hex());
                let _ = writeln!(out, "public key: {}", s.public_key().to_hex());
                let _ = writeln!(out, "key id: {}", s.public_key().key_id());
                0
            }
            Err(e) => {
                let _ = writeln!(err, "error: {e}");
                2
            }
        },
        "verify-audit" | "export-audit" => {
            let a = match parse(rest) {
                Ok(a) => a,
                Err(e) => {
                    let _ = writeln!(err, "error: {e}\n\n{USAGE}");
                    return 2;
                }
            };
            if cmd == "export-audit" {
                return match export(&a, out).await {
                    Ok(n) => {
                        let _ = writeln!(err, "exported {n} entries");
                        0
                    }
                    Err(e) => {
                        let _ = writeln!(err, "error: {e}");
                        2
                    }
                };
            }
            match verify(&a).await {
                Ok(report) => {
                    let _ = if a.json {
                        writeln!(
                            out,
                            "{}",
                            serde_json::to_string_pretty(&report).unwrap_or_default()
                        )
                    } else {
                        writeln!(out, "{report}")
                    };
                    if report.is_ok() {
                        0
                    } else {
                        1
                    }
                }
                Err(e) => {
                    let _ = writeln!(err, "error: {e}");
                    2
                }
            }
        }
        other => {
            let _ = writeln!(err, "unknown command `{other}`\n\n{USAGE}");
            2
        }
    }
}
