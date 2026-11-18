//! `sinew-scope` command line.
//!
//! ```text
//! sinew-scope serve  [--db scope.db] [--addr 127.0.0.1:7878] [--token T]
//!                    [--pricing prices.json] [--retention-days N] [--no-engine]
//!                    [--metrics-auth] [--playground-providers LIST]
//! sinew-scope import <traces.jsonl> [--db scope.db] [--project P]
//! sinew-scope view   <traces.jsonl> [--trace <id-prefix>] [--io]
//! sinew-scope <traces.jsonl> [--trace <id-prefix>] [--io]   # same as `view`
//! ```

use std::{net::SocketAddr, process::ExitCode};

use sinew_scope::{store::Store, view};

const USAGE: &str = "usage:
  sinew-scope serve  [--db scope.db] [--addr 127.0.0.1:7878] [--token T]
                     [--pricing prices.json] [--retention-days N] [--no-engine]
                     [--metrics-auth]   (require the token on GET /metrics)
                     [--playground-providers ollama,openai,...|local|all|none]
                     (providers the playground may call with this server's
                      API keys; default: local providers only; also read from
                      SINEW_SCOPE_PLAYGROUND_PROVIDERS)
                     (token also read from SINEW_SCOPE_TOKEN)
  sinew-scope import <traces.jsonl> [--db scope.db] [--project default]
  sinew-scope view   <traces.jsonl> [--trace <id-prefix>] [--io]
  sinew-scope <traces.jsonl> [--trace <id-prefix>] [--io]";

enum Cmd {
    Serve {
        db: String,
        addr: String,
        token: Option<String>,
        pricing: Option<String>,
        retention_days: Option<u32>,
        no_engine: bool,
        metrics_auth: bool,
        playground_providers: Option<Vec<String>>,
    },
    Import {
        path: String,
        db: String,
        project: String,
    },
    View {
        path: String,
        trace: Option<String>,
        io: bool,
    },
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Cmd, String> {
    let first = args.next().ok_or("missing command")?;
    let mut rest: Vec<String> = args.collect();
    let sub = match first.as_str() {
        "serve" | "import" | "view" => first,
        "-h" | "--help" | "help" => return Err(String::new()),
        // Backward compatibility: a bare path means `view <path>`.
        _ => {
            rest.insert(0, first);
            "view".into()
        }
    };
    let mut it = rest.into_iter();
    let (mut db, mut addr, mut token, mut project) = (
        "scope.db".to_string(),
        "127.0.0.1:7878".to_string(),
        None,
        "default".to_string(),
    );
    let (mut path, mut trace, mut io) = (None, None, false);
    let (mut pricing, mut retention_days, mut no_engine) = (None, None, false);
    let mut metrics_auth = false;
    let mut playground_providers = None;
    while let Some(a) = it.next() {
        let mut value = |flag: &str| it.next().ok_or(format!("{flag} needs a value"));
        match a.as_str() {
            "--db" => db = value("--db")?,
            "--addr" => addr = value("--addr")?,
            "--token" => token = Some(value("--token")?),
            "--project" => project = value("--project")?,
            "--trace" => trace = Some(value("--trace")?),
            "--io" => io = true,
            "--pricing" => pricing = Some(value("--pricing")?),
            "--retention-days" => {
                let v = value("--retention-days")?;
                retention_days = Some(
                    v.parse::<u32>()
                        .ok()
                        .filter(|d| *d > 0)
                        .ok_or(format!("invalid --retention-days `{v}`"))?,
                );
            }
            "--no-engine" => no_engine = true,
            "--metrics-auth" => metrics_auth = true,
            "--playground-providers" => {
                playground_providers = Some(sinew_scope::playground::parse_providers(&value(
                    "--playground-providers",
                )?)?)
            }
            "-h" | "--help" => return Err(String::new()),
            _ if !a.starts_with("--") && path.is_none() => path = Some(a),
            _ => return Err(format!("unexpected argument `{a}`")),
        }
    }
    Ok(match sub.as_str() {
        "serve" => Cmd::Serve {
            db,
            addr,
            token: token.or_else(|| std::env::var("SINEW_SCOPE_TOKEN").ok()),
            pricing,
            retention_days,
            no_engine,
            metrics_auth,
            playground_providers: match playground_providers {
                Some(p) => Some(p),
                None => std::env::var("SINEW_SCOPE_PLAYGROUND_PROVIDERS")
                    .ok()
                    .map(|v| sinew_scope::playground::parse_providers(&v))
                    .transpose()
                    .map_err(|e| format!("SINEW_SCOPE_PLAYGROUND_PROVIDERS: {e}"))?,
            },
        },
        "import" => Cmd::Import {
            path: path.ok_or("missing trace file")?,
            db,
            project,
        },
        _ => Cmd::View {
            path: path.ok_or("missing trace file")?,
            trace,
            io,
        },
    })
}

fn main() -> ExitCode {
    let cmd = match parse_args(std::env::args().skip(1)) {
        Ok(c) => c,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match cmd {
        Cmd::View { path, trace, io } => match view::load_jsonl(&path) {
            Ok(runs) => {
                view::print_traces(runs, trace.as_deref(), io);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: cannot read {path}: {e}");
                ExitCode::from(1)
            }
        },
        Cmd::Serve {
            db,
            addr,
            token,
            pricing,
            retention_days,
            no_engine,
            metrics_auth,
            playground_providers,
        } => run_async(async move {
            let addr: SocketAddr = addr
                .parse()
                .map_err(|e| format!("invalid --addr `{addr}`: {e}"))?;
            let token = token.filter(|t| !t.is_empty());
            if !addr.ip().is_loopback() && token.is_none() {
                eprintln!(
                    "warning: listening on a non-loopback address without --token; \
                     anyone who can reach it can read and write traces"
                );
            }
            let store = Store::open(&db).await.map_err(|e| e.to_string())?;
            eprintln!("database: {db}");
            if let Some(path) = pricing {
                let table = read_pricing(&path)?;
                let n = store.set_pricing(table).await.map_err(|e| e.to_string())?;
                eprintln!("pricing: {path} ({n} runs priced)");
            }
            if let Some(days) = retention_days {
                let mut s = store.settings().await.map_err(|e| e.to_string())?;
                s.retention_days = Some(days);
                store.set_settings(&s).await.map_err(|e| e.to_string())?;
                eprintln!("retention: {days} days");
            }
            let mut playground = sinew_scope::PlaygroundConfig::default();
            if let Some(p) = playground_providers {
                playground = playground.allow_providers(p);
            }
            eprintln!(
                "playground providers: {}",
                if playground.allowed_providers().is_empty() {
                    "none".to_string()
                } else {
                    playground.allowed_providers().join(", ")
                }
            );
            sinew_scope::server::serve_with_config(
                addr,
                store,
                sinew_scope::ServerConfig {
                    token,
                    disable_engine: no_engine,
                    metrics_auth,
                    playground,
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| e.to_string())
        }),
        Cmd::Import { path, db, project } => run_async(async move {
            let n = import(&path, &db, &project).await?;
            println!("imported {n} runs into {db} (project `{project}`)");
            Ok(())
        }),
    }
}

/// Reads a pricing file: a `Pricing` object or a bare list of model prices.
fn read_pricing(path: &str) -> Result<sinew_scope::Pricing, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    serde_json::from_str(&text)
        .or_else(|_| {
            serde_json::from_str::<Vec<sinew_scope::ModelPrice>>(&text)
                .map(|models| sinew_scope::Pricing { models })
        })
        .map_err(|e| format!("invalid pricing file {path}: {e}"))
}

/// Imports a JSONL trace file into the database; returns the runs accepted.
async fn import(path: &str, db: &str, project: &str) -> Result<usize, String> {
    let runs = view::load_jsonl(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let store = Store::open(db).await.map_err(|e| e.to_string())?;
    let pairs: Vec<_> = runs.into_iter().map(|r| (project.to_string(), r)).collect();
    let mut accepted = 0;
    for chunk in pairs.chunks(500) {
        accepted += store
            .ingest(chunk)
            .await
            .map_err(|e| e.to_string())?
            .accepted;
    }
    Ok(accepted)
}

fn run_async(fut: impl std::future::Future<Output = Result<(), String>>) -> ExitCode {
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    match rt.block_on(fut) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cmd, String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn serve_defaults_and_flags() {
        match parse(&["serve"]).unwrap() {
            Cmd::Serve {
                db,
                addr,
                pricing,
                retention_days,
                no_engine,
                metrics_auth,
                ..
            } => {
                assert_eq!(db, "scope.db");
                assert_eq!(addr, "127.0.0.1:7878");
                assert!(pricing.is_none() && retention_days.is_none());
                assert!(!no_engine && !metrics_auth);
            }
            _ => panic!("expected serve"),
        }
        match parse(&[
            "serve",
            "--db",
            "x.db",
            "--addr",
            "0.0.0.0:9000",
            "--token",
            "secret",
            "--pricing",
            "p.json",
            "--retention-days",
            "30",
            "--no-engine",
            "--metrics-auth",
            "--playground-providers",
            "ollama, openai,ollama",
        ])
        .unwrap()
        {
            Cmd::Serve {
                db,
                addr,
                token,
                pricing,
                retention_days,
                no_engine,
                metrics_auth,
                playground_providers,
            } => {
                assert_eq!(
                    playground_providers,
                    Some(vec!["ollama".to_string(), "openai".to_string()])
                );
                assert_eq!(db, "x.db");
                assert_eq!(addr, "0.0.0.0:9000");
                assert_eq!(token.as_deref(), Some("secret"));
                assert_eq!(pricing.as_deref(), Some("p.json"));
                assert_eq!(retention_days, Some(30));
                assert!(no_engine && metrics_auth);
            }
            _ => panic!("expected serve"),
        }
    }

    #[test]
    fn import_and_view() {
        match parse(&["import", "t.jsonl", "--project", "p", "--db", "d.db"]).unwrap() {
            Cmd::Import { path, db, project } => {
                assert_eq!((path.as_str(), db.as_str()), ("t.jsonl", "d.db"));
                assert_eq!(project, "p");
            }
            _ => panic!("expected import"),
        }
        match parse(&["import", "t.jsonl"]).unwrap() {
            Cmd::Import { project, .. } => assert_eq!(project, "default"),
            _ => panic!("expected import"),
        }
        match parse(&["view", "t.jsonl", "--trace", "ab", "--io"]).unwrap() {
            Cmd::View { path, trace, io } => {
                assert_eq!(path, "t.jsonl");
                assert_eq!(trace.as_deref(), Some("ab"));
                assert!(io);
            }
            _ => panic!("expected view"),
        }
        // A bare path is `view <path>`.
        match parse(&["t.jsonl"]).unwrap() {
            Cmd::View { path, trace, io } => {
                assert_eq!(path, "t.jsonl");
                assert!(trace.is_none() && !io);
            }
            _ => panic!("expected view"),
        }
    }

    #[test]
    fn pricing_file_object_list_and_errors() {
        let dir = tempfile::tempdir().unwrap();
        let obj = dir.path().join("obj.json");
        std::fs::write(
            &obj,
            r#"{"models":[{"model":"m1","input_per_mtok":1.0,"output_per_mtok":2.0}]}"#,
        )
        .unwrap();
        let list = dir.path().join("list.json");
        std::fs::write(
            &list,
            r#"[{"model":"m2","input_per_mtok":3.0,"output_per_mtok":4.0,"provider":"p"}]"#,
        )
        .unwrap();
        let bad = dir.path().join("bad.json");
        std::fs::write(&bad, r#"{"nope": 1}"#).unwrap();

        let p = read_pricing(obj.to_str().unwrap()).unwrap();
        assert_eq!(p.models[0].model, "m1");
        let p = read_pricing(list.to_str().unwrap()).unwrap();
        assert_eq!(p.models[0].provider.as_deref(), Some("p"));
        let e = read_pricing(bad.to_str().unwrap()).unwrap_err();
        assert!(e.starts_with("invalid pricing file"), "{e}");
        let e = read_pricing("/definitely/missing.json").unwrap_err();
        assert!(e.starts_with("cannot read"), "{e}");
    }

    #[tokio::test]
    async fn import_writes_runs_into_the_database() {
        let dir = tempfile::tempdir().unwrap();
        let trace = dir.path().join("t.jsonl");
        let runs: Vec<String> = (0..3)
            .map(|i| {
                serde_json::json!({
                    "id": format!("r{i}"), "trace_id": "t1",
                    "parent_id": if i == 0 { None } else { Some("r0") },
                    "name": "n", "kind": "chain", "start_ms": i, "end_ms": i + 1,
                    "inputs": {}, "outputs": null, "error": null,
                    "input_tokens": null, "output_tokens": null
                })
                .to_string()
            })
            .collect();
        std::fs::write(&trace, runs.join("\n")).unwrap();
        let db = dir.path().join("scope.db");
        let (trace, db) = (trace.to_str().unwrap(), db.to_str().unwrap());
        assert_eq!(import(trace, db, "proj").await.unwrap(), 3);
        let stats = Store::open(db).await.unwrap().stats().await.unwrap();
        assert_eq!((stats.runs, stats.traces), (3, 1));
        let e = import("/definitely/missing.jsonl", db, "p")
            .await
            .unwrap_err();
        assert!(e.starts_with("cannot read"), "{e}");
    }

    #[test]
    fn errors_and_help() {
        assert_eq!(parse(&[]).err().as_deref(), Some("missing command"));
        for help in [&["--help"][..], &["help"], &["-h"], &["serve", "--help"]] {
            assert_eq!(parse(help).err().as_deref(), Some(""));
        }
        assert_eq!(
            parse(&["import"]).err().as_deref(),
            Some("missing trace file")
        );
        assert_eq!(
            parse(&["view"]).err().as_deref(),
            Some("missing trace file")
        );
        assert_eq!(
            parse(&["serve", "--db"]).err().as_deref(),
            Some("--db needs a value")
        );
        for bad in ["0", "-1", "x"] {
            assert_eq!(
                parse(&["serve", "--retention-days", bad]).err(),
                Some(format!("invalid --retention-days `{bad}`"))
            );
        }
        let e = parse(&["serve", "--playground-providers", "nope"])
            .err()
            .unwrap();
        assert!(e.contains("unknown provider `nope`"), "{e}");
        assert_eq!(
            parse(&["serve", "--bogus"]).err().as_deref(),
            Some("unexpected argument `--bogus`")
        );
        assert_eq!(
            parse(&["view", "a.jsonl", "b.jsonl"]).err().as_deref(),
            Some("unexpected argument `b.jsonl`")
        );
    }
}
