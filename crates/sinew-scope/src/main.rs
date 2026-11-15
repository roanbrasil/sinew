//! `sinew-scope` command line.
//!
//! ```text
//! sinew-scope serve  [--db scope.db] [--addr 127.0.0.1:7878] [--token T]
//!                    [--pricing prices.json] [--retention-days N] [--no-engine]
//! sinew-scope import <traces.jsonl> [--db scope.db] [--project P]
//! sinew-scope view   <traces.jsonl> [--trace <id-prefix>] [--io]
//! sinew-scope <traces.jsonl> [--trace <id-prefix>] [--io]   # same as `view`
//! ```

use std::{net::SocketAddr, process::ExitCode};

use sinew_scope::{store::Store, view};

const USAGE: &str = "usage:
  sinew-scope serve  [--db scope.db] [--addr 127.0.0.1:7878] [--token T]
                     [--pricing prices.json] [--retention-days N] [--no-engine]
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
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("cannot read {path}: {e}"))?;
                let table: sinew_scope::Pricing = serde_json::from_str(&text)
                    .or_else(|_| {
                        serde_json::from_str::<Vec<sinew_scope::ModelPrice>>(&text)
                            .map(|models| sinew_scope::Pricing { models })
                    })
                    .map_err(|e| format!("invalid pricing file {path}: {e}"))?;
                let n = store.set_pricing(table).await.map_err(|e| e.to_string())?;
                eprintln!("pricing: {path} ({n} runs priced)");
            }
            if let Some(days) = retention_days {
                let mut s = store.settings().await.map_err(|e| e.to_string())?;
                s.retention_days = Some(days);
                store.set_settings(&s).await.map_err(|e| e.to_string())?;
                eprintln!("retention: {days} days");
            }
            sinew_scope::server::serve_with_config(
                addr,
                store,
                sinew_scope::ServerConfig {
                    token,
                    disable_engine: no_engine,
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| e.to_string())
        }),
        Cmd::Import { path, db, project } => run_async(async move {
            let runs = view::load_jsonl(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
            let store = Store::open(&db).await.map_err(|e| e.to_string())?;
            let pairs: Vec<_> = runs.into_iter().map(|r| (project.clone(), r)).collect();
            let mut total = (0, 0);
            for chunk in pairs.chunks(500) {
                let r = store.ingest(chunk).await.map_err(|e| e.to_string())?;
                total.0 += r.accepted;
                total.1 += r.traces;
            }
            println!("imported {} runs into {db} (project `{project}`)", total.0);
            Ok(())
        }),
    }
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
