//! Runs the OWASP agentic attack suite and exits with status 1 when a
//! scenario regresses, so it can gate CI.
//!
//! ```bash
//! cargo run -p sample-23-owasp-agentic                                  # offline, scripted models
//! cargo run -p sample-23-owasp-agentic -- --only ASI01                  # one risk
//! cargo run -p sample-23-owasp-agentic -- --json report.json --markdown report.md
//! cargo run -p sample-23-owasp-agentic -- --live ollama:qwen3:4b-instruct-2507-q4_K_M
//! SINEW_SECURITY_LIVE=anthropic:claude-haiku-4-5 cargo run -p sample-23-owasp-agentic
//! ```

use sinew::eval::security::SuiteMode;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match flag(&args, "--live") {
        Some(spec) => SuiteMode::from_spec(&spec)?,
        None => SuiteMode::from_env()?,
    };
    let mut suite = sample_23_owasp_agentic::suite();
    if let Some(prefix) = flag(&args, "--only") {
        suite = suite.filter(&prefix);
    }
    let report = suite.run_with(mode).await;
    println!("{}", report.to_markdown());
    if let Some(path) = flag(&args, "--json") {
        std::fs::write(&path, report.to_json())?;
        eprintln!("wrote {path}");
    }
    if let Some(path) = flag(&args, "--markdown") {
        std::fs::write(&path, report.to_markdown())?;
        eprintln!("wrote {path}");
    }
    eprintln!("{}", report.summary());
    if !report.passed() {
        std::process::exit(1);
    }
    Ok(())
}
