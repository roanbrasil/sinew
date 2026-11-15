//! `sinew-mcp --traces <path>`: an MCP server (stdio) that lets coding
//! assistants inspect Sinew traces. See `docs/mcp.md`.

use std::{path::PathBuf, process::ExitCode};

use sinew_mcp::{traces::trace_toolbox, McpServer};

const USAGE: &str = "usage: sinew-mcp --traces <traces.jsonl>

Serves the Sinew trace file over the Model Context Protocol (stdio) with the
tools list_traces, get_trace and search_runs. The file may not exist yet;
it is read on every call.";

fn parse_args() -> Result<PathBuf, String> {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--traces" => path = Some(args.next().ok_or("--traces needs a path")?),
            "-h" | "--help" => return Err(String::new()),
            "-V" | "--version" => {
                println!("sinew-mcp {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }
    path.map(PathBuf::from)
        .ok_or_else(|| "missing --traces <path>".into())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let path = match parse_args() {
        Ok(p) => p,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let server = McpServer::new("sinew-mcp", env!("CARGO_PKG_VERSION"))
        .with_instructions(format!(
            "Inspect Sinew agent traces from {}. Start with list_traces, then get_trace with an id prefix; use search_runs to find failing tools or slow model calls.",
            path.display()
        ))
        .with_tools(trace_toolbox(path));
    match server.serve_stdio().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sinew-mcp: {e}");
            ExitCode::FAILURE
        }
    }
}
