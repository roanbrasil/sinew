//! `sinew-agent`: a terminal coding agent on the Sinew harness.
//!
//! ```text
//! sinew-agent                                  # interactive REPL in the current directory
//! sinew-agent -p "summarize README.md"         # one-shot
//! sinew-agent --model anthropic:claude-sonnet-4-5 --workspace ~/code/app --allow-shell
//! ```

use std::{
    io::{IsTerminal, Write},
    path::PathBuf,
    process::ExitCode,
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use serde_json::Value;
use sinew_harness::{
    tools::todo::render_todos, Agent, AgentEvent, AgentStream, ApprovalDecision, ApprovalRequest,
    InjectionDetector, PermissionPolicy, RunResult, ShellOptions,
};
use tokio::sync::mpsc;

const DEFAULT_MODEL: &str = "ollama:qwen3:4b-instruct-2507-q4_K_M";

const CLI_INSTRUCTIONS: &str = "You are sinew-agent, a coding and research assistant working in the user's workspace through tools. Read before you change things, make focused edits with edit_file, and verify your work (re-read files, run tests when a shell is available). Keep the user informed with short messages; give a concise final answer. Never invent file contents: read them.";

const HELP: &str = "sinew-agent: an AI agent in your terminal, built on Sinew.

USAGE:
    sinew-agent [OPTIONS]                 interactive session
    sinew-agent [OPTIONS] -p <PROMPT>     one-shot: answer and exit

OPTIONS:
    -p, --prompt <TEXT>       Run one prompt and exit
    -m, --model <SPEC>        provider:model [default: ollama:qwen3:4b-instruct-2507-q4_K_M]
                              (env SINEW_AGENT_MODEL), e.g. anthropic:claude-sonnet-4-5,
                              openai:gpt-4.1-mini, lmstudio:qwen3-8b
    -w, --workspace <DIR>     Workspace root for file tools and the shell [default: .]
        --read-only           Only read_file, list_dir, glob, grep
        --allow-shell         Enable the shell tool (asks before each command unless --yes)
        --shell-allow <PAT>   Restrict the shell to commands matching PAT (repeatable), e.g. 'cargo *'
        --http                Enable http_fetch
        --think               Enable the think scratchpad tool
        --no-subagents        Disable the task tool
        --skills <DIR>        Skills folder (also auto: .sinew/skills, .claude/skills)
    -y, --yes                 Approve every tool call without asking
        --max-steps <N>       Model calls per turn [default: 40]
        --max-tokens <N>      Token budget per turn
        --timeout <SECS>      Wall-clock limit per turn
        --max-context <N>     Context budget in tokens before compaction
        --instructions <TEXT> Replace the default system instructions
        --trace <FILE>        Write a JSONL trace (view with `sinew-scope view FILE`)
        --mcp <CMD>           Attach an MCP server over stdio (repeatable; needs feature `mcp`)
    -q, --quiet               Only print the answer
    -h, --help                Show this help
    -V, --version             Show the version

IN THE REPL:
    /help  /tools  /todos  /usage  /reset  /exit      Ctrl-C cancels the running turn
";

#[derive(Debug, Default)]
struct Args {
    prompt: Option<String>,
    model: Option<String>,
    workspace: Option<PathBuf>,
    read_only: bool,
    allow_shell: bool,
    shell_allow: Vec<String>,
    http: bool,
    think: bool,
    no_subagents: bool,
    skills: Vec<PathBuf>,
    yes: bool,
    max_steps: Option<u32>,
    max_tokens: Option<u64>,
    timeout: Option<u64>,
    max_context: Option<usize>,
    instructions: Option<String>,
    trace: Option<PathBuf>,
    mcp: Vec<String>,
    quiet: bool,
}

fn parse_args(raw: Vec<String>) -> Result<Option<Args>, String> {
    let mut a = Args::default();
    let mut it = raw.into_iter();
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            inline
                .clone()
                .or_else(|| it.next())
                .ok_or_else(|| format!("{name} needs a value"))
        };
        let num = |v: String, name: &str| -> Result<u64, String> {
            v.parse()
                .map_err(|_| format!("{name}: `{v}` is not a number"))
        };
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("sinew-agent {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "-p" | "--prompt" => a.prompt = Some(value(&flag)?),
            "-m" | "--model" => a.model = Some(value(&flag)?),
            "-w" | "--workspace" => a.workspace = Some(value(&flag)?.into()),
            "--read-only" => a.read_only = true,
            "--allow-shell" => a.allow_shell = true,
            "--shell-allow" => {
                a.allow_shell = true;
                a.shell_allow.push(value(&flag)?);
            }
            "--http" => a.http = true,
            "--think" => a.think = true,
            "--no-subagents" => a.no_subagents = true,
            "--skills" => a.skills.push(value(&flag)?.into()),
            "-y" | "--yes" => a.yes = true,
            "--max-steps" => a.max_steps = Some(num(value(&flag)?, &flag)? as u32),
            "--max-tokens" => a.max_tokens = Some(num(value(&flag)?, &flag)?),
            "--timeout" => a.timeout = Some(num(value(&flag)?, &flag)?),
            "--max-context" => a.max_context = Some(num(value(&flag)?, &flag)? as usize),
            "--instructions" => a.instructions = Some(value(&flag)?),
            "--trace" => a.trace = Some(value(&flag)?.into()),
            "--mcp" => a.mcp.push(value(&flag)?),
            "-q" | "--quiet" => a.quiet = true,
            other => return Err(format!("unknown option `{other}` (see --help)")),
        }
    }
    Ok(Some(a))
}

/// Terminal styling (plain when stdout is not a terminal).
struct Ui {
    color: bool,
    quiet: bool,
}

impl Ui {
    fn paint(&self, code: &str, s: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    fn dim(&self, s: &str) -> String {
        self.paint("2", s)
    }
    fn cyan(&self, s: &str) -> String {
        self.paint("36", s)
    }
    fn red(&self, s: &str) -> String {
        self.paint("31", s)
    }
    fn yellow(&self, s: &str) -> String {
        self.paint("33", s)
    }
}

fn out(s: &str) {
    let mut o = std::io::stdout().lock();
    let _ = o.write_all(s.as_bytes());
    let _ = o.flush();
}

fn one_line(s: &str, max: usize) -> String {
    let line = s
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    if line.chars().count() > max {
        format!("{}…", line.chars().take(max).collect::<String>())
    } else {
        line.to_string()
    }
}

/// `read_file(notes.md)`-style summary of a call.
fn call_summary(name: &str, args: &Value) -> String {
    let key = ["path", "command", "url", "pattern", "description", "name"]
        .iter()
        .find_map(|k| args.get(*k).and_then(Value::as_str));
    let detail = match key {
        Some(v) => v.to_string(),
        None => args.to_string(),
    };
    format!("{name}({})", one_line(&detail, 90))
}

type Lines = mpsc::UnboundedReceiver<Option<String>>;

/// Reads stdin lines on a thread (None = EOF).
fn spawn_line_reader() -> Lines {
    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        loop {
            let mut line = String::new();
            match stdin.read_line(&mut line) {
                Ok(0) | Err(_) => {
                    let _ = tx.send(None);
                    break;
                }
                Ok(_) => {
                    if tx
                        .send(Some(line.trim_end_matches(['\n', '\r']).to_string()))
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });
    rx
}

enum Terminal {
    Done(Box<RunResult>),
    Approval(Vec<ApprovalRequest>),
    Error(String),
    Cancelled,
}

async fn render(stream: &mut AgentStream, ui: &Ui) -> Terminal {
    let mut at_line_start = true;
    loop {
        let ev = tokio::select! {
            ev = stream.next() => ev,
            _ = tokio::signal::ctrl_c() => return Terminal::Cancelled,
        };
        let Some(ev) = ev else {
            return Terminal::Error("the run ended unexpectedly".into());
        };
        match ev {
            AgentEvent::Token { text } => {
                out(&text);
                at_line_start = text.ends_with('\n');
            }
            AgentEvent::ToolStart {
                name, arguments, ..
            } if !ui.quiet => {
                let nl = if at_line_start { "" } else { "\n" };
                out(&format!(
                    "{nl}{}\n",
                    ui.cyan(&format!("> {}", call_summary(&name, &arguments)))
                ));
                at_line_start = true;
            }
            AgentEvent::ToolEnd {
                name,
                output,
                is_error,
                ..
            } if !ui.quiet => {
                let text = if is_error {
                    ui.red(&format!("  ! {}", one_line(&output, 160)))
                } else if name == "write_todos" {
                    let body: Vec<String> =
                        output.lines().skip(1).map(|l| format!("  {l}")).collect();
                    ui.dim(&body.join("\n"))
                } else {
                    let n = output.lines().count();
                    if n <= 1 {
                        ui.dim(&format!("  = {}", one_line(&output, 160)))
                    } else {
                        ui.dim(&format!("  = {} ({n} lines)", one_line(&output, 100)))
                    }
                };
                out(&format!("{text}\n"));
                at_line_start = true;
            }
            AgentEvent::Compacted(info) if !ui.quiet => {
                let nl = if at_line_start { "" } else { "\n" };
                out(&format!(
                    "{nl}{}\n",
                    ui.dim(&format!(
                        "[context compacted: {} messages summarized, ~{} -> ~{} tokens]",
                        info.removed_messages, info.tokens_before, info.tokens_after
                    ))
                ));
                at_line_start = true;
            }
            AgentEvent::Done { result } => {
                if !at_line_start {
                    out("\n");
                }
                return Terminal::Done(Box::new(result));
            }
            AgentEvent::ApprovalRequired { requests } => {
                if !at_line_start {
                    out("\n");
                }
                return Terminal::Approval(requests);
            }
            AgentEvent::Error { message } => {
                if !at_line_start {
                    out("\n");
                }
                return Terminal::Error(message);
            }
            _ => {}
        }
    }
}

async fn ask_approvals(
    requests: &[ApprovalRequest],
    lines: &mut Option<Lines>,
    ui: &Ui,
) -> Option<Vec<ApprovalDecision>> {
    let Some(rx) = lines.as_mut() else {
        out(&ui.yellow(
            "[approval needed but stdin is not interactive; denying. Use --yes to approve]\n",
        ));
        return Some(
            requests
                .iter()
                .map(|r| ApprovalDecision::deny(&r.call_id, "not approved (non-interactive)"))
                .collect(),
        );
    };
    let mut decisions = Vec::new();
    for r in requests {
        let detail = match r.arguments.get("command").and_then(Value::as_str) {
            Some(cmd) => cmd.to_string(),
            None => serde_json::to_string_pretty(&r.arguments).unwrap_or_default(),
        };
        let shown: String = detail.chars().take(1500).collect();
        out(&format!(
            "{}\n{}\n{} ",
            ui.yellow(&format!("? allow {}", r.tool)),
            ui.dim(&shown),
            ui.yellow("[y]es / [n]o / [a]lways for this tool:")
        ));
        let answer = tokio::select! {
            l = rx.recv() => l.flatten(),
            _ = tokio::signal::ctrl_c() => { out("\n"); return None }
        };
        let answer = answer?;
        decisions.push(match answer.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" | "" => ApprovalDecision::approve(&r.call_id),
            "a" | "always" => ApprovalDecision::approve_always(&r.call_id),
            other => {
                let reason = if other == "n" || other == "no" {
                    "the user declined".to_string()
                } else {
                    format!("the user declined and said: {other}")
                };
                ApprovalDecision::deny(&r.call_id, reason)
            }
        });
    }
    Some(decisions)
}

/// Runs one turn, handling approvals and Ctrl-C.
async fn run_turn(
    agent: &Agent,
    thread: &str,
    input: &str,
    lines: &mut Option<Lines>,
    ui: &Ui,
) -> Result<Option<RunResult>, String> {
    let started = Instant::now();
    let mut stream = agent
        .stream(thread, input)
        .await
        .map_err(|e| e.to_string())?;
    loop {
        match render(&mut stream, ui).await {
            Terminal::Done(r) => {
                if let Some(stop) = &r.stop {
                    out(&ui.yellow(&format!("[stopped: {stop}]\n")));
                }
                if !ui.quiet {
                    out(&ui.dim(&format!(
                        "[{} step{} · {} in / {} out tokens · {:.1}s]\n",
                        r.steps,
                        if r.steps == 1 { "" } else { "s" },
                        r.usage.input_tokens,
                        r.usage.output_tokens,
                        started.elapsed().as_secs_f64()
                    )));
                }
                return Ok(Some(*r));
            }
            Terminal::Approval(requests) => {
                drop(stream);
                let Some(decisions) = ask_approvals(&requests, lines, ui).await else {
                    out(&ui.yellow("[cancelled]\n"));
                    return Ok(None);
                };
                stream = agent
                    .resume_stream(thread, decisions)
                    .await
                    .map_err(|e| e.to_string())?;
            }
            Terminal::Error(e) => return Err(e),
            Terminal::Cancelled => {
                drop(stream);
                out(&ui.yellow("\n[cancelled]\n"));
                return Ok(None);
            }
        }
    }
}

async fn build_agent(args: &Args) -> Result<Agent, String> {
    let model = args
        .model
        .clone()
        .or_else(|| std::env::var("SINEW_AGENT_MODEL").ok())
        .unwrap_or_else(|| DEFAULT_MODEL.into());
    let workspace = args.workspace.clone().unwrap_or_else(|| PathBuf::from("."));
    let mut b = Agent::builder()
        .model_spec(&model)
        .name("sinew-agent")
        .instructions(
            args.instructions
                .clone()
                .unwrap_or_else(|| CLI_INSTRUCTIONS.into()),
        )
        .workspace(&workspace)
        .todos()
        .project_instructions()
        .guardrail(InjectionDetector::new())
        .permissions(PermissionPolicy::interactive())
        .auto_approve(args.yes)
        .max_steps(args.max_steps.unwrap_or(40));
    b = if args.read_only {
        b.filesystem_read_only()
    } else {
        b.filesystem()
    };
    if args.allow_shell {
        let mut opts = ShellOptions::default();
        for p in &args.shell_allow {
            opts = opts.allow(p.clone());
        }
        b = b.shell(opts);
    }
    if args.http {
        b = b.http_fetch(Default::default());
    }
    if args.think {
        b = b.think();
    }
    if !args.no_subagents {
        b = b.general_subagent();
    }
    let mut skills = args.skills.clone();
    for auto in [".sinew/skills", ".claude/skills"] {
        let d = workspace.join(auto);
        if d.is_dir() {
            skills.push(d);
        }
    }
    for d in skills {
        b = b.skills_dir(d);
    }
    if let Some(n) = args.max_tokens {
        b = b.max_total_tokens(n);
    }
    if let Some(s) = args.timeout {
        b = b.timeout(Duration::from_secs(s));
    }
    let ctx = args.max_context.unwrap_or(if model.starts_with("ollama:") {
        8_000
    } else {
        24_000
    });
    b = b.max_context_tokens(ctx);
    if let Some(path) = &args.trace {
        let sink = sinew_trace::JsonlSink::open(path)
            .map_err(|e| format!("trace file {}: {e}", path.display()))?;
        b = b.tracer(sinew_trace::Tracer::new().with_sink(sink));
    }
    #[cfg(feature = "mcp")]
    for cmd in &args.mcp {
        let client = sinew_harness::mcp::connect_command_line(cmd)
            .await
            .map_err(|e| format!("MCP server `{cmd}`: {e}"))?;
        b = b
            .mcp_server(&client)
            .await
            .map_err(|e| format!("MCP server `{cmd}`: {e}"))?;
    }
    #[cfg(not(feature = "mcp"))]
    if !args.mcp.is_empty() {
        return Err("--mcp needs sinew-agent built with the `mcp` feature".into());
    }
    b.build().map_err(|e| e.to_string())
}

async fn repl(agent: Agent, ui: Ui) -> ExitCode {
    let mut lines = Some(spawn_line_reader());
    let mut thread = uuid_like();
    out(&ui.dim(&format!(
        "sinew-agent {} · model {} · {} tools · /help for commands, Ctrl-C cancels a turn, Ctrl-D exits\n",
        env!("CARGO_PKG_VERSION"),
        agent.model_name(),
        agent.tool_names().len()
    )));
    loop {
        out(&ui.paint("1", "\n> "));
        let line = {
            let rx = lines.as_mut().expect("reader");
            tokio::select! {
                l = rx.recv() => l.flatten(),
                _ = tokio::signal::ctrl_c() => None,
            }
        };
        let Some(line) = line else {
            out("\n");
            return ExitCode::SUCCESS;
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            "/exit" | "/quit" => return ExitCode::SUCCESS,
            "/help" => {
                out(&ui.dim("/tools  list tools\n/todos  show the plan\n/usage  token usage of this session\n/reset  start a new conversation\n/exit   quit\n"));
                continue;
            }
            "/tools" => {
                out(&format!("{}\n", agent.tool_names().join(", ")));
                continue;
            }
            "/reset" => {
                thread = uuid_like();
                out(&ui.dim("[new conversation]\n"));
                continue;
            }
            "/todos" | "/usage" => {
                let state = agent
                    .state(&thread)
                    .await
                    .ok()
                    .flatten()
                    .unwrap_or_default();
                if input == "/todos" {
                    out(&format!("{}\n", render_todos(&state.todos)));
                } else {
                    out(&format!(
                        "{} model calls · {} in / {} out tokens · {} compactions\n",
                        state.usage.model_calls,
                        state.usage.input_tokens,
                        state.usage.output_tokens,
                        state.compactions
                    ));
                }
                continue;
            }
            _ => {}
        }
        if let Err(e) = run_turn(&agent, &thread, input, &mut lines, &ui).await {
            out(&ui.red(&format!("error: {e}\n")));
        }
    }
}

fn uuid_like() -> String {
    format!(
        "cli-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    )
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = match parse_args(std::env::args().skip(1).collect()) {
        Ok(Some(a)) => a,
        Ok(None) => return ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sinew-agent: {e}");
            return ExitCode::from(2);
        }
    };
    let ui = Ui {
        color: std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
        quiet: args.quiet,
    };
    let agent = match build_agent(&args).await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("sinew-agent: {e}");
            return ExitCode::from(2);
        }
    };
    match &args.prompt {
        Some(prompt) => {
            let mut lines = if std::io::stdin().is_terminal() {
                Some(spawn_line_reader())
            } else {
                None
            };
            match run_turn(&agent, &uuid_like(), prompt, &mut lines, &ui).await {
                Ok(Some(_)) => ExitCode::SUCCESS,
                Ok(None) => ExitCode::from(130),
                Err(e) => {
                    eprintln!("{}", ui.red(&format!("error: {e}")));
                    ExitCode::FAILURE
                }
            }
        }
        None => repl(agent, ui).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Args {
        parse_args(v.iter().map(|s| s.to_string()).collect())
            .unwrap()
            .unwrap()
    }

    #[test]
    fn parses_flags() {
        let a = args(&[
            "-p",
            "hi",
            "--model=openai:gpt-4.1-mini",
            "--shell-allow",
            "cargo *",
            "-y",
            "--max-steps",
            "7",
        ]);
        assert_eq!(a.prompt.as_deref(), Some("hi"));
        assert_eq!(a.model.as_deref(), Some("openai:gpt-4.1-mini"));
        assert!(a.allow_shell && a.yes);
        assert_eq!(a.shell_allow, vec!["cargo *"]);
        assert_eq!(a.max_steps, Some(7));
        assert!(parse_args(vec!["--bogus".into()]).is_err());
        assert!(parse_args(vec!["--max-steps".into(), "x".into()]).is_err());
    }

    #[test]
    fn summarizes_calls() {
        assert_eq!(
            call_summary("read_file", &serde_json::json!({"path": "src/main.rs"})),
            "read_file(src/main.rs)"
        );
        assert_eq!(
            call_summary("x", &serde_json::json!({"a": 1})),
            "x({\"a\":1})"
        );
    }
}
