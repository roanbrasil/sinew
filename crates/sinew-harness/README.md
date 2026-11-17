# sinew-harness

An agent harness for [Sinew](https://github.com/roanbrasil/sinew): the loop,
tools and safety rails that turn a chat model into a working agent, in the
spirit of Claude Code, LangChain deepagents and the OpenAI Agents SDK. It
ships as a library and as the `sinew-agent` terminal app.

```rust
use futures_util::StreamExt;
use sinew_harness::{Agent, AgentEvent, PermissionPolicy, ShellOptions};

let agent = Agent::builder()
    .model_spec("ollama:qwen3:4b-instruct-2507-q4_K_M")   // or any ChatModel
    .instructions("You are a coding assistant.")
    .workspace("./my-project")
    .filesystem()                                          // read/write/edit/list/glob/grep
    .shell(ShellOptions::default().allow("cargo *"))
    .todos()
    .general_subagent()
    .permissions(PermissionPolicy::interactive())          // ask before writes and shell
    .max_steps(30)
    .build()?;

let mut events = agent.stream("thread-1", "Fix the failing test").await?;
while let Some(ev) = events.next().await {
    match ev {
        AgentEvent::Token { text } => print!("{text}"),
        AgentEvent::ToolStart { name, .. } => println!("\n> {name}"),
        AgentEvent::ApprovalRequired { requests } => { /* ask, then agent.resume_stream(...) */ }
        AgentEvent::Done { result } => println!("\n{} steps", result.steps),
        _ => {}
    }
}
```

## What's in it

| Area | What you get |
|---|---|
| Loop | `agent → (approval) → tools → agent` on a `sinew-graph` `StateGraph`, checkpointed after every node: threads resume across turns and after approvals; any `Checkpointer` (e.g. `sinew-checkpoint` SQLite) works |
| Models | any `ChatModel`, or a `provider:model` spec (`from_spec`): Ollama, OpenAI, Anthropic, Gemini, xAI, Groq, vLLM, LM Studio… |
| Streaming | `AgentEvent`: `Step`, `Token`, `ToolStart`, `ToolEnd`, `Compacted`, `ApprovalRequired`, `Done`, `Error`; dropping the stream cancels the turn |
| Structured output | `output_schema(StructuredOutput)`; the parsed value lands in `RunResult::output` |
| Filesystem | `read_file` (line ranges), `write_file`, `edit_file` (exact replace, uniqueness check), `list_dir`, `glob`, `grep` (regex); confined to the workspace, `..` and symlink escapes rejected |
| Shell | `sh -c` in the workspace, built-in denylist (`sudo`, `rm -rf /`, `mkfs`, `curl … \| sh`, …), optional allowlist, timeout, head+tail truncation |
| Web | `http_fetch`: GET, byte limit, HTML → text, private-address guard, optional host allowlist |
| Planning | `write_todos`, stored in the thread state; `think` scratchpad |
| Subagents | `task` tool: named subagents with their own instructions, tools and model; context isolation; depth limit; parallel when several are called at once |
| Hooks | `before_model`, `after_model`, `before_tool` (allow / ask / deny, may rewrite arguments), `after_tool` |
| Permissions | `PermissionPolicy` rules like `write_file(src/**)`, `shell(git *)`, `mcp_*`; deny > ask > allow > default; "always allow" per thread |
| Approvals | `Ask` pauses the graph (interrupt); `Agent::resume(thread, decisions)` continues it |
| Context | token estimates; compaction of older turns with the model (tool calls and results never split); large tool outputs offloaded to `.sinew/tool-outputs/` with a preview |
| Budgets | max steps, max tokens, max cost (with `Pricing`), wall-clock timeout; a budget ends the turn with a `StopReason` |
| Guardrails | `Guardrail` trait on input, final output and tool output; `PiiRedactor` (e-mail, phone, Luhn-checked cards), `InjectionDetector` |
| Project memory | `AGENTS.md`, `CLAUDE.md`, `.claude/CLAUDE.md`; skills folders (`*/SKILL.md` with `name`/`description` frontmatter) listed in the prompt and loaded with `load_skill` |
| MCP | feature `mcp`: `builder.mcp_server(&client).await?` adds an MCP server's tools |
| Tracing | `.tracer(tracer)`: graph, node, every model call (with usage) and every tool call become `sinew-trace` runs |

## The CLI

```bash
cargo install --path crates/sinew-harness        # or: cargo run -p sinew-harness --
sinew-agent                                       # REPL in the current directory
sinew-agent -p "what does src/lib.rs do?"         # one-shot
sinew-agent --model anthropic:claude-sonnet-4-5 --workspace ~/code/app --allow-shell
sinew-agent --allow-shell --shell-allow 'cargo *' --yes -p "run the tests and fix failures"
```

| Flag | Meaning |
|---|---|
| `-p, --prompt TEXT` | one-shot mode |
| `-m, --model SPEC` | `provider:model`, default `ollama:qwen3:4b-instruct-2507-q4_K_M` (or `SINEW_AGENT_MODEL`) |
| `-w, --workspace DIR` | workspace root (default `.`) |
| `--allow-shell`, `--shell-allow PAT` | enable the shell, optionally restricted to patterns |
| `--http`, `--think`, `--read-only`, `--no-subagents` | tool selection |
| `-y, --yes` | approve every tool call |
| `--max-steps`, `--max-tokens`, `--timeout`, `--max-context` | budgets |
| `--skills DIR` | skills folder (`.sinew/skills` and `.claude/skills` are picked up automatically) |
| `--trace FILE` | JSONL trace, view with `sinew-scope view FILE` |
| `--mcp CMD` | attach an MCP server over stdio (build with `--features mcp`) |

Without `--yes`, `write_file`, `edit_file`, `shell` and `http_fetch` ask for
approval (`y`, `n`, `a` = always for this tool). Tokens stream as they
arrive; Ctrl-C cancels the running turn and keeps the conversation. REPL
commands: `/tools`, `/todos`, `/usage`, `/reset`, `/exit`.

## Approvals in code

```rust
use sinew_harness::{AgentOutcome, ApprovalDecision};

match agent.run("t1", "Delete the build folder").await? {
    AgentOutcome::Completed(result) => println!("{}", result.text),
    AgentOutcome::ApprovalRequired { requests, .. } => {
        let decisions = requests.iter().map(|r| ApprovalDecision::approve(&r.call_id)).collect();
        let result = agent.resume("t1", decisions).await?;
    }
}
```

## Features

- `cli` (default): the `sinew-agent` binary.
- `mcp`: MCP client integration via `sinew-mcp`.

See [docs/harness.md](../../docs/harness.md) for the design and every option,
and `samples/19-harness` for a runnable example.

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

MIT OR Apache-2.0
