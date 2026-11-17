# Agent harness (`sinew-harness`)

`sinew-harness` is the layer between "a model that can call tools" and "an
agent you can point at a repository": the loop, a set of opt-in tools,
permissions and approvals, context management, budgets, guardrails, project
memory and a terminal app. It is built only on Sinew primitives:
`sinew-core` models and tools, a `sinew-graph` state graph with
checkpoints and interrupts, `sinew-trace` runs and (optionally) the
`sinew-mcp` client.

Run `cargo run -p sample-19-harness` (offline) or
`cargo run -p sample-19-harness -- --ollama` to see it work, or try
`cargo run -p sinew-harness -- -p "what is in this folder?"`.

## The loop

```text
            ┌──────────────────────────────────────────┐
            ▼                                          │
 user ─► agent ──(no tool calls / stop)──► END          │
            │                                          │
            ├──(every call allowed or denied)──► tools ─┘
            │                                   ▲
            └──(some call needs approval)──► approval
                         (interrupt_before: the run pauses here)
```

Each node is a `sinew-graph` node over `HarnessState`:

- **agent**: checks the budgets, compacts the history if it is over the
  context budget, runs `before_model` hooks, calls the model (streaming when
  the run is streamed), runs `after_model` hooks, accounts usage, and turns
  each requested tool call into a `PendingCall` with a decision (allow / ask
  / deny) from `before_tool` hooks and the permission policy.
- **approval**: the graph is compiled with `interrupt_before(["approval"])`.
  `Agent::run` returns `AgentOutcome::ApprovalRequired`; `Agent::resume`
  writes the human decisions into the checkpoint (`update_state`) and
  resumes the thread.
- **tools**: runs the pending calls concurrently (`write_todos` is applied
  to the state first), then `after_tool` hooks, tool-output guardrails and
  offloading, and appends one tool result per call.

The graph checkpoints after every node (in memory by default; pass any
`Checkpointer`, e.g. `sinew_checkpoint::SqliteCheckpointer`, with
`.checkpointer(...)`). A thread is therefore resumable: `agent.run("t", ...)`
twice continues the same conversation, and a paused approval survives a
restart with a durable checkpointer.

The system prompt is not stored in the state. It is rebuilt for every model
call from the instructions, a short "working style" section for the enabled
tools, the workspace path, project instruction files and the skills list.

### Cancellation and repair

`Agent::stream` drives the graph by polling; dropping the stream cancels the
turn (the CLI does this on Ctrl-C). The last checkpoint may then hold an
assistant message whose tool calls never ran. Before the next turn the
harness repairs the history: every unanswered call gets a synthetic
`error: not executed (the turn was interrupted)` result, so the conversation
stays valid for every provider. The same happens when the user sends a new
message instead of answering an approval.

## Builder reference

| Method | Effect |
|---|---|
| `model(m)`, `model_arc(m)`, `model_spec("provider:model")` | the model; specs use `sinew_core::providers::from_spec` |
| `instructions(text)`, `name(n)` | system instructions (default `DEFAULT_INSTRUCTIONS`), name in traces and hooks |
| `temperature(t)`, `max_tokens(n)` | request parameters |
| `output_schema(StructuredOutput)` | structured final answer in `RunResult::output` (extra model call after the loop) |
| `tool(t)`, `tool_arc(t)`, `tools(iter)`, `toolbox(&tb)` | your own tools |
| `workspace(dir)` | root for file tools, shell and project files |
| `filesystem()`, `filesystem_read_only()` | file tools |
| `shell(ShellOptions)` | shell tool |
| `http_fetch(HttpOptions)` | web fetch tool |
| `todos()`, `think()` | planning tools |
| `skills_dir(dir)`, `project_instructions()`, `instructions_file(path)` | project memory |
| `subagent(SubagentSpec)`, `general_subagent()`, `max_subagent_depth(n)` | `task` tool |
| `hook(h)`, `permissions(policy)`, `auto_approve(bool)`, `guardrail(g)` | control |
| `context(ContextConfig)`, `max_context_tokens(n)` | context budget, compaction, offloading |
| `budget(Budget)`, `max_steps(n)`, `max_total_tokens(n)`, `max_cost_usd(x)`, `pricing(Pricing)`, `timeout(d)` | budgets |
| `checkpointer(cp)`, `shared_checkpointer(arc)`, `tracer(t)` | persistence and tracing |

`Agent` methods: `ask(input)` (fresh thread, fails if approval is needed),
`run(thread, input)`, `resume(thread, decisions)`, `stream(thread, input)`,
`resume_stream(thread, decisions)`, `state(thread)`, `system_prompt()`,
`tool_names()`, `graph()`.

## Tools

All tools are opt-in. Tool failures come back to the model as
`error: ...` so it can recover; they never abort the run.

### Filesystem

`Workspace::resolve` joins the path to the root, normalizes `.`/`..`
lexically, rejects anything outside the root, then canonicalizes the deepest
existing ancestor and rejects it if a symlink leads outside. Absolute paths
inside the workspace are accepted.

| Tool | Arguments | Notes |
|---|---|---|
| `read_file` | `path`, `offset` (1-based), `limit` (default 2000) | numbered lines, long lines cut at 2000 chars, refuses binaries and files over 10 MB |
| `write_file` | `path`, `content` | creates parent directories |
| `edit_file` | `path`, `old_string`, `new_string`, `replace_all` | `old_string` must be unique unless `replace_all` |
| `list_dir` | `path` | directories first, sizes |
| `glob` | `pattern`, `path` | `**` crosses directories; a pattern without `/` matches file names at any depth |
| `grep` | `pattern` (regex), `path`, `glob`, `case_insensitive`, `max_results` | skips binaries, `.git`, `target`, `node_modules` |

### Shell

`ShellOptions { allow, deny, timeout, max_output, cwd, no_default_deny }`.
A command is split into segments on `;`, `&&`, `||`, `|`, `&` and newlines
(redirections like `2>&1` are kept). The whole command and every segment are
checked against the deny patterns (`DEFAULT_DENY` plus yours). With an
allowlist, every segment must match an allow pattern and `$(...)` / backticks
are refused. The process gets no stdin, is killed on timeout, and its
output (exit code, stdout, stderr) is truncated keeping head and tail. This
is a guard against accidents, not a sandbox: run untrusted agents in a
container.

### `http_fetch`

GET with a timeout and a byte limit read in chunks. HTML is reduced to text
(scripts, styles and comments dropped, block tags become line breaks,
entities decoded). `localhost`, private, loopback and link-local IP literals
are refused unless `allow_private` is set; `allowed_hosts` restricts the
hosts further.

### `write_todos` and `think`

`write_todos` replaces the plan (`content` + `pending` / `in_progress` /
`completed`) in `HarnessState::todos`; it is part of the checkpoint and of
`RunResult::todos`. `think` records a thought and does nothing else.

### Subagents (`task`)

`SubagentSpec::new(name, description, instructions)` with
`only_tools([...])`, `no_inherited_tools()`, `tool(extra)`, `model(m)` and
`max_steps(n)`. `general_subagent()` adds a `general-purpose` subagent with
the parent's tools. The `task` tool takes `prompt`, `subagent_type` and a
short `description`, builds the subagent, runs it on a fresh thread and
returns only its final answer. The subagent sees its own system prompt
(plus the workspace, project instructions and skills sections) and the
prompt, nothing of the parent conversation.

Depth: the top-level agent has depth 0 and gets `task` when
`depth < max_subagent_depth` (default 1), so by default subagents cannot
spawn subagents. Subagents inherit hooks, policy, guardrails, budgets and
the tracer. A subagent cannot pause for approval, so "ask" decisions become
"deny" inside it (unless the parent runs with `auto_approve`).

## Hooks

```rust
#[async_trait]
impl Hook for AuditLog {
    async fn before_tool(&self, ctx: &HookContext, call: &mut ToolCall) -> ToolDecision {
        if call.name == "shell" && call.arguments["command"].as_str().unwrap_or("").contains("prod") {
            return ToolDecision::Ask("touches production".into());
        }
        ToolDecision::Allow
    }
}
```

`before_model(&mut ChatRequest)` and `after_model(&mut ChatResponse)` can
rewrite the request and reply (an error aborts the run). `before_tool` can
rewrite the arguments (the assistant message is updated to match) and
returns `Allow`, `Ask(reason)` or `Deny(reason)`. `after_tool` can rewrite
the output. Hooks run in registration order; the most restrictive verdict
of all hooks and the policy wins.

`sinew_memory::MemoryHook` (crate `sinew-memory`, feature `harness`) is a
ready-made hook for long-term memory: it recalls a user's memories into the
system message before a turn and extracts new ones when the turn ends. See
[long-term memory](memory.md#harness-hook).

## Permissions

```rust
PermissionPolicy::ask_all()
    .allow("read_file").allow("grep").allow("glob").allow("list_dir")
    .allow("write_file(docs/**)")
    .allow("shell(cargo test*)")
    .deny("shell(git push*)")
    .deny("write_file(**/.env)")
```

A rule is `tool` or `tool(subject)`. Tool names use wildcards (`mcp_*`).
The subject is the call's `path`/`file_path` (path glob), `command` or `url`
(wildcard). Evaluation: any matching deny rule → deny; else any ask → ask;
else any allow → allow; else the default. `PermissionPolicy::interactive()`
allows everything but asks for `write_file`, `edit_file`, `shell` and
`http_fetch`. `auto_approve(true)` turns every ask into allow. Approving with
`ApprovalDecision::approve_always` allows that tool for the rest of the
thread.

## Context management

`ContextConfig { max_context_tokens, keep_recent, offload_threshold,
offload_preview, offload_dir }`.

- Token counts are estimated at about four characters per token (plus a
  small per-message overhead), for the system prompt, tool schemas and
  history. Provider-reported usage is used for budgets when available.
- **Compaction**: when the estimate exceeds `max_context_tokens`, the
  messages before the last `keep_recent` are summarized by the model and
  replaced with one system message starting with `SUMMARY_HEADER`. The split
  moves back so a tool call and its results are never separated. A
  `Compacted` event reports the before/after estimates.
- **Offloading**: a tool output longer than `offload_threshold` characters
  is saved to `<workspace>/.sinew/tool-outputs/<call_id>.txt` and replaced
  with its first `offload_preview` characters and the path, so the model can
  page through it with `read_file` or search it with `grep`.

For other trimming, tool-result clearing, running summaries and semantic selection outside the harness
(for example in a `ReactAgent`), see [Context engineering](context.md). Plugging a `ContextPolicy` into
the harness is tracked in [#160](https://github.com/roanbrasil/sinew/issues/160).

## Budgets

`Budget { max_steps, max_total_tokens, max_cost_usd, timeout }` applies per
turn (one user message, including resumes). Before each model call the
agent node checks them; when one is exhausted the turn ends with
`RunResult::stop = Some(StopReason::...)` instead of an error. Cost needs
`Pricing { input_per_mtok, output_per_mtok }`. The timeout is also enforced
around the whole run, so a hung model call is cancelled. When a provider
reports zero usage, the harness falls back to its estimates.

## Guardrails

```rust
pub trait Guardrail: Send + Sync {
    fn name(&self) -> &str;
    fn check(&self, stage: GuardrailStage, text: &str) -> GuardrailVerdict; // Pass | Modify | Block
}
```

Stages: `Input` (the user message: a block ends the turn without calling the
model), `Output` (the final answer: a block withholds it) and `ToolOutput`
(a block replaces the output with an error). `PiiRedactor` replaces e-mails,
phone numbers (10–15 digits with separators) and Luhn-valid card numbers;
`InjectionDetector` prefixes tool outputs that contain instruction-override
phrases or chat-template tokens with a warning (or blocks them with
`InjectionDetector::blocking()`). Both are heuristics.

The trait and the built-in guardrails are defined in `sinew_core::guardrails` and re-exported here, so
`SecretRedactor`, `DenyList`, `JsonSchemaGuard` and `MaxLength` can be registered too, and the same guardrails
work on any model, chain or `ReactAgent`. The harness applies them with the `Redact` action. For per-rule
actions (warn, refuse, retry with feedback) or an LLM judge, wrap the model with `with_guardrails`; see
[Guardrails](guardrails.md).

## Project instructions and skills

`project_instructions()` loads `AGENTS.md`, `CLAUDE.md`, `.claude/CLAUDE.md`
and `.sinew/AGENTS.md` from the workspace into a "Project instructions"
section. `skills_dir(dir)` scans `dir/*/SKILL.md`:

```markdown
---
name: release-notes
description: Write release notes from the git log
---
1. Run `git log --oneline <last-tag>..HEAD` ...
```

Names and descriptions are listed in the system prompt; `load_skill(name)`
returns the body and lists the other files in the skill's folder.

## MCP

With the `mcp` feature:

```rust
let client = sinew_harness::mcp::connect_command_line("my-mcp-server --flag").await?;
let agent = Agent::builder().model_spec(spec).mcp_server(&client).await?.build()?;
```

MCP tools go through the same policy, hooks, guardrails, offloading and
tracing as built-in tools. The CLI accepts `--mcp "<command line>"`.

## Tracing

`.tracer(tracer)` passes the tracer to the graph: each turn is a `graph`
run with `node` children, every model call is an `llm` run with token usage
(streamed calls record the chunk count), every tool call a `tool` run, and
compaction summaries and structured-output calls are recorded too. View
with `sinew-scope`.

## Testing

Everything above is covered offline with `ScriptedModel` and temporary
directories (`cargo test -p sinew-harness`). Live tests against Ollama are
ignored by default:

```bash
cargo test -p sinew-harness --test live_ollama -- --ignored --test-threads=1
```

## Loop detection

On by default: a tool call repeated with the same arguments is denied with a nudge, and a turn whose tool calls keep failing stops with `StopReason::Guardrail`. Configure it with `.loop_guard(LoopGuard::default().max_repeats(3))` or turn it off with `.loop_guard(None)`. See [Loop detection](loop-detection.md).
