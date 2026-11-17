# Troubleshooting

One entry per error variant of `sinew_core::Error`, `sinew_graph::GraphError` and `sinew_harness::HarnessError`,
followed by situations that do not map to a single variant (provider quirks, local servers, agents that stop).
Each heading is the error's `Display` text, so you can paste the start of an error message into your browser's
search. Placeholders such as `{provider}` stand for the values in your message.

Useful helpers on `sinew_core::Error` while debugging:

- `err.status()`: the HTTP status for `Provider`, `RateLimited` and `Http` errors.
- `err.retry_after()`: the parsed `Retry-After` delay of a `RateLimited` error.
- `err.is_retryable()`: whether the default `RetryPolicy` retries it (429, 408, 409, 5xx, timeouts, connection
  errors, mid-stream errors).

Contents:

- [Model and tool errors (`sinew_core::Error`)](#model-and-tool-errors-sinew_coreerror)
- [Graph errors (`sinew_graph::GraphError`)](#graph-errors-sinew_graphgrapherror)
- [Harness errors (`sinew_harness::HarnessError`)](#harness-errors-sinew_harnessharnesserror)
- [Common situations](#common-situations)

## Model and tool errors (`sinew_core::Error`)

<a id="error-http"></a>
### `http error: {0}` (`Error::Http`)

**Meaning.** The HTTP request did not produce a response Sinew could read: DNS failure, refused connection, TLS
error, connect timeout (10 s on the shared client), or the body could not be read.

**Common causes.**

- The server is not running or listens on another address (most often a local server; see
  [Ollama or another local server is not reachable](#local-server-not-reachable)).
- A proxy or firewall blocks the host.
- A custom endpoint URL (`http://host:port/v1|model`) without the `/v1` suffix the server expects, or with a typo.

**How to fix.** Run `curl` against the same base URL to confirm it answers. Connection and timeout errors are
retryable, so `with_retry(RetryPolicy::default())` covers short outages; a persistent failure is a configuration
problem.

<a id="error-provider"></a>
### `{provider} returned HTTP {status}: {body}` (`Error::Provider`)

**Meaning.** The provider answered with a non-success status other than 429 and without a `Retry-After` header.
`{provider}` is `anthropic`, `gemini` or `openai-compat` (every OpenAI-compatible preset, Azure and Ollama
included). `{body}` is the provider's own error text.

**Common causes and fixes.**

| Status | Usual cause | Fix |
|---|---|---|
| 400 | A parameter the model does not accept: sampling or forced tool choice on newer Claude models, a JSON schema the provider rejects, images sent to a text-only model | Read `{body}`. Check `model.capabilities()` before sending images, PDF, audio or a forced tool choice. See [Anthropic returns 400](#anthropic-400-sampling-tool-choice). |
| 401 / 403 | Wrong, expired or missing API key; key without access to the model | Check the environment variable from the [provider table](providers.md). Local servers that need a key read `LOCAL_API_KEY`. |
| 404 | Unknown model id, wrong base URL, model not pulled (Ollama: `model "x" not found, try pulling it first`), Azure deployment name typo | Check the model id against the provider's list; `ollama pull <model>` for Ollama. |
| 408, 409, 5xx | Transient provider failure | Retried by `RetryPolicy`; add `with_fallbacks` to switch provider after the retries. |
| 413 | Prompt or attachment too large | Trim the context (`sinew-harness` compacts automatically with `max_context_tokens`), or send files by URL / Files API id. |

Only 408, 409 and 5xx are retried by default. Other 4xx responses are not, because retrying the same request
gives the same answer.

<a id="error-rate-limited"></a>
### `{provider} returned HTTP {status} (rate limited): {body}` (`Error::RateLimited`)

**Meaning.** The provider returned 429, or another error status (typically 503) with a `Retry-After` header.
`retry_after` holds the parsed delay when the header was present (seconds or an HTTP date).

**Common causes.** Requests or tokens per minute above your quota; many agents sharing one key; bursts at
start-up.

**How to fix.** See [HTTP 429 and rate limits](#http-429-and-rate-limits).

<a id="error-timeout"></a>
### `timed out after {0:?}` (`Error::Timeout`)

**Meaning.** A `with_timeout(d)` wrapper elapsed before `chat` finished, before a stream opened, or (with
`with_stream_idle_timeout(d)`) while waiting for the next stream event.

**Common causes.** Long generations (reasoning models, large `max_tokens`), a local model loading into memory on
its first request, a timeout set lower than the provider's typical latency.

**How to fix.** Raise the timeout or lower `max_tokens`. Warm up local models once before timing them. Timeouts
are retryable, so put `with_timeout` inside `with_retry` (each attempt gets its own deadline), as in
[Resilience](resilience.md#wrappers).

<a id="error-invalid-response"></a>
### `invalid provider response: {0}` (`Error::InvalidResponse`)

**Meaning.** The provider returned success, but the body is missing something Sinew needs. Messages include
`missing content` (Anthropic), `missing choices[0].message` and `tool call without name` (OpenAI-compatible),
`no candidates` and `prompt blocked: {reason}` (Gemini).

**Common causes.**

- Gemini blocked the prompt with its safety filters (`prompt blocked`), or returned no candidate.
- An "OpenAI-compatible" server that does not follow the Chat Completions response format, or a proxy that
  rewrites the body.
- The base URL points at a different API (for example the Ollama native API instead of `/v1`).

**How to fix.** For `prompt blocked`, change the prompt or the Gemini safety settings. For the others, call the
endpoint with `curl` and compare the body with the OpenAI Chat Completions format; if a server deviates, open an
issue with the recorded body (see [Adding a provider](contributing/provider.md)).

<a id="error-json"></a>
### `json error: {0}` (`Error::Json`)

**Meaning.** `serde_json` failed: a response body that is not valid JSON, tool arguments or results that do not
deserialize, or a cassette file that does not parse.

**Common causes.** An HTML error page from a proxy or gateway returned with status 200; a tool whose argument type
does not match what the model sent; a hand-edited cassette.

**How to fix.** Log the raw body. For tools, make argument types accept what models send (`Option<T>` for optional
fields, `#[serde(default)]`). Re-record broken cassettes with `SINEW_VCR=record`.

<a id="error-tool-not-found"></a>
### `` tool `{0}` is not registered `` (`Error::ToolNotFound`)

**Meaning.** `ToolBox::call` got a tool call whose name is not in the toolbox.

**Common causes.** The model invented a tool name or misspelled one; the tool list sent to the model differs from
the toolbox that runs the calls; a tool was renamed with `#[tool(name = "...")]`.

**How to fix.** Inside agents, `ToolBox::run` turns this error into an `error: ...` tool message, and the model
usually corrects itself on the next turn. If it does not, make sure the same `ToolBox` provides both the specs
(`ToolBox::specs`) and the execution, and give tools distinct, descriptive names.

<a id="error-tool"></a>
### `` tool `{name}` failed: {message} `` (`Error::Tool`)

**Meaning.** A tool returned an error. For `#[sinew::tool]` functions, the `Err` value's `Display` text is the
message; built-in harness tools use it for invalid arguments (for example `` `path` (string) is required ``).

**Common causes.** Invalid arguments from the model, a failing backend, or (in replay) a tool call that is not in
the cassette (see [Cassette mismatch](#cassette-mismatch)).

**How to fix.** Inside agents the model sees `error: tool ... failed: ...` and can retry. Write error messages the
model can act on ("`id` must look like A-1001"). Repeated failures in one turn trigger the
[loop guard](#loop-guard-stops).

<a id="error-missing-variable"></a>
### `` prompt template is missing variable `{0}` `` (`Error::MissingVariable`)

**Meaning.** A `PromptTemplate` was formatted without a value for one of its `{variable}` placeholders.

**How to fix.** Pass every variable the template uses. To put a literal brace in a template, double it: `{{` and
`}}`.

<a id="error-template"></a>
### `malformed prompt template: {0}` (`Error::Template`)

**Meaning.** `PromptTemplate::new` could not parse the template: `` unclosed `{` at byte {i} ``,
`` unmatched `}` at byte {i} `` or `` invalid variable name `{name}` ``.

**Common causes.** JSON examples or code inside the template text, whose braces are read as placeholders.

**How to fix.** Escape literal braces as `{{` and `}}`, or pass the JSON example as a variable.

<a id="error-config"></a>
### `missing configuration: {0}` (`Error::Config`)

**Meaning.** Something needed to build a client or open a file is missing or malformed. Typical messages:

- `ANTHROPIC_API_KEY not set` (or any other key variable): see [Missing API key](#missing-api-key).
- `` expected `provider:model`, got `...` ``, `` no model in `...` ``, `` expected `url|model`, got `...` ``: a
  malformed spec passed to `providers::from_spec`.
- `` unknown provider `x`; known: ... ``: the provider name is not in `providers::provider_names()` or its
  aliases.
- `cannot read cassette ...`, `cassette ... has version N, expected M`, `` SINEW_VCR must be record, replay or
  auto, not `...` ``: record/replay setup (see [Cassette mismatch](#cassette-mismatch)).

**How to fix.** Follow the message. The full list of providers, aliases and variables is in
[docs/providers.md](providers.md).

<a id="error-structured"></a>
### `structured output failed: {message}` (`Error::Structured`)

**Meaning.** `StructuredOutput::generate` (or a JSON output parser) could not get output that parses and validates
against the schema after all retries. `raw` holds the model's last output.

**Common causes.** Small models that do not follow schemas; a schema with constructs the model or provider handles
poorly (deep nesting, `oneOf`); `StructuredMode::Tool` on a model without tool calling.

**How to fix.** Inspect `raw`. Raise `with_retries(n)`; each retry sends the validation error back to the model.
Switch `with_mode(StructuredMode::Json)` for models without tools, or `StructuredMode::Tool` for models that are
better at tool calls than at JSON text. Simplify the schema.

<a id="error-stream"></a>
### `stream error: {0}` (`Error::Stream`)

**Meaning.** The provider sent an error event in the middle of a stream (OpenAI-compatible and Gemini decoders),
for example an `overloaded_error`.

**How to fix.** Stream errors are retryable, but `RetryPolicy` retries a stream only when it fails before the first
event; after output has started the error is passed through, because the caller has already seen partial output.
Restart the request in your code if partial output is acceptable to discard.

<a id="error-script-exhausted"></a>
### `scripted model has no responses left` (`Error::ScriptExhausted`)

**Meaning.** A `ScriptedModel` (offline test model) was called more times than it has scripted responses.

**Common causes.** The code under test makes an extra model call: a retry, a structured-output repair, an agent
loop that did not stop where the test expected.

**How to fix.** Add the missing responses, or inspect `ScriptedModel::requests()` to see which call was
unexpected.

<a id="error-guardrail"></a>
### `blocked by guardrail` (`Error::Guardrail`)

**Meaning.** A guardrail rule blocked a model input or output: the message names the guardrail, the stage
(`input`, `tool_output`, `output`) and the guardrail's reason. With a `Retry` rule, the reason ends with
`(still failing after N retries)` when the model kept producing answers that failed the check.

**Common causes.** A `Block` rule, a guardrail that returns `Block` under the default `Redact` action
(`MaxLength`, `DenyList`, `JsonSchemaGuard`, `LlmJudge`), or an `InjectionDetector::blocking()` match. In
`StreamGuardMode::Final`, any rewrite or refusal after the text was streamed also ends the stream with this error.

**How to fix.** If the block is expected, catch the error and answer the user, or use `.refuse("...")` to return
a fixed message instead. If it is a false positive, narrow the rule (`.at(...)`, fewer keywords) or switch it to
`.warn()` while you tune it. For schema failures, give the model more retries (`.retry(n)`) or a clearer prompt.
See [Guardrails](guardrails.md).

<a id="error-other"></a>
### `{0}` (`Error::Other`)

**Meaning.** A free-form error raised by a runnable, user code, or the record/replay layer. Messages that start
with `cassette ...` come from `sinew_core::vcr`; see [Cassette mismatch](#cassette-mismatch).

**How to fix.** Search for the message text in your code first; it was not produced by a provider.

## Graph errors (`sinew_graph::GraphError`)

<a id="graph-error-build"></a>
### `invalid graph:` (`GraphError::Build`)

**Meaning.** `StateGraph::compile()` found one or more problems. Every problem is listed, one per line, sorted.

| Message | Fix |
|---|---|
| `no entry point set` | Call `set_entry_point`, `set_conditional_entry_point` or `add_edge(START, ...)`. |
| `` entry point `x` is not a node `` | Add the node, or fix the name. `END` cannot be the entry point. |
| `` node `x` added twice `` | Node names are unique; rename one. |
| `` node `x` has more than one outgoing edge definition `` | One `add_edge` / `add_conditional_edges` / `add_send_edges` per source node; list several targets in one call instead. |
| `` edge from unknown node `x` `` / `` edge `a` -> `b` points to an unknown node `` | Fix the names or add the node. |
| `` edge from `x` lists no targets `` | Give the edge at least one target. |
| `` node `x` has no outgoing edge (add one to END to finish) `` | Add `add_edge("x", END)` or route it somewhere. Command nodes (`add_command_node`) are exempt. |
| `` node `x` lists unknown destination `y` `` | Fix `set_node_destinations` for the command node. |
| `` node `x` is unreachable from `entry` `` | Connect it, or remove it. |
| `` node `x` fans out but no reducer is set (call `with_reducer`) `` / `the entry point fans out but no reducer is set` | See [Fan-out without a reducer](#graph-error-no-reducer). |
| `` retry policy / cache policy / timeout set for unknown node `x` `` | Fix the node name passed to `set_node_retry`, `set_node_cache` or `set_node_timeout`. |

<a id="graph-error-node"></a>
### `` node `{node}` failed: {source} `` (`GraphError::Node`)

**Meaning.** A node returned an error, panicked, or exceeded its timeout. `source` is the node's error. During a
fan-out the other branches of the superstep are cancelled and no checkpoint is written for it.

**Common causes.** A model or tool error inside the node (for example `Error::Provider`; see the entries above),
a node timeout set with `set_node_timeout`, a bug in the node.

**How to fix.** Read `source` and look up its own entry. Transient failures can be retried per node with
`set_node_retry(name, RetryPolicy)`. With a checkpointer, fix the cause and resume the thread with
`run(thread_id, None)`: the run continues from the last completed superstep.

<a id="graph-error-recursion-limit"></a>
### `recursion limit of {0} supersteps reached before the graph hit END` (`GraphError::RecursionLimit`)

**Meaning.** The thread ran the maximum number of supersteps (default 25) without reaching `END`. A superstep that
runs ten parallel branches counts once.

**Common causes.** A router that never returns `END`; an agent loop where the model keeps calling tools (see
[Loop guard stops](#loop-guard-stops)); a legitimately long workflow.

**How to fix.** If the loop is intended, raise the limit for the graph with `with_recursion_limit(n)` or for one
run with `RunConfig::new(thread).with_recursion_limit(n)` and `run_with`. Otherwise, check the routing condition
that should lead to `END`, and for tool-calling agents enable `LoopGuard`.

<a id="graph-error-unknown-route"></a>
### `` router of `{from}` returned unknown target `{to}` `` (`GraphError::UnknownRoute`)

**Meaning.** At run time, a conditional edge, a `SendTo` router or a `Command::goto` from node `{from}` named a
node that is not among the declared targets (or does not exist).

**How to fix.** Add `{to}` to the `targets` list passed to `add_conditional_edges` / `add_send_edges`, or to the
destinations of the command node (`set_node_destinations`). Route to `END` with the `END` constant, not a string
like `"end"`.

<a id="graph-error-interrupted"></a>
### `` graph was interrupted at `{0}`; use `run` with a checkpointer to resume `` (`GraphError::Interrupted`)

**Meaning.** The graph paused (`interrupt_before`, `interrupt_after` or a dynamic `interrupt()` in a node) but
there is nowhere to keep the paused state: `invoke` was used, or no checkpointer is configured.

**How to fix.** See [Interrupts without a checkpointer](#interrupts-without-a-checkpointer).

<a id="graph-error-no-checkpoint"></a>
### `` no checkpoint for thread `{0}` `` (`GraphError::NoCheckpoint`)

**Meaning.** A resume (`run(thread, None)`, `resume`, `update_state` or a `Command` without a state update) was
requested for a thread that has no checkpoint.

**Common causes.** A typo in the thread id; a `MemoryCheckpointer` that was recreated (its data lives only in the
process); a different database file or schema than the one the first run used; `prune`/`delete_thread` removed
the thread.

**How to fix.** Start the thread with `run(thread, Some(state))`. Use the same checkpointer instance (or the same
database URL) for the first run and the resume.

<a id="graph-error-checkpoint-not-found"></a>
### `` no checkpoint `{checkpoint_id}` in thread `{thread_id}` `` (`GraphError::CheckpointNotFound`)

**Meaning.** Time travel (`get_state_at`, `RunConfig::at_checkpoint` for replay or fork) referenced a checkpoint
id that the thread does not have.

**How to fix.** List valid ids with `get_state_history(thread)`. Check that retention (`with_retention`, `prune`)
did not remove it.

<a id="graph-error-no-checkpointer"></a>
### `this operation needs a checkpointer` (`GraphError::NoCheckpointer`)

**Meaning.** Resuming, `update_state`, `get_state*` and time travel read stored checkpoints; the compiled graph has
none.

**How to fix.** `compiled.with_checkpointer(MemoryCheckpointer::new())` for tests, or a durable one from
`sinew-checkpoint` (`SqliteCheckpointer`, `PostgresCheckpointer`); see [Checkpointing](checkpointing.md).

<a id="graph-error-checkpoint"></a>
### `checkpointer error: {0}` (`GraphError::Checkpoint`)

**Meaning.** The checkpointer's `put`, `latest`, `history` or `get` returned an error (a `String`).

**Common causes.** Database unreachable, wrong URL, permissions, a locked SQLite file shared by processes without
WAL, a full disk.

**How to fix.** Read the database error in the message. For several processes or hosts, use
`PostgresCheckpointer`; see [Checkpointing](checkpointing.md#concurrency).

<a id="graph-error-serde"></a>
### `state serialization failed: {0}` (`GraphError::Serde`)

**Meaning.** The state could not be converted to or from JSON: when writing a checkpoint, resuming from one,
deserializing a `SendTo` input, or reading a resume value for `interrupt()`.

**Common causes.** The state type changed (renamed or new required field) and old checkpoints no longer match;
`f64::NAN` or map keys that are not strings; a resume value of the wrong type.

**How to fix.** Add `#[serde(default)]` to new fields so older checkpoints still load, or start a new thread.
Resume `interrupt::<T>()` with a value that deserializes into `T`.

<a id="graph-error-no-reducer"></a>
### `` superstep runs several branches (...) but no reducer is set; call `StateGraph::with_reducer` `` (`GraphError::NoReducer`)

**Meaning.** A superstep ran more than one node, and there is no reducer to merge their output states. `compile()`
rejects this when it can see the fan-out (several static targets, multi-target routers, `add_send_edges`); this
run-time error is the safety net for fan-outs created by `Command::goto` with several nodes.

**How to fix.** Set a reducer that merges branch results into the shared state:

```rust
graph.with_reducer(|mut acc: MyState, branch: MyState| {
    acc.results.extend(branch.results);
    acc
});
```

or `with_reducer_from_base` when the merge needs the state from before the superstep. See
[Fan-out](fan-out.md#writing-reducers).

<a id="graph-error-invalid-command"></a>
### `invalid command: {0}` (`GraphError::InvalidCommand`)

**Meaning.** A `Command` cannot be applied. Messages: `a run input cannot target the parent graph`,
`a node returned a command for the parent graph, but it runs in the root graph`,
`` `as_node` names unknown node `x` ``.

**How to fix.** Use `Command::to_parent` only from nodes that run inside a subgraph. Pass an existing node name to
`update_state_with`'s `as_node`.

<a id="graph-error-parent-command"></a>
### `a subgraph node sent a command to its parent graph` (`GraphError::ParentCommand`)

**Meaning.** Internal signal: a node inside a subgraph returned `Command::to_parent`. The subgraph node handles it,
so you only see it when you run a subgraph's compiled graph directly instead of through `add_subgraph`.

**How to fix.** Add the subgraph to the parent with `add_subgraph` / `add_subgraph_mapped`. See
[Subgraphs](subgraphs.md#routing-the-parent-commandto_parent).

## Harness errors (`sinew_harness::HarnessError`)

<a id="harness-error-model"></a>
### Model errors (`HarnessError::Model`)

**Meaning.** A `sinew_core::Error` from the model, a tool or `from_spec`, shown with the original message
(`#[error(transparent)]`).

**How to fix.** Look up the message in [Model and tool errors](#model-and-tool-errors-sinew_coreerror). The most
common at start-up is `missing configuration: ..._API_KEY not set` from `model_spec`.

<a id="harness-error-graph"></a>
### Graph errors (`HarnessError::Graph`)

**Meaning.** A `sinew_graph::GraphError` from the harness's agent graph, shown with the original message.

**How to fix.** Look up the message in [Graph errors](#graph-errors-sinew_graphgrapherror). Long runs stop with a
`StopReason` from budgets (`max_steps`, `max_total_tokens`, `max_cost_usd`, `timeout`) rather than with a
recursion-limit error.

<a id="harness-error-io"></a>
### `io error: {0}` (`HarnessError::Io`)

**Meaning.** A file-system operation of the harness failed (reading instructions, skills or files).

**How to fix.** Check that the path exists and the process can read it.

<a id="harness-error-config"></a>
### `configuration error: {0}` (`HarnessError::Config`)

**Meaning.** `AgentBuilder::build` (or an MCP server definition) is inconsistent. Messages:

| Message | Fix |
|---|---|
| `no model configured` | Call `model(...)`, `model_arc(...)` or `model_spec("provider:model")`. |
| `workspace {dir}: {error}` | The workspace directory does not exist or is not accessible. |
| `filesystem tools need a workspace` | Call `workspace(dir)` before `filesystem()` / `filesystem_read_only()`. |
| `project_instructions needs a workspace` | Same, for `project_instructions()`. |
| `skills dir {dir}: {error}` | Fix the path passed to `skills_dir`. |
| ``two tools are named `x` `` | Tool names must be unique across built-in, custom and MCP tools; rename one. |
| `empty MCP command` | The MCP server command line is empty. |

<a id="harness-error-no-pending-approval"></a>
### `` thread `{0}` has no pending approval `` (`HarnessError::NoPendingApproval`)

**Meaning.** `Agent::resume` (or `resume_stream`) was called for a thread that is not waiting for approval.

**Common causes.** The thread already resumed; a typo in the thread id; a new process with a `MemoryCheckpointer`
that lost the paused state.

**How to fix.** Resume only after `run` returned `AgentOutcome::ApprovalRequired`. Use a durable checkpointer
(`AgentBuilder::checkpointer`) when approvals can arrive after a restart.

<a id="harness-error-approval-required"></a>
### `the run needs human approval for: {0}` (`HarnessError::ApprovalRequired`)

**Meaning.** `Agent::ask` ran a tool call that the permission policy, a hook or the loop guard sent to approval.
`ask` is the one-shot API and cannot pause.

**How to fix.** Use `run(thread_id, input)` and handle `AgentOutcome::ApprovalRequired` with `resume`, or allow the
tools up front with `permissions(...)` / `auto_approve(...)`. See [Harness permissions](harness.md#permissions).

<a id="harness-error-hook"></a>
### `hook failed: {0}` (`HarnessError::Hook`)

**Meaning.** A `Hook` returned this error from `before_model` or `after_model`; an error from a hook aborts the run.

**How to fix.** The message is your hook's. To block a single tool call without aborting the run, return
`ToolDecision` from `before_tool` instead.

<a id="harness-error-other"></a>
### `{0}` (`HarnessError::Other`)

**Meaning.** Any other harness failure, for example `mcp: ...` (an MCP server failed to start or answer) or
`corrupt thread state: ...` (a checkpoint that does not deserialize into `HarnessState`, typically written by
another version).

**How to fix.** For MCP, run the server command by hand to see its output. For corrupt state, start a new thread.

## Common situations

<a id="anthropic-400-sampling-tool-choice"></a>
### Anthropic returns 400 for `temperature` or `tool_choice`

**Symptom.** `anthropic returned HTTP 400: ...` mentioning `temperature`, `top_p` or `tool_choice`.

**Cause.** Newer Claude models reject sampling parameters, and some reject forced tool choice (`any` / `tool`).
Sinew handles this per model id prefix in `crates/sinew-core/src/providers/anthropic.rs`:

- no sampling: `claude-fable`, `claude-mythos`, `claude-opus-5`, `claude-opus-4-8`, `claude-opus-4-7`,
  `claude-sonnet-5`, `claude-haiku-5`; Sinew drops `temperature`;
- no forced tool choice: `claude-fable-5-1`, `claude-mythos-5-1`, `claude-opus-5-5`, `claude-sonnet-5-5`; Sinew
  sends `auto` and adds an instruction to the system prompt.

`model.capabilities()` reports both (`sampling`, `forced_tool_choice`).

**How to fix.** If you still get the 400:

- You call the API through another client (an OpenAI-compatible gateway such as OpenRouter) that forwards the
  parameters unchanged: omit `with_temperature` and use `ToolChoice::Auto` for those models.
- The model id is new and not in the lists yet: open an issue or a pull request that adds the prefix (see
  [Adding a provider](contributing/provider.md)).

### HTTP 429 and rate limits

**Symptom.** `Error::RateLimited` (`... returned HTTP 429 (rate limited): ...`).

**How to fix.**

1. Wrap the model in `with_retry(RetryPolicy::default())`. A `RateLimited` error with `Retry-After` waits exactly
   that long (capped at `max_retry_after`, 60 s by default) instead of the exponential backoff.
2. Pace requests below the quota with `with_rate_limit(RateLimiter::per_minute(n).with_tokens_per_minute(t))`.
   Clones of a `RateLimiter` share their buckets: pass clones to every model that uses the same key.
3. Add `with_fallbacks([...])` with `.fallback_if(|e| e.is_retryable())` to move to another provider after the
   retries.
4. Cache repeated requests with `with_cache`.

Order matters: `with_timeout` inside `with_retry`, the limiter outside the retry. See
[Resilience](resilience.md).

<a id="local-server-not-reachable"></a>
### Ollama or another local server is not reachable

**Symptom.** `http error: error sending request for url (http://127.0.0.1:11434/v1/chat/completions)`, or
`openai-compat returned HTTP 404: ... model "x" not found, try pulling it first`.

**Causes and fixes.**

- Ollama is not running: start it (`ollama serve`, or the desktop app) and check `curl $OLLAMA_HOST/api/tags`
  (default `http://127.0.0.1:11434`).
- `OLLAMA_HOST` points elsewhere: Sinew reads it for `ollama:` specs and `OpenAiCompat::ollama`. Use `host:port` or
  `http://host:port`. `from_spec` also accepts a trailing `/v1`; `OpenAiCompat::ollama` appends `/v1` itself, so
  leave it out there. Ollama itself binds to `127.0.0.1` unless started with `OLLAMA_HOST=0.0.0.0`; set that on
  the server when Sinew runs on another machine.
- The model is not pulled: `ollama pull qwen3:4b`. Keep the full tag in the spec: `ollama:qwen3:4b`.
- Other local servers use their own variable and default port (`LMSTUDIO_HOST`, `LLAMACPP_HOST`, `VLLM_HOST`, ...);
  see the table in [docs/providers.md](providers.md).
- The first request to a large model can take longer than a short `with_timeout` while the model loads.

### Missing API key

**Symptom.** `missing configuration: ANTHROPIC_API_KEY not set` (or `GEMINI_API_KEY`, `OPENAI_API_KEY`,
`XAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, ...).

**Cause.** Hosted providers read their key from the environment when built with `from_spec`, `from_env` or a
preset constructor. The variable for each provider is in `providers::catalog()` (`ProviderInfo::env_var`) and in
the second table of [docs/providers.md](providers.md). Gemini also accepts `GOOGLE_API_KEY`; Azure needs
`AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` (and optionally `AZURE_OPENAI_API_VERSION`).

**How to fix.** Export the variable in the environment of the process (a `.env` file is not read by Sinew), or
build the client with the key in code: `Anthropic::new(key, model)`, `OpenAiCompat::new(url, model).with_api_key(key)`.
For tests, use `ScriptedModel` or a cassette instead of a key.

### Interrupts without a checkpointer

**Symptom.** `` graph was interrupted at `node`; use `run` with a checkpointer to resume ``.

**Cause.** A paused run must be stored to be resumed later. `invoke` starts a throw-away thread and has no way to
return a paused state, and `run` without a checkpointer cannot save one.

**How to fix.**

```rust
let graph = builder.compile()?
    .with_checkpointer(MemoryCheckpointer::new())   // or SqliteCheckpointer / PostgresCheckpointer
    .interrupt_before(["review"]);
let out = graph.run("thread-1", Some(state)).await?;
if out.is_interrupted() {
    // inspect out.interrupts(), then:
    let out = graph.resume("thread-1", json!({"approved": true})).await?;   // dynamic interrupt()
    // or graph.run("thread-1", None) for interrupt_before / interrupt_after
}
```

See [Commands and interrupts](commands-and-interrupts.md).

### Loop guard stops

**Symptom.** The agent answers "I stopped because I was repeating the same actions without making progress."
(`LOOP_STOP_REPLY`), the harness returns `StopReason::Guardrail("loop detected: ...")`, a tool result reads
`error: loop detected. You already called ...`, or the model sees `permission denied: loop detected ...`.

**Cause.** `LoopGuard` caught repeated identical tool calls (default: the third identical call in a turn) or a
streak of failing tool results (default: three in a row). It is on by default in `sinew-harness` and opt-in on
`ReactAgent`.

**How to fix.**

- Find out why the model repeats itself: a tool result it cannot use, an error message it cannot act on, a model
  too small for the task. The `loop_detected` custom event in the stream or trace has the details.
- Polling tools legitimately repeat: raise `max_repeats`, put a changing value (an attempt counter) in the
  arguments, or turn the guard off for that agent (`AgentBuilder::loop_guard(None)`).
- Flaky backends: raise `max_consecutive_errors`, or retry inside the tool.

See [Loop detection](loop-detection.md).

### Cassette mismatch

**Symptom.** Errors from `sinew_core::vcr` in replay mode:

| Message | Cause | Fix |
|---|---|---|
| `cassette {path}: model call #N differs from the recording: {diff} (re-record with SINEW_VCR=record)` | The request changed: prompt, tools, temperature, max tokens, tool choice or response format | If the change is intended, re-record. If only tool descriptions or options changed, use `.matching(MatchRequests::Messages)`. Trace cassettes need `Messages`. |
| `` cassette {path}: no recorded model call #N for `model` (re-record ...) `` | The code makes more model calls than were recorded | Re-record, or find the extra call (a retry, a new step). |
| `` tool `x` failed: cassette {path}: no recorded call with arguments {...} `` | A tool was called with arguments that are not in the cassette | Re-record, or check what changed the arguments. |
| `cassette {path}: N of M model calls and K of L tool calls were replayed` | `assert_all_played` found recorded calls that did not happen | The code stopped making a call; re-record if intended. |
| `cassette interaction has no response` | A hand-edited or truncated cassette | Re-record. |
| `cannot read cassette ...` / `cassette ... has version N, expected M` | Missing file in replay mode, or a cassette from another Sinew version | Record it (`SINEW_VCR=record cargo test`), or re-record with the current version. |
| `` SINEW_VCR must be record, replay or auto, not `x` `` | Invalid environment value | Use `record`, `replay` or `auto` (default). |

See [Testing](testing.md#what-replay-checks).
