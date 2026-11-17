# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Secret and PII redaction in traces: `sinew_trace::Redactor`, set with `Tracer::with_redactor`, rewrites every run's `inputs`, `outputs`, `error` and `metadata` before any sink (JSONL, memory, HTTP, OTLP) sees it. Built-in rules for API keys and tokens (`sk-`, `sk-ant-`, `xai-`, `gsk_`, `AIza`, `ghp_`/`github_pat_`, AWS `AKIA`, Slack, `Bearer`, JWTs, PEM private keys), e-mail addresses, Luhn-checked card numbers and (opt-in) phone numbers, each toggleable; custom regex rules; field rules by key name (`password`, `api_key`, `authorization`, `cookie`, ...); an allow-list of paths. The harness `PiiRedactor` now shares its patterns and validators from `sinew_trace::redact` (#45). Guide: [docs/redaction.md](docs/redaction.md).
- `prompt!` and `chat_prompt!` (feature `macros` on `sinew-core`, on by default in `sinew` and in its prelude): prompt templates checked at compile time. The macro parses the template with `PromptTemplate`'s rules and returns a builder with one setter per variable (any `Display` value); `format()` compiles only when every variable is set, so a missing or misspelled variable is a compile error instead of `Error::MissingVariable`. `prompt!(file = "...")` reads the template at compile time relative to `CARGO_MANIFEST_DIR` and rebuilds when the file changes. Converts into `PromptTemplate` / `ChatPromptTemplate` (`to_template()`, `From`) and `Vars` (`to_vars()`) for runnables and the prompt hub. `PromptTemplate::as_str()` (#63). Guide: [docs/prompts.md](docs/prompts.md).
- MCP resources and prompts in `sinew-mcp` (#10). Server: `McpServer::with_resource`, `with_text_resource`, `with_resource_template` (RFC 6570 `UriTemplate`), `with_prompt`, `with_prompt_template` (from a `PromptTemplate`), runtime `add_*`/`remove_*`, `with_page_size`; handles `resources/list`, `resources/templates/list`, `resources/read`, `resources/subscribe`/`unsubscribe`, `prompts/list` and `prompts/get` (required-argument checks, `-32002` for unknown resources) and sends `notifications/resources/updated` and the `list_changed` notifications (`notify_resource_updated`, `notify_resource_list_changed`, `notify_prompt_list_changed`). The `resources` and `prompts` capabilities are advertised only when registered. Client: `list_resources`, `list_resource_templates`, `read_resource`, `subscribe_resource`/`unsubscribe_resource`, `notifications()` (`McpNotification`), `list_prompts`, `get_prompt`, `get_prompt_messages` (Sinew `Message`s), `prompt_template` and `chat_prompt_template`; feature `rag` adds `read_resource_documents` and `ResourceContents::to_document` (`sinew_rag::Document`). Streamable HTTP: the server answers `GET` with the session's SSE notification stream (previously `405`; stateless mode still answers `405`), and the HTTP client opens it on demand. Sample 11 serves and reads a resource template and a prompt.
- `#[sinew::tool]` attribute (new crate `sinew-macros`, feature `macros`, on by default in `sinew`): a plain sync or async function becomes a `Tool`; name from the function, description from the doc comment, JSON Schema from the argument types (argument doc comments, `Option`, enums, `#[serde]`/`#[schemars]` attributes), `#[state]` for injected application state, helpful compile errors. `tools![...]` builds a `ToolBox` (#4).
- Guide: [docs/tools.md](docs/tools.md).
- Anthropic prompt caching: `Anthropic::with_prompt_caching(PromptCaching::agent() | system_only())` places a breakpoint on the system prompt (or last tool) and enables automatic conversation caching, 5-minute or 1-hour TTL (`CacheTtl`); `SINEW_ANTHROPIC_CACHE` for `from_spec`. Batch APIs: `batch::{BatchModel, BatchStatus, run_batch}`, implemented for Anthropic Message Batches (#36).
- Durable tool calls: a tool ledger in the checkpointer (`ToolEntry`, `Checkpointer::{supports_tool_ledger, get_tool_entry, put_tool_entry}`, implemented by the memory, SQLite and Postgres checkpointers) so calls that finished before a crash are replayed instead of re-run; `Tool::side_effects` and `#[tool(side_effects)]`; `sinew_core::tool::idempotency_key()`; `UnknownOutcome::{Fail, Retry, Interrupt}` on `ReactAgent` and the harness; `sinew_graph::durable::{durable_tool_call, durable_run}`; `harness::Agent::recover` (#25). Guide: [docs/durable-execution.md](docs/durable-execution.md).
- `#[derive(State)]` (feature `macros` on `sinew-graph`, on by default in `sinew`): field-wise reducers for parallel branches with `#[state(replace | overwrite | keep | append | add | merge | reducer = "path")]`, relative to the pre-superstep state so clones and minimal `SendTo` inputs both merge correctly; `Reduce` trait and `StateGraph::with_derived_reducer()`. Sample 09 uses it (#5).
- `LoopGuard` for tool-calling agents: catches repeated identical tool calls and turns whose tools keep failing, then nudges the model, stops the run or asks a human (`LoopAction`). Opt-in on `ReactAgent::loop_guard`, on by default in `sinew-harness` (`AgentBuilder::loop_guard`, `None` to disable). Detections are emitted as `loop_detected` custom events (#26). Guide: [docs/loop-detection.md](docs/loop-detection.md).
- Record/replay cassettes (`sinew_core::vcr`): `Vcr` wraps models and tools, records requests, responses, stream events and tool results to JSON, and replays them offline with request matching (`MatchRequests::{Exact, Messages, Sequential}`), `assert_all_played` and `SINEW_VCR=record|replay|auto`. `sinew_trace::cassette_from_runs` and `GET /api/traces/{id}/cassette` in sinew-scope turn any trace into a cassette. `ToolBox::tools` and `ToolBox::register_arc` (#27). Guide: [docs/testing.md](docs/testing.md).
- Graph diagrams: `CompiledGraph::view()` (serializable `GraphView`), `to_mermaid()` and `to_dot()`, with command nodes, `Send` edges, subgraphs drawn nested, interrupts and policies marked. The root `graph` trace run carries the view in `metadata.graph` (#65). Guide: [docs/visualization.md](docs/visualization.md).
- Guide: [docs/migrating-from-langchain.md](docs/migrating-from-langchain.md), a concept map from LangChain / LangGraph / LangSmith to Sinew, ten side-by-side Python/Rust patterns, and what is intentionally different (#52).
- `Capabilities` and `ChatModel::capabilities()` (tools, streaming, images, PDF, audio, JSON schema, cached tokens, sampling, forced tool choice), forwarded by every wrapper; per-preset values, `OpenAiCompat::with_capabilities`, `providers::capabilities_for`, `providers::catalog()` (`ProviderInfo`). `docs/providers.md` is generated from the catalog by a test (#54).
- Architecture guide: [docs/architecture.md](docs/architecture.md) with the crate layer diagram, request lifecycle, concurrency and error models, cargo feature map and extension points, linked from the README and each crate README (#51).
- Architecture decision records: [docs/adr/](docs/adr/README.md) with an index, a MADR-style template and eight ADRs for decisions already in the code (#59).
- Troubleshooting guide [docs/troubleshooting.md](docs/troubleshooting.md): cause and fix for every variant of `sinew_core::Error`, `sinew_graph::GraphError` and `sinew_harness::HarnessError`, plus Anthropic 400s on sampling and forced tool choice, rate limits, local servers, missing API keys, interrupts without a checkpointer, loop guard stops and cassette mismatches. Each variant's rustdoc links to its entry (#56).
- Contributor guides in [docs/contributing/](docs/contributing/README.md) for adding a provider, vector store, document loader, tool, and checkpointer or store, plus "New provider" and "New integration" issue templates (#58).

### Fixed

- `OpenAiCompat::ollama` accepts an `OLLAMA_HOST` ending in `/v1` (it produced `/v1/v1`), like `from_spec("ollama:...")`.
- `sinew-scope`: opening a database that needs schema migrations could panic in sqlx (`index out of bounds`) when a pooled connection had cached a statement prepared before the `ALTER TABLE`; migrations now run on a separate connection.

### Changed

- CONTRIBUTING.md documents the protected-branch workflow and commit signing; `.github/CODEOWNERS` added (#104).
- Releases are automated with release-plz and crates.io trusted publishing; see [RELEASING.md](RELEASING.md) (#101).
- Tool parameter schemas from `TypedTool` and `#[tool]` inline nested types instead of using `$ref`/`$defs`, which several providers reject.
- Samples 03 and 22 use `#[sinew::tool]`.
- `sinew-harness`: loop detection is on by default (see above).
- `ReactAgent`: the `tools` node now routes with `Command::goto("agent")` instead of a static edge.
- `sinew-harness`: `PiiRedactor` only masks Luhn-valid card numbers that start with 2–6, so Luhn-valid identifiers and millisecond timestamps starting with 1 are kept (#45).

## [0.1.0] - 2026-11-16

First release on crates.io.

### Added

- Crates `sinew-core`, `sinew-graph`, `sinew-trace`, `sinew-checkpoint`, `sinew-mcp`, `sinew-scope`, `sinew-eval` and the `sinew` umbrella; samples 01–13.
- Native Google Gemini provider (`Gemini`) with tools, streaming, structured output and thought-signature round trips.
- OpenAI-compatible presets: xAI Grok, Groq, Mistral, DeepSeek, OpenRouter, Together, Fireworks, Perplexity, Cerebras, NVIDIA NIM, Hugging Face; local LM Studio, llama.cpp, vLLM, SGLang, LocalAI, Jan, llamafile, TGI.
- `providers::from_spec("provider:model")`; samples accept `--model provider:model`.
- `sinew-core`: multimodal messages (images, PDF, audio), LCEL-style `Runnable` pipelines and output parsers, retry / fallback / rate-limit / timeout / response-cache wrappers, token cost table, Azure OpenAI, shared tuned HTTP client.
- `sinew-graph`: `Command`, dynamic `interrupt()` and resume, `interrupt_after`, subgraphs, time travel (history, replay, fork), node retry / cache / timeout, long-term `Store`, stream modes, supervisor and swarm multi-agent prebuilts.
- `sinew-checkpoint`: checkpoint ids and namespaces (with in-place migrations), `SqliteStore`.
- New crates: `sinew-rag` (retrieval), `sinew-harness` (agent harness and `sinew-agent` CLI), `sinew-serve` (deployment server, OpenAI-compatible endpoint), `sinew-a2a` (Agent2Agent 0.3).
- `sinew-scope`/`sinew-trace`/`sinew-eval`: prompt hub, feedback and annotation queues, cost and TTFT tracking, monitoring dashboards, online evaluator rules and alerts, OTLP export (JSON and protobuf), full-text run search, retention, pairwise experiments.
- `sinew-mcp`: Streamable HTTP server transport (`McpServer::http_router`, `serve_http`, `McpHttpConfig`: sessions or stateless, bearer auth, origin allow-list, health route); HTTP client with token or custom headers.
- `sinew-core`: `AsyncFnTool` (async closures) and `TypedTool` (schema derived from the argument type, feature `schemars`).
- Samples 14–22; umbrella features `rag`, `harness`, `serve`, `a2a`, `otlp`.
- Per-crate READMEs, logo, banner; workspace lints (`unsafe_code = "forbid"`), release profile with thin LTO; CI on Linux and macOS, docs, MSRV and cargo-deny.

### Changed

- Breaking: `Message` and `Usage` have new fields, `Error` gains `RateLimited`, `Timeout` and `Other`, `GraphEvent`/`RunOutcome::Interrupted` carry `interrupts`, and new `GraphEvent` variants exist; use `..` / `..Default::default()` in patterns and literals.
- Anthropic and Gemini `input_tokens` now include cached and tool-use prompt tokens, matching OpenAI.

### Security

- `pdf-extract` 0.12 (lopdf 0.42) fixes RUSTSEC-2026-0187. RUSTSEC-2026-0459 (bundled SQLite) is acknowledged until the `sqlx` 0.9 upgrade.

[Unreleased]: https://github.com/roanbrasil/sinew/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/roanbrasil/sinew/releases/tag/v0.1.0
