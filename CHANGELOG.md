# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `#[sinew::tool]` attribute (new crate `sinew-macros`, feature `macros`, on by default in `sinew`): a plain sync or async function becomes a `Tool`; name from the function, description from the doc comment, JSON Schema from the argument types (argument doc comments, `Option`, enums, `#[serde]`/`#[schemars]` attributes), `#[state]` for injected application state, helpful compile errors. `tools![...]` builds a `ToolBox` (#4).
- Guide: [docs/tools.md](docs/tools.md).
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

### Changed

- Releases are automated with release-plz and crates.io trusted publishing; see [RELEASING.md](RELEASING.md) (#101).
- Tool parameter schemas from `TypedTool` and `#[tool]` inline nested types instead of using `$ref`/`$defs`, which several providers reject.
- Samples 03 and 22 use `#[sinew::tool]`.
- `sinew-harness`: loop detection is on by default (see above).
- `ReactAgent`: the `tools` node now routes with `Command::goto("agent")` instead of a static edge.

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
