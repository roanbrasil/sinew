# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Native Google Gemini provider (`Gemini`) with tools, streaming, structured output and thought-signature round trips.
- OpenAI-compatible presets: xAI Grok, Groq, Mistral, DeepSeek, OpenRouter, Together, Fireworks, Perplexity, Cerebras, NVIDIA NIM, Hugging Face; local LM Studio, llama.cpp, vLLM, SGLang, LocalAI, Jan, llamafile, TGI.
- `providers::from_spec("provider:model")`; samples accept `--model provider:model`.
- `sinew-core`: multimodal messages (images, PDF, audio), LCEL-style `Runnable` pipelines and output parsers, retry / fallback / rate-limit / timeout / response-cache wrappers, token cost table, Azure OpenAI, shared tuned HTTP client.
- `sinew-graph`: `Command`, dynamic `interrupt()` and resume, `interrupt_after`, subgraphs, time travel (history, replay, fork), node retry / cache / timeout, long-term `Store`, stream modes, supervisor and swarm multi-agent prebuilts.
- `sinew-checkpoint`: checkpoint ids and namespaces (with in-place migrations), `SqliteStore`.
- New crates: `sinew-rag` (retrieval), `sinew-harness` (agent harness and `sinew-agent` CLI), `sinew-serve` (deployment server, OpenAI-compatible endpoint), `sinew-a2a` (Agent2Agent 0.3).
- `sinew-scope`/`sinew-trace`/`sinew-eval`: prompt hub, feedback and annotation queues, cost and TTFT tracking, monitoring dashboards, online evaluator rules and alerts, OTLP export (JSON and protobuf), full-text run search, retention, pairwise experiments.
- Samples 14–21; umbrella features `rag`, `harness`, `serve`, `a2a`, `otlp`.
- Per-crate READMEs, logo, banner; workspace lints (`unsafe_code = "forbid"`), release profile with thin LTO; CI on Linux and macOS, docs, MSRV and cargo-deny.

### Changed

- Breaking: `Message` and `Usage` have new fields, `Error` gains `RateLimited`, `Timeout` and `Other`, `GraphEvent`/`RunOutcome::Interrupted` carry `interrupts`, and new `GraphEvent` variants exist; use `..` / `..Default::default()` in patterns and literals.
- Anthropic and Gemini `input_tokens` now include cached and tool-use prompt tokens, matching OpenAI.

### Security

- `pdf-extract` 0.12 (lopdf 0.42) fixes RUSTSEC-2026-0187. RUSTSEC-2026-0459 (bundled SQLite) is acknowledged until the `sqlx` 0.9 upgrade.

## [0.1.0]

- Initial crates: `sinew-core`, `sinew-graph`, `sinew-trace`, `sinew-checkpoint`, `sinew-mcp`, `sinew-scope`, `sinew-eval`, `sinew`; samples 01–13.
