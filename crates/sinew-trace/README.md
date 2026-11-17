<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew-trace

[![crates.io](https://img.shields.io/crates/v/sinew-trace.svg)](https://crates.io/crates/sinew-trace)
[![docs.rs](https://docs.rs/sinew-trace/badge.svg)](https://docs.rs/sinew-trace)
[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

Tracing for every graph, node, model call and tool call: the LangSmith SDK side of Sinew.

Part of [Sinew](https://github.com/roanbrasil/sinew): LLM chains, agent graphs, tracing and evaluation in Rust.

## What is inside

- `Run`, `RunKind`, `RunRef`: a tree of runs with timings, inputs, outputs, token usage and errors
- `Tracer` and `RunHandle` to open and close runs
- Sinks: `MemorySink`, `JsonlSink`, and `HttpSink` (feature `http`) which batches runs to a [`sinew-scope`](../sinew-scope) server in the background
- Helpers: `traced_chat`, `traced_chat_stream`, `traced_tool`
- `Redactor` (`Tracer::with_redactor`): masks API keys, tokens, e-mail addresses, card numbers, sensitive fields and custom patterns before any sink sees a run ([guide](https://github.com/roanbrasil/sinew/blob/main/docs/redaction.md))

## Cargo features

| Feature | Adds |
|---|---|
| `http` | `HttpSink`, `HttpSinkConfig` |

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/roanbrasil/sinew/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/roanbrasil/sinew/blob/main/LICENSE-MIT) at your option.
