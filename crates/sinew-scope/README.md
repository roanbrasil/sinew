<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew-scope

[![crates.io](https://img.shields.io/crates/v/sinew-scope.svg)](https://crates.io/crates/sinew-scope)
[![docs.rs](https://docs.rs/sinew-scope/badge.svg)](https://docs.rs/sinew-scope)
[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

A local, self-hosted trace viewer, dataset store and experiment dashboard: the LangSmith app side of Sinew.

Part of [Sinew](https://github.com/roanbrasil/sinew): LLM chains, agent graphs, tracing and evaluation in Rust.

## Run it

```sh
cargo install sinew-scope
sinew-scope serve --db scope.db --addr 127.0.0.1:7878   # web UI at http://127.0.0.1:7878/
sinew-scope import traces.jsonl               # load a JSONL trace file
sinew-scope view traces.jsonl                 # tree view in the terminal
```

Send runs from your app with `sinew_trace::HttpSink`, or point any OpenTelemetry exporter (OTLP/HTTP JSON, GenAI semantic conventions) at `/v1/traces`.

## Embed it

```rust
let store = sinew_scope::Store::memory().await.unwrap();
let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
let addr = listener.local_addr()?;
tokio::spawn(sinew_scope::server::serve_with_shutdown(
    listener,
    sinew_scope::router(store, None),
    std::future::pending(),
));
println!("UI at http://{addr}/");
```

See the [scope guide](../../docs/scope.md).

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/roanbrasil/sinew/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/roanbrasil/sinew/blob/main/LICENSE-MIT) at your option.
