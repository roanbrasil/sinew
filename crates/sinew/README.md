<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew

[![crates.io](https://img.shields.io/crates/v/sinew.svg)](https://crates.io/crates/sinew)
[![docs.rs](https://docs.rs/sinew/badge.svg)](https://docs.rs/sinew)
[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

The umbrella crate: one dependency for the whole Sinew stack, with feature flags for the optional parts.

Part of [Sinew](https://github.com/roanbrasil/sinew): LLM chains, agent graphs, tracing and evaluation in Rust.

```toml
[dependencies]
sinew = { version = "0.1", features = ["full"] }
```

```rust
use sinew::prelude::*;
```

## Cargo features

| Feature | Adds |
|---|---|
| (default) | `sinew::core`, `sinew::graph`, `sinew::trace` |
| `sqlite` / `postgres` | `sinew::checkpoint`: durable checkpointers |
| `mcp` | `sinew::mcp`: Model Context Protocol client and server |
| `eval` | `sinew::eval`: datasets, evaluators, experiments |
| `http-trace` | `trace::HttpSink` for `sinew-scope serve` |
| `govern` | `sinew::govern`: agent control plane: policies before every tool and model call, registry, kill switch |
| `full` | all of the above except Postgres and `govern-cedar` |

See the [main README](https://github.com/roanbrasil/sinew#readme) and the [samples](https://github.com/roanbrasil/sinew/tree/main/samples).

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/roanbrasil/sinew/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/roanbrasil/sinew/blob/main/LICENSE-MIT) at your option.
