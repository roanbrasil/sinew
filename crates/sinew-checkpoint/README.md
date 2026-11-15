<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew-checkpoint

[![crates.io](https://img.shields.io/crates/v/sinew-checkpoint.svg)](https://crates.io/crates/sinew-checkpoint)
[![docs.rs](https://docs.rs/sinew-checkpoint/badge.svg)](https://docs.rs/sinew-checkpoint)
[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

Durable checkpointers for `sinew-graph`: agents that survive restarts.

Part of [Sinew](https://github.com/roanbrasil/sinew): LLM chains, agent graphs, tracing and evaluation in Rust.

## Quick example

```rust
use sinew_checkpoint::{CheckpointStore, SqliteCheckpointer};

// Keep at most 50 checkpoints per thread.
let store = SqliteCheckpointer::connect("sqlite://agent.db")
    .await?
    .with_retention(50);
println!("threads: {:?}", store.list_threads().await?);
// graph.with_checkpointer(store) ...
```

## Backends

| Backend | Feature | Use it for |
|---|---|---|
| `SqliteCheckpointer` | `sqlite` (default) | one process, a single file, or in-memory for tests |
| `PostgresCheckpointer` | `postgres` | several processes or hosts sharing state |

Both implement `Checkpointer` and `CheckpointStore` (`list_threads`, `delete_thread`, `prune`, `with_retention`).
See the [checkpointing guide](../../docs/checkpointing.md).

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/roanbrasil/sinew/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/roanbrasil/sinew/blob/main/LICENSE-MIT) at your option.
