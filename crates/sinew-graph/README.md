<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew-graph

[![crates.io](https://img.shields.io/crates/v/sinew-graph.svg)](https://crates.io/crates/sinew-graph)
[![docs.rs](https://docs.rs/sinew-graph/badge.svg)](https://docs.rs/sinew-graph)
[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

A typed, async state graph for agents: the LangGraph side of Sinew.

Part of [Sinew](https://github.com/roanbrasil/sinew): LLM chains, agent graphs, tracing and evaluation in Rust.

## Quick example

```rust
let mut g = StateGraph::<S>::new();
g.add_node("inc", |mut s: S| async move { s.n += 1; Ok(s) });
g.add_conditional_edges("inc", |s: &S| if s.n < 3 { "inc".into() } else { END.into() }, ["inc", END]);
g.set_entry_point("inc");
let out = g.compile()?.invoke(S { n: 0 }).await?;
assert_eq!(out.n, 3);
```

## Features

- Fixed and conditional edges, cycles with a recursion limit
- Fan-out to several nodes or one branch per item (`SendTo`), merged with reducers
- Concurrent supersteps
- `stream()` with per-node events and token deltas
- Checkpoints after every step, `interrupt_before`, `get_state` / `update_state` for human-in-the-loop
- Prebuilt ReAct agent (`prebuilt::react_agent`, `AgentState`)

Durable checkpointers (SQLite, Postgres) live in [`sinew-checkpoint`](../sinew-checkpoint).
See the [streaming](../../docs/streaming.md), [fan-out](../../docs/fan-out.md) and [checkpointing](../../docs/checkpointing.md) guides.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/roanbrasil/sinew/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/roanbrasil/sinew/blob/main/LICENSE-MIT) at your option.
