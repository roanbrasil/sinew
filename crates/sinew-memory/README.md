<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew-memory

[![crates.io](https://img.shields.io/crates/v/sinew-memory.svg)](https://crates.io/crates/sinew-memory)
[![docs.rs](https://docs.rs/sinew-memory/badge.svg)](https://docs.rs/sinew-memory)
[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

Long-term memory for Sinew agents: facts, preferences and episodes that survive across threads,
stored in any `sinew-graph` `Store` (LangMem / Mem0 style).

Part of [Sinew](https://github.com/roanbrasil/sinew): LLM chains, agent graphs, tracing and evaluation in Rust.

## Quick example

```rust
use std::sync::Arc;
use sinew_checkpoint::SqliteStore;
use sinew_core::{ChatModel, Message};
use sinew_graph::prebuilt::{AgentState, ReactAgent};
use sinew_memory::{MemoryAgent, MemoryManager};
use sinew_rag::embeddings_from_spec;

let model: Arc<dyn ChatModel> = /* any chat model */;
let memory = MemoryManager::new(Arc::new(SqliteStore::connect("sqlite://memory.db").await?), model.clone())
    .with_embeddings_arc(embeddings_from_spec("ollama:bge-m3")?);

// Low level: extract from a conversation, recall later.
memory.remember("ana", &conversation, Some("thread-1")).await?;
let hits = memory.recall("ana", "what does she eat?", 5).await?;

// Prebuilt: recall before every turn, save after it (in the background).
let agent = MemoryAgent::new(ReactAgent::new(model, tools), memory)?;
let out = agent.invoke("ana", AgentState::new(vec![Message::user("Dinner ideas?")])).await?;
```

Feature `harness` adds `MemoryHook`, the same behaviour as a `sinew-harness` hook.

See the [memory guide](https://github.com/roanbrasil/sinew/blob/main/docs/memory.md).

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/roanbrasil/sinew/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/roanbrasil/sinew/blob/main/LICENSE-MIT) at your option.
