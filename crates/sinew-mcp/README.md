<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew-mcp

[![crates.io](https://img.shields.io/crates/v/sinew-mcp.svg)](https://crates.io/crates/sinew-mcp)
[![docs.rs](https://docs.rs/sinew-mcp/badge.svg)](https://docs.rs/sinew-mcp)
[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

[Model Context Protocol](https://modelcontextprotocol.io) client and server for Sinew.

Part of [Sinew](https://github.com/roanbrasil/sinew): LLM chains, agent graphs, tracing and evaluation in Rust.

## Quick example

```rust
use std::sync::Arc;
use sinew_core::{mock::ScriptedModel, Message};
use sinew_graph::prebuilt::{react_agent, AgentState};
use sinew_mcp::McpClient;

let client = McpClient::connect_stdio("my-mcp-server", ["--flag"]).await?;
let tools = client.toolbox().await?;
let model = Arc::new(ScriptedModel::new([Message::assistant("hi")]));
let agent = react_agent(model, tools)?;
agent.invoke(AgentState::new(vec![Message::user("hello")])).await?;
```

## What is inside

- `McpClient`: stdio (and Streamable HTTP with feature `http`); MCP tools become ordinary Sinew tools; resources, resource templates and subscriptions; prompts as `Message`s or Sinew `PromptTemplate`s; server notifications
- `McpServer`: serve any `ToolBox`, or a whole agent as one tool (`AgentTool`), plus resources, resource templates and prompts, over stdio, or over Streamable HTTP as an axum `Router` you can merge into a microservice (`http_router`, feature `http-server`) with sessions or stateless mode, bearer auth, origin checks and a health route
- HTTP client with auth: `McpClient::connect_http_with_token` / `connect_http_with_headers` (feature `http`)
- `sinew-mcp` binary: exposes a Sinew JSONL trace file to MCP clients such as Claude Code (`list_traces`, `get_trace`, `search_runs`)

```sh
cargo install sinew-mcp
claude mcp add sinew-traces -- sinew-mcp --traces ./traces.jsonl
```

See the [MCP guide](../../docs/mcp.md).

## Cargo features

| Feature | Adds |
|---|---|
| `http` | Streamable HTTP client transport |
| `http-server` | Streamable HTTP server: `McpServer::http_router`, `serve_http`, `McpHttpConfig` (with the SSE `GET` stream for notifications) |
| `rag` | `ResourceContents::to_document` and `McpClient::read_resource_documents` (`sinew-rag` `Document`s) |

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/roanbrasil/sinew/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/roanbrasil/sinew/blob/main/LICENSE-MIT) at your option.
