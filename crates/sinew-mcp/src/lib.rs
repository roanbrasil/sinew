//! [Model Context Protocol](https://modelcontextprotocol.io) for Sinew.
//!
//! - **Client** ([`McpClient`]): connect to an MCP server over stdio (or
//!   Streamable HTTP with the `http` feature), and use its tools as ordinary
//!   Sinew [`Tool`](sinew_core::Tool)s, so `react_agent` calls them
//!   transparently. Read its resources, subscribe to changes, and load its
//!   prompts as [`Message`](sinew_core::Message)s or
//!   [`PromptTemplate`](sinew_core::PromptTemplate)s.
//! - **Server** ([`McpServer`]): serve any [`ToolBox`](sinew_core::ToolBox),
//!   optionally an agent graph as a single tool ([`AgentTool`]), resources,
//!   resource templates and prompts to MCP clients such as Claude Code.
//! - **OAuth 2.1** (features `oauth-server`, `oauth-client`, `oauth`): the
//!   MCP authorization specification for the Streamable HTTP transport, in
//!   the `oauth` module: JWT access-token validation and protected resource
//!   metadata on the server, authorization code + PKCE on the client.
//! - **`sinew-mcp` binary**: an MCP server exposing a Sinew JSONL trace
//!   file through `list_traces`, `get_trace` and `search_runs` ([`traces`]).
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use std::sync::Arc;
//! use sinew_core::{mock::ScriptedModel, Message};
//! use sinew_graph::prebuilt::{react_agent, AgentState};
//! use sinew_mcp::McpClient;
//!
//! let client = McpClient::connect_stdio("my-mcp-server", ["--flag"]).await?;
//! let tools = client.toolbox().await?;
//! let model = Arc::new(ScriptedModel::new([Message::assistant("hi")]));
//! let agent = react_agent(model, tools)?;
//! agent.invoke(AgentState::new(vec![Message::user("hello")])).await?;
//! # Ok(()) }
//! ```
//!
//! # Why a hand-written JSON-RPC layer instead of `rmcp`
//!
//! The official Rust SDK (`rmcp`) covers the whole protocol (resources,
//! prompts, sampling, elicitation, OAuth) and brings proc macros, `schemars`
//! and a large dependency tree with it. Sinew uses the server features
//! (tools, resources, prompts), `ping`, cancellation and server
//! notifications, which is JSON-RPC over tokio. Writing it directly keeps
//! `sinew-mcp` free of new dependencies (everything it uses is already in the
//! workspace), compiles in seconds, and lets the mapping to Sinew's `Tool`,
//! `ToolBox`, `Message` and `PromptTemplate` stay exact. The [`protocol`]
//! module holds the message types.
//!
//! # Protocol notes
//!
//! - Speaks protocol version `2025-06-18` and accepts `2025-03-26` and
//!   `2024-11-05` from peers (the messages used are compatible across them).
//! - Resource reads for unknown URIs are `-32002` errors; unknown prompts and
//!   missing required prompt arguments are `-32602`.
//! - Stdio framing is newline-delimited JSON. Batches are rejected, as in
//!   `2025-06-18`.
//! - Tool failures are reported in-band (`isError: true`) so a model can see
//!   them; unknown tools and malformed params are JSON-RPC `-32602` errors.
//! - The server is lenient about requests sent before `initialize`.
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

pub mod agent;
pub mod client;
pub mod error;
#[cfg(feature = "http")]
mod http;
#[cfg(feature = "http-server")]
pub mod http_server;
mod hub;
#[cfg(any(feature = "oauth-client", feature = "oauth-server"))]
pub mod oauth;
pub mod protocol;
pub mod server;
pub mod traces;

pub use agent::AgentTool;
pub use client::{McpClient, McpTool};
pub use error::{McpError, Result};
#[cfg(feature = "http-server")]
pub use http_server::McpHttpConfig;
pub use protocol::{
    CallToolResult, Content, GetPromptResult, InitializeResult, McpNotification, McpPrompt,
    McpResource, McpResourceTemplate, McpToolInfo, PromptArgument, PromptMessage, PromptRole,
    ResourceContents, RpcError, UriTemplate,
};
pub use server::{HandlerResult, McpServer, PromptRequest, ResourceRequest};
