//! Attach Model Context Protocol servers' tools (feature `mcp`).
//!
//! ```no_run
//! # async fn demo() -> sinew_harness::Result<()> {
//! use sinew_harness::{mcp, Agent};
//!
//! let client = mcp::connect_command_line("npx -y @modelcontextprotocol/server-everything").await?;
//! let agent = Agent::builder()
//!     .model_spec("ollama:qwen3:4b-instruct-2507-q4_K_M")
//!     .mcp_server(&client)
//!     .await?
//!     .build()?;
//! # Ok(()) }
//! ```

use std::sync::Arc;

use sinew_core::Tool;
pub use sinew_mcp::McpClient;

use crate::{AgentBuilder, HarnessError, Result};

fn mcp_err(e: sinew_mcp::McpError) -> HarnessError {
    HarnessError::Other(format!("mcp: {e}"))
}

/// Every tool of an MCP server as harness tools.
pub async fn mcp_tools(client: &McpClient) -> Result<Vec<Arc<dyn Tool>>> {
    Ok(client
        .tools()
        .await
        .map_err(mcp_err)?
        .into_iter()
        .map(|t| Arc::new(t) as Arc<dyn Tool>)
        .collect())
}

/// Spawns an MCP server over stdio from a command line (split on
/// whitespace; no shell quoting).
pub async fn connect_command_line(command_line: &str) -> Result<McpClient> {
    let mut parts = command_line.split_whitespace();
    let program = parts
        .next()
        .ok_or_else(|| HarnessError::Config("empty MCP command".into()))?;
    McpClient::connect_stdio(program, parts.collect::<Vec<_>>())
        .await
        .map_err(mcp_err)
}

impl AgentBuilder {
    /// Adds every tool of a connected MCP server. Tool calls go through
    /// the same permission policy, hooks, guardrails and tracing as
    /// built-in tools (rule `mcp_tool_name` or a wildcard).
    pub async fn mcp_server(self, client: &McpClient) -> Result<Self> {
        Ok(self.tools(mcp_tools(client).await?))
    }
}
