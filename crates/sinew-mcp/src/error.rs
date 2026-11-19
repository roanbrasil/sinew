use thiserror::Error;

use crate::protocol::RpcError;

#[derive(Debug, Error)]
pub enum McpError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("JSON-RPC error {}: {}", .0.code, .0.message)]
    Rpc(RpcError),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("connection closed")]
    Closed,
    #[error("request `{0}` timed out")]
    Timeout(String),
    /// A tool vetting check set to `VetAction::Fail` refused a server tool
    /// (see [`McpClient::vetted_toolbox`](crate::McpClient::vetted_toolbox)).
    #[error("tool vetting: {0}")]
    Vetting(sinew_core::Error),
    #[cfg(feature = "http")]
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, McpError>;
