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
    #[cfg(feature = "http")]
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, McpError>;
