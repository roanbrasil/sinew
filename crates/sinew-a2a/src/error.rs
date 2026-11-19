use thiserror::Error;

use crate::jsonrpc::RpcError;

/// Client-side errors.
#[derive(Debug, Error)]
pub enum A2aError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("HTTP {status} from {url}: {body}")]
    Status {
        url: String,
        status: u16,
        body: String,
    },
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("A2A error {}: {}", .0.code, .0.message)]
    Rpc(RpcError),
    #[error("protocol error: {0}")]
    Protocol(String),
    /// A card or identity proof that does not verify.
    #[error("identity: {0}")]
    Identity(#[from] sinew_core::identity::IdentityError),
}

impl From<RpcError> for A2aError {
    fn from(e: RpcError) -> Self {
        A2aError::Rpc(e)
    }
}

pub type Result<T> = std::result::Result<T, A2aError>;
