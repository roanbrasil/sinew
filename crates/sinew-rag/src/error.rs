use std::path::PathBuf;

use thiserror::Error;

/// Errors from the retrieval stack.
#[derive(Debug, Error)]
pub enum Error {
    /// A chat model or other `sinew-core` call failed.
    #[error(transparent)]
    Core(#[from] sinew_core::Error),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("{provider} returned HTTP {status}: {body}")]
    Provider {
        provider: String,
        status: u16,
        body: String,
    },
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
    #[error("{path}: {message}")]
    Load { path: PathBuf, message: String },
    #[cfg(any(feature = "sqlite", feature = "pgvector"))]
    #[error("database error: {0}")]
    Sql(#[from] sqlx::Error),
    #[error("embedding has {got} dimensions, the store expects {expected}")]
    Dimensions { expected: usize, got: usize },
    #[error("missing configuration: {0}")]
    Config(String),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
