use thiserror::Error;

/// Errors returned by the harness.
#[derive(Debug, Error)]
pub enum HarnessError {
    #[error(transparent)]
    Model(#[from] sinew_core::Error),
    #[error(transparent)]
    Graph(#[from] sinew_graph::GraphError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("configuration error: {0}")]
    Config(String),
    #[error("thread `{0}` has no pending approval")]
    NoPendingApproval(String),
    #[error("the run needs human approval for: {0}")]
    ApprovalRequired(String),
    #[error("hook failed: {0}")]
    Hook(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, HarnessError>;
