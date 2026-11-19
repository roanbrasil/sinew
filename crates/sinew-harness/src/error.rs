use thiserror::Error;

/// Errors returned by the harness.
#[derive(Debug, Error)]
pub enum HarnessError {
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#harness-error-model).
    #[error(transparent)]
    Model(#[from] sinew_core::Error),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#harness-error-graph).
    #[error(transparent)]
    Graph(#[from] sinew_graph::GraphError),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#harness-error-io).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#harness-error-config).
    #[error("configuration error: {0}")]
    Config(String),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#harness-error-no-pending-approval).
    #[error("thread `{0}` has no pending approval")]
    NoPendingApproval(String),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#harness-error-approval-required).
    #[error("the run needs human approval for: {0}")]
    ApprovalRequired(String),
    /// An approval did not satisfy the request's
    /// [`Confirmation`](crate::Confirmation) (a retyped value differs, or
    /// too few approvers). Nothing was applied; the thread is still paused.
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#harness-error-confirmation).
    #[error("approval of call `{call_id}` refused: {reason}")]
    Confirmation { call_id: String, reason: String },
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#harness-error-hook).
    #[error("hook failed: {0}")]
    Hook(String),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#harness-error-other).
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, HarnessError>;
