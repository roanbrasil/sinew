use thiserror::Error;

/// Errors from building policies, reading policy files and evaluating
/// external policy engines.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GovernError {
    /// A condition such as `amount <= 100` could not be parsed.
    #[error("invalid condition `{condition}`: {message}")]
    Condition { condition: String, message: String },
    /// A time or UTC offset could not be parsed.
    #[error("invalid time: {0}")]
    Time(String),
    /// A policy file is malformed or invalid. `location` names the entry
    /// (`policies[2] (id "refund-limit")`) or the line and column.
    #[error("policy file: {location}: {message}")]
    PolicyFile { location: String, message: String },
    /// A policy could not be evaluated (e.g. an OPA server is down). The
    /// control plane turns this into a `Deny` (fail closed).
    #[error("policy `{policy}` failed: {message}")]
    Evaluation { policy: String, message: String },
    /// Reading a policy file failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl GovernError {
    pub(crate) fn file(location: impl Into<String>, message: impl Into<String>) -> Self {
        GovernError::PolicyFile {
            location: location.into(),
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, GovernError>;
