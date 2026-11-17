//! The error type returned by every tool in this crate.

/// Why a tool call failed. Inside an agent the message becomes a tool
/// message, so it is written for the model: it says what was wrong and, when
/// possible, what to do instead.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ToolError {
    /// The arguments are malformed or out of range.
    #[error("invalid input: {0}")]
    Invalid(String),
    /// The host, address, method or recipient is not permitted by the
    /// tool's policy.
    #[error("blocked: {0}")]
    Blocked(String),
    /// The tool is not configured (missing API key, bad base URL).
    #[error("configuration error: {0}")]
    Config(String),
    /// A remote service answered with an error status.
    #[error("{service} returned HTTP {status}: {message}")]
    Api {
        service: String,
        status: u16,
        message: String,
    },
    /// The request could not be sent or the response could not be read.
    #[error("request failed: {0}")]
    Request(String),
}

impl ToolError {
    /// Converts an error from a client built with
    /// [`HostPolicy::client_builder`](crate::net::HostPolicy::client_builder):
    /// a refusal by the guarded resolver or redirect policy becomes
    /// [`ToolError::Blocked`]; anything else becomes [`ToolError::Request`]
    /// with the messages of the error's sources appended.
    pub fn from_reqwest(err: &reqwest::Error) -> Self {
        let marker = crate::net::BLOCKED_MARKER;
        let mut msg = err.to_string();
        let mut source = std::error::Error::source(err);
        while let Some(s) = source {
            let s_msg = s.to_string();
            if let Some(i) = s_msg.find(marker) {
                return Self::Blocked(s_msg[i + marker.len()..].to_string());
            }
            if !msg.contains(&s_msg) {
                msg.push_str(": ");
                msg.push_str(&s_msg);
            }
            source = s.source();
        }
        Self::Request(msg)
    }

    /// The `sinew_core` error a hand-written `Tool::call` returns.
    #[allow(dead_code)]
    pub(crate) fn into_core(self, tool: &str) -> sinew_core::Error {
        sinew_core::Error::Tool {
            name: tool.into(),
            message: self.to_string(),
        }
    }
}
