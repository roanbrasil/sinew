use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("{provider} returned HTTP {status}: {body}")]
    Provider {
        provider: String,
        status: u16,
        body: String,
    },
    #[error("invalid provider response: {0}")]
    InvalidResponse(String),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tool `{0}` is not registered")]
    ToolNotFound(String),
    #[error("tool `{name}` failed: {message}")]
    Tool { name: String, message: String },
    #[error("prompt template is missing variable `{0}`")]
    MissingVariable(String),
    #[error("malformed prompt template: {0}")]
    Template(String),
    #[error("missing configuration: {0}")]
    Config(String),
    #[error("structured output failed: {message}")]
    Structured { message: String, raw: String },
    #[error("stream error: {0}")]
    Stream(String),
    #[error("scripted model has no responses left")]
    ScriptExhausted,
}

pub type Result<T> = std::result::Result<T, Error>;
