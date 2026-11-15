//! Core abstractions: messages, chat models, tools and prompt templates.
//!
//! Providers live in [`providers`]: an OpenAI-compatible client (OpenAI, Ollama,
//! vLLM, llama.cpp server) and a native Anthropic client.

pub mod error;
pub mod message;
pub mod mock;
pub mod model;
pub mod prompt;
pub mod providers;
pub mod stream;
pub mod structured;
pub mod tool;

pub use error::{Error, Result};
pub use message::{Message, Role, ToolCall};
pub use model::{ChatModel, ChatRequest, ChatResponse, ResponseFormat, ToolChoice, Usage};
pub use prompt::PromptTemplate;
pub use stream::{
    chat_streaming, collect_stream, BoxStream, ChatStream, StreamAccumulator, StreamEvent,
};
pub use structured::{structured, StructuredMode, StructuredOutput};
pub use tool::{FnTool, Tool, ToolBox, ToolSpec};
