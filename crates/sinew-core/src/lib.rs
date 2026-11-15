//! Core abstractions: messages, chat models, tools and prompt templates.
//!
//! Providers live in [`providers`]: native Anthropic and Gemini clients, and an
//! OpenAI-compatible client with presets for OpenAI, xAI Grok, Groq, Mistral,
//! DeepSeek, OpenRouter and other hosted APIs, plus local servers (Ollama,
//! LM Studio, llama.cpp, vLLM, SGLang, LocalAI, Jan, llamafile, TGI).
//! [`providers::from_spec`] builds any of them from `"provider:model"`.
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

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
