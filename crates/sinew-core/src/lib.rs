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

pub mod cache;
pub mod error;
pub mod message;
pub mod mock;
pub mod model;
pub mod parsers;
pub mod pricing;
pub mod prompt;
pub mod providers;
pub mod resilience;
pub mod runnable;
pub mod stream;
pub mod structured;
pub mod tool;

pub use cache::{CachedModel, InMemoryCache, ResponseCache};
pub use error::{Error, Result};
pub use message::{ContentPart, ImageDetail, MediaSource, Message, Role, ToolCall};
pub use model::{ChatModel, ChatRequest, ChatResponse, ResponseFormat, ToolChoice, Usage};
pub use parsers::{JsonOutputParser, ListOutputParser, StrOutputParser};
pub use prompt::PromptTemplate;
pub use resilience::{ChatModelExt, RateLimiter, RetryPolicy};
pub use runnable::{BoxRunnable, ChatPromptTemplate, Runnable, RunnableExt, Vars};
pub use stream::{
    chat_streaming, collect_stream, BoxStream, ChatStream, StreamAccumulator, StreamEvent,
};
pub use structured::{structured, StructuredMode, StructuredOutput};
#[cfg(feature = "schemars")]
pub use tool::TypedTool;
pub use tool::{AsyncFnTool, FnTool, Tool, ToolBox, ToolSpec};
