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

// Lets `#[tool]` expansions name this crate as `::sinew_core` from inside it too.
#[allow(unused_extern_crates)]
extern crate self as sinew_core;

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

/// Turns a function into a [`Tool`] (feature `macros`).
///
/// The function name becomes the tool name, its doc comment the description,
/// and its arguments the JSON Schema of the parameters (via `schemars`).
/// A unit-like struct with the same name is generated next to the function,
/// so the function stays callable and the struct is the tool:
///
/// ```
/// use sinew_core::{tool, tools, Tool};
///
/// #[derive(serde::Serialize)]
/// struct Weather { city: String, celsius: f32 }
///
/// /// Current weather for a city.
/// #[tool]
/// async fn weather(
///     /// City name, e.g. "Montreal".
///     city: String,
///     /// Use Fahrenheit instead of Celsius.
///     fahrenheit: Option<bool>,
/// ) -> sinew_core::Result<Weather> {
///     let _ = fahrenheit;
///     Ok(Weather { city, celsius: -3.0 })
/// }
///
/// /// Adds two numbers.
/// #[tool(name = "add")]
/// fn add_numbers(a: f64, b: f64) -> f64 { a + b }
///
/// let spec = weather::tool().spec();               // the generated struct
/// assert_eq!(spec.name, "weather");
/// assert_eq!(spec.description, "Current weather for a city.");
/// assert_eq!(spec.parameters["required"], serde_json::json!(["city"]));
/// assert_eq!(add_numbers(1.0, 2.0), 3.0);           // still a normal function
///
/// let toolbox = tools![weather, add_numbers];      // a ToolBox with both
/// assert_eq!(toolbox.specs().len(), 2);
/// ```
///
/// # Shared state
///
/// Mark one argument `#[state]` to inject application state (a database pool,
/// an HTTP client, an `Arc<...>`) that the model never sees. The tool is then
/// built with that value; it is cloned for each call, so wrap big things in
/// `Arc`:
///
/// ```
/// use std::sync::{Arc, Mutex};
/// use sinew_core::{tool, Tool, ToolBox};
///
/// type Counter = Arc<Mutex<u32>>;
///
/// /// Adds to the shared counter and returns the new total.
/// #[tool]
/// fn bump(#[state] counter: Counter, by: u32) -> u32 {
///     let mut n = counter.lock().unwrap();
///     *n += by;
///     *n
/// }
///
/// let counter = Counter::default();
/// let tools = ToolBox::new().with(bump::tool(counter.clone()));
/// assert_eq!(bump::tool(counter).spec().parameters["required"], serde_json::json!(["by"]));
/// ```
///
/// # Rules
///
/// - Sync or `async` free functions; no `self` (use `#[state]`), no generics.
/// - Arguments must be owned and implement `serde::Deserialize` and
///   `schemars::JsonSchema` (`String`, numbers, `bool`, `Vec<T>`, `Option<T>`
///   for optional parameters, or your own `#[derive(Deserialize, JsonSchema)]`
///   types). Doc comments, `#[serde(...)]` and `#[schemars(...)]` on an
///   argument describe that parameter.
/// - The return type is any `T: Serialize`, or `Result<T, E>` with
///   `E: Display`; an `Err` becomes [`Error::Tool`] so the agent sees the
///   message and can recover.
/// - Unknown argument names are rejected, so a model that invents parameters
///   gets an error it can correct.
///
/// # Attributes
///
/// - `name = "..."`: tool name (default: the function name).
/// - `description = "..."`: overrides the doc comment.
/// - `crate = "..."`: path to `sinew_core`, for re-exports under another name.
#[cfg(feature = "macros")]
pub use sinew_macros::tool;

/// Builds a [`ToolBox`] from `#[tool]` functions or any `Tool` types that
/// implement `Default`: `tools![get_order, update_order]`.
#[macro_export]
macro_rules! tools {
    ($($tool:ty),* $(,)?) => {
        $crate::ToolBox::new() $(.with(<$tool as ::core::default::Default>::default()))*
    };
}

/// Support code for macro expansions. Not public API.
#[doc(hidden)]
#[cfg(feature = "macros")]
pub mod __private {
    pub use async_trait::async_trait;
    pub use schemars;
    pub use serde;
    pub use serde_json;

    use crate::{Error, Result};

    pub fn schema_for<T: schemars::JsonSchema>() -> serde_json::Value {
        let mut v = crate::tool::tool_schema::<T>();
        if let Some(obj) = v.as_object_mut() {
            obj.entry("properties")
                .or_insert_with(|| serde_json::Value::Object(Default::default()));
        }
        v
    }

    pub fn parse_args<T: serde::de::DeserializeOwned>(
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<T> {
        let arguments = match arguments {
            serde_json::Value::Null => serde_json::Value::Object(Default::default()),
            v => v,
        };
        serde_json::from_value(arguments).map_err(|e| Error::Tool {
            name: name.into(),
            message: format!("invalid arguments: {e}"),
        })
    }

    pub fn to_value<T: serde::Serialize>(name: &str, v: T) -> Result<serde_json::Value> {
        serde_json::to_value(v).map_err(|e| Error::Tool {
            name: name.into(),
            message: format!("could not serialize the result: {e}"),
        })
    }

    pub fn tool_error<E: std::fmt::Display>(name: &str, e: E) -> Error {
        Error::Tool {
            name: name.into(),
            message: e.to_string(),
        }
    }
}
