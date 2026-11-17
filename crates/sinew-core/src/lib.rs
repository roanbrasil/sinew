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

pub mod batch;
pub mod cache;
pub mod context;
pub mod error;
pub mod message;
pub mod mock;
pub mod model;
pub mod parsers;
pub mod pricing;
pub mod prompt;
pub mod providers;
pub mod resilience;
pub mod router;
pub mod runnable;
pub mod stream;
pub mod structured;
pub mod tool;
pub mod vcr;

pub use cache::{CachedModel, InMemoryCache, ResponseCache};
pub use error::{Error, Result};
pub use message::{ContentPart, ImageDetail, MediaSource, Message, Role, ToolCall};
pub use model::{
    Capabilities, ChatModel, ChatRequest, ChatResponse, ResponseFormat, ToolChoice, Usage,
};
pub use parsers::{JsonOutputParser, ListOutputParser, StrOutputParser};
pub use prompt::PromptTemplate;
pub use resilience::{ChatModelExt, RateLimiter, RetryPolicy};
pub use router::Router;
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

/// A prompt template checked at compile time (feature `macros`).
///
/// `prompt!("...")` parses the template while your crate compiles, with the
/// same rules as [`PromptTemplate`]: `{name}` is a variable (surrounding
/// spaces are ignored), `{{` and `}}` are literal braces, and an unclosed
/// `{`, a stray `}` or a variable name with characters other than letters,
/// digits and `_` is an error. The macro returns a builder with one setter per
/// variable. `format()` compiles only once every variable has a value, so a
/// missing or misspelled variable is a compile error instead of
/// [`Error::MissingVariable`] at runtime.
///
/// ```
/// use sinew_core::prompt;
///
/// let doc = "Rust is a systems programming language.";
/// let text = prompt!("Summarize {text} in {lang}. Reply as JSON: {{\"summary\": ...}}")
///     .text(doc)
///     .lang("pt-BR")
///     .format();
/// assert_eq!(
///     text,
///     "Summarize Rust is a systems programming language. in pt-BR. Reply as JSON: {\"summary\": ...}"
/// );
/// ```
///
/// Leaving out `.lang(...)` fails to compile:
///
/// ```compile_fail
/// let text = sinew_core::prompt!("Summarize {text} in {lang}").text("...").format();
/// // error[E0277]: prompt variable `lang` is not set
/// ```
///
/// # The builder
///
/// - One setter per variable, named after it (`{type}` becomes `r#type`). It
///   takes any `impl Display` (`&str`, `String`, numbers, your own types) and
///   can be called again to replace the value. A variable used several times in
///   the template is set once.
/// - `format(&self) -> String` renders the prompt. It borrows the builder, so
///   it can render several times.
/// - `to_vars(&self) -> Vars` returns the values, to render a runtime template
///   (for example one pulled from the prompt hub) with values whose names were
///   checked at compile time.
/// - `to_template(&self) -> PromptTemplate`, and `From<_> for PromptTemplate`,
///   give the runtime template so it can be piped like any other runnable;
///   `PromptTemplate::as_str` gives the text to push to the prompt hub.
/// - `variables(&self) -> &'static [&'static str]`, in order of first
///   appearance.
///
/// ```
/// use sinew_core::{prompt, PromptTemplate};
///
/// let p = prompt!("Translate {text} to {lang}");
/// let runtime: PromptTemplate = p.clone().into();
/// assert_eq!(runtime.variables(), p.variables());
///
/// let filled = p.text("bonjour").lang("English");
/// assert_eq!(runtime.render_vars(&filled.to_vars()).unwrap(), filled.format());
/// ```
///
/// # Templates from files
///
/// `prompt!(file = "prompts/summarize.txt")` reads the file at compile time,
/// relative to the calling crate's `CARGO_MANIFEST_DIR` (absolute paths work
/// too). The expansion uses `include_str!` on the file, so editing it
/// triggers a rebuild. A missing file is a compile error.
///
/// ```ignore
/// let text = prompt!(file = "prompts/summarize.txt").text(doc).format();
/// ```
///
/// # Differences from `PromptTemplate`
///
/// Every template `prompt!` accepts renders exactly like
/// [`PromptTemplate::render`]. Two kinds of variable names that
/// `PromptTemplate` accepts are rejected because they cannot be method names:
/// names that are not Rust identifiers (`{1}`, `{_}`, `{self}`), and the
/// builder's own method names (`format`, `to_template`, `to_vars`,
/// `to_messages`, `variables`).
///
/// The builder's type is generated at the call site and has no name, like a
/// closure's. Pass it to generic code or wrap the call in a function that
/// returns the `String`.
///
/// # Arguments
///
/// - `"..."` or `file = "..."`: the template.
/// - `crate = "..."` (optional, after the template): path to `sinew_core`,
///   for re-exports under another name.
#[cfg(feature = "macros")]
pub use sinew_macros::prompt;

/// A chat prompt checked at compile time (feature `macros`).
///
/// Each argument is a role (`system`, `user` or `assistant`) and a template
/// literal or `file = "..."`. Templates follow the [`prompt!`] rules; the
/// builder has one setter per variable across all messages.
///
/// ```
/// use sinew_core::{chat_prompt, ChatPromptTemplate, Role};
///
/// let p = chat_prompt!(
///     system: "You are a {persona}.",
///     user: "Explain {topic} in one sentence.",
/// );
/// let request = p.clone().persona("teacher").topic("borrowing").format();
/// assert_eq!(request.messages.len(), 2);
/// assert_eq!(request.messages[0].role, Role::System);
/// assert_eq!(request.messages[1].text(), "Explain borrowing in one sentence.");
///
/// let runtime: ChatPromptTemplate = p.into();
/// assert_eq!(runtime.variables(), ["persona", "topic"]);
/// ```
///
/// - `format(&self) -> ChatRequest`: the rendered messages in a request,
///   like [`ChatPromptTemplate::format`]. Set temperature or other options on
///   the returned request.
/// - `to_messages(&self) -> Vec<Message>`: only the messages.
/// - `to_template(&self) -> ChatPromptTemplate`, `From<_> for
///   ChatPromptTemplate`, `to_vars()` and `variables()`, as in [`prompt!`].
#[cfg(feature = "macros")]
pub use sinew_macros::chat_prompt;

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

    /// Support for `prompt!` and `chat_prompt!` expansions.
    pub mod prompt {
        use crate::{ChatPromptTemplate, Message, PromptTemplate, Role};

        /// A variable that has no value yet.
        #[derive(Clone, Copy, Debug, Default)]
        pub struct Unset;

        /// A variable with its rendered value.
        #[derive(Clone, Debug)]
        pub struct Set(String);

        /// Implemented only by [`Set`]; generated per-variable traits extend it.
        pub trait Value {
            fn get(&self) -> &str;
        }

        impl Value for Set {
            fn get(&self) -> &str {
                &self.0
            }
        }

        pub fn set(value: impl std::fmt::Display) -> Set {
            Set(value.to_string())
        }

        pub fn template(text: &str) -> PromptTemplate {
            PromptTemplate::new(text)
                .expect("prompt! checked this template at compile time with the same rules")
        }

        pub fn chat_template(messages: &[(Role, &str)]) -> ChatPromptTemplate {
            ChatPromptTemplate::from_messages(messages.iter().map(|(r, t)| (*r, *t)))
                .expect("chat_prompt! checked these templates at compile time with the same rules")
        }

        /// Same role-to-message mapping as `ChatPromptTemplate::format`.
        pub fn message(role: Role, text: String) -> Message {
            match role {
                Role::System => Message::system(text),
                Role::Assistant => Message::assistant(text),
                _ => Message::user(text),
            }
        }
    }
}
