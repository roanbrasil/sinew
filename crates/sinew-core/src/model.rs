use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    stream::{events_from_response, ChatStream},
    Message, Result, ToolSpec,
};

/// Constrains the shape of the model's text reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Any JSON object (OpenAI `json_object`).
    Json,
    /// JSON matching a schema (OpenAI `json_schema`). Providers without native
    /// support ignore it; [`StructuredOutput`](crate::StructuredOutput) also
    /// puts the schema in the prompt so the request still works.
    JsonSchema {
        name: String,
        schema: serde_json::Value,
    },
}

/// Whether and which tool the model must call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    /// Must call some tool.
    Required,
    /// Must call this tool.
    Tool(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

impl From<Vec<Message>> for ChatRequest {
    fn from(messages: Vec<Message>) -> Self {
        Self::new(messages)
    }
}

impl From<Message> for ChatRequest {
    fn from(m: Message) -> Self {
        Self::new(vec![m])
    }
}

/// A single user message.
impl From<String> for ChatRequest {
    fn from(text: String) -> Self {
        Self::new(vec![Message::user(text)])
    }
}

/// A single user message.
impl From<&str> for ChatRequest {
    fn from(text: &str) -> Self {
        Self::new(vec![Message::user(text)])
    }
}

impl ChatRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Default::default()
        }
    }
    pub fn with_tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools = tools;
        self
    }
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }
}

/// Token counts for one response (or a sum of several).
///
/// `input_tokens` is the whole prompt, including any part served from the
/// provider's prompt cache (`cached_input_tokens`) or written to it
/// (`cache_creation_input_tokens`). `output_tokens` includes reasoning /
/// thinking tokens, which are also reported on their own in
/// `reasoning_tokens`. The extra fields are zero when the provider does not
/// report them; build values with `..Default::default()`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Prompt tokens read from the provider's cache (subset of `input_tokens`).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cached_input_tokens: u32,
    /// Prompt tokens written to the cache (Anthropic; subset of `input_tokens`).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_creation_input_tokens: u32,
    /// Reasoning / thinking tokens (subset of `output_tokens`).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub reasoning_tokens: u32,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

impl Usage {
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            ..Default::default()
        }
    }

    pub fn total_tokens(&self) -> u32 {
        self.input_tokens.saturating_add(self.output_tokens)
    }

    /// Estimated cost in USD from the built-in [price table](crate::pricing),
    /// or `None` when the model is unknown (local models, new releases).
    pub fn cost(&self, model: &str) -> Option<f64> {
        crate::pricing::price_for(model).map(|p| p.cost(self))
    }
}

impl std::ops::Add for Usage {
    type Output = Usage;
    fn add(mut self, rhs: Usage) -> Usage {
        self += rhs;
        self
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Usage) {
        self.input_tokens = self.input_tokens.saturating_add(rhs.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(rhs.output_tokens);
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(rhs.cached_input_tokens);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .saturating_add(rhs.cache_creation_input_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(rhs.reasoning_tokens);
    }
}

impl std::iter::Sum for Usage {
    fn sum<I: Iterator<Item = Usage>>(iter: I) -> Usage {
        iter.fold(Usage::default(), |a, b| a + b)
    }
}

impl ChatResponse {
    /// Estimated cost of this response in USD (see [`Usage::cost`]); uses the
    /// model name the provider reported.
    pub fn cost(&self) -> Option<f64> {
        self.usage.cost(&self.model)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    pub usage: Usage,
    pub model: String,
}

/// Anything that can answer a chat request.
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// Model identifier used in traces.
    fn name(&self) -> &str;
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;

    /// Streams the response as [`StreamEvent`](crate::StreamEvent)s.
    ///
    /// The default implementation calls [`chat`](Self::chat) and replays the
    /// whole response as a single text delta plus complete tool calls, so
    /// every model can be streamed; providers override it with real
    /// incremental output. Errors that happen before the first byte are
    /// returned directly; later ones arrive as `Err` items.
    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let resp = self.chat(request).await?;
        let events = events_from_response(resp).into_iter().map(Ok);
        Ok(Box::pin(futures_util::stream::iter(events)))
    }
}

#[async_trait]
impl<T: ChatModel + ?Sized> ChatModel for Arc<T> {
    fn name(&self) -> &str {
        (**self).name()
    }
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        (**self).chat(request).await
    }
    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        (**self).chat_stream(request).await
    }
}
