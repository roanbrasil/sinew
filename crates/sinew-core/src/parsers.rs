//! Output parsers: runnables that turn a model reply into a `String`, JSON
//! (or any `Deserialize` type) or a list of strings.
//!
//! Each parser accepts a [`ChatResponse`], a [`Message`] or a `String`, so it
//! can follow a chat model, a [`StrOutputParser`] or a closure in a chain.

use std::marker::PhantomData;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{
    runnable::{Runnable, RunnableStream},
    structured::extract_json,
    ChatResponse, Error, Message, Result,
};

/// The reply text. Streams: every model chunk becomes its text delta, so
/// `prompt.pipe(model).pipe(StrOutputParser).stream(..)` yields tokens.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrOutputParser;

/// What parsers accept: a [`ChatResponse`], a [`Message`] or a `String`.
pub trait ModelOutput: Send + 'static {
    fn into_message(self) -> Message;
}

impl ModelOutput for ChatResponse {
    fn into_message(self) -> Message {
        self.message
    }
}

impl ModelOutput for Message {
    fn into_message(self) -> Message {
        self
    }
}

impl ModelOutput for String {
    fn into_message(self) -> Message {
        Message::assistant(self)
    }
}

#[async_trait]
impl<I: ModelOutput> Runnable<I, String> for StrOutputParser {
    async fn invoke(&self, input: I) -> Result<String> {
        Ok(input.into_message().content)
    }
    async fn transform(&self, input: RunnableStream<I>) -> Result<RunnableStream<String>> {
        Ok(input
            .filter_map(|c| async move {
                match c.map(|c| c.into_message().content) {
                    Ok(text) if text.is_empty() => None,
                    other => Some(other),
                }
            })
            .boxed())
    }
}

/// Text to parse: the content, or a tool call's arguments when the model
/// answered with a tool call and no text.
fn reply_text(m: &Message) -> String {
    if m.content.trim().is_empty() {
        if let Some(c) = m.tool_calls.first() {
            return match &c.arguments {
                Value::String(s) => s.clone(),
                v => v.to_string(),
            };
        }
    }
    m.content.clone()
}

/// Parses JSON from the reply (Markdown fences and surrounding prose are
/// stripped) into `T` (default [`serde_json::Value`]). For schema-validated
/// output with automatic repair use [`StructuredOutput`](crate::StructuredOutput).
pub struct JsonOutputParser<T = Value>(PhantomData<fn() -> T>);

impl<T> JsonOutputParser<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Default for JsonOutputParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for JsonOutputParser<T> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<T: DeserializeOwned> JsonOutputParser<T> {
    pub fn parse(&self, text: &str) -> Result<T> {
        let json = extract_json(text).ok_or_else(|| Error::Structured {
            message: "no JSON value found".into(),
            raw: text.to_string(),
        })?;
        serde_json::from_str(json).map_err(|e| Error::Structured {
            message: e.to_string(),
            raw: text.to_string(),
        })
    }
}

#[async_trait]
impl<I, T> Runnable<I, T> for JsonOutputParser<T>
where
    I: ModelOutput,
    T: DeserializeOwned + Send + 'static,
{
    async fn invoke(&self, input: I) -> Result<T> {
        self.parse(&reply_text(&input.into_message()))
    }
}

/// How [`ListOutputParser`] splits the reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ListFormat {
    /// Lines if the reply has several non-empty lines, commas otherwise.
    #[default]
    Auto,
    /// `a, b, c` (LangChain `CommaSeparatedListOutputParser`).
    Comma,
    /// One item per line; bullets (`-`, `*`, `•`) and numbering (`1.`,
    /// `2)`) are stripped.
    Lines,
}

/// Splits the reply into a list of trimmed, non-empty strings.
#[derive(Debug, Clone, Copy, Default)]
pub struct ListOutputParser {
    format: ListFormat,
}

impl ListOutputParser {
    pub fn new(format: ListFormat) -> Self {
        Self { format }
    }
    pub fn comma() -> Self {
        Self::new(ListFormat::Comma)
    }
    pub fn lines() -> Self {
        Self::new(ListFormat::Lines)
    }

    pub fn parse(&self, text: &str) -> Vec<String> {
        let lines: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        let use_lines = match self.format {
            ListFormat::Lines => true,
            ListFormat::Comma => false,
            ListFormat::Auto => lines.len() > 1,
        };
        if use_lines {
            lines
                .into_iter()
                .map(strip_marker)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        } else {
            text.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        }
    }
}

fn strip_marker(line: &str) -> &str {
    let l = line.trim_start_matches(['-', '*', '•']).trim_start();
    let digits = l.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0 {
        if let Some(rest) = l[digits..]
            .strip_prefix('.')
            .or_else(|| l[digits..].strip_prefix(')'))
        {
            return rest.trim();
        }
    }
    l.trim()
}

#[async_trait]
impl<I: ModelOutput> Runnable<I, Vec<String>> for ListOutputParser {
    async fn invoke(&self, input: I) -> Result<Vec<String>> {
        Ok(self.parse(&input.into_message().content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn json_parser_handles_fences_and_tool_calls() {
        let p = JsonOutputParser::<Value>::new();
        let v = p
            .invoke("Sure:\n```json\n{\"a\": [1, 2]}\n```".to_string())
            .await
            .unwrap();
        assert_eq!(v, json!({"a": [1, 2]}));
        let m = Message::assistant_with_tools(
            "",
            vec![crate::ToolCall {
                id: "1".into(),
                name: "x".into(),
                arguments: json!({"ok": true}),
            }],
        );
        let v: Value = Runnable::<Message, Value>::invoke(&p, m).await.unwrap();
        assert_eq!(v, json!({"ok": true}));
        assert!(Runnable::<String, Value>::invoke(&p, "no json here".into())
            .await
            .is_err());

        #[derive(serde::Deserialize)]
        struct P {
            n: u8,
        }
        let typed: P = JsonOutputParser::<P>::new().parse("{\"n\": 4}").unwrap();
        assert_eq!(typed.n, 4);
    }

    #[test]
    fn list_parser_formats() {
        let p = ListOutputParser::default();
        assert_eq!(p.parse("red, green ,blue,"), vec!["red", "green", "blue"]);
        assert_eq!(
            p.parse("1. red\n2) green\n- blue\n* cyan\n\n"),
            vec!["red", "green", "blue", "cyan"]
        );
        assert_eq!(ListOutputParser::lines().parse("a, b"), vec!["a, b"]);
        assert_eq!(
            ListOutputParser::comma().parse("a\nb, c"),
            vec!["a\nb", "c"]
        );
    }
}
