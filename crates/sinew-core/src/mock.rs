//! Deterministic model for tests and offline examples.

use std::{collections::VecDeque, sync::Mutex};

use async_trait::async_trait;

use crate::{
    stream::ChatStream, ChatModel, ChatRequest, ChatResponse, Error, Message, Result, StreamEvent,
    Usage,
};

/// Returns pre-scripted messages in order and records every request.
///
/// `chat_stream` replays each scripted message the way a real provider
/// streams: the text word by word, every tool call as an id/name delta plus
/// argument fragments followed by the complete call, then usage and `Done`.
pub struct ScriptedModel {
    responses: Mutex<VecDeque<Message>>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl ScriptedModel {
    pub fn new(responses: impl IntoIterator<Item = Message>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatModel for ScriptedModel {
    fn name(&self) -> &str {
        "scripted"
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let input_tokens = request
            .messages
            .iter()
            .map(|m| m.content.split_whitespace().count() as u32)
            .sum();
        self.requests.lock().unwrap().push(request);
        let message = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or(Error::ScriptExhausted)?;
        let output_tokens = message.content.split_whitespace().count() as u32;
        Ok(ChatResponse {
            message,
            usage: Usage {
                input_tokens,
                output_tokens,
            },
            model: "scripted".into(),
        })
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let resp = self.chat(request).await?;
        let mut events = Vec::new();
        let text = &resp.message.content;
        let mut start = 0;
        for (i, c) in text.char_indices() {
            if c.is_whitespace() && i > start {
                events.push(StreamEvent::text(&text[start..i]));
                start = i;
            }
        }
        if start < text.len() {
            events.push(StreamEvent::text(&text[start..]));
        }
        for (index, call) in resp.message.tool_calls.iter().enumerate() {
            let args = call.arguments.to_string();
            let mid = args
                .char_indices()
                .nth(args.chars().count() / 2)
                .map_or(0, |(i, _)| i);
            events.push(StreamEvent::ToolCallDelta {
                index,
                id: Some(call.id.clone()),
                name: Some(call.name.clone()),
                arguments: args[..mid].to_string(),
            });
            events.push(StreamEvent::ToolCallDelta {
                index,
                id: None,
                name: None,
                arguments: args[mid..].to_string(),
            });
            events.push(StreamEvent::ToolCall {
                index,
                call: call.clone(),
            });
        }
        events.push(StreamEvent::Usage { usage: resp.usage });
        events.push(StreamEvent::Done {
            model: resp.model,
            stop_reason: None,
        });
        Ok(Box::pin(futures_util::stream::iter(
            events.into_iter().map(Ok),
        )))
    }
}
