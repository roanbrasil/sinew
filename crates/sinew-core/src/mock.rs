//! Deterministic model for tests and offline examples.

use std::{collections::VecDeque, sync::Mutex};

use async_trait::async_trait;

use crate::{ChatModel, ChatRequest, ChatResponse, Error, Message, Result, Usage};

/// Returns pre-scripted messages in order and records every request.
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
}
