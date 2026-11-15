//! Native Anthropic Messages API client.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::{ChatModel, ChatRequest, ChatResponse, Error, Message, Result, Role, ToolCall, Usage};

const API_VERSION: &str = "2023-06-01";

#[derive(Clone)]
pub struct Anthropic {
    base_url: String,
    api_key: String,
    model: String,
    default_max_tokens: u32,
    client: reqwest::Client,
}

impl Anthropic {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: "https://api.anthropic.com/v1".into(),
            api_key: api_key.into(),
            model: model.into(),
            default_max_tokens: 4096,
            client: reqwest::Client::new(),
        }
    }

    /// Reads `ANTHROPIC_API_KEY`.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| Error::Config("ANTHROPIC_API_KEY not set".into()))?;
        Ok(Self::new(key, model))
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    pub fn with_default_max_tokens(mut self, n: u32) -> Self {
        self.default_max_tokens = n;
        self
    }
}

/// System messages go to the top-level `system` field; consecutive tool
/// results are merged into a single user turn, as the API requires.
pub(crate) fn encode_request(model: &str, default_max: u32, req: &ChatRequest) -> Value {
    let system: Vec<&str> = req
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.content.as_str())
        .collect();
    let mut turns: Vec<Value> = Vec::new();
    for m in req.messages.iter().filter(|m| m.role != Role::System) {
        match m.role {
            Role::Tool => {
                let block = json!({"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content});
                let merged = turns.last_mut().filter(|t| {
                    t["role"] == "user"
                        && t["content"]
                            .as_array()
                            .is_some_and(|a| a.iter().all(|b| b["type"] == "tool_result"))
                });
                match merged {
                    Some(t) => t["content"].as_array_mut().unwrap().push(block),
                    None => turns.push(json!({"role": "user", "content": [block]})),
                }
            }
            Role::Assistant => {
                let mut blocks = Vec::new();
                if !m.content.is_empty() {
                    blocks.push(json!({"type": "text", "text": m.content}));
                }
                for c in &m.tool_calls {
                    blocks.push(json!({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}));
                }
                turns.push(json!({"role": "assistant", "content": blocks}));
            }
            _ => turns
                .push(json!({"role": "user", "content": [{"type": "text", "text": m.content}]})),
        }
    }
    let mut body = json!({
        "model": model,
        "max_tokens": req.max_tokens.unwrap_or(default_max),
        "messages": turns,
    });
    if !system.is_empty() {
        body["system"] = json!(system.join("\n\n"));
    }
    if !req.tools.is_empty() {
        body["tools"] = req
            .tools
            .iter()
            .map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.parameters}))
            .collect();
    }
    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }
    body
}

pub(crate) fn decode_response(model: &str, v: &Value) -> Result<ChatResponse> {
    let blocks = v["content"]
        .as_array()
        .ok_or_else(|| Error::InvalidResponse("missing content".into()))?;
    let mut text = String::new();
    let mut calls = Vec::new();
    for b in blocks {
        match b["type"].as_str() {
            Some("text") => text.push_str(b["text"].as_str().unwrap_or_default()),
            Some("tool_use") => calls.push(ToolCall {
                id: b["id"].as_str().unwrap_or_default().to_string(),
                name: b["name"].as_str().unwrap_or_default().to_string(),
                arguments: b["input"].clone(),
            }),
            _ => {}
        }
    }
    Ok(ChatResponse {
        message: Message::assistant_with_tools(text, calls),
        usage: Usage {
            input_tokens: v
                .pointer("/usage/input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32,
            output_tokens: v
                .pointer("/usage/output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32,
        },
        model: v["model"].as_str().unwrap_or(model).to_string(),
    })
}

#[async_trait]
impl ChatModel for Anthropic {
    fn name(&self) -> &str {
        &self.model
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let req = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION);
        let v = super::post_json(
            "anthropic",
            req,
            &encode_request(&self.model, self.default_max_tokens, &request),
        )
        .await?;
        decode_response(&self.model, &v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_hoisted_and_tool_results_merged() {
        let calls = vec![
            ToolCall {
                id: "a".into(),
                name: "t".into(),
                arguments: json!({}),
            },
            ToolCall {
                id: "b".into(),
                name: "t".into(),
                arguments: json!({}),
            },
        ];
        let req = ChatRequest::new(vec![
            Message::system("be brief"),
            Message::user("go"),
            Message::assistant_with_tools("", calls),
            Message::tool_result("a", "1"),
            Message::tool_result("b", "2"),
        ]);
        let body = encode_request("claude-sonnet-5-5", 1024, &req);
        assert_eq!(body["system"], "be brief");
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2]["content"].as_array().unwrap().len(), 2);
        assert_eq!(msgs[1]["content"][0]["type"], "tool_use");
    }

    #[test]
    fn decodes_text_and_tool_use() {
        let v = json!({"model": "claude-sonnet-5-5", "content": [
            {"type": "text", "text": "Let me add."},
            {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1}}
        ], "usage": {"input_tokens": 20, "output_tokens": 5}});
        let r = decode_response("x", &v).unwrap();
        assert_eq!(r.message.content, "Let me add.");
        assert_eq!(r.message.tool_calls[0].name, "add");
        assert_eq!(r.usage.output_tokens, 5);
    }
}
