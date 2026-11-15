//! OpenAI Chat Completions wire format. Works with OpenAI, Ollama, vLLM,
//! llama.cpp server and any other compatible endpoint.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::{ChatModel, ChatRequest, ChatResponse, Error, Message, Result, Role, ToolCall, Usage};

#[derive(Clone)]
pub struct OpenAiCompat {
    base_url: String,
    api_key: Option<String>,
    model: String,
    client: reqwest::Client,
}

impl OpenAiCompat {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: None,
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    /// OpenAI itself; reads `OPENAI_API_KEY`.
    pub fn openai(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| Error::Config("OPENAI_API_KEY not set".into()))?;
        Ok(Self::new("https://api.openai.com/v1", model).with_api_key(key))
    }

    /// Local Ollama (`OLLAMA_HOST` or 127.0.0.1:11434).
    pub fn ollama(model: impl Into<String>) -> Self {
        let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
        let host = if host.starts_with("http") {
            host
        } else {
            format!("http://{host}")
        };
        Self::new(format!("{}/v1", host.trim_end_matches('/')), model)
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

pub(crate) fn encode_message(m: &Message) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut v = json!({ "role": role, "content": m.content });
    if !m.tool_calls.is_empty() {
        v["tool_calls"] = m
            .tool_calls
            .iter()
            .map(|c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments.to_string()}}))
            .collect();
    }
    if let Some(id) = &m.tool_call_id {
        v["tool_call_id"] = json!(id);
    }
    v
}

pub(crate) fn encode_request(model: &str, req: &ChatRequest) -> Value {
    let mut body = json!({
        "model": model,
        "messages": req.messages.iter().map(encode_message).collect::<Vec<_>>(),
    });
    if !req.tools.is_empty() {
        body["tools"] = req
            .tools
            .iter()
            .map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}))
            .collect();
    }
    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(n) = req.max_tokens {
        body["max_tokens"] = json!(n);
    }
    body
}

pub(crate) fn decode_response(model: &str, v: &Value) -> Result<ChatResponse> {
    let msg = v
        .pointer("/choices/0/message")
        .ok_or_else(|| Error::InvalidResponse("missing choices[0].message".into()))?;
    let content = msg["content"].as_str().unwrap_or_default().to_string();
    let mut calls = Vec::new();
    for c in msg["tool_calls"].as_array().into_iter().flatten() {
        let raw = &c["function"]["arguments"];
        let arguments = match raw {
            Value::String(s) => {
                serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
            }
            other => other.clone(),
        };
        calls.push(ToolCall {
            id: c["id"].as_str().unwrap_or_default().to_string(),
            name: c["function"]["name"]
                .as_str()
                .ok_or_else(|| Error::InvalidResponse("tool call without name".into()))?
                .to_string(),
            arguments,
        });
    }
    let usage = Usage {
        input_tokens: v
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        output_tokens: v
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
    };
    Ok(ChatResponse {
        message: Message::assistant_with_tools(content, calls),
        usage,
        model: v["model"].as_str().unwrap_or(model).to_string(),
    })
}

#[async_trait]
impl ChatModel for OpenAiCompat {
    fn name(&self) -> &str {
        &self.model
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let mut req = self
            .client
            .post(format!("{}/chat/completions", self.base_url));
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        let v =
            super::post_json("openai-compat", req, &encode_request(&self.model, &request)).await?;
        decode_response(&self.model, &v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolSpec;

    #[test]
    fn encodes_tools_and_tool_results() {
        let call = ToolCall {
            id: "c1".into(),
            name: "add".into(),
            arguments: json!({"a": 1}),
        };
        let req = ChatRequest::new(vec![
            Message::user("2+2?"),
            Message::assistant_with_tools("", vec![call]),
            Message::tool_result("c1", "4"),
        ])
        .with_tools(vec![ToolSpec {
            name: "add".into(),
            description: "d".into(),
            parameters: json!({}),
        }]);
        let body = encode_request("m", &req);
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{\"a\":1}"
        );
        assert_eq!(body["messages"][2]["tool_call_id"], "c1");
        assert_eq!(body["tools"][0]["function"]["name"], "add");
    }

    #[test]
    fn decodes_tool_calls_and_usage() {
        let v = json!({
            "model": "llama",
            "choices": [{"message": {"role": "assistant", "content": null,
                "tool_calls": [{"id": "x", "type": "function", "function": {"name": "add", "arguments": "{\"a\":2}"}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}
        });
        let r = decode_response("m", &v).unwrap();
        assert_eq!(r.message.tool_calls[0].arguments, json!({"a": 2}));
        assert_eq!(
            r.usage,
            Usage {
                input_tokens: 10,
                output_tokens: 3
            }
        );
        assert_eq!(r.model, "llama");
    }
}
