//! OpenAI Chat Completions wire format. Works with OpenAI, Ollama, vLLM,
//! llama.cpp server and any other compatible endpoint.

use async_trait::async_trait;
use serde_json::{json, Value};

use std::collections::BTreeMap;

use crate::{
    message::{ContentPart, MediaSource},
    stream::{parse_arguments, sse_stream, ChatStream, SseDecoder, SseEvent, StreamEvent},
    ChatModel, ChatRequest, ChatResponse, Error, Message, ResponseFormat, Result, Role, ToolCall,
    ToolChoice, Usage,
};

#[derive(Clone)]
pub struct OpenAiCompat {
    base_url: String,
    api_key: Option<String>,
    /// Header that carries `api_key`; `None` means `Authorization: Bearer`.
    key_header: Option<String>,
    headers: Vec<(String, String)>,
    query: Vec<(String, String)>,
    model: String,
    stream_usage: bool,
    capabilities: crate::Capabilities,
    client: reqwest::Client,
}

impl OpenAiCompat {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: None,
            key_header: None,
            headers: Vec::new(),
            query: Vec::new(),
            model: model.into(),
            stream_usage: true,
            capabilities: crate::Capabilities::all(),
            client: super::shared_client(),
        }
    }

    /// Declares what this endpoint supports (presets set it; the default
    /// assumes everything the Chat Completions format can express).
    pub fn with_capabilities(mut self, capabilities: crate::Capabilities) -> Self {
        self.capabilities = capabilities;
        self
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
        // Same normalization as `from_spec("ollama:...")`: scheme optional,
        // trailing `/` or `/v1` accepted.
        Self::new(super::presets::local_url(host), model)
            .with_capabilities(super::capabilities_for("ollama", "").unwrap_or_default())
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Whether streaming requests ask for a final usage chunk
    /// (`stream_options.include_usage`, on by default). Turn it off for
    /// servers that reject the option.
    pub fn with_stream_usage(mut self, on: bool) -> Self {
        self.stream_usage = on;
        self
    }

    /// Sends the API key in this header instead of `Authorization: Bearer`
    /// (Azure OpenAI uses `api-key`).
    pub fn with_api_key_header(mut self, header: impl Into<String>) -> Self {
        self.key_header = Some(header.into());
        self
    }

    /// Adds a header to every request (e.g. OpenRouter's `HTTP-Referer`).
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Adds a query parameter to every request (e.g. Azure `api-version`).
    pub fn with_query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.push((key.into(), value.into()));
        self
    }

    /// Uses this HTTP client instead of the [shared](super::shared_client) one.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// The endpoint base URL (without `/chat/completions`).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub(crate) fn request(&self) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .post(format!("{}/chat/completions", self.base_url));
        if !self.query.is_empty() {
            req = req.query(&self.query);
        }
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }
        match (&self.api_key, &self.key_header) {
            (Some(k), Some(h)) => req.header(h, k),
            (Some(k), None) => req.bearer_auth(k),
            (None, _) => req,
        }
    }
}

/// The audio `format` OpenAI expects for a MIME type.
fn audio_format(media_type: &str) -> &str {
    match media_type {
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/wav" | "audio/x-wav" | "audio/wave" => "wav",
        other => other.strip_prefix("audio/").unwrap_or(other),
    }
}

/// One `content` array entry. Sources OpenAI cannot take (a document or
/// audio by URL) become a text reference so nothing is silently dropped.
fn encode_part(p: &ContentPart) -> Value {
    match p {
        ContentPart::Text { text } => json!({"type": "text", "text": text}),
        ContentPart::Image { source, detail } => {
            let mut img = json!({ "url": source.to_url() });
            if let Some(d) = detail {
                img["detail"] = json!(d);
            }
            json!({"type": "image_url", "image_url": img})
        }
        ContentPart::Document { source, name } => match source {
            MediaSource::Base64 { .. } => json!({"type": "file", "file": {
                "filename": name.as_deref().unwrap_or("document.pdf"),
                "file_data": source.to_url(),
            }}),
            // An uploaded OpenAI file id.
            MediaSource::Url { url, .. } if url.starts_with("file-") => {
                json!({"type": "file", "file": {"file_id": url}})
            }
            MediaSource::Url { url, .. } => {
                json!({"type": "text", "text": format!("[document: {url}]")})
            }
        },
        ContentPart::Audio { source } => match source {
            MediaSource::Base64 { media_type, data } => {
                json!({"type": "input_audio", "input_audio": {
                    "data": data, "format": audio_format(media_type),
                }})
            }
            MediaSource::Url { url, .. } => {
                json!({"type": "text", "text": format!("[audio: {url}]")})
            }
        },
    }
}

/// `content` as a plain string for text-only messages (the compact form
/// every server accepts) or as a parts array when the message has parts.
pub(crate) fn encode_content(m: &Message) -> Value {
    if m.parts.is_empty() {
        return json!(m.content);
    }
    let mut parts = Vec::with_capacity(m.parts.len() + 1);
    if !m.content.is_empty() {
        parts.push(json!({"type": "text", "text": m.content}));
    }
    parts.extend(m.parts.iter().map(encode_part));
    Value::Array(parts)
}

fn usage_from(u: &Value) -> Usage {
    let n = |p: &str| u.pointer(p).and_then(super::token_count).unwrap_or(0);
    Usage {
        input_tokens: n("/prompt_tokens"),
        output_tokens: n("/completion_tokens"),
        cached_input_tokens: n("/prompt_tokens_details/cached_tokens"),
        cache_creation_input_tokens: 0,
        reasoning_tokens: n("/completion_tokens_details/reasoning_tokens"),
    }
}

pub(crate) fn encode_message(m: &Message) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut v = json!({ "role": role, "content": encode_content(m) });
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
    if let Some(choice) = &req.tool_choice {
        body["tool_choice"] = match choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::Required => json!("required"),
            ToolChoice::Tool(name) => json!({"type": "function", "function": {"name": name}}),
        };
    }
    match &req.response_format {
        Some(ResponseFormat::Json) => body["response_format"] = json!({"type": "json_object"}),
        Some(ResponseFormat::JsonSchema { name, schema }) => {
            body["response_format"] =
                json!({"type": "json_schema", "json_schema": {"name": name, "schema": schema}})
        }
        None => {}
    }
    body
}

pub(crate) fn encode_stream_request(model: &str, req: &ChatRequest, usage: bool) -> Value {
    let mut body = encode_request(model, req);
    body["stream"] = json!(true);
    if usage {
        body["stream_options"] = json!({"include_usage": true});
    }
    body
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

/// Decodes Chat Completions `stream: true` chunks.
pub(crate) struct OpenAiStreamDecoder {
    model: String,
    calls: BTreeMap<usize, PartialCall>,
    usage: Option<Usage>,
    stop_reason: Option<String>,
    done: bool,
}

impl OpenAiStreamDecoder {
    pub(crate) fn new(model: &str) -> Self {
        Self {
            model: model.into(),
            calls: BTreeMap::new(),
            usage: None,
            stop_reason: None,
            done: false,
        }
    }
}

impl SseDecoder for OpenAiStreamDecoder {
    fn on_event(&mut self, ev: &SseEvent) -> Result<Vec<StreamEvent>> {
        let data = ev.data.trim();
        if data == "[DONE]" {
            self.done = true;
            return Ok(vec![]);
        }
        let v: Value = serde_json::from_str(data)?;
        if let Some(err) = v.get("error") {
            return Err(Error::Stream(err.to_string()));
        }
        if let Some(m) = v["model"].as_str() {
            self.model = m.to_string();
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(usage_from(u));
        }
        let mut out = Vec::new();
        for choice in v["choices"].as_array().into_iter().flatten() {
            if let Some(r) = choice["finish_reason"].as_str() {
                self.stop_reason = Some(r.to_string());
            }
            let delta = &choice["delta"];
            if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
                out.push(StreamEvent::text(text));
            }
            for (pos, tc) in delta["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                let index = tc["index"].as_u64().map(|i| i as usize).unwrap_or(pos);
                let id = tc["id"].as_str().map(str::to_string);
                let name = tc["function"]["name"].as_str().map(str::to_string);
                let arguments = match &tc["function"]["arguments"] {
                    Value::String(s) => s.clone(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                };
                let p = self.calls.entry(index).or_default();
                if let Some(id) = id.as_ref().filter(|s| !s.is_empty()) {
                    p.id.clone_from(id);
                }
                if let Some(name) = name.as_ref().filter(|s| !s.is_empty()) {
                    p.name.clone_from(name);
                }
                p.arguments.push_str(&arguments);
                out.push(StreamEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments,
                });
            }
        }
        Ok(out)
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        let mut out: Vec<StreamEvent> = std::mem::take(&mut self.calls)
            .into_iter()
            .map(|(index, p)| StreamEvent::ToolCall {
                index,
                call: ToolCall {
                    id: p.id,
                    name: p.name,
                    arguments: parse_arguments(&p.arguments),
                },
            })
            .collect();
        if let Some(usage) = self.usage.take() {
            out.push(StreamEvent::Usage { usage });
        }
        out.push(StreamEvent::Done {
            model: self.model.clone(),
            stop_reason: self.stop_reason.take(),
        });
        out
    }

    fn is_done(&self) -> bool {
        self.done
    }
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
    let usage = usage_from(&v["usage"]);
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
    fn capabilities(&self) -> crate::Capabilities {
        self.capabilities
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let body = encode_request(&self.model, &request);
        let v = super::post_json("openai-compat", self.request(), &body).await?;
        decode_response(&self.model, &v)
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let body = encode_stream_request(&self.model, &request, self.stream_usage);
        let resp = super::post_sse("openai-compat", self.request(), &body).await?;
        Ok(sse_stream(resp, OpenAiStreamDecoder::new(&self.model)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{decode_all, StreamAccumulator};
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
        assert_eq!(r.usage, Usage::new(10, 3));
        assert_eq!(r.model, "llama");
    }

    // Recorded from an OpenAI-compatible server (shape as sent by OpenAI and
    // Ollama): text, a tool call split over several chunks, finish_reason,
    // the include_usage chunk and the [DONE] sentinel.
    const TOOL_STREAM: &str = r#"data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"content":"Let me "},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"content":"check."},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9","type":"function","function":{"name":"multiply","arguments":""}}]},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\": 37"}}]},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":", \"b\": 3}"}}]},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[],"usage":{"prompt_tokens":52,"completion_tokens":18,"total_tokens":70}}

data: [DONE]

"#;

    #[test]
    fn stream_decoder_handles_text_tool_deltas_and_usage() {
        let events = decode_all(OpenAiStreamDecoder::new("m"), TOOL_STREAM).unwrap();
        let deltas = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolCallDelta { .. }))
            .count();
        assert_eq!(deltas, 3);
        assert_eq!(events[0], StreamEvent::text("Let me "));
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Done { stop_reason: Some(r), model }) if r == "tool_calls" && model == "gpt-x"
        ));
        let mut acc = StreamAccumulator::new();
        events.iter().for_each(|e| acc.push(e));
        let r = acc.finish();
        assert_eq!(r.message.content, "Let me check.");
        assert_eq!(r.message.tool_calls.len(), 1);
        assert_eq!(r.message.tool_calls[0].id, "call_9");
        assert_eq!(r.message.tool_calls[0].arguments, json!({"a": 37, "b": 3}));
        assert_eq!(r.usage, Usage::new(52, 18));
    }

    #[test]
    fn encodes_multimodal_content_arrays() {
        use crate::message::ImageDetail;
        let req = ChatRequest::new(vec![
            Message::system("sys"),
            Message::user_with_images(
                "What is in these?",
                [
                    ContentPart::image_url("https://x/cat.png").with_detail(ImageDetail::High),
                    ContentPart::image_bytes(b"abc", "image/png"),
                ],
            )
            .with_part(ContentPart::pdf_bytes(b"%PDF").with_name("r.pdf"))
            .with_part(ContentPart::audio_bytes(b"RIFF", "audio/wav")),
        ]);
        let body = encode_request("gpt-4o", &req);
        assert_eq!(body["messages"][0]["content"], "sys");
        assert_eq!(
            body["messages"][1]["content"],
            json!([
                {"type": "text", "text": "What is in these?"},
                {"type": "image_url", "image_url": {"url": "https://x/cat.png", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
                {"type": "file", "file": {"filename": "r.pdf", "file_data": "data:application/pdf;base64,JVBERg=="}},
                {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}
            ])
        );
    }

    #[test]
    fn decodes_cached_and_reasoning_tokens() {
        let v = json!({
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 64},
                "completion_tokens_details": {"reasoning_tokens": 30}}
        });
        let u = decode_response("o3", &v).unwrap().usage;
        assert_eq!(u.cached_input_tokens, 64);
        assert_eq!(u.reasoning_tokens, 30);
        assert_eq!(u.input_tokens, 100);
    }

    #[test]
    fn custom_auth_header_and_query() {
        let m = OpenAiCompat::new("https://r.openai.azure.com/openai/deployments/d", "d")
            .with_api_key("k")
            .with_api_key_header("api-key")
            .with_query("api-version", "2024-10-21");
        let req = m.request().build().unwrap();
        assert_eq!(
            req.url().as_str(),
            "https://r.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-10-21"
        );
        assert_eq!(req.headers()["api-key"], "k");
        assert!(req.headers().get("authorization").is_none());
    }

    #[test]
    fn stream_decoder_reports_errors_and_tolerates_missing_index() {
        let err = decode_all(
            OpenAiStreamDecoder::new("m"),
            "data: {\"error\":{\"message\":\"boom\"}}\n\n",
        );
        assert!(matches!(err, Err(Error::Stream(m)) if m.contains("boom")));

        // Some servers send whole tool calls in one chunk without `index`.
        let body = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"a\",\"function\":{\"name\":\"x\",\"arguments\":\"{}\"}},{\"id\":\"b\",\"function\":{\"name\":\"y\",\"arguments\":\"{\\\"k\\\":1}\"}}]}}]}\n\n";
        let events = decode_all(OpenAiStreamDecoder::new("m"), body).unwrap();
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCall { call, .. } => Some(call.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].arguments, json!({"k": 1}));
    }

    #[test]
    fn encodes_stream_options_tool_choice_and_format() {
        let req = ChatRequest::new(vec![Message::user("hi")])
            .with_tool_choice(ToolChoice::Tool("respond".into()))
            .with_response_format(ResponseFormat::Json);
        let body = encode_stream_request("m", &req, true);
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["tool_choice"]["function"]["name"], "respond");
        assert_eq!(body["response_format"]["type"], "json_object");
        assert!(encode_stream_request("m", &req, false)
            .get("stream_options")
            .is_none());
    }
}
