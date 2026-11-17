//! Native Anthropic Messages API client.

use async_trait::async_trait;
use serde_json::{json, Value};

use std::collections::BTreeMap;

use crate::{
    message::{ContentPart, MediaSource},
    stream::{parse_arguments, sse_stream, ChatStream, SseDecoder, SseEvent, StreamEvent},
    ChatModel, ChatRequest, ChatResponse, Error, Message, Result, Role, ToolCall, ToolChoice,
    Usage,
};

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
            client: super::shared_client(),
        }
    }

    /// Uses this HTTP client instead of the [shared](super::shared_client) one.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
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

    fn request(&self) -> reqwest::RequestBuilder {
        self.client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
    }
}

fn encode_source(source: &MediaSource) -> Value {
    match source {
        MediaSource::Base64 { media_type, data } => {
            json!({"type": "base64", "media_type": media_type, "data": data})
        }
        // Files API id.
        MediaSource::Url { url, .. } if url.starts_with("file_") => {
            json!({"type": "file", "file_id": url})
        }
        MediaSource::Url { url, .. } => json!({"type": "url", "url": url}),
    }
}

/// One content block. Audio is not accepted by the Messages API; it becomes
/// a text reference so the request still goes through.
fn encode_part(p: &ContentPart) -> Value {
    match p {
        ContentPart::Text { text } => json!({"type": "text", "text": text}),
        ContentPart::Image { source, .. } => {
            json!({"type": "image", "source": encode_source(source)})
        }
        ContentPart::Document { source, name } => {
            let mut b = json!({"type": "document", "source": encode_source(source)});
            if let Some(n) = name {
                b["title"] = json!(n);
            }
            b
        }
        ContentPart::Audio { source } => {
            let what = match source {
                MediaSource::Url { url, .. } => url.clone(),
                MediaSource::Base64 { media_type, .. } => format!("inline {media_type}"),
            };
            json!({"type": "text", "text": format!("[audio not supported by this provider: {what}]")})
        }
    }
}

/// Text block (if any) followed by the message's parts.
fn content_blocks(m: &Message) -> Vec<Value> {
    let mut blocks = Vec::with_capacity(m.parts.len() + 1);
    if !m.content.is_empty() || m.parts.is_empty() {
        blocks.push(json!({"type": "text", "text": m.content}));
    }
    blocks.extend(m.parts.iter().map(encode_part));
    blocks
}

/// System messages go to the top-level `system` field; consecutive tool
/// results are merged into a single user turn, as the API requires.
pub(crate) fn encode_request(model: &str, default_max: u32, req: &ChatRequest) -> Value {
    let system: Vec<String> = req
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(Message::text)
        .collect();
    let mut turns: Vec<Value> = Vec::new();
    for m in req.messages.iter().filter(|m| m.role != Role::System) {
        match m.role {
            Role::Tool => {
                let content = if m.parts.is_empty() {
                    json!(m.content)
                } else {
                    Value::Array(content_blocks(m))
                };
                let block = json!({"type": "tool_result", "tool_use_id": m.tool_call_id, "content": content});
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
                blocks.extend(m.parts.iter().map(encode_part));
                for c in &m.tool_calls {
                    blocks.push(json!({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}));
                }
                turns.push(json!({"role": "assistant", "content": blocks}));
            }
            _ => turns.push(json!({"role": "user", "content": content_blocks(m)})),
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
    if let Some(t) = req.temperature.filter(|_| accepts_sampling(model)) {
        body["temperature"] = json!(t);
    }
    if let Some(choice) = &req.tool_choice {
        if accepts_forced_tool_choice(model) {
            body["tool_choice"] = match choice {
                ToolChoice::Auto => json!({"type": "auto"}),
                ToolChoice::Required => json!({"type": "any"}),
                ToolChoice::Tool(name) => json!({"type": "tool", "name": name}),
            };
        } else {
            // These models reject forced tool use with a 400: ask for `auto`
            // and steer from the system prompt instead.
            body["tool_choice"] = json!({"type": "auto"});
            let steer = match choice {
                ToolChoice::Auto => None,
                ToolChoice::Required => {
                    Some("You must answer by calling one of the tools.".to_string())
                }
                ToolChoice::Tool(name) => {
                    Some(format!("You must answer by calling the `{name}` tool."))
                }
            };
            if let Some(steer) = steer {
                body["system"] = match body["system"].as_str() {
                    Some(s) => json!(format!("{s}\n\n{steer}")),
                    None => json!(steer),
                };
            }
        }
    }
    // `response_format` has no direct equivalent here; StructuredOutput puts
    // the schema in the system prompt (JSON mode) or uses a forced tool.
    body
}

/// Claude models whose request surface removed sampling parameters
/// (`temperature`, `top_p`, `top_k`): sending them returns a 400.
/// What the Messages API supports for `model` through this client.
pub(crate) fn anthropic_capabilities(model: &str) -> crate::Capabilities {
    crate::Capabilities::all()
        .with_audio(false)
        // StructuredOutput uses a forced tool (or a steer) instead.
        .with_json_schema(false)
        .with_sampling(accepts_sampling(model))
        .with_forced_tool_choice(accepts_forced_tool_choice(model))
}

fn accepts_sampling(model: &str) -> bool {
    const NO_SAMPLING: &[&str] = &[
        "claude-fable",
        "claude-mythos",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-haiku-5",
    ];
    !NO_SAMPLING.iter().any(|p| model.starts_with(p))
}

/// Claude models that return a 400 for `tool_choice` `any` / `tool`.
fn accepts_forced_tool_choice(model: &str) -> bool {
    const NO_FORCED: &[&str] = &[
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
    ];
    !NO_FORCED.iter().any(|p| model.starts_with(p))
}

struct ToolBlock {
    id: String,
    name: String,
    input: Value,
    json: String,
}

/// Decodes Messages API streaming events (`message_start`,
/// `content_block_*`, `message_delta`, `message_stop`, `ping`, `error`).
pub(crate) struct AnthropicStreamDecoder {
    model: String,
    tools: BTreeMap<usize, ToolBlock>,
    usage: RawUsage,
    stop_reason: Option<String>,
    done: bool,
}

/// Anthropic reports uncached input, cache reads and cache writes
/// separately; [`Usage::input_tokens`] is their sum.
#[derive(Default, Clone, Copy)]
struct RawUsage {
    input: u32,
    cache_read: u32,
    cache_write: u32,
    output: u32,
}

impl RawUsage {
    fn update(&mut self, u: &Value) {
        let get = |k: &str| u[k].as_u64().map(|n| n as u32);
        if let Some(n) = get("input_tokens") {
            self.input = n;
        }
        if let Some(n) = get("cache_read_input_tokens") {
            self.cache_read = n;
        }
        if let Some(n) = get("cache_creation_input_tokens") {
            self.cache_write = n;
        }
        if let Some(n) = get("output_tokens") {
            self.output = n;
        }
    }

    fn usage(self) -> Usage {
        Usage {
            input_tokens: self.input + self.cache_read + self.cache_write,
            output_tokens: self.output,
            cached_input_tokens: self.cache_read,
            cache_creation_input_tokens: self.cache_write,
            reasoning_tokens: 0,
        }
    }
}

impl AnthropicStreamDecoder {
    pub(crate) fn new(model: &str) -> Self {
        Self {
            model: model.into(),
            tools: BTreeMap::new(),
            usage: RawUsage::default(),
            stop_reason: None,
            done: false,
        }
    }

    fn take_usage(&mut self, u: &Value) {
        self.usage.update(u);
    }
}

impl SseDecoder for AnthropicStreamDecoder {
    fn on_event(&mut self, ev: &SseEvent) -> Result<Vec<StreamEvent>> {
        let v: Value = serde_json::from_str(&ev.data)?;
        let kind = v["type"].as_str().or(ev.event.as_deref()).unwrap_or("");
        let index = v["index"].as_u64().unwrap_or(0) as usize;
        let mut out = Vec::new();
        match kind {
            "message_start" => {
                let m = &v["message"];
                if let Some(model) = m["model"].as_str() {
                    self.model = model.to_string();
                }
                self.take_usage(&m["usage"]);
            }
            "content_block_start" => {
                let b = &v["content_block"];
                match b["type"].as_str() {
                    Some("tool_use") => {
                        let id = b["id"].as_str().unwrap_or_default().to_string();
                        let name = b["name"].as_str().unwrap_or_default().to_string();
                        out.push(StreamEvent::ToolCallDelta {
                            index,
                            id: Some(id.clone()),
                            name: Some(name.clone()),
                            arguments: String::new(),
                        });
                        self.tools.insert(
                            index,
                            ToolBlock {
                                id,
                                name,
                                input: b["input"].clone(),
                                json: String::new(),
                            },
                        );
                    }
                    Some("text") => {
                        if let Some(t) = b["text"].as_str().filter(|t| !t.is_empty()) {
                            out.push(StreamEvent::text(t));
                        }
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let d = &v["delta"];
                match d["type"].as_str() {
                    Some("text_delta") => {
                        if let Some(t) = d["text"].as_str().filter(|t| !t.is_empty()) {
                            out.push(StreamEvent::text(t));
                        }
                    }
                    Some("input_json_delta") => {
                        let part = d["partial_json"].as_str().unwrap_or_default();
                        if let Some(t) = self.tools.get_mut(&index) {
                            t.json.push_str(part);
                        }
                        out.push(StreamEvent::ToolCallDelta {
                            index,
                            id: None,
                            name: None,
                            arguments: part.to_string(),
                        });
                    }
                    _ => {} // thinking / signature deltas are not surfaced
                }
            }
            "content_block_stop" => {
                if let Some(t) = self.tools.remove(&index) {
                    let arguments = if t.json.trim().is_empty() && t.input.is_object() {
                        t.input
                    } else {
                        parse_arguments(&t.json)
                    };
                    out.push(StreamEvent::ToolCall {
                        index,
                        call: ToolCall {
                            id: t.id,
                            name: t.name,
                            arguments,
                        },
                    });
                }
            }
            "message_delta" => {
                if let Some(r) = v["delta"]["stop_reason"].as_str() {
                    self.stop_reason = Some(r.to_string());
                }
                self.take_usage(&v["usage"]);
            }
            "message_stop" => self.done = true,
            "error" => return Err(Error::Stream(v["error"].to_string())),
            _ => {} // ping and future event types
        }
        Ok(out)
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        vec![
            StreamEvent::Usage {
                usage: self.usage.usage(),
            },
            StreamEvent::Done {
                model: self.model.clone(),
                stop_reason: self.stop_reason.take(),
            },
        ]
    }

    fn is_done(&self) -> bool {
        self.done
    }
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
        usage: {
            let mut raw = RawUsage::default();
            raw.update(&v["usage"]);
            raw.usage()
        },
        model: v["model"].as_str().unwrap_or(model).to_string(),
    })
}

#[async_trait]
impl ChatModel for Anthropic {
    fn name(&self) -> &str {
        &self.model
    }
    fn capabilities(&self) -> crate::Capabilities {
        anthropic_capabilities(&self.model)
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let body = encode_request(&self.model, self.default_max_tokens, &request);
        let v = super::post_json("anthropic", self.request(), &body).await?;
        decode_response(&self.model, &v)
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let mut body = encode_request(&self.model, self.default_max_tokens, &request);
        body["stream"] = json!(true);
        let resp = super::post_sse("anthropic", self.request(), &body).await?;
        Ok(sse_stream(resp, AnthropicStreamDecoder::new(&self.model)))
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

    // Recorded Messages API stream: text block, ping, tool_use block with
    // input_json_delta fragments, message_delta with stop_reason and usage.
    const TOOL_STREAM: &str = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Okay, "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"checking."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\": \"San Fra"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"ncisco, CA\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_02","name":"now","input":{}}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}

event: message_stop
data: {"type":"message_stop"}

"#;

    #[test]
    fn stream_decoder_handles_text_tool_use_and_usage() {
        use crate::stream::{decode_all, StreamAccumulator};
        let events = decode_all(AnthropicStreamDecoder::new("x"), TOOL_STREAM).unwrap();
        assert_eq!(events[0], StreamEvent::text("Okay, "));
        let mut acc = StreamAccumulator::new();
        events.iter().for_each(|e| acc.push(e));
        assert_eq!(acc.stop_reason(), Some("tool_use"));
        let r = acc.finish();
        assert_eq!(r.message.content, "Okay, checking.");
        assert_eq!(r.model, "claude-sonnet-5-5");
        assert_eq!(r.message.tool_calls.len(), 2);
        assert_eq!(r.message.tool_calls[0].id, "toolu_01");
        assert_eq!(
            r.message.tool_calls[0].arguments,
            json!({"location": "San Francisco, CA"})
        );
        assert_eq!(r.message.tool_calls[1].arguments, json!({}));
        assert_eq!(r.usage, Usage::new(472, 89));
    }

    #[test]
    fn stream_decoder_surfaces_error_events() {
        use crate::stream::decode_all;
        let body = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";
        let err = decode_all(AnthropicStreamDecoder::new("x"), body).unwrap_err();
        assert!(err.to_string().contains("Overloaded"));
    }

    #[test]
    fn newer_models_get_auto_tool_choice_and_no_temperature() {
        let req = ChatRequest::new(vec![Message::system("be brief"), Message::user("go")])
            .with_tool_choice(ToolChoice::Tool("respond".into()))
            .with_temperature(0.2);
        for m in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"] {
            let body = encode_request(m, 10, &req);
            assert_eq!(body["tool_choice"], json!({"type": "auto"}), "{m}");
            assert!(body["system"].as_str().unwrap().contains("`respond` tool"));
            assert!(body.get("temperature").is_none(), "{m}");
        }
        let old = encode_request("claude-sonnet-4-6", 10, &req);
        assert_eq!(old["tool_choice"]["type"], "tool");
        assert!(old.get("temperature").is_some());
        let opus5 = encode_request("claude-opus-5", 10, &req);
        assert_eq!(opus5["tool_choice"]["type"], "tool");
        assert!(opus5.get("temperature").is_none());
    }

    #[test]
    fn encodes_tool_choice() {
        let req = ChatRequest::new(vec![Message::user("go")])
            .with_tool_choice(ToolChoice::Tool("respond".into()));
        let body = encode_request("m", 10, &req);
        assert_eq!(
            body["tool_choice"],
            json!({"type": "tool", "name": "respond"})
        );
    }

    #[test]
    fn encodes_images_documents_and_tool_result_images() {
        let req = ChatRequest::new(vec![
            Message::user_with_parts(
                "Summarise",
                [
                    ContentPart::pdf_bytes(b"%PDF").with_name("report"),
                    ContentPart::image_url("https://x/a.jpg"),
                    ContentPart::image_bytes(b"abc", "image/png"),
                ],
            ),
            Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "t1".into(),
                    name: "screenshot".into(),
                    arguments: json!({}),
                }],
            ),
            Message::tool_result("t1", "here")
                .with_part(ContentPart::image_base64("image/jpeg", "Zm9v")),
        ]);
        let body = encode_request("claude-sonnet-5-5", 1024, &req);
        assert_eq!(
            body["messages"][0]["content"],
            json!([
                {"type": "text", "text": "Summarise"},
                {"type": "document", "title": "report", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERg=="}},
                {"type": "image", "source": {"type": "url", "url": "https://x/a.jpg"}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"}}
            ])
        );
        assert_eq!(
            body["messages"][2]["content"][0]["content"],
            json!([
                {"type": "text", "text": "here"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "Zm9v"}}
            ])
        );
        // Text-only messages are unchanged.
        let plain = encode_request("m", 1, &ChatRequest::new(vec![Message::user("hi")]));
        assert_eq!(
            plain["messages"][0]["content"],
            json!([{"type": "text", "text": "hi"}])
        );
    }

    #[test]
    fn cache_tokens_are_folded_into_input() {
        let v = json!({"content": [{"type": "text", "text": "ok"}], "usage": {
            "input_tokens": 10, "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 200, "output_tokens": 5}});
        let u = decode_response("m", &v).unwrap().usage;
        assert_eq!(u.input_tokens, 1210);
        assert_eq!(u.cached_input_tokens, 1000);
        assert_eq!(u.cache_creation_input_tokens, 200);
        assert_eq!(u.output_tokens, 5);
    }
}
