//! Native Google Gemini API client (`generateContent` /
//! `streamGenerateContent`).
//!
//! Works with Google AI Studio keys out of the box. For Vertex AI, point
//! [`Gemini::with_base_url`] at your regional endpoint and pass an access
//! token with [`Gemini::with_bearer_token`].

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::{json, Map, Value};

use crate::{
    message::{ContentPart, MediaSource},
    stream::{sse_stream, ChatStream, SseDecoder, SseEvent, StreamEvent},
    ChatModel, ChatRequest, ChatResponse, Error, Message, ResponseFormat, Result, Role, ToolCall,
    ToolChoice, Usage,
};

/// Separator between a call id and the Gemini thought signature carried
/// inside [`ToolCall::id`]. Gemini 3 models reject a follow-up turn whose
/// function calls lost their signature, so it rides along in the id and is
/// put back when the conversation is sent again.
const SIG: &str = "#ts=";

#[derive(Clone)]
pub struct Gemini {
    base_url: String,
    auth: Auth,
    model: String,
    client: reqwest::Client,
}

#[derive(Clone)]
enum Auth {
    ApiKey(String),
    Bearer(String),
}

impl Gemini {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
            auth: Auth::ApiKey(api_key.into()),
            model: model.into(),
            client: super::shared_client(),
        }
    }

    /// Uses this HTTP client instead of the [shared](super::shared_client) one.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| Error::Config("GEMINI_API_KEY not set".into()))?;
        Ok(Self::new(key, model))
    }

    /// Base URL up to (not including) `/models`, e.g. a Vertex AI
    /// `https://{region}-aiplatform.googleapis.com/v1/projects/{p}/locations/{region}/publishers/google`.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// OAuth access token instead of an API key (Vertex AI).
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.auth = Auth::Bearer(token.into());
        self
    }

    fn request(&self, method: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/models/{}:{method}", self.base_url, self.model);
        let req = self.client.post(url);
        match &self.auth {
            Auth::ApiKey(k) => req.header("x-goog-api-key", k),
            Auth::Bearer(t) => req.bearer_auth(t),
        }
    }
}

fn split_id(id: &str) -> (&str, Option<&str>) {
    match id.split_once(SIG) {
        Some((base, sig)) => (base, Some(sig)),
        None => (id, None),
    }
}

/// Tool results must be objects; plain text is wrapped as `{"result": ...}`.
fn tool_response(content: &str) -> Value {
    match serde_json::from_str::<Value>(content) {
        Ok(v @ Value::Object(_)) => v,
        Ok(other) => json!({ "result": other }),
        Err(_) => json!({ "result": content }),
    }
}

/// Drops JSON Schema keys the Gemini schema dialect does not accept.
fn clean_schema(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, val) in map {
                if matches!(
                    k.as_str(),
                    "$schema" | "$id" | "title" | "additionalProperties"
                ) {
                    continue;
                }
                out.insert(k.clone(), clean_schema(val));
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(clean_schema).collect()),
        other => other.clone(),
    }
}

/// `inlineData` for inline bytes, `fileData` for URIs (Files API, `gs://`,
/// public URLs). Gemini needs a MIME type for both; URLs without one get a
/// guess from the extension.
fn encode_part(p: &ContentPart) -> Value {
    let source = match p {
        ContentPart::Text { text } => return json!({ "text": text }),
        ContentPart::Image { source, .. }
        | ContentPart::Document { source, .. }
        | ContentPart::Audio { source } => source,
    };
    match source {
        MediaSource::Base64 { media_type, data } => {
            json!({"inlineData": {"mimeType": media_type, "data": data}})
        }
        MediaSource::Url { url, .. } => {
            let fallback = match p {
                ContentPart::Image { .. } => "image/jpeg",
                ContentPart::Audio { .. } => "audio/mpeg",
                _ => "application/pdf",
            };
            let mime = source.media_type().unwrap_or_else(|| fallback.into());
            json!({"fileData": {"mimeType": mime, "fileUri": url}})
        }
    }
}

/// Text part (if any) followed by the message's parts.
fn user_parts(m: &Message) -> Vec<Value> {
    let mut parts = Vec::with_capacity(m.parts.len() + 1);
    if !m.content.is_empty() || m.parts.is_empty() {
        parts.push(json!({"text": m.content}));
    }
    parts.extend(m.parts.iter().map(encode_part));
    parts
}

pub(crate) fn encode_request(req: &ChatRequest) -> Value {
    let system: Vec<String> = req
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(Message::text)
        .collect();

    // functionResponse needs the function name; the tool message only has the
    // call id, so remember which name each id belonged to.
    let mut names: HashMap<&str, &str> = HashMap::new();
    let mut contents: Vec<Value> = Vec::new();
    for m in req.messages.iter().filter(|m| m.role != Role::System) {
        match m.role {
            Role::Assistant => {
                let mut parts = Vec::new();
                if !m.content.is_empty() {
                    parts.push(json!({"text": m.content}));
                }
                parts.extend(m.parts.iter().map(encode_part));
                for c in &m.tool_calls {
                    names.insert(c.id.as_str(), c.name.as_str());
                    let mut part = json!({"functionCall": {"name": c.name, "args": c.arguments}});
                    if let (_, Some(sig)) = split_id(&c.id) {
                        part["thoughtSignature"] = json!(sig);
                    }
                    parts.push(part);
                }
                if parts.is_empty() {
                    parts.push(json!({"text": ""}));
                }
                contents.push(json!({"role": "model", "parts": parts}));
            }
            Role::Tool => {
                let id = m.tool_call_id.as_deref().unwrap_or_default();
                let name = names.get(id).copied().unwrap_or(id);
                let part = json!({"functionResponse": {"name": name, "response": tool_response(&m.content)}});
                // Parallel results go back together in one user turn; media
                // attached to a tool result follows as extra parts.
                let merged = contents.last_mut().filter(|c| {
                    c["role"] == "user"
                        && c["parts"]
                            .as_array()
                            .is_some_and(|p| p.iter().all(|x| x.get("functionResponse").is_some()))
                });
                let extra = m.parts.iter().map(encode_part);
                match merged {
                    Some(c) => {
                        let arr = c["parts"].as_array_mut().unwrap();
                        arr.push(part);
                        arr.extend(extra);
                    }
                    None => {
                        let mut parts = vec![part];
                        parts.extend(extra);
                        contents.push(json!({"role": "user", "parts": parts}));
                    }
                }
            }
            _ => contents.push(json!({"role": "user", "parts": user_parts(m)})),
        }
    }

    let mut body = json!({ "contents": contents });
    if !system.is_empty() {
        body["systemInstruction"] = json!({"parts": [{"text": system.join("\n\n")}]});
    }
    if !req.tools.is_empty() {
        let decls: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({"name": t.name, "description": t.description, "parameters": clean_schema(&t.parameters)})
            })
            .collect();
        body["tools"] = json!([{ "functionDeclarations": decls }]);
    }
    if let Some(choice) = &req.tool_choice {
        body["toolConfig"] = json!({"functionCallingConfig": match choice {
            ToolChoice::Auto => json!({"mode": "AUTO"}),
            ToolChoice::Required => json!({"mode": "ANY"}),
            ToolChoice::Tool(name) => json!({"mode": "ANY", "allowedFunctionNames": [name]}),
        }});
    }
    let mut gen = Map::new();
    if let Some(t) = req.temperature {
        gen.insert("temperature".into(), json!(t));
    }
    if let Some(n) = req.max_tokens {
        gen.insert("maxOutputTokens".into(), json!(n));
    }
    match &req.response_format {
        Some(ResponseFormat::Json) => {
            gen.insert("responseMimeType".into(), json!("application/json"));
        }
        Some(ResponseFormat::JsonSchema { schema, .. }) => {
            gen.insert("responseMimeType".into(), json!("application/json"));
            gen.insert("responseSchema".into(), clean_schema(schema));
        }
        None => {}
    }
    if !gen.is_empty() {
        body["generationConfig"] = Value::Object(gen);
    }
    body
}

fn usage_of(v: &Value) -> Usage {
    let u = &v["usageMetadata"];
    let n = |k: &str| u[k].as_u64().unwrap_or(0) as u32;
    Usage {
        input_tokens: n("promptTokenCount") + n("toolUsePromptTokenCount"),
        // Thinking tokens are billed as output.
        output_tokens: n("candidatesTokenCount") + n("thoughtsTokenCount"),
        cached_input_tokens: n("cachedContentTokenCount"),
        cache_creation_input_tokens: 0,
        reasoning_tokens: n("thoughtsTokenCount"),
    }
}

/// Text and function calls of the first candidate. Thought summaries
/// (`"thought": true`) are skipped.
fn parts_of(v: &Value, first_index: usize) -> (String, Vec<ToolCall>) {
    let mut text = String::new();
    let mut calls = Vec::new();
    let parts = v
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array);
    for p in parts.into_iter().flatten() {
        if p["thought"].as_bool() == Some(true) {
            continue;
        }
        if let Some(t) = p["text"].as_str() {
            text.push_str(t);
        }
        if let Some(fc) = p.get("functionCall") {
            let index = first_index + calls.len();
            let base = fc["id"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| format!("call_{index}"));
            let id = match p["thoughtSignature"].as_str() {
                Some(sig) => format!("{base}{SIG}{sig}"),
                None => base,
            };
            calls.push(ToolCall {
                id,
                name: fc["name"].as_str().unwrap_or_default().to_string(),
                arguments: match &fc["args"] {
                    Value::Null => json!({}),
                    a => a.clone(),
                },
            });
        }
    }
    (text, calls)
}

fn check_blocked(v: &Value) -> Result<()> {
    if let Some(reason) = v
        .pointer("/promptFeedback/blockReason")
        .and_then(Value::as_str)
    {
        return Err(Error::InvalidResponse(format!("prompt blocked: {reason}")));
    }
    Ok(())
}

pub(crate) fn decode_response(model: &str, v: &Value) -> Result<ChatResponse> {
    check_blocked(v)?;
    if v["candidates"].as_array().is_none_or(|c| c.is_empty()) {
        return Err(Error::InvalidResponse("no candidates".into()));
    }
    let (text, calls) = parts_of(v, 0);
    Ok(ChatResponse {
        message: Message::assistant_with_tools(text, calls),
        usage: usage_of(v),
        model: v["modelVersion"].as_str().unwrap_or(model).to_string(),
    })
}

/// Each SSE event is a full `GenerateContentResponse` chunk. Text arrives in
/// pieces; function calls arrive whole, so they are emitted as complete
/// `ToolCall` events.
pub(crate) struct GeminiStreamDecoder {
    model: String,
    calls: usize,
    usage: Usage,
    stop_reason: Option<String>,
}

impl GeminiStreamDecoder {
    pub(crate) fn new(model: &str) -> Self {
        Self {
            model: model.into(),
            calls: 0,
            usage: Usage::default(),
            stop_reason: None,
        }
    }
}

impl SseDecoder for GeminiStreamDecoder {
    fn on_event(&mut self, ev: &SseEvent) -> Result<Vec<StreamEvent>> {
        let v: Value = serde_json::from_str(&ev.data)?;
        if let Some(err) = v.get("error") {
            return Err(Error::Stream(err.to_string()));
        }
        check_blocked(&v)?;
        let mut out = Vec::new();
        let (text, calls) = parts_of(&v, self.calls);
        if !text.is_empty() {
            out.push(StreamEvent::text(text));
        }
        for call in calls {
            out.push(StreamEvent::ToolCall {
                index: self.calls,
                call,
            });
            self.calls += 1;
        }
        if v.get("usageMetadata").is_some() {
            self.usage = usage_of(&v);
        }
        if let Some(m) = v["modelVersion"].as_str() {
            self.model = m.to_string();
        }
        if let Some(r) = v
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
        {
            self.stop_reason = Some(r.to_string());
        }
        Ok(out)
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        vec![
            StreamEvent::Usage { usage: self.usage },
            StreamEvent::Done {
                model: self.model.clone(),
                stop_reason: self.stop_reason.take(),
            },
        ]
    }

    fn is_done(&self) -> bool {
        false // the stream ends when the body does
    }
}

#[async_trait]
impl ChatModel for Gemini {
    fn name(&self) -> &str {
        &self.model
    }
    fn capabilities(&self) -> crate::Capabilities {
        crate::Capabilities::all()
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let body = encode_request(&request);
        let v = super::post_json("gemini", self.request("generateContent"), &body).await?;
        decode_response(&self.model, &v)
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let body = encode_request(&request);
        let req = self
            .request("streamGenerateContent")
            .query(&[("alt", "sse")]);
        let resp = super::post_sse("gemini", req, &body).await?;
        Ok(sse_stream(resp, GeminiStreamDecoder::new(&self.model)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{decode_all, StreamAccumulator};
    use crate::ToolSpec;

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: json!({"city": "Rio"}),
        }
    }

    #[test]
    fn encodes_system_roles_tools_and_results() {
        let req = ChatRequest::new(vec![
            Message::system("be brief"),
            Message::user("weather?"),
            Message::assistant_with_tools(
                "",
                vec![call("call_0#ts=SIGA", "weather"), call("call_1", "time")],
            ),
            Message::tool_result("call_0#ts=SIGA", r#"{"temp": 31}"#),
            Message::tool_result("call_1", "noon"),
        ])
        .with_tools(vec![ToolSpec {
            name: "weather".into(),
            description: "Weather".into(),
            parameters: json!({"$schema": "x", "title": "W", "type": "object",
                "additionalProperties": false,
                "properties": {"city": {"type": "string", "title": "City"}}}),
        }])
        .with_tool_choice(ToolChoice::Tool("weather".into()))
        .with_temperature(0.2)
        .with_max_tokens(100);
        let b = encode_request(&req);

        assert_eq!(b["systemInstruction"]["parts"][0]["text"], "be brief");
        let c = b["contents"].as_array().unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c[0]["role"], "user");
        assert_eq!(c[1]["role"], "model");
        let fc = &c[1]["parts"][0];
        assert_eq!(fc["functionCall"]["name"], "weather");
        assert_eq!(fc["thoughtSignature"], "SIGA");
        assert!(c[1]["parts"][1].get("thoughtSignature").is_none());
        // both results merged into one user turn, mapped back to names
        let parts = c[2]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["functionResponse"]["name"], "weather");
        assert_eq!(
            parts[0]["functionResponse"]["response"],
            json!({"temp": 31})
        );
        assert_eq!(parts[1]["functionResponse"]["name"], "time");
        assert_eq!(
            parts[1]["functionResponse"]["response"],
            json!({"result": "noon"})
        );

        let decl = &b["tools"][0]["functionDeclarations"][0];
        assert_eq!(decl["name"], "weather");
        assert!(decl["parameters"].get("$schema").is_none());
        assert!(decl["parameters"].get("additionalProperties").is_none());
        assert!(decl["parameters"]["properties"]["city"]
            .get("title")
            .is_none());
        assert_eq!(
            b["toolConfig"]["functionCallingConfig"],
            json!({"mode": "ANY", "allowedFunctionNames": ["weather"]})
        );
        assert_eq!(b["generationConfig"]["maxOutputTokens"], 100);
    }

    #[test]
    fn encodes_json_schema_format() {
        let req = ChatRequest::new(vec![Message::user("x")]).with_response_format(
            ResponseFormat::JsonSchema {
                name: "r".into(),
                schema: json!({"type": "object", "title": "R"}),
            },
        );
        let g = &encode_request(&req)["generationConfig"];
        assert_eq!(g["responseMimeType"], "application/json");
        assert_eq!(g["responseSchema"], json!({"type": "object"}));
    }

    #[test]
    fn decodes_text_calls_signature_and_usage() {
        let v = json!({
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": "thinking...", "thought": true},
                {"text": "Checking."},
                {"functionCall": {"name": "weather", "args": {"city": "Rio"}}, "thoughtSignature": "S1"},
                {"functionCall": {"name": "now"}}
            ]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 7, "thoughtsTokenCount": 3},
            "modelVersion": "gemini-2.5-flash"
        });
        let r = decode_response("x", &v).unwrap();
        assert_eq!(r.message.content, "Checking.");
        assert_eq!(r.message.tool_calls.len(), 2);
        assert_eq!(r.message.tool_calls[0].id, "call_0#ts=S1");
        assert_eq!(r.message.tool_calls[1].id, "call_1");
        assert_eq!(r.message.tool_calls[1].arguments, json!({}));
        assert_eq!(
            r.usage,
            Usage {
                input_tokens: 12,
                output_tokens: 10,
                reasoning_tokens: 3,
                ..Default::default()
            }
        );
        assert_eq!(r.model, "gemini-2.5-flash");
    }

    #[test]
    fn blocked_prompt_is_an_error() {
        let v = json!({"promptFeedback": {"blockReason": "SAFETY"}});
        let e = decode_response("x", &v).unwrap_err();
        assert!(e.to_string().contains("SAFETY"));
    }

    #[test]
    fn signature_survives_a_round_trip() {
        let v = json!({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "abc+/="}
        ]}}]});
        let r = decode_response("x", &v).unwrap();
        let req = ChatRequest::new(vec![Message::user("go"), r.message]);
        let b = encode_request(&req);
        assert_eq!(b["contents"][1]["parts"][0]["thoughtSignature"], "abc+/=");
    }

    // Recorded `streamGenerateContent?alt=sse` body.
    const STREAM: &str = r#"data: {"candidates": [{"content": {"parts": [{"text": "The weather"}],"role": "model"},"index": 0}],"usageMetadata": {"promptTokenCount": 30,"totalTokenCount": 30},"modelVersion": "gemini-2.5-flash"}

data: {"candidates": [{"content": {"parts": [{"text": " in Rio:"}],"role": "model"},"index": 0}],"modelVersion": "gemini-2.5-flash"}

data: {"candidates": [{"content": {"parts": [{"functionCall": {"name": "weather","args": {"city": "Rio"}},"thoughtSignature": "SIG"}],"role": "model"},"finishReason": "STOP","index": 0}],"usageMetadata": {"promptTokenCount": 30,"candidatesTokenCount": 9,"totalTokenCount": 39},"modelVersion": "gemini-2.5-flash"}

"#;

    #[test]
    fn stream_decoder_assembles_text_and_calls() {
        let events = decode_all(GeminiStreamDecoder::new("x"), STREAM).unwrap();
        assert_eq!(events[0], StreamEvent::text("The weather"));
        let mut acc = StreamAccumulator::new();
        events.iter().for_each(|e| acc.push(e));
        assert_eq!(acc.stop_reason(), Some("STOP"));
        let r = acc.finish();
        assert_eq!(r.message.content, "The weather in Rio:");
        assert_eq!(r.message.tool_calls[0].name, "weather");
        assert_eq!(r.message.tool_calls[0].id, "call_0#ts=SIG");
        assert_eq!(r.usage, Usage::new(30, 9));
        assert_eq!(r.model, "gemini-2.5-flash");
    }

    #[test]
    fn stream_decoder_surfaces_errors() {
        let body = "data: {\"error\": {\"code\": 429, \"message\": \"quota\"}}\n\n";
        let err = decode_all(GeminiStreamDecoder::new("x"), body).unwrap_err();
        assert!(err.to_string().contains("quota"));
    }

    #[test]
    fn encodes_inline_and_file_data() {
        let req = ChatRequest::new(vec![Message::user_with_parts(
            "Describe",
            [
                ContentPart::image_bytes(b"abc", "image/png"),
                ContentPart::image_url("https://x/photo.webp"),
                ContentPart::pdf_url("gs://bucket/doc"),
                ContentPart::audio_bytes(b"ID3", "audio/mpeg"),
                ContentPart::file_uri(
                    "https://generativelanguage.googleapis.com/v1beta/files/abc",
                    "video/mp4",
                ),
            ],
        )]);
        let body = encode_request(&req);
        assert_eq!(
            body["contents"][0]["parts"],
            json!([
                {"text": "Describe"},
                {"inlineData": {"mimeType": "image/png", "data": "YWJj"}},
                {"fileData": {"mimeType": "image/webp", "fileUri": "https://x/photo.webp"}},
                {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/doc"}},
                {"inlineData": {"mimeType": "audio/mpeg", "data": "SUQz"}},
                {"fileData": {"mimeType": "video/mp4", "fileUri": "https://generativelanguage.googleapis.com/v1beta/files/abc"}}
            ])
        );
    }

    #[test]
    fn decodes_cached_tokens() {
        let v = json!({"candidates": [{"content": {"parts": [{"text": "x"}]}}],
            "usageMetadata": {"promptTokenCount": 100, "cachedContentTokenCount": 80, "candidatesTokenCount": 2}});
        let u = decode_response("x", &v).unwrap().usage;
        assert_eq!((u.input_tokens, u.cached_input_tokens), (100, 80));
    }
}
