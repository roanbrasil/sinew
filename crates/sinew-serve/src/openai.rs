//! OpenAI-compatible façade: `GET /v1/models` and `POST /v1/chat/completions`
//! (streaming and not), backed by registered agents or plain chat models.

use std::{sync::Arc, time::SystemTime};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::{BufMut, BytesMut};
use futures_util::{stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use sinew_core::{
    BoxStream, ChatModel, ChatRequest, Message, Role, StreamEvent, ToolCall, ToolSpec, Usage,
};

use crate::{
    assistant::{ChatPiece, ErasedGraph},
    server::App,
    sse::{write_json_str, FrameWriter},
};

/// What answers a model name on the façade.
#[derive(Clone)]
pub(crate) enum ChatBackend {
    Model(Arc<dyn ChatModel>),
    Graph(Arc<dyn ErasedGraph>),
}

#[derive(Deserialize)]
struct CompletionRequest {
    model: String,
    messages: Vec<OaMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    tools: Vec<OaTool>,
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
struct OaMessage {
    role: String,
    #[serde(default)]
    content: Option<OaContent>,
    #[serde(default)]
    tool_calls: Vec<OaToolCall>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OaContent {
    Text(String),
    Parts(Vec<OaPart>),
}

#[derive(Deserialize)]
struct OaPart {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct OaToolCall {
    id: String,
    function: OaFunction,
}

#[derive(Deserialize)]
struct OaFunction {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct OaTool {
    function: OaToolFn,
}

#[derive(Deserialize)]
struct OaToolFn {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<Value>,
}

fn convert(m: OaMessage) -> Result<Message, String> {
    let content = match m.content {
        Some(OaContent::Text(s)) => s,
        Some(OaContent::Parts(parts)) => parts
            .into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>()
            .join(""),
        None => String::new(),
    };
    Ok(match m.role.as_str() {
        "system" | "developer" => Message::system(content),
        "user" => Message::user(content),
        "assistant" => {
            let calls = m
                .tool_calls
                .into_iter()
                .map(|c| ToolCall {
                    id: c.id,
                    arguments: serde_json::from_str(&c.function.arguments)
                        .unwrap_or(Value::String(c.function.arguments)),
                    name: c.function.name,
                })
                .collect();
            Message::assistant_with_tools(content, calls)
        }
        "tool" => Message::tool_result(m.tool_call_id.unwrap_or_default(), content),
        other => return Err(format!("unsupported role `{other}`")),
    })
}

/// OpenAI-style error body.
pub(crate) fn oa_error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": {"message": message.into(), "type": kind, "code": Value::Null}})),
    )
        .into_response()
}

pub(crate) async fn list_models(State(app): State<App>) -> Json<Value> {
    let data: Vec<Value> = app
        .chat
        .keys()
        .map(|id| json!({"id": id, "object": "model", "created": 0, "owned_by": "sinew"}))
        .collect();
    Json(json!({"object": "list", "data": data}))
}

/// Pieces of a reply, whatever produced it.
enum Piece {
    Text(String),
    Tool(usize, ToolCall),
    Usage(Usage),
    Finish(Option<String>),
    Error(String),
}

fn model_pieces(s: sinew_core::ChatStream) -> BoxStream<'static, Piece> {
    Box::pin(s.filter_map(|ev| {
        futures_util::future::ready(match ev {
            Ok(StreamEvent::TextDelta { text }) => Some(Piece::Text(text)),
            Ok(StreamEvent::ToolCall { index, call }) => Some(Piece::Tool(index, call)),
            Ok(StreamEvent::Usage { usage }) => Some(Piece::Usage(usage)),
            Ok(StreamEvent::Done { stop_reason, .. }) => Some(Piece::Finish(stop_reason)),
            Ok(StreamEvent::ToolCallDelta { .. }) => None,
            Err(e) => Some(Piece::Error(e.to_string())),
        })
    }))
}

fn graph_pieces(s: BoxStream<'static, ChatPiece>) -> BoxStream<'static, Piece> {
    Box::pin(s.flat_map(|p| {
        stream::iter(match p {
            ChatPiece::Delta(d) => vec![Piece::Text(d)],
            ChatPiece::Done(Ok(m)) => {
                let mut v: Vec<Piece> = m
                    .tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(i, c)| Piece::Tool(i, c))
                    .collect();
                v.push(Piece::Finish(None));
                v
            }
            ChatPiece::Done(Err(e)) => vec![Piece::Error(e)],
        })
    }))
}

/// Encodes `chat.completion.chunk`s. The constant part of every chunk is
/// rendered once; per token only the escaped delta is written.
struct ChunkEncoder {
    w: FrameWriter,
    /// `{"id":..,"object":"chat.completion.chunk","created":..,"model":..,"choices":[`
    head: Vec<u8>,
    include_usage: bool,
    saw_tools: bool,
    usage: Usage,
}

impl ChunkEncoder {
    fn new(id: &str, created: u64, model: &str, include_usage: bool) -> Self {
        let mut head = BytesMut::new();
        head.put_slice(b"{\"id\":");
        write_json_str(&mut head, id);
        head.put_slice(b",\"object\":\"chat.completion.chunk\",\"created\":");
        head.put_slice(created.to_string().as_bytes());
        head.put_slice(b",\"model\":");
        write_json_str(&mut head, model);
        head.put_slice(b",\"choices\":[");
        Self {
            w: FrameWriter::default(),
            head: head.to_vec(),
            include_usage,
            saw_tools: false,
            usage: Usage::default(),
        }
    }

    fn choice(&mut self, delta: impl FnOnce(&mut BytesMut), finish: Option<&str>) -> Bytes {
        let head = &self.head;
        self.w.data_with(|b| {
            b.put_slice(head);
            b.put_slice(b"{\"index\":0,\"delta\":");
            delta(b);
            b.put_slice(b",\"finish_reason\":");
            match finish {
                Some(f) => write_json_str(b, f),
                None => b.put_slice(b"null"),
            }
            b.put_slice(b"}]}");
        })
    }

    fn role(&mut self) -> Bytes {
        self.choice(
            |b| b.put_slice(b"{\"role\":\"assistant\",\"content\":\"\"}"),
            None,
        )
    }

    /// Frames for one piece; empty when there is nothing to send. Returns
    /// `true` as second value when the stream is over.
    fn piece(&mut self, p: Piece) -> (Bytes, bool) {
        match p {
            Piece::Text(t) if t.is_empty() => (Bytes::new(), false),
            Piece::Text(t) => (
                self.choice(
                    |b| {
                        b.put_slice(b"{\"content\":");
                        write_json_str(b, &t);
                        b.put_u8(b'}');
                    },
                    None,
                ),
                false,
            ),
            Piece::Tool(index, call) => {
                self.saw_tools = true;
                let delta = json!({"tool_calls": [{
                    "index": index, "id": call.id, "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments.to_string()}
                }]});
                (
                    self.choice(
                        |b| {
                            let _ = serde_json::to_writer(b.writer(), &delta);
                        },
                        None,
                    ),
                    false,
                )
            }
            Piece::Usage(u) => {
                self.usage = u;
                (Bytes::new(), false)
            }
            Piece::Finish(reason) => {
                let reason = if self.saw_tools {
                    "tool_calls"
                } else if matches!(reason.as_deref(), Some("length" | "max_tokens")) {
                    "length"
                } else {
                    "stop"
                };
                let mut out = BytesMut::new();
                out.put_slice(&self.choice(|b| b.put_slice(b"{}"), Some(reason)));
                if self.include_usage {
                    let head = &self.head;
                    let u = self.usage;
                    out.put_slice(&self.w.data_with(|b| {
                        b.put_slice(head);
                        b.put_slice(b"],\"usage\":");
                        let _ = serde_json::to_writer(b.writer(), &usage_json(u));
                        b.put_u8(b'}');
                    }));
                }
                out.put_slice(b"data: [DONE]\n\n");
                (out.freeze(), true)
            }
            Piece::Error(e) => {
                let mut out = BytesMut::new();
                out.put_slice(&self.w.data_with(|b| {
                    let _ = serde_json::to_writer(
                        b.writer(),
                        &json!({"error": {"message": e, "type": "server_error"}}),
                    );
                }));
                out.put_slice(b"data: [DONE]\n\n");
                (out.freeze(), true)
            }
        }
    }
}

fn usage_json(u: Usage) -> Value {
    json!({
        "prompt_tokens": u.input_tokens,
        "completion_tokens": u.output_tokens,
        "total_tokens": u.input_tokens + u.output_tokens,
    })
}

fn sse_response(pieces: BoxStream<'static, Piece>, mut enc: ChunkEncoder) -> Response {
    let first = enc.role();
    let body = stream::once(futures_util::future::ready(first))
        .chain(stream::unfold(
            (pieces, enc, false),
            |(mut pieces, mut enc, done)| async move {
                if done {
                    return None;
                }
                loop {
                    let (bytes, end) = match pieces.next().await {
                        Some(p) => enc.piece(p),
                        // The source ended without a finish marker.
                        None => enc.piece(Piece::Finish(None)),
                    };
                    if !bytes.is_empty() || end {
                        return Some((bytes, (pieces, enc, end)));
                    }
                }
            },
        ))
        .map(Ok::<_, std::convert::Infallible>);
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(body))
        .unwrap()
}

pub(crate) async fn chat_completions(State(app): State<App>, body: Bytes) -> Response {
    let req: CompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return oa_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("invalid request: {e}"),
            )
        }
    };
    let Some(backend) = app.chat.get(&req.model).cloned() else {
        return oa_error(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            format!("model `{}` does not exist", req.model),
        );
    };
    let messages = match req.messages.into_iter().map(convert).collect() {
        Ok(m) => m,
        Err(e) => return oa_error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let include_usage = req.stream_options.is_some_and(|o| o.include_usage);

    match backend {
        ChatBackend::Model(model) => {
            let mut creq = ChatRequest::new(messages);
            creq.temperature = req.temperature;
            creq.max_tokens = req.max_tokens.or(req.max_completion_tokens);
            creq.tools = req
                .tools
                .into_iter()
                .map(|t| ToolSpec {
                    name: t.function.name,
                    description: t.function.description.unwrap_or_default(),
                    parameters: t
                        .function
                        .parameters
                        .unwrap_or_else(|| json!({"type": "object"})),
                })
                .collect();
            if req.stream {
                match model.chat_stream(creq).await {
                    Ok(s) => sse_response(
                        model_pieces(s),
                        ChunkEncoder::new(&id, created, &req.model, include_usage),
                    ),
                    Err(e) => oa_error(StatusCode::BAD_GATEWAY, "server_error", e.to_string()),
                }
            } else {
                match model.chat(creq).await {
                    Ok(r) => completion(&id, created, &req.model, &r.message, r.usage),
                    Err(e) => oa_error(StatusCode::BAD_GATEWAY, "server_error", e.to_string()),
                }
            }
        }
        ChatBackend::Graph(graph) => {
            let Some(mut s) = graph.chat(messages, req.stream) else {
                return oa_error(
                    StatusCode::NOT_FOUND,
                    "invalid_request_error",
                    "assistant is not chat-capable",
                );
            };
            if req.stream {
                return sse_response(
                    graph_pieces(s),
                    ChunkEncoder::new(&id, created, &req.model, include_usage),
                );
            }
            while let Some(p) = s.next().await {
                if let ChatPiece::Done(res) = p {
                    return match res {
                        Ok(m) => completion(&id, created, &req.model, &m, Usage::default()),
                        Err(e) => oa_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e),
                    };
                }
            }
            oa_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "agent produced no reply",
            )
        }
    }
}

fn completion(id: &str, created: u64, model: &str, m: &Message, usage: Usage) -> Response {
    let mut message = json!({"role": "assistant", "content": m.content});
    if m.role != Role::Assistant {
        message["role"] = json!("assistant");
    }
    if !m.tool_calls.is_empty() {
        message["tool_calls"] = m
            .tool_calls
            .iter()
            .map(|c| {
                json!({"id": c.id, "type": "function",
                       "function": {"name": c.name, "arguments": c.arguments.to_string()}})
            })
            .collect();
    }
    let finish = if m.tool_calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage_json(usage),
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_openai_messages() {
        let raw = json!([
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "2"}
        ]);
        let msgs: Vec<OaMessage> = serde_json::from_value(raw).unwrap();
        let m: Vec<Message> = msgs.into_iter().map(|m| convert(m).unwrap()).collect();
        assert_eq!(m[0].role, Role::System);
        assert_eq!(m[1].content, "hi there");
        assert_eq!(m[2].tool_calls[0].arguments, json!({"a": 1}));
        assert_eq!(m[3].tool_call_id.as_deref(), Some("c1"));
    }

    #[test]
    fn chunks_are_valid_json() {
        let mut enc = ChunkEncoder::new("chatcmpl-1", 7, "m\"x", true);
        let role = enc.role();
        let (text, _) = enc.piece(Piece::Text("a \"q\"\n".into()));
        let (_, _) = enc.piece(Piece::Usage(Usage {
            input_tokens: 2,
            output_tokens: 3,
        }));
        let (fin, end) = enc.piece(Piece::Finish(None));
        assert!(end);
        let all = [role, text, fin].concat();
        let all = String::from_utf8(all).unwrap();
        let mut datas = all
            .split("\n\n")
            .filter_map(|f| f.strip_prefix("data: "))
            .collect::<Vec<_>>();
        assert_eq!(datas.pop(), Some("[DONE]"));
        let parsed: Vec<Value> = datas
            .iter()
            .map(|d| serde_json::from_str(d).unwrap())
            .collect();
        assert_eq!(parsed[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(parsed[1]["choices"][0]["delta"]["content"], "a \"q\"\n");
        assert_eq!(parsed[1]["model"], "m\"x");
        assert_eq!(parsed[2]["choices"][0]["finish_reason"], "stop");
        assert_eq!(parsed[3]["usage"]["total_tokens"], 5);
        assert_eq!(parsed[3]["choices"], json!([]));
    }
}
