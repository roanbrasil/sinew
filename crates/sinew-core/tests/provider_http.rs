//! Provider HTTP behaviour against a local mock server: status mapping,
//! malformed bodies, streams that fail midway, and request shape (auth
//! headers, paths, query). Fully offline.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use futures_util::StreamExt;
use serde_json::{json, Value};
use sinew_core::{
    batch::{run_batch, BatchModel},
    providers::{Anthropic, Gemini, OpenAiCompat},
    ChatModel, ChatRequest, Error, Message, StreamEvent,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

/// One canned HTTP response.
#[derive(Clone)]
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    /// Advertise more bytes than are sent, then close: the client sees the
    /// body end early (a dropped connection mid-stream).
    truncate: bool,
}

impl Reply {
    fn json(status: u16, v: Value) -> Self {
        Self {
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: v.to_string().into_bytes(),
            truncate: false,
        }
    }
    fn text(status: u16, body: &str) -> Self {
        Self {
            status,
            headers: vec![("content-type".into(), "text/plain".into())],
            body: body.as_bytes().to_vec(),
            truncate: false,
        }
    }
    fn sse(events: &[&str]) -> Self {
        let body: String = events.iter().map(|e| format!("{e}\n\n")).collect();
        Self {
            status: 200,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            body: body.into_bytes(),
            truncate: false,
        }
    }
    fn header(mut self, k: &str, v: &str) -> Self {
        self.headers.push((k.into(), v.into()));
        self
    }
    fn truncated(mut self) -> Self {
        self.truncate = true;
        self
    }
}

/// What the server received.
#[derive(Debug, Clone)]
struct Seen {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

/// Serves `replies` in order, one per connection, and records requests.
async fn mock(replies: Vec<Reply>) -> (String, Arc<Mutex<Vec<Seen>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    tokio::spawn(async move {
        for reply in replies {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let req = read_request(&mut sock).await;
            log.lock().unwrap().push(req);
            let reason = if reply.status < 400 { "OK" } else { "ERR" };
            let len = reply.body.len() + if reply.truncate { 1000 } else { 0 };
            let mut head = format!(
                "HTTP/1.1 {} {reason}\r\ncontent-length: {len}\r\nconnection: close\r\n",
                reply.status
            );
            for (k, v) in &reply.headers {
                head.push_str(&format!("{k}: {v}\r\n"));
            }
            head.push_str("\r\n");
            let _ = sock.write_all(head.as_bytes()).await;
            let _ = sock.write_all(&reply.body).await;
            let _ = sock.flush().await;
            let _ = sock.shutdown().await;
        }
    });
    (url, seen)
}

async fn read_request(sock: &mut tokio::net::TcpStream) -> Seen {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = sock.read(&mut chunk).await.unwrap();
        assert!(n > 0, "client closed before sending headers");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap().split(' ');
    let method = first.next().unwrap().to_string();
    let path = first.next().unwrap().to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(": "))
        .map(|(k, v)| (k.to_ascii_lowercase(), v.to_string()))
        .collect();
    let len: usize = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .map(|(_, v)| v.parse().unwrap())
        .unwrap_or(0);
    let mut body = buf[header_end..].to_vec();
    while body.len() < len {
        let n = sock.read(&mut chunk).await.unwrap();
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Seen {
        method,
        path,
        headers,
        body,
    }
}

fn req() -> ChatRequest {
    ChatRequest::new(vec![Message::system("be brief"), Message::user("hi")])
}

async fn collect(model: &dyn ChatModel) -> Vec<Result<StreamEvent, Error>> {
    model
        .chat_stream(req())
        .await
        .expect("stream opens")
        .collect()
        .await
}

fn openai_ok() -> Value {
    json!({
        "model": "served-model",
        "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1}
    })
}

// ---------------------------------------------------------------- OpenAI

#[tokio::test]
async fn openai_chat_sends_auth_headers_query_and_decodes() {
    let (url, seen) = mock(vec![Reply::json(200, openai_ok())]).await;
    let m = OpenAiCompat::new(format!("{url}/v1/"), "m")
        .with_api_key("sk-test")
        .with_header("x-extra", "1")
        .with_query("api-version", "2024");
    let r = m.chat(req()).await.unwrap();
    assert_eq!(r.message.content, "hello");
    assert_eq!(r.model, "served-model");
    assert_eq!(r.usage.input_tokens, 3);
    let s = seen.lock().unwrap()[0].clone();
    assert_eq!(s.method, "POST");
    assert_eq!(s.path, "/v1/chat/completions?api-version=2024");
    assert_eq!(s.header("authorization"), Some("Bearer sk-test"));
    assert_eq!(s.header("x-extra"), Some("1"));
    assert_eq!(s.header("content-type"), Some("application/json"));
    assert_eq!(s.json()["model"], "m");
}

#[tokio::test]
async fn openai_custom_key_header_and_no_key() {
    let (url, seen) = mock(vec![
        Reply::json(200, openai_ok()),
        Reply::json(200, openai_ok()),
    ])
    .await;
    OpenAiCompat::new(&url, "m")
        .with_api_key("k")
        .with_api_key_header("api-key")
        .chat(req())
        .await
        .unwrap();
    OpenAiCompat::new(&url, "m").chat(req()).await.unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0].header("api-key"), Some("k"));
    assert_eq!(seen[0].header("authorization"), None);
    assert_eq!(seen[1].header("authorization"), None);
}

#[tokio::test]
async fn http_status_mapping() {
    let (url, _) = mock(vec![
        Reply::text(400, "bad request body"),
        Reply::text(401, "invalid key"),
        Reply::text(429, "slow down").header("retry-after", "7"),
        Reply::text(503, "overloaded").header("retry-after", "2"),
        Reply::text(500, "boom"),
    ])
    .await;
    let m = OpenAiCompat::new(&url, "m");

    let e = m.chat(req()).await.unwrap_err();
    match &e {
        Error::Provider {
            provider,
            status,
            body,
        } => {
            assert_eq!(provider, "openai-compat");
            assert_eq!(*status, 400);
            assert_eq!(body, "bad request body");
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(!e.is_retryable());

    let e = m.chat(req()).await.unwrap_err();
    assert_eq!(e.status(), Some(401));
    assert!(!e.is_retryable());

    let e = m.chat(req()).await.unwrap_err();
    assert!(matches!(e, Error::RateLimited { status: 429, .. }), "{e:?}");
    assert_eq!(e.retry_after(), Some(Duration::from_secs(7)));
    assert!(e.is_retryable());

    // Retry-After on a 5xx is a rate limit too.
    let e = m.chat(req()).await.unwrap_err();
    assert!(matches!(e, Error::RateLimited { status: 503, .. }), "{e:?}");
    assert_eq!(e.retry_after(), Some(Duration::from_secs(2)));

    let e = m.chat(req()).await.unwrap_err();
    assert!(matches!(e, Error::Provider { status: 500, .. }));
    assert!(e.is_retryable());
}

#[tokio::test]
async fn malformed_success_bodies() {
    let (url, _) = mock(vec![
        Reply::text(200, "<html>not json</html>"),
        Reply::json(200, json!({"choices": []})),
        Reply::json(
            200,
            json!({"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "c1", "function": {"arguments": "{}"}}
            ]}}]}),
        ),
    ])
    .await;
    let m = OpenAiCompat::new(&url, "m");
    assert!(matches!(m.chat(req()).await, Err(Error::Json(_))));
    let e = m.chat(req()).await.unwrap_err();
    assert!(matches!(e, Error::InvalidResponse(ref s) if s.contains("choices")));
    let e = m.chat(req()).await.unwrap_err();
    assert!(matches!(e, Error::InvalidResponse(ref s) if s.contains("without name")));
}

#[tokio::test]
async fn connection_refused_is_a_retryable_http_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let e = OpenAiCompat::new(&url, "m").chat(req()).await.unwrap_err();
    assert!(matches!(e, Error::Http(_)), "{e:?}");
    assert!(e.is_retryable());
    assert_eq!(e.status(), None);
    let e = OpenAiCompat::new(&url, "m")
        .chat_stream(req())
        .await
        .err()
        .unwrap();
    assert!(matches!(e, Error::Http(_)));
}

#[tokio::test]
async fn openai_stream_happy_path_with_tool_call_and_usage() {
    let (url, seen) = mock(vec![Reply::sse(&[
        r#"data: {"model":"m2","choices":[{"delta":{"content":"Hel"}}]}"#,
        r#"data: {"choices":[{"delta":{"content":"lo"}}]}"#,
        r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"add","arguments":"{\"a\":"}}]}}]}"#,
        r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}"#,
        r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#,
        "data: [DONE]",
        // Anything after [DONE] is ignored.
        r#"data: {"choices":[{"delta":{"content":"ignored"}}]}"#,
    ])])
    .await;
    let m = OpenAiCompat::new(&url, "m");
    let events: Vec<StreamEvent> = collect(&m)
        .await
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::TextDelta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello");
    let call = events
        .iter()
        .find_map(|e| match e {
            StreamEvent::ToolCall { call, .. } => Some(call.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(call.name, "add");
    assert_eq!(call.arguments, json!({"a": 1}));
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::Usage { usage } if usage.output_tokens == 2)));
    match events.last().unwrap() {
        StreamEvent::Done { model, stop_reason } => {
            assert_eq!(model, "m2");
            assert_eq!(stop_reason.as_deref(), Some("tool_calls"));
        }
        other => panic!("{other:?}"),
    }
    let s = seen.lock().unwrap()[0].clone();
    assert_eq!(s.header("accept"), Some("text/event-stream"));
    let body = s.json();
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn openai_stream_without_usage_option() {
    let (url, seen) = mock(vec![Reply::sse(&["data: [DONE]"])]).await;
    let m = OpenAiCompat::new(&url, "m").with_stream_usage(false);
    let events = collect(&m).await;
    assert!(matches!(events.last(), Some(Ok(StreamEvent::Done { .. }))));
    assert!(seen.lock().unwrap()[0]
        .json()
        .get("stream_options")
        .is_none());
}

#[tokio::test]
async fn stream_status_errors_happen_before_the_stream() {
    let (url, _) = mock(vec![Reply::text(429, "later").header("retry-after", "1")]).await;
    let e = OpenAiCompat::new(&url, "m")
        .chat_stream(req())
        .await
        .err()
        .unwrap();
    assert!(matches!(e, Error::RateLimited { status: 429, .. }));
}

#[tokio::test]
async fn stream_error_event_midway_stops_the_stream() {
    let (url, _) = mock(vec![Reply::sse(&[
        r#"data: {"choices":[{"delta":{"content":"partial"}}]}"#,
        r#"data: {"error":{"message":"overloaded"}}"#,
        r#"data: {"choices":[{"delta":{"content":"never"}}]}"#,
    ])])
    .await;
    let events = collect(&OpenAiCompat::new(&url, "m")).await;
    assert!(matches!(&events[0], Ok(StreamEvent::TextDelta { text }) if text == "partial"));
    assert!(matches!(&events[1], Err(Error::Stream(s)) if s.contains("overloaded")));
    assert_eq!(events.len(), 2, "nothing after the error: {events:?}");
}

#[tokio::test]
async fn stream_malformed_chunk_is_a_json_error() {
    let (url, _) = mock(vec![Reply::sse(&[
        r#"data: {"choices":[{"delta":{"content":"a"}}]}"#,
        "data: {not json",
    ])])
    .await;
    let events = collect(&OpenAiCompat::new(&url, "m")).await;
    assert!(events[0].is_ok());
    assert!(matches!(events[1], Err(Error::Json(_))));
    assert_eq!(events.len(), 2);
}

#[tokio::test]
async fn stream_connection_dropped_midway_is_an_error() {
    let (url, _) = mock(vec![Reply::sse(&[
        r#"data: {"choices":[{"delta":{"content":"partial"}}]}"#,
    ])
    .truncated()])
    .await;
    let events = collect(&OpenAiCompat::new(&url, "m")).await;
    assert!(matches!(&events[0], Ok(StreamEvent::TextDelta { text }) if text == "partial"));
    let last = events.last().unwrap();
    assert!(matches!(last, Err(Error::Http(_))), "{events:?}");
    // Regression: a connection dropped mid-body was not retryable.
    assert!(last.as_ref().unwrap_err().is_retryable());
    assert!(!events
        .iter()
        .any(|e| matches!(e, Ok(StreamEvent::Done { .. }))));
}

// ------------------------------------------------------------- Anthropic

fn anthropic_ok() -> Value {
    json!({
        "model": "claude-x",
        "content": [{"type": "text", "text": "hi there"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 2}
    })
}

#[tokio::test]
async fn anthropic_chat_headers_and_errors() {
    let (url, seen) = mock(vec![
        Reply::json(200, anthropic_ok()),
        Reply::json(
            400,
            json!({"type":"error","error":{"type":"invalid_request_error","message":"bad"}}),
        ),
        Reply::text(529, "overloaded").header("retry-after", "3"),
    ])
    .await;
    let m = Anthropic::new("ak", "claude-haiku-4-5")
        .with_base_url(format!("{url}/v1/"))
        .with_default_max_tokens(77);
    let r = m.chat(req()).await.unwrap();
    assert_eq!(r.message.content, "hi there");
    let e = m.chat(req()).await.unwrap_err();
    assert!(
        matches!(&e, Error::Provider { provider, status: 400, body } if provider == "anthropic" && body.contains("invalid_request_error"))
    );
    let e = m.chat(req()).await.unwrap_err();
    assert!(matches!(e, Error::RateLimited { status: 529, .. }));

    let s = seen.lock().unwrap()[0].clone();
    assert_eq!(s.path, "/v1/messages");
    assert_eq!(s.header("x-api-key"), Some("ak"));
    assert!(s.header("anthropic-version").is_some());
    let body = s.json();
    assert_eq!(body["max_tokens"], 77);
    assert_eq!(body["system"], "be brief");
}

#[tokio::test]
async fn anthropic_stream_and_error_event() {
    let (url, seen) = mock(vec![
        Reply::sse(&[
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-x\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}",
        ]),
        Reply::sse(&[
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}",
        ]),
    ])
    .await;
    let m = Anthropic::new("ak", "claude-haiku-4-5").with_base_url(&url);
    let events: Vec<StreamEvent> = collect(&m)
        .await
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta { text } if text == "Hi")));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::Done { stop_reason: Some(r), .. }) if r == "end_turn"
    ));
    assert_eq!(seen.lock().unwrap()[0].json()["stream"], true);

    let events = collect(&m).await;
    assert!(events[0].is_ok());
    assert!(
        matches!(events.last(), Some(Err(Error::Stream(s))) if s.contains("overloaded")),
        "{events:?}"
    );
}

#[tokio::test]
async fn anthropic_batches_over_http() {
    let results = [
        json!({"custom_id": "r1", "result": {"type": "errored", "error": {"type": "invalid_request"}}}),
        json!({"custom_id": "r0", "result": {"type": "succeeded", "message": anthropic_ok()}}),
        json!({"custom_id": "r2", "result": {"type": "expired"}}),
    ]
    .iter()
    .map(Value::to_string)
    .collect::<Vec<_>>()
    .join("\n");
    let (url, seen) = mock(vec![
        Reply::json(200, json!({"id": "b1", "processing_status": "in_progress"})),
        Reply::json(
            200,
            json!({"id": "b1", "processing_status": "in_progress", "request_counts": {"processing": 3}}),
        ),
        Reply::json(
            200,
            json!({"id": "b1", "processing_status": "ended", "request_counts": {"succeeded": 1, "errored": 1, "expired": 1}}),
        ),
        Reply::text(200, &results),
    ])
    .await;
    let m = Anthropic::new("ak", "claude-haiku-4-5").with_base_url(&url);
    let out = run_batch(&m, vec![req(), req(), req()], Duration::from_millis(1))
        .await
        .unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].as_ref().unwrap().message.content, "hi there");
    assert!(matches!(out[1], Err(Error::Provider { status: 400, .. })));
    assert!(matches!(&out[2], Err(Error::Other(s)) if s.contains("expired")));

    let seen = seen.lock().unwrap();
    assert_eq!(seen[0].path, "/messages/batches");
    assert_eq!(seen[0].json()["requests"][2]["custom_id"], "r2");
    assert_eq!(seen[1].method, "GET");
    assert_eq!(seen[1].path, "/messages/batches/b1");
    assert_eq!(seen[1].header("x-api-key"), Some("ak"));
    assert_eq!(seen[3].path, "/messages/batches/b1/results");
}

#[tokio::test]
async fn anthropic_batch_errors_keep_status_body_and_retry_after() {
    // Regression: batch status / results used `error_for_status`, which
    // dropped the error body and the `Retry-After` header.
    let (url, _) = mock(vec![
        Reply::text(404, "batch not found"),
        Reply::text(429, "slow").header("retry-after", "5"),
        Reply::json(200, json!({"processing_status": "ended"})),
        Reply::text(200, "{\"custom_id\": \"r0\", \"result\": {}}"),
        Reply::text(200, "not json"),
    ])
    .await;
    let m = Anthropic::new("ak", "claude-haiku-4-5").with_base_url(&url);
    let e = m.batch_status("nope").await.unwrap_err();
    assert!(
        matches!(&e, Error::Provider { status: 404, body, .. } if body == "batch not found"),
        "{e:?}"
    );
    let e = m.batch_results("b1").await.unwrap_err();
    assert!(matches!(e, Error::RateLimited { status: 429, .. }), "{e:?}");
    assert_eq!(e.retry_after(), Some(Duration::from_secs(5)));
    let e = m.batch_status("b1").await.unwrap_err();
    assert!(matches!(e, Error::InvalidResponse(ref s) if s.contains("without id")));
    let out = m.batch_results("b1").await.unwrap();
    assert!(matches!(&out[0], Err(Error::InvalidResponse(s)) if s.contains("without type")));
    assert!(matches!(m.batch_results("b1").await, Err(Error::Json(_))));
}

#[tokio::test]
async fn empty_batch_does_not_call_the_provider() {
    // No server at all: an empty batch must not touch the network.
    let m = Anthropic::new("ak", "m").with_base_url("http://127.0.0.1:9");
    assert!(run_batch(&m, vec![], Duration::from_millis(1))
        .await
        .unwrap()
        .is_empty());
}

// ---------------------------------------------------------------- Gemini

#[tokio::test]
async fn gemini_chat_auth_paths_and_errors() {
    let ok = json!({
        "candidates": [{"content": {"parts": [{"text": "hola"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
        "modelVersion": "gemini-x"
    });
    let (url, seen) = mock(vec![
        Reply::json(200, ok.clone()),
        Reply::json(200, ok),
        Reply::json(200, json!({"promptFeedback": {"blockReason": "SAFETY"}})),
        Reply::json(200, json!({"candidates": []})),
        Reply::text(403, "permission denied"),
    ])
    .await;
    let m = Gemini::new("gk", "gemini-2.5-flash").with_base_url(format!("{url}/v1beta/"));
    let r = m.chat(req()).await.unwrap();
    assert_eq!(r.message.content, "hola");
    assert_eq!(r.model, "gemini-x");
    let vertex = Gemini::new("unused", "gemini-2.5-flash")
        .with_base_url(&url)
        .with_bearer_token("tok");
    vertex.chat(req()).await.unwrap();
    let e = m.chat(req()).await.unwrap_err();
    assert!(matches!(e, Error::InvalidResponse(ref s) if s.contains("SAFETY")));
    let e = m.chat(req()).await.unwrap_err();
    assert!(matches!(e, Error::InvalidResponse(ref s) if s.contains("no candidates")));
    let e = m.chat(req()).await.unwrap_err();
    assert!(matches!(&e, Error::Provider { provider, status: 403, .. } if provider == "gemini"));

    let seen = seen.lock().unwrap();
    assert_eq!(
        seen[0].path,
        "/v1beta/models/gemini-2.5-flash:generateContent"
    );
    assert_eq!(seen[0].header("x-goog-api-key"), Some("gk"));
    assert_eq!(seen[1].header("authorization"), Some("Bearer tok"));
    assert_eq!(seen[1].header("x-goog-api-key"), None);
}

#[tokio::test]
async fn gemini_stream_path_events_and_midway_block() {
    let (url, seen) = mock(vec![
        Reply::sse(&[
            r#"data: {"candidates":[{"content":{"parts":[{"text":"Ho"}]}}]}"#,
            r#"data: {"candidates":[{"content":{"parts":[{"text":"la"},{"functionCall":{"name":"f","args":{"x":1}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3},"modelVersion":"g2"}"#,
        ]),
        Reply::sse(&[
            r#"data: {"candidates":[{"content":{"parts":[{"text":"a"}]}}]}"#,
            r#"data: {"promptFeedback":{"blockReason":"OTHER"}}"#,
        ]),
    ])
    .await;
    let m = Gemini::new("gk", "gemini-2.5-flash").with_base_url(&url);
    let events: Vec<StreamEvent> = collect(&m)
        .await
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamEvent::ToolCall { index: 0, call } if call.name == "f")));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::Done { model, stop_reason: Some(r) }) if model == "g2" && r == "STOP"
    ));
    assert_eq!(
        seen.lock().unwrap()[0].path,
        "/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    );

    let events = collect(&m).await;
    assert!(events[0].is_ok());
    assert!(matches!(&events[1], Err(Error::InvalidResponse(s)) if s.contains("OTHER")));
    assert_eq!(events.len(), 2);
}
