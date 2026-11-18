//! MCP and A2A JSON-RPC message handling on in-process servers.

use std::time::Duration;

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header, Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sinew_a2a::{A2aServer, AgentCard, AgentExecutor, BoxError, EventQueue, Part, RequestContext};
use sinew_core::{FnTool, PromptTemplate};
use sinew_mcp::{
    McpHttpConfig, McpResource, McpResourceTemplate, McpServer, ResourceContents, ResourceRequest,
    RpcError,
};
use tower::ServiceExt;

use crate::{assert_json_rpc_response, block_on};

/// Upper bound for one request; a request that takes longer hangs.
const LIMIT: Duration = Duration::from_secs(10);

fn mcp_server() -> McpServer {
    McpServer::new("fuzz", "1.0")
        .with_tool(FnTool::new(
            "echo",
            "Echo `text`",
            json!({"type": "object", "properties": {"text": {"type": "string"}}}),
            |a: Value| match a["text"].as_str() {
                Some(t) => Ok(json!(t)),
                None => Err(sinew_core::Error::Tool {
                    name: "echo".into(),
                    message: "`text` is required".into(),
                }),
            },
        ))
        .with_page_size(2)
        .with_text_resource(McpResource::new("notes://a", "a"), "note a")
        .with_text_resource(McpResource::new("notes://b", "b"), "note b")
        .with_resource_template(
            McpResourceTemplate::new("notes://{title}/rows/{+rest}", "note"),
            |r: ResourceRequest| async move {
                match r.params.get("title").map(String::as_str) {
                    Some("missing") => Err(RpcError::resource_not_found(&r.uri)),
                    t => Ok(vec![ResourceContents::text(
                        r.uri.clone(),
                        format!("{t:?} {:?}", r.params.get("rest")),
                    )]),
                }
            },
        )
        .with_prompt_template(
            "review",
            "Review code",
            PromptTemplate::new("Review this {lang} code:\n{code}").expect("template"),
        )
}

/// A notification (no `id`) gets no answer; a request gets a response with
/// its own id; anything else gets an error response.
fn check_mcp_reply(msg: &Value, reply: Option<&Value>) {
    let is_request_shape = msg.get("jsonrpc") == Some(&json!("2.0"))
        && msg.get("method").is_some_and(Value::is_string);
    match (msg.get("id"), reply) {
        (None, None) if is_request_shape => {}
        (Some(id), Some(r)) if is_request_shape && (id.is_string() || id.is_number()) => {
            assert_json_rpc_response(r);
            assert_eq!(&r["id"], id, "reply to another id");
        }
        (_, Some(r)) => {
            assert_json_rpc_response(r);
        }
        // Responses to requests the server never sent are ignored.
        (Some(_), None) if msg.get("method").is_none() => {}
        (_, None) => panic!("no reply to {msg}"),
    }
}

/// Posts `body` to `router` and returns the status, content type and body.
async fn post(router: Router, path: &str, body: &[u8]) -> (StatusCode, String, Vec<u8>) {
    let req = Request::post(path)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream")
        .body(Body::from(body.to_vec()))
        .expect("request");
    let resp = tokio::time::timeout(LIMIT, router.oneshot(req))
        .await
        .expect("request hung")
        .expect("infallible");
    let status = resp.status();
    let ctype = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    if ctype.starts_with("text/event-stream") {
        // Streams stay open until the task ends; the frames are checked by
        // the server's own tests.
        return (status, ctype, Vec::new());
    }
    let bytes = tokio::time::timeout(LIMIT, resp.into_body().collect())
        .await
        .expect("body hung")
        .expect("body")
        .to_bytes()
        .to_vec();
    (status, ctype, bytes)
}

/// MCP: each line is one message for `handle_message`; the whole input is
/// also posted to the stateless Streamable HTTP endpoint.
pub fn mcp(data: &[u8]) {
    block_on(async {
        let server = mcp_server();
        for line in data.split(|&b| b == b'\n').take(16) {
            let Ok(msg) = serde_json::from_slice::<Value>(line) else {
                continue;
            };
            let reply = tokio::time::timeout(LIMIT, server.handle_message(msg.clone()))
                .await
                .expect("handle_message hung");
            check_mcp_reply(&msg, reply.as_ref());
        }

        let router = mcp_server().http_router(McpHttpConfig::default().stateless());
        let (status, ctype, body) = post(router, "/mcp", data).await;
        if body.is_empty() {
            assert!(
                status == StatusCode::ACCEPTED || ctype.starts_with("text/event-stream"),
                "empty {status} reply"
            );
            return;
        }
        let v: Value = serde_json::from_slice(&body)
            .unwrap_or_else(|e| panic!("{status} reply is not JSON ({e})"));
        match &v {
            Value::Array(items) => {
                assert!(!items.is_empty(), "empty batch reply");
                items.iter().for_each(assert_json_rpc_response);
            }
            one => assert_json_rpc_response(one),
        }
    });
}

/// Answers every message with its own text and completes the task.
struct Echo;

#[async_trait]
impl AgentExecutor for Echo {
    async fn execute(&self, ctx: RequestContext, events: EventQueue) -> Result<(), BoxError> {
        let reply = events.agent_message(vec![Part::text(ctx.text())]);
        events.complete(Some(reply));
        Ok(())
    }
}

/// A2A: the input is posted to the JSON-RPC endpoint; the reply is always
/// `200` with a JSON-RPC response (or an SSE stream for streaming methods).
pub fn a2a(data: &[u8]) {
    block_on(async {
        let server =
            A2aServer::new(AgentCard::new("fuzz", "Echo agent", ""), Echo).with_rpc_path("/rpc");
        let (status, ctype, body) = post(server.router(), "/rpc", data).await;
        assert_eq!(status, StatusCode::OK);
        if ctype.starts_with("text/event-stream") {
            return;
        }
        let v: Value = serde_json::from_slice(&body)
            .unwrap_or_else(|e| panic!("reply is not JSON ({e}): {body:?}"));
        assert_json_rpc_response(&v);
        if let Ok(Value::Object(req)) = serde_json::from_slice::<Value>(data) {
            if let Some(id) = req.get("id").filter(|id| id.is_string() || id.is_number()) {
                if req.get("jsonrpc") == Some(&json!("2.0"))
                    && req.get("method").is_some_and(Value::is_string)
                {
                    assert_eq!(&v["id"], id, "reply to another id");
                }
            }
        }
    });
}
