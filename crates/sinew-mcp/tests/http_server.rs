//! `McpServer::http_router` end to end: Sinew's own Streamable HTTP client and
//! raw HTTP requests against a server on 127.0.0.1:0.
#![cfg(all(feature = "http", feature = "http-server"))]

mod common;

use std::time::Duration;

use serde_json::{json, Value};
use sinew_core::{AsyncFnTool, FnTool, ToolBox};
use sinew_mcp::{McpClient, McpHttpConfig, McpServer};
use tokio::sync::oneshot;

fn tools() -> ToolBox {
    ToolBox::new()
        .with(FnTool::new(
            "upper",
            "Uppercase `text`",
            json!({"type": "object", "properties": {"text": {"type": "string"}}}),
            |a: Value| Ok(json!(a["text"].as_str().unwrap_or("").to_uppercase())),
        ))
        .with(AsyncFnTool::new(
            "slow_double",
            "Double `n` after a short wait",
            json!({"type": "object", "properties": {"n": {"type": "number"}}}),
            |a: Value| async move {
                tokio::time::sleep(Duration::from_millis(20)).await;
                Ok(json!(a["n"].as_f64().unwrap_or(0.0) * 2.0))
            },
        ))
}

async fn spawn(config: McpHttpConfig) -> (String, oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = oneshot::channel::<()>();
    let server = McpServer::new("svc", "1.0").with_tools(tools());
    tokio::spawn(server.serve_http_with(listener, config, async {
        let _ = rx.await;
    }));
    (base, tx)
}

#[tokio::test]
async fn sinew_client_lists_and_calls_tools_over_http() {
    let (base, _stop) = spawn(McpHttpConfig::default()).await;
    let client = McpClient::connect_http(format!("{base}/mcp"))
        .await
        .unwrap();
    let mut names: Vec<_> = client
        .list_tools()
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    names.sort();
    assert_eq!(names, ["slow_double", "upper"]);
    let r = client
        .call_tool("upper", json!({"text": "rust"}))
        .await
        .unwrap();
    assert!(r.to_json().to_string().contains("RUST"));
    let r = client
        .call_tool("slow_double", json!({"n": 21}))
        .await
        .unwrap();
    assert!(r.to_json().to_string().contains("42"));
    // MCP tools plug straight into a Sinew agent.
    assert_eq!(client.toolbox().await.unwrap().specs().len(), 2);
    client.close().await;
}

async fn init(http: &reqwest::Client, url: &str, token: Option<&str>) -> reqwest::Response {
    let mut req = http.post(url).header("accept", "application/json, text/event-stream").json(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}
    }));
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    req.send().await.unwrap()
}

#[tokio::test]
async fn sessions_are_issued_required_and_ended() {
    let (base, _stop) = spawn(McpHttpConfig::default()).await;
    let url = format!("{base}/mcp");
    let http = reqwest::Client::new();
    let r = init(&http, &url, None).await;
    assert_eq!(r.status(), 200);
    let sid = r.headers()["mcp-session-id"].to_str().unwrap().to_string();

    let list = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"});
    let no_sid = http.post(&url).json(&list).send().await.unwrap();
    assert_eq!(no_sid.status(), 400);
    let bad_sid = http
        .post(&url)
        .header("mcp-session-id", "nope")
        .json(&list)
        .send()
        .await
        .unwrap();
    assert_eq!(bad_sid.status(), 404);

    let ok = http
        .post(&url)
        .header("mcp-session-id", &sid)
        .json(&list)
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), 200);
    let body: Value = ok.json().await.unwrap();
    assert_eq!(body["result"]["tools"].as_array().unwrap().len(), 2);

    let note = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    let n = http
        .post(&url)
        .header("mcp-session-id", &sid)
        .json(&note)
        .send()
        .await
        .unwrap();
    assert_eq!(n.status(), 202);

    // GET opens the session's SSE stream: it needs a valid session and must
    // accept text/event-stream.
    let sse = |sid: &str, accept: &str| {
        http.get(&url)
            .header("mcp-session-id", sid)
            .header("accept", accept)
            .send()
    };
    assert_eq!(http.get(&url).send().await.unwrap().status(), 400);
    assert_eq!(
        sse("nope", "text/event-stream").await.unwrap().status(),
        404
    );
    assert_eq!(sse(&sid, "application/json").await.unwrap().status(), 406);
    let stream = sse(&sid, "text/event-stream").await.unwrap();
    assert_eq!(stream.status(), 200);
    assert!(stream.headers()["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let del = http
        .delete(&url)
        .header("mcp-session-id", &sid)
        .send()
        .await
        .unwrap();
    assert_eq!(del.status(), 200);
    let after = http
        .post(&url)
        .header("mcp-session-id", &sid)
        .json(&list)
        .send()
        .await
        .unwrap();
    assert_eq!(after.status(), 404);
}

#[tokio::test]
async fn bearer_origin_and_version_checks() {
    let cfg = McpHttpConfig::default()
        .with_bearer_token("s3cret")
        .with_allowed_origins(["https://app.example.com"]);
    let (base, _stop) = spawn(cfg).await;
    let url = format!("{base}/mcp");
    let http = reqwest::Client::new();

    let r = init(&http, &url, None).await;
    assert_eq!(r.status(), 401);
    assert_eq!(r.headers()["www-authenticate"], "Bearer");
    assert_eq!(init(&http, &url, Some("wrong")).await.status(), 401);
    assert_eq!(init(&http, &url, Some("s3cret")).await.status(), 200);

    let evil = http
        .post(&url)
        .bearer_auth("s3cret")
        .header("origin", "https://evil.example")
        .json(&json!({"jsonrpc": "2.0", "id": 1, "method": "ping"}))
        .send()
        .await
        .unwrap();
    assert_eq!(evil.status(), 403);

    let old = http
        .post(&url)
        .bearer_auth("s3cret")
        .header("mcp-protocol-version", "1999-01-01")
        .json(&json!({"jsonrpc": "2.0", "id": 1, "method": "ping"}))
        .send()
        .await
        .unwrap();
    assert_eq!(old.status(), 400);

    // Health stays open for load balancers and probes.
    assert_eq!(
        http.get(format!("{base}/health"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );

    // The Sinew client works end to end once it sends the token.
    let client = McpClient::connect_http_with_token(&url, "s3cret")
        .await
        .unwrap();
    assert_eq!(client.list_tools().await.unwrap().len(), 2);
    client.close().await;
    assert!(McpClient::connect_http(&url).await.is_err());
}

#[tokio::test]
async fn stateless_mode_needs_no_session_and_merges_into_an_app() {
    let mcp = McpServer::new("svc", "1.0")
        .with_tools(tools())
        .http_router(McpHttpConfig::default().stateless().with_path("/tools/mcp"));
    let app = axum::Router::new()
        .route("/orders", axum::routing::get(|| async { "[]" }))
        .merge(mcp);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await });

    let http = reqwest::Client::new();
    assert_eq!(
        http.get(format!("{base}/orders"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "[]"
    );
    let r: Value = http
        .post(format!("{base}/tools/mcp"))
        .json(&json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "slow_double", "arguments": {"n": 4}}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["id"], 7);
    assert!(r["result"].to_string().contains('8'));

    // Old-style batches still work.
    let batch: Value = http
        .post(format!("{base}/tools/mcp"))
        .json(&json!([
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        ]))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(batch.as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn stateless_mode_has_no_get_stream_and_accepts_subscriptions() {
    let mcp = common::docs_server().http_router(McpHttpConfig::default().stateless());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, mcp).await });
    let http = reqwest::Client::new();
    let get = http
        .get(&url)
        .header("accept", "text/event-stream")
        .send()
        .await
        .unwrap();
    assert_eq!(get.status(), 405);
    // The Sinew client treats a missing GET stream as "no notifications".
    let client = McpClient::connect_http(&url).await.unwrap();
    client.subscribe_resource("docs://readme").await.unwrap();
    let license = client.read_resource("docs://license").await.unwrap();
    assert_eq!(license[0].text.as_deref(), Some("MIT"));
    client.close().await;
}

async fn spawn_docs() -> (String, McpServer, oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let (tx, rx) = oneshot::channel::<()>();
    let server = common::docs_server();
    let handle = server.clone();
    tokio::spawn(
        server.serve_http_with(listener, McpHttpConfig::default(), async {
            let _ = rx.await;
        }),
    );
    (url, handle, tx)
}

#[tokio::test]
async fn every_resource_and_prompt_method_over_http() {
    let (url, _handle, _stop) = spawn_docs().await;
    let client = McpClient::connect_http(&url).await.unwrap();
    common::exercise_resources_and_prompts(&client).await;
    client.close().await;
}

#[tokio::test]
async fn notifications_arrive_on_the_get_stream() {
    let (url, handle, _stop) = spawn_docs().await;
    let client = McpClient::connect_http(&url).await.unwrap();
    common::exercise_notifications(&client, &handle).await;

    // Closing the session (DELETE) drops its stream and subscriptions.
    client.close().await;
    assert_eq!(handle.notify_resource_list_changed(), 0);
}

#[tokio::test]
async fn raw_http_subscription_and_sse_event() {
    let (url, handle, _stop) = spawn_docs().await;
    let http = reqwest::Client::new();
    let r = init(&http, &url, None).await;
    let sid = r.headers()["mcp-session-id"].to_str().unwrap().to_string();
    let mut stream = http
        .get(&url)
        .header("mcp-session-id", &sid)
        .header("accept", "text/event-stream")
        .send()
        .await
        .unwrap();
    let sub: Value = http
        .post(&url)
        .header("mcp-session-id", &sid)
        .json(
            &json!({"jsonrpc": "2.0", "id": 2, "method": "resources/subscribe",
            "params": {"uri": "docs://pages/intro"}}),
        )
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(sub["result"], json!({}));
    assert_eq!(handle.notify_resource_updated("docs://pages/intro"), 1);
    let mut body = String::new();
    while !body.contains("\n\n") {
        let chunk = tokio::time::timeout(Duration::from_secs(5), stream.chunk())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        body.push_str(&String::from_utf8_lossy(&chunk));
    }
    assert!(body.contains("event: message"));
    assert!(body.contains(r#""method":"notifications/resources/updated""#));
    assert!(body.contains(r#""uri":"docs://pages/intro""#));
}
