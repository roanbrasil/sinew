//! Streamable HTTP client against a minimal local HTTP server that wraps
//! `McpServer::handle_message`. Run with `cargo test -p sinew-mcp --features http`.
#![cfg(feature = "http")]

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use sinew_core::FnTool;
use sinew_mcp::{McpClient, McpServer};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::TcpListener,
};

/// (label, Mcp-Session-Id, MCP-Protocol-Version) of every request received.
type Seen = Arc<Mutex<Vec<(String, Option<String>, Option<String>)>>>;

/// Serves one HTTP/1.1 request per connection. `initialize` gets a JSON
/// reply plus a session id; other requests get an SSE reply; notifications 202.
async fn mock_server(seen: Seen) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = Arc::new(McpServer::new("http-test", "1.0").with_tool(FnTool::new(
        "upper",
        "Uppercase `text`",
        json!({"type": "object"}),
        |a: Value| Ok(json!(a["text"].as_str().unwrap_or("").to_uppercase())),
    )));
    tokio::spawn(async move {
        loop {
            let (sock, _) = listener.accept().await.unwrap();
            let (server, seen) = (server.clone(), seen.clone());
            tokio::spawn(async move {
                let mut reader = BufReader::new(sock);
                let mut start = String::new();
                reader.read_line(&mut start).await.unwrap();
                let (mut len, mut session, mut version) = (0usize, None, None);
                loop {
                    let mut h = String::new();
                    reader.read_line(&mut h).await.unwrap();
                    let h = h.trim_end();
                    if h.is_empty() {
                        break;
                    }
                    let (k, v) = h.split_once(':').unwrap();
                    let v = v.trim().to_string();
                    match k.to_ascii_lowercase().as_str() {
                        "content-length" => len = v.parse().unwrap(),
                        "mcp-session-id" => session = Some(v),
                        "mcp-protocol-version" => version = Some(v),
                        _ => {}
                    }
                }
                let mut body = vec![0; len];
                reader.read_exact(&mut body).await.unwrap();
                let method = start.split_whitespace().next().unwrap().to_string();
                let msg: Value = if body.is_empty() {
                    Value::Null
                } else {
                    serde_json::from_slice(&body).unwrap()
                };
                let label = msg["method"].as_str().unwrap_or(&method).to_string();
                seen.lock().unwrap().push((label.clone(), session, version));
                let resp = if method == "DELETE" {
                    "HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n".to_string()
                } else {
                    match server.handle_message(msg).await {
                        None => "HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\n\r\n".to_string(),
                        Some(reply) if label == "initialize" => {
                            let b = reply.to_string();
                            format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nmcp-session-id: s-123\r\ncontent-length: {}\r\n\r\n{b}", b.len())
                        }
                        Some(reply) => {
                            let b = format!("event: message\r\ndata: {reply}\r\n\r\n");
                            format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{b}", b.len())
                        }
                    }
                };
                let mut sock = reader.into_inner();
                sock.write_all(resp.as_bytes()).await.unwrap();
                sock.shutdown().await.ok();
            });
        }
    });
    format!("http://{addr}/mcp")
}

#[tokio::test]
async fn streamable_http_round_trip() {
    let seen: Seen = Arc::default();
    let url = mock_server(Arc::clone(&seen)).await;
    let client = McpClient::connect_http(url).await.unwrap();
    assert_eq!(client.server_info().server_info.name, "http-test");
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools[0].name, "upper");
    let r = client
        .call_tool("upper", json!({"text": "mcp"}))
        .await
        .unwrap();
    assert_eq!(r.text(), "MCP");
    assert!(client.call_tool("missing", json!({})).await.is_err());
    client.close().await;

    let seen = seen.lock().unwrap().clone();
    let labels: Vec<&str> = seen.iter().map(|s| s.0.as_str()).collect();
    assert_eq!(
        labels,
        [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
            "tools/call",
            "DELETE"
        ]
    );
    // No session header before the server assigned one; afterwards both headers.
    assert_eq!(seen[0].1, None);
    for s in &seen[1..] {
        assert_eq!(s.1.as_deref(), Some("s-123"));
        assert_eq!(s.2.as_deref(), Some("2025-06-18"));
    }
}
