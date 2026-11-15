//! Client and server connected through in-memory pipes: the agent tool,
//! `react_agent` over MCP tools, timeouts and cancellation.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, FnTool, Message, Role, Tool, ToolBox, ToolCall, ToolSpec};
use sinew_graph::prebuilt::{react_agent, AgentState};
use sinew_mcp::{McpClient, McpError, McpServer};

struct Slow;

#[async_trait]
impl Tool for Slow {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "slow".into(),
            description: "Sleeps for `ms` milliseconds".into(),
            parameters: json!({"type": "object", "properties": {"ms": {"type": "integer"}}}),
        }
    }
    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        let ms = args["ms"].as_u64().unwrap_or(0);
        tokio::time::sleep(Duration::from_millis(ms)).await;
        Ok(json!(format!("slept {ms}ms")))
    }
}

fn math() -> ToolBox {
    ToolBox::new()
        .with(FnTool::new(
            "add",
            "Add a and b",
            json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
            |a: Value| match (a["a"].as_f64(), a["b"].as_f64()) {
                (Some(x), Some(y)) => Ok(json!(x + y)),
                _ => Err(sinew_core::Error::Tool {
                    name: "add".into(),
                    message: "a and b must be numbers".into(),
                }),
            },
        ))
        .with(Slow)
}

async fn connect(server: McpServer) -> McpClient {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let (sr, sw) = tokio::io::split(server_io);
    tokio::spawn(server.serve(sr, sw));
    let (cr, cw) = tokio::io::split(client_io);
    McpClient::connect(cr, cw).await.unwrap()
}

#[tokio::test]
async fn react_agent_uses_mcp_tools_transparently() {
    let client = connect(McpServer::new("math", "1.0").with_tools(math())).await;
    let tools = client.toolbox().await.unwrap();
    let model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "",
            vec![
                ToolCall {
                    id: "c1".into(),
                    name: "add".into(),
                    arguments: json!({"a": 40, "b": 2}),
                },
                ToolCall {
                    id: "c2".into(),
                    name: "add".into(),
                    arguments: json!({"a": "x"}),
                },
            ],
        ),
        Message::assistant("40 + 2 = 42."),
    ]));
    let agent = react_agent(model.clone(), tools).unwrap();
    let out = agent
        .invoke(AgentState::new(vec![Message::user("40 + 2?")]))
        .await
        .unwrap();
    let roles: Vec<Role> = out.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        [
            Role::User,
            Role::Assistant,
            Role::Tool,
            Role::Tool,
            Role::Assistant
        ]
    );
    assert_eq!(out.messages[2].content, "42.0");
    assert_eq!(
        out.messages[3].content,
        "error: tool `add` failed: a and b must be numbers"
    );
    // The model saw the MCP tool schemas.
    let specs = &model.requests()[0].tools;
    let add = specs.iter().find(|s| s.name == "add").unwrap();
    assert_eq!(add.parameters["required"], json!(["a", "b"]));
}

#[tokio::test]
async fn serves_an_agent_as_one_tool() {
    let model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "add".into(),
                arguments: json!({"a": 1, "b": 2}),
            }],
        ),
        Message::assistant("It is 3."),
    ]));
    let graph = react_agent(model, math()).unwrap();
    let client = connect(McpServer::new("agent", "1.0").with_agent(
        "math_agent",
        "Answers arithmetic questions",
        graph,
    ))
    .await;
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].input_schema["required"], json!(["input"]));
    let r = client
        .call_tool("math_agent", json!({"input": "1 + 2?"}))
        .await
        .unwrap();
    assert_eq!((r.text().as_str(), r.is_error), ("It is 3.", false));
    let bad = client.call_tool("math_agent", json!({})).await.unwrap();
    assert!(bad.is_error);
}

#[tokio::test]
async fn timeouts_cancel_and_calls_run_concurrently() {
    let client = connect(McpServer::new("slow", "1.0").with_tools(math())).await;
    // Concurrent: a slow call does not block a fast one.
    let c2 = client.clone();
    let slow = tokio::spawn(async move { c2.call_tool("slow", json!({"ms": 300})).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    let started = std::time::Instant::now();
    let fast = client
        .call_tool("add", json!({"a": 1, "b": 1}))
        .await
        .unwrap();
    assert_eq!(fast.text(), "2.0");
    assert!(started.elapsed() < Duration::from_millis(200));
    assert_eq!(slow.await.unwrap().unwrap().text(), "slept 300ms");

    // Timeout: the client gives up and cancels; the connection stays usable.
    client.set_request_timeout(Duration::from_millis(50));
    let err = client
        .call_tool("slow", json!({"ms": 5_000}))
        .await
        .unwrap_err();
    assert!(matches!(err, McpError::Timeout(ref m) if m == "tools/call"));
    client.set_request_timeout(Duration::from_secs(5));
    client.ping().await.unwrap();
}

#[tokio::test]
async fn server_shutdown_fails_pending_requests() {
    let (client_io, server_io) = tokio::io::duplex(4096);
    let (sr, sw) = tokio::io::split(server_io);
    let server = tokio::spawn(McpServer::new("x", "1").with_tools(math()).serve(sr, sw));
    let (cr, cw) = tokio::io::split(client_io);
    let client = McpClient::connect(cr, cw).await.unwrap();
    server.abort();
    let _ = server.await;
    assert!(matches!(client.ping().await, Err(McpError::Closed)));
}
