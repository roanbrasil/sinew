//! A `react_agent` whose tools live in another process, reached over MCP.
//!
//! The example is both sides: run with `--serve` it is an MCP server
//! exposing a small Sinew toolbox on stdio; run normally it spawns itself
//! with `--serve`, connects with `McpClient` and hands the remote tools to
//! the agent.
//!
//! Offline (scripted model):  cargo run -p sinew-mcp --example mcp_agent
//! Local Ollama:              SINEW_MODEL=qwen3:4b-instruct-2507-q4_K_M \
//!                              cargo run -p sinew-mcp --example mcp_agent -- --ollama

use std::sync::Arc;

use serde_json::{json, Value};
use sinew_core::{
    mock::ScriptedModel, providers::OpenAiCompat, ChatModel, FnTool, Message, ToolBox, ToolCall,
};
use sinew_graph::prebuilt::{react_agent, AgentState};
use sinew_mcp::{McpClient, McpServer};

fn server_tools() -> ToolBox {
    let num = |v: &Value, k: &str| {
        v[k].as_f64().ok_or_else(|| sinew_core::Error::Tool {
            name: "calculator".into(),
            message: format!("`{k}` must be a number"),
        })
    };
    let schema = json!({
        "type": "object",
        "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
        "required": ["a", "b"]
    });
    ToolBox::new()
        .with(FnTool::new(
            "multiply",
            "Multiply two numbers",
            schema.clone(),
            move |a: Value| Ok(json!(num(&a, "a")? * num(&a, "b")?)),
        ))
        .with(FnTool::new(
            "add",
            "Add two numbers",
            schema,
            move |a: Value| Ok(json!(num(&a, "a")? + num(&a, "b")?)),
        ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_default();
    if mode == "--serve" {
        McpServer::new("calculator", "0.1.0")
            .with_tools(server_tools())
            .serve_stdio()
            .await?;
        return Ok(());
    }

    let client = McpClient::connect_stdio(std::env::current_exe()?, ["--serve"]).await?;
    let info = client.server_info();
    println!(
        "connected to {} {} (protocol {})",
        info.server_info.name, info.server_info.version, info.protocol_version
    );
    let tools = client.toolbox().await?;
    let names: Vec<String> = tools.specs().into_iter().map(|s| s.name).collect();
    println!("remote tools: {names:?}\n");

    let model: Arc<dyn ChatModel> = if mode == "--ollama" {
        Arc::new(OpenAiCompat::ollama(
            std::env::var("SINEW_MODEL").unwrap_or_else(|_| "qwen3:4b-instruct-2507-q4_K_M".into()),
        ))
    } else {
        Arc::new(ScriptedModel::new([
            Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "call_1".into(),
                    name: "multiply".into(),
                    arguments: json!({"a": 37, "b": 3}),
                }],
            ),
            Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "call_2".into(),
                    name: "add".into(),
                    arguments: json!({"a": 111, "b": 4}),
                }],
            ),
            Message::assistant("37 × 3 + 4 = 115."),
        ]))
    };

    let agent = react_agent(model, tools)?;
    let out = agent
        .invoke(AgentState::new(vec![
            Message::system("Use the tools for arithmetic. Answer briefly."),
            Message::user("What is 37 times 3, plus 4?"),
        ]))
        .await?;
    for m in &out.messages {
        let calls: Vec<String> = m
            .tool_calls
            .iter()
            .map(|c| format!("{}({})", c.name, c.arguments))
            .collect();
        println!(
            "{:<9} {}{}",
            format!("{:?}", m.role),
            m.content,
            if calls.is_empty() {
                String::new()
            } else {
                format!(" {}", calls.join(", "))
            }
        );
    }
    client.close().await;
    Ok(())
}
