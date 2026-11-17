//! Model Context Protocol, both directions:
//!
//! - **server**: `--serve` exposes a Sinew `ToolBox`, a resource, a resource
//!   template and a prompt as an MCP server over stdio, usable from Claude
//!   Code, Cursor or any MCP client;
//! - **client**: the default mode spawns that server as a child process,
//!   hands its tools to a `react_agent` like local tools, then reads a
//!   resource and renders the prompt.
//!
//! ```bash
//! cargo run -p sample-11-mcp
//! cargo run -p sample-11-mcp -- --ollama
//! # register the server with Claude Code (build first):
//! claude mcp add sinew-notes -- target/debug/sample-11-mcp --serve
//! ```

use std::{collections::BTreeMap, sync::Mutex};

use sample_common::{pick_model, BoxError};
use serde_json::{json, Value};
use sinew::mcp::{
    McpClient, McpResource, McpResourceTemplate, McpServer, ResourceContents, ResourceRequest,
    RpcError,
};
use sinew::prelude::*;

/// The notes the server stores, shared by its tools and resources.
static NOTES: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

/// A tiny notes service: the tools the MCP server exposes.
fn notes_tools() -> ToolBox {
    let text = |v: &Value, k: &str| v[k].as_str().unwrap_or_default().to_string();
    ToolBox::new()
        .with(FnTool::new(
            "save_note",
            "Saves a note under a title",
            json!({"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}}, "required": ["title", "body"]}),
            move |a: Value| {
                NOTES.lock().unwrap().insert(text(&a, "title"), text(&a, "body"));
                Ok(json!("saved"))
            },
        ))
        .with(FnTool::new(
            "list_notes",
            "Lists saved note titles",
            json!({"type": "object", "properties": {}}),
            |_| Ok(json!(NOTES.lock().unwrap().keys().cloned().collect::<Vec<_>>())),
        ))
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    if std::env::args().any(|a| a == "--serve") {
        // stdout carries the protocol, so never print to it in server mode.
        McpServer::new("sinew-notes", "0.1.0")
            .with_tools(notes_tools())
            // A static resource and a template: every note is readable as `notes://<title>`.
            .with_text_resource(
                McpResource::new("notes://readme", "readme").with_mime_type("text/plain"),
                "Notes are saved with save_note and readable as notes://<title>.",
            )
            .with_resource_template(
                McpResourceTemplate::new("notes://{title}", "note").with_mime_type("text/plain"),
                |r: ResourceRequest| async move {
                    let body = NOTES.lock().unwrap().get(&r.params["title"]).cloned();
                    match body {
                        Some(b) => Ok(vec![ResourceContents::text(r.uri, b)]),
                        None => Err(RpcError::resource_not_found(&r.uri)),
                    }
                },
            )
            // A prompt with one required argument, rendered from a Sinew template.
            .with_prompt_template(
                "standup",
                "Draft a stand-up update about a topic",
                PromptTemplate::new("Write a two-line stand-up update about {topic}.")?,
            )
            .serve_stdio()
            .await?;
        return Ok(());
    }

    let client = McpClient::connect_stdio(std::env::current_exe()?, ["--serve"]).await?;
    let info = client.server_info();
    println!(
        "connected to {} (protocol {})",
        info.server_info.name, info.protocol_version
    );
    let tools = client.toolbox().await?;
    println!(
        "remote tools: {:?}\n",
        tools.specs().iter().map(|s| &s.name).collect::<Vec<_>>()
    );

    let model = pick_model(vec![
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "1".into(),
                name: "save_note".into(),
                arguments: json!({"title": "standup", "body": "ship the MCP sample"}),
            }],
        ),
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "2".into(),
                name: "list_notes".into(),
                arguments: json!({}),
            }],
        ),
        Message::assistant("Saved the note \"standup\"; it is now the only note on the server."),
    ])?;

    let agent = react_agent(model, tools)?;
    let out = agent
        .invoke(AgentState::new(vec![
            Message::system("Use the tools. Answer in one sentence."),
            Message::user("Save a note titled 'standup' saying 'ship the MCP sample', then tell me which notes exist."),
        ]))
        .await?;
    for m in &out.messages[2..] {
        match m.role {
            Role::Assistant if m.has_tool_calls() => m
                .tool_calls
                .iter()
                .for_each(|c| println!("→ mcp call {}({})", c.name, c.arguments)),
            Role::Tool => println!("← {}", m.content),
            _ => println!("\n{}", m.content),
        }
    }

    // Resources: list them, then read the note the agent just saved.
    let resources = client.list_resources().await?;
    let templates = client.list_resource_templates().await?;
    println!(
        "\nresources: {:?}, templates: {:?}",
        resources.iter().map(|r| &r.uri).collect::<Vec<_>>(),
        templates
            .iter()
            .map(|t| &t.uri_template)
            .collect::<Vec<_>>()
    );
    for c in client.read_resource("notes://standup").await? {
        println!("notes://standup → {}", c.text.unwrap_or_default());
    }

    // Prompts: render on the server, or load as a local Sinew template.
    let messages = client
        .get_prompt_messages("standup", json!({"topic": "MCP resources"}))
        .await?;
    println!("prompt standup → {}", messages[0].content);
    let template = client.prompt_template("standup").await?;
    println!("as PromptTemplate, variables {:?}", template.variables());
    client.close().await;
    Ok(())
}
