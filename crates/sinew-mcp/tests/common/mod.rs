//! A server with resources, a resource template and prompts, shared by the
//! in-process and HTTP tests, plus assertions both transports must pass.
#![allow(dead_code)]

use std::time::Duration;

use serde_json::{json, Value};
use sinew_core::{PromptTemplate, Role};
use sinew_mcp::{
    protocol::error_code, GetPromptResult, McpClient, McpError, McpNotification, McpPrompt,
    McpResource, McpResourceTemplate, McpServer, PromptMessage, PromptRequest, ResourceContents,
    ResourceRequest, RpcError,
};
use tokio::sync::broadcast;

pub fn docs_server() -> McpServer {
    McpServer::new("docs", "1.0")
        .with_page_size(2)
        .with_text_resource(
            McpResource::new("docs://readme", "readme")
                .with_mime_type("text/markdown")
                .with_description("Project readme"),
            "# Sinew\nRust agents.",
        )
        .with_text_resource(McpResource::new("docs://license", "license"), "MIT")
        .with_resource(
            McpResource::new("docs://logo", "logo").with_mime_type("image/png"),
            |r: ResourceRequest| async move {
                Ok(vec![ResourceContents::blob_bytes(
                    r.uri,
                    "image/png",
                    b"\x89PNG",
                )])
            },
        )
        .with_resource_template(
            McpResourceTemplate::new("docs://pages/{slug}", "page").with_mime_type("text/plain"),
            |r: ResourceRequest| async move {
                match r.params["slug"].as_str() {
                    "intro" | "tools" => Ok(vec![ResourceContents::text(
                        r.uri.clone(),
                        format!("page {}", r.params["slug"]),
                    )
                    .with_mime_type("text/plain")]),
                    _ => Err(RpcError::resource_not_found(&r.uri)),
                }
            },
        )
        .with_prompt_template(
            "summarize",
            "Summarize a text in a given style",
            PromptTemplate::new("Summarize in a {style} style (use {{braces}} literally):\n{text}")
                .unwrap(),
        )
        .with_prompt(
            McpPrompt::new("review")
                .with_description("Code review conversation")
                .with_argument("code", "The code to review", true)
                .with_argument("focus", "What to focus on", false),
            |r: PromptRequest| async move {
                let focus = r.arguments.get("focus").cloned().unwrap_or("bugs".into());
                Ok(GetPromptResult::new([
                    PromptMessage::user(format!("Review for {focus}:\n{}", r.arguments["code"])),
                    PromptMessage::assistant("I will review it."),
                ])
                .with_description("review"))
            },
        )
}

fn rpc_code(e: McpError) -> i64 {
    match e {
        McpError::Rpc(e) => e.code,
        other => panic!("expected a JSON-RPC error, got {other:?}"),
    }
}

/// Every resource and prompt method, including the error cases.
pub async fn exercise_resources_and_prompts(client: &McpClient) {
    let caps = &client.server_info().capabilities;
    assert_eq!(caps["resources"]["subscribe"], true);
    assert_eq!(caps["prompts"]["listChanged"], true);

    // Pagination (page size 2) is followed transparently.
    let resources = client.list_resources().await.unwrap();
    let uris: Vec<&str> = resources.iter().map(|r| r.uri.as_str()).collect();
    assert_eq!(uris, ["docs://readme", "docs://license", "docs://logo"]);
    assert_eq!(resources[0].mime_type.as_deref(), Some("text/markdown"));
    let templates = client.list_resource_templates().await.unwrap();
    assert_eq!(templates[0].uri_template, "docs://pages/{slug}");

    let readme = client.read_resource("docs://readme").await.unwrap();
    assert_eq!(readme[0].text.as_deref(), Some("# Sinew\nRust agents."));
    assert_eq!(readme[0].mime_type.as_deref(), Some("text/markdown"));
    let logo = client.read_resource("docs://logo").await.unwrap();
    assert!(logo[0].is_blob());
    assert_eq!(logo[0].blob.as_deref(), Some("iVBORw=="));
    let page = client.read_resource("docs://pages/intro").await.unwrap();
    assert_eq!(page[0].text.as_deref(), Some("page intro"));

    let unknown = client.read_resource("docs://nope").await.unwrap_err();
    assert_eq!(rpc_code(unknown), error_code::RESOURCE_NOT_FOUND);
    let missing_page = client.read_resource("docs://pages/zzz").await.unwrap_err();
    assert_eq!(rpc_code(missing_page), error_code::RESOURCE_NOT_FOUND);
    let bad_sub = client.subscribe_resource("docs://nope").await.unwrap_err();
    assert_eq!(rpc_code(bad_sub), error_code::RESOURCE_NOT_FOUND);

    let prompts = client.list_prompts().await.unwrap();
    let names: Vec<&str> = prompts.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["summarize", "review"]);
    assert!(prompts[1].arguments[0].required && !prompts[1].arguments[1].required);

    let r = client
        .get_prompt("review", json!({"code": "fn f() {}"}))
        .await
        .unwrap();
    assert_eq!(r.description.as_deref(), Some("review"));
    assert_eq!(r.messages.len(), 2);
    let msgs = client
        .get_prompt_messages("review", json!({"code": "x", "focus": "style"}))
        .await
        .unwrap();
    assert_eq!(msgs[0].role, Role::User);
    assert_eq!(msgs[0].content, "Review for style:\nx");
    assert_eq!(msgs[1].role, Role::Assistant);

    let missing = client
        .get_prompt("review", json!({"focus": "style"}))
        .await
        .unwrap_err();
    assert_eq!(rpc_code(missing), error_code::INVALID_PARAMS);
    let unknown = client.get_prompt("nope", Value::Null).await.unwrap_err();
    assert_eq!(rpc_code(unknown), error_code::INVALID_PARAMS);

    // MCP prompt -> Sinew templates, rendered locally.
    let t = client.prompt_template("summarize").await.unwrap();
    assert_eq!(t.variables(), ["style", "text"]);
    let vars = std::collections::HashMap::from([
        ("style".to_string(), "terse".to_string()),
        ("text".to_string(), "Long text.".to_string()),
    ]);
    assert_eq!(
        t.render_vars(&vars).unwrap(),
        "Summarize in a terse style (use {braces} literally):\nLong text."
    );
    let chat = client.chat_prompt_template("review").await.unwrap();
    assert_eq!(chat.variables(), ["focus", "code"]);
}

/// Next notification, failing after a timeout.
pub async fn next_notification(rx: &mut broadcast::Receiver<McpNotification>) -> McpNotification {
    tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("notification timed out")
        .unwrap()
}

/// Subscriptions, updates and list changes pushed by `handle`.
pub async fn exercise_notifications(client: &McpClient, handle: &McpServer) {
    let mut rx = client.notifications().await.unwrap();
    client.subscribe_resource("docs://readme").await.unwrap();
    assert_eq!(handle.notify_resource_updated("docs://readme"), 1);
    assert_eq!(
        next_notification(&mut rx).await,
        McpNotification::ResourceUpdated {
            uri: "docs://readme".into()
        }
    );
    // Not subscribed: nothing is sent.
    assert_eq!(handle.notify_resource_updated("docs://license"), 0);

    handle.add_resource(
        McpResource::new("docs://new", "new"),
        |r: ResourceRequest| async move { Ok(vec![ResourceContents::text(r.uri, "fresh")]) },
    );
    assert_eq!(
        next_notification(&mut rx).await,
        McpNotification::ResourceListChanged
    );
    assert_eq!(client.list_resources().await.unwrap().len(), 4);
    assert!(handle.remove_prompt("review"));
    assert_eq!(
        next_notification(&mut rx).await,
        McpNotification::PromptListChanged
    );
    assert_eq!(client.list_prompts().await.unwrap().len(), 1);

    client.unsubscribe_resource("docs://readme").await.unwrap();
    assert_eq!(handle.notify_resource_updated("docs://readme"), 0);
}
