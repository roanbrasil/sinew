//! Resources, resource templates, prompts and notifications between a client
//! and a server connected through in-memory pipes (stdio framing).

mod common;

use serde_json::json;
use sinew_mcp::{McpClient, McpServer};

async fn connect(server: McpServer) -> McpClient {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let (sr, sw) = tokio::io::split(server_io);
    tokio::spawn(server.serve(sr, sw));
    let (cr, cw) = tokio::io::split(client_io);
    McpClient::connect(cr, cw).await.unwrap()
}

#[tokio::test]
async fn every_resource_and_prompt_method_over_stdio_framing() {
    let client = connect(common::docs_server()).await;
    common::exercise_resources_and_prompts(&client).await;
}

#[tokio::test]
async fn subscriptions_and_list_changes_over_stdio_framing() {
    let server = common::docs_server();
    let handle = server.clone();
    let client = connect(server).await;
    common::exercise_notifications(&client, &handle).await;
}

#[tokio::test]
async fn servers_without_resources_or_prompts_reject_the_methods() {
    let client = connect(McpServer::new("tools-only", "1.0")).await;
    let caps = &client.server_info().capabilities;
    assert!(caps.get("resources").is_none() && caps.get("prompts").is_none());
    assert!(client.list_resources().await.is_err());
    assert!(client.list_prompts().await.is_err());
    assert!(client.get_prompt("x", json!({})).await.is_err());
}

#[cfg(feature = "rag")]
#[tokio::test]
async fn resources_become_rag_documents() {
    let client = connect(common::docs_server()).await;
    let docs = client
        .read_resource_documents("docs://readme")
        .await
        .unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].content, "# Sinew\nRust agents.");
    assert_eq!(docs[0].source(), Some("docs://readme"));
    assert_eq!(docs[0].metadata["mime_type"], "text/markdown");
    // Blob contents are skipped.
    assert!(client
        .read_resource_documents("docs://logo")
        .await
        .unwrap()
        .is_empty());
}
