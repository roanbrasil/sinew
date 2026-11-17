#![cfg(feature = "harness")]

use std::sync::Arc;

use serde_json::json;
use sinew_core::{mock::ScriptedModel, FnTool, Message, Role, ToolCall};
use sinew_graph::InMemoryStore;
use sinew_harness::Agent;
use sinew_memory::{MemoryHook, MemoryKind, MemoryManager, MemoryOptions};

fn call(name: &str, args: serde_json::Value) -> Message {
    Message::assistant_with_tools(
        "",
        vec![ToolCall {
            id: format!("{name}-1"),
            name: name.into(),
            arguments: args,
        }],
    )
}

#[tokio::test]
async fn hook_recalls_once_per_turn_and_saves_when_the_turn_ends() {
    let model = Arc::new(ScriptedModel::new([
        call("echo", json!({})),
        Message::assistant("Here is a vegetarian lasagna."),
        call(
            "save_memories",
            json!({"memories": [{"content": "The user wants dinner ideas on Fridays.", "kind": "episode"}]}),
        ),
        // "dinner" matches the stored preference, so the model consolidates.
        call(
            "update_memories",
            json!({"operations": [{"action": "add", "content": "The user wants dinner ideas on Fridays.", "kind": "episode"}]}),
        ),
    ]));
    let memory = MemoryManager::new(Arc::new(InMemoryStore::new()), model.clone());
    memory
        .add(
            "ana",
            "The user is vegetarian and cooks dinner.",
            MemoryKind::Preference,
        )
        .await
        .unwrap();
    let hook = MemoryHook::new(memory.clone(), "ana");
    let agent = Agent::builder()
        .model_arc(model.clone())
        .tool(FnTool::new(
            "echo",
            "echo",
            json!({"type": "object"}),
            |_| Ok(json!("ok")),
        ))
        .hook(hook.clone())
        .build()
        .unwrap();

    let out = agent.ask("Suggest a dinner for Friday").await.unwrap();
    assert_eq!(out.text, "Here is a vegetarian lasagna.");
    let saved = hook.flush().await;
    assert_eq!(saved.len(), 1);
    let report = saved.into_iter().next().unwrap().unwrap();
    assert_eq!(report.added[0].kind, MemoryKind::Episode);
    assert_eq!(
        report.added[0].source_thread.as_deref(),
        Some(out.thread_id.as_str())
    );

    let reqs = model.requests();
    assert_eq!(reqs.len(), 4);
    for r in &reqs[..2] {
        assert_eq!(r.messages[0].role, Role::System);
        assert!(r.messages[0]
            .content
            .ends_with("- The user is vegetarian and cooks dinner."));
    }
    // The extraction saw the user message and the final answer, not the tool output.
    let transcript = &reqs[2].messages.last().unwrap().content;
    assert!(transcript.contains("user: Suggest a dinner for Friday"));
    assert!(transcript.contains("assistant: Here is a vegetarian lasagna."));
    assert_eq!(memory.list("ana").await.unwrap().len(), 2);
}

#[tokio::test]
async fn hook_resolves_the_user_per_thread_and_can_save_synchronously() {
    let model = Arc::new(ScriptedModel::new([
        Message::assistant("Hello!"),
        call(
            "save_memories",
            json!({"memories": [{"content": "The user is Bob.", "kind": "fact"}]}),
        ),
    ]));
    let memory = MemoryManager::new(Arc::new(InMemoryStore::new()), model.clone());
    let hook = MemoryHook::with_user_resolver(memory.clone(), |ctx| {
        ctx.thread_id
            .split('/')
            .next()
            .unwrap_or_default()
            .to_string()
    })
    .with_options(MemoryOptions {
        background: false,
        ..Default::default()
    });
    let agent = Agent::builder()
        .model_arc(model.clone())
        .hook(hook.clone())
        .build()
        .unwrap();
    agent.run("bob/chat-1", "I'm Bob").await.unwrap();
    assert!(hook.flush().await.is_empty());
    assert_eq!(
        memory.list("bob").await.unwrap()[0].content,
        "The user is Bob."
    );
    assert!(memory.list("ana").await.unwrap().is_empty());
}
