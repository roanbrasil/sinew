use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_checkpoint::SqliteStore;
use sinew_core::{mock::ScriptedModel, Message, Role, ToolBox, ToolCall};
use sinew_graph::{
    prebuilt::{AgentState, ReactAgent},
    InMemoryStore, MemoryCheckpointer, Store,
};
use sinew_memory::{Memory, MemoryAgent, MemoryKind, MemoryManager, MemoryOptions, RememberScope};
use sinew_rag::Embeddings;

/// Deterministic embeddings: one dimension per topic word.
struct TopicEmbeddings;

const TOPICS: &[&str] = &["coffee", "tea", "cat", "dog", "lisbon", "montreal", "rust"];

#[async_trait]
impl Embeddings for TopicEmbeddings {
    fn model(&self) -> &str {
        "topics-v1"
    }
    async fn embed_documents(&self, texts: &[String]) -> sinew_rag::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let t = t.to_lowercase();
                TOPICS
                    .iter()
                    .map(|w| if t.contains(w) { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect())
    }
}

fn call(name: &str, args: Value) -> Message {
    Message::assistant_with_tools(
        "",
        vec![ToolCall {
            id: format!("{name}-call"),
            name: name.into(),
            arguments: args,
        }],
    )
}

fn extract(memories: Value) -> Message {
    call("save_memories", json!({ "memories": memories }))
}

fn update(ops: Value) -> Message {
    call("update_memories", json!({ "operations": ops }))
}

fn manager(model: Arc<ScriptedModel>) -> MemoryManager {
    MemoryManager::new(Arc::new(InMemoryStore::new()), model)
}

fn chat(user: &str) -> Vec<Message> {
    vec![Message::user(user), Message::assistant("Got it.")]
}

#[tokio::test]
async fn extraction_adds_memories_with_metadata() {
    let model = Arc::new(ScriptedModel::new([extract(json!([
        {"content": "The user drinks coffee every morning.", "kind": "preference", "importance": 0.4},
        {"content": "The user lives in Lisbon.", "kind": "fact", "importance": 1.7},
        {"content": "  ", "kind": "fact"},
    ]))]));
    let memory = manager(model.clone());
    let report = memory
        .remember(
            "ana",
            &[
                Message::system("ignored"),
                Message::user("I live in Lisbon and drink coffee every morning."),
                Message::assistant("Nice!"),
            ],
            Some("t1"),
        )
        .await
        .unwrap();

    assert_eq!(report.added.len(), 2);
    assert!(report.updated.is_empty() && report.deleted.is_empty());
    let lisbon = &report.added[1];
    assert_eq!(lisbon.kind, MemoryKind::Fact);
    assert_eq!(lisbon.importance, Some(1.0)); // clamped
    assert_eq!(lisbon.source_thread.as_deref(), Some("t1"));
    assert_eq!(lisbon.created_at, lisbon.updated_at);

    // One model call (nothing to consolidate against), with the transcript.
    let reqs = model.requests();
    assert_eq!(reqs.len(), 1);
    let prompt = &reqs[0].messages.last().unwrap().content;
    assert!(prompt.contains("user: I live in Lisbon"));
    assert!(prompt.contains("assistant: Nice!"));
    assert!(!prompt.contains("ignored"));

    let stored = memory.get("ana", &lisbon.id).await.unwrap().unwrap();
    assert_eq!(&stored, lisbon);
    assert_eq!(memory.list("ana").await.unwrap().len(), 2);
}

#[tokio::test]
async fn custom_extraction_prompt_is_sent() {
    let model = Arc::new(ScriptedModel::new([extract(json!([]))]));
    let memory = manager(model.clone()).with_extraction_prompt("Only remember pets.");
    let report = memory
        .remember("ana", &chat("I like tea"), None)
        .await
        .unwrap();
    assert!(report.is_empty());
    assert_eq!(
        model.requests()[0].messages[1].content,
        "Only remember pets."
    );
}

#[tokio::test]
async fn exact_duplicates_are_skipped_without_a_second_call() {
    let model = Arc::new(ScriptedModel::new([extract(json!([
        {"content": "the user has a cat.", "kind": "fact"},
        {"content": "The user has a cat", "kind": "fact"},
    ]))]));
    let memory = manager(model.clone());
    memory
        .add("ana", "The user has a cat.", MemoryKind::Fact)
        .await
        .unwrap();
    let report = memory
        .remember("ana", &chat("I have a cat"), None)
        .await
        .unwrap();
    assert!(report.is_empty());
    assert_eq!(model.requests().len(), 1);
    assert_eq!(memory.list("ana").await.unwrap().len(), 1);
}

#[tokio::test]
async fn consolidation_updates_deletes_and_adds() {
    let model = Arc::new(ScriptedModel::new([
        extract(json!([
            {"content": "The user moved to Montreal and now drinks tea instead of coffee.", "kind": "fact"},
            {"content": "The user no longer has a dog.", "kind": "fact"},
            {"content": "The user is learning Rust.", "kind": "fact", "importance": 0.6},
        ])),
        // Candidates are numbered in the order they were found.
        update(json!([
            {"action": "update", "id": "0", "content": "The user drinks tea, not coffee.", "kind": "preference"},
            {"action": "delete", "id": "1"},
            {"action": "add", "content": "The user is learning Rust.", "kind": "fact", "importance": 0.6},
            {"action": "update", "id": "0", "content": "ignored second update"},
            {"action": "delete", "id": "99"},
            {"action": "none", "id": "2"},
        ])),
    ]));
    let memory = manager(model.clone()).with_embeddings(TopicEmbeddings);
    let coffee = memory
        .add("ana", "The user drinks coffee.", MemoryKind::Preference)
        .await
        .unwrap();
    let dog = memory
        .add("ana", "The user has a dog.", MemoryKind::Fact)
        .await
        .unwrap();
    let cat = memory
        .add("ana", "The user has a cat named Miso.", MemoryKind::Fact)
        .await
        .unwrap();

    let report = memory
        .remember("ana", &chat("..."), Some("t2"))
        .await
        .unwrap();

    // The update call saw the similar memories with short ids, not the cat.
    let reqs = model.requests();
    assert_eq!(reqs.len(), 2);
    let prompt = &reqs[1].messages.last().unwrap().content;
    assert!(prompt.contains("\"id\": \"0\""));
    assert!(prompt.contains("The user drinks coffee."));
    assert!(prompt.contains("The user has a dog."));
    assert!(!prompt.contains("Miso"));
    assert!(!prompt.contains(&coffee.id));

    assert_eq!(report.updated.len(), 1);
    let updated = &report.updated[0];
    assert_eq!(updated.id, coffee.id);
    assert_eq!(updated.content, "The user drinks tea, not coffee.");
    assert_eq!(updated.created_at, coffee.created_at);
    assert_eq!(updated.source_thread.as_deref(), Some("t2"));
    assert_eq!(report.deleted, vec![dog.clone()]);
    assert_eq!(report.added.len(), 1);
    assert_eq!(report.added[0].importance, Some(0.6));

    let all = memory.list("ana").await.unwrap();
    assert_eq!(all.len(), 3);
    assert!(memory.get("ana", &dog.id).await.unwrap().is_none());
    assert!(memory.get("ana", &cat.id).await.unwrap().is_some());

    // The updated memory's vector was refreshed: it is now found by "tea".
    let hits = memory.recall("ana", "tea", 1).await.unwrap();
    assert_eq!(hits[0].memory.id, coffee.id);
}

#[tokio::test]
async fn update_with_unknown_id_becomes_an_add() {
    let model = Arc::new(ScriptedModel::new([
        extract(json!([{"content": "The user has a cat called Rex.", "kind": "fact"}])),
        update(
            json!([{"action": "update", "id": "7", "content": "The user has a cat called Rex."}]),
        ),
    ]));
    let memory = manager(model);
    memory
        .add("ana", "The user has a cat.", MemoryKind::Fact)
        .await
        .unwrap();
    let report = memory
        .remember("ana", &chat("my cat is Rex"), None)
        .await
        .unwrap();
    assert_eq!(report.added.len(), 1);
    assert_eq!(memory.list("ana").await.unwrap().len(), 2);
}

#[tokio::test]
async fn forget_and_forget_all() {
    let memory = manager(Arc::new(ScriptedModel::new([])));
    let a = memory
        .add("ana", "The user likes tea.", MemoryKind::Preference)
        .await
        .unwrap();
    memory
        .add("ana", "The user has a cat.", MemoryKind::Fact)
        .await
        .unwrap();
    memory
        .add("bob", "The user has a dog.", MemoryKind::Fact)
        .await
        .unwrap();
    assert!(memory.forget("ana", &a.id).await.unwrap());
    assert!(!memory.forget("ana", &a.id).await.unwrap());
    assert_eq!(memory.list("ana").await.unwrap().len(), 1);
    assert_eq!(memory.forget_all("ana").await.unwrap(), 1);
    assert!(memory.list("ana").await.unwrap().is_empty());
    assert_eq!(memory.list("bob").await.unwrap().len(), 1);
}

#[tokio::test]
async fn semantic_recall_ranks_by_similarity() {
    let memory = manager(Arc::new(ScriptedModel::new([]))).with_embeddings(TopicEmbeddings);
    for (c, k) in [
        ("The user lives in Montreal.", MemoryKind::Fact),
        ("The user prefers tea.", MemoryKind::Preference),
        (
            "The user writes Rust and drinks tea while coding.",
            MemoryKind::Fact,
        ),
    ] {
        memory.add("ana", c, k).await.unwrap();
    }
    let hits = memory.recall("ana", "what tea?", 0).await.unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].memory.content, "The user prefers tea.");
    assert!(hits[0].score > hits[2].score);
    assert_eq!(hits[2].memory.content, "The user lives in Montreal.");
    assert_eq!(memory.recall("ana", "tea", 2).await.unwrap().len(), 2);

    let strict = memory.clone().with_min_score(0.5);
    assert_eq!(strict.recall("ana", "tea", 0).await.unwrap().len(), 2);
}

#[tokio::test]
async fn keyword_recall_without_embeddings() {
    let memory = manager(Arc::new(ScriptedModel::new([])));
    let important = Memory::new("The user plays piano.", MemoryKind::Fact).with_importance(0.9);
    memory.put("ana", important.clone()).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    memory
        .add("ana", "The user plays chess.", MemoryKind::Fact)
        .await
        .unwrap();
    memory
        .add("ana", "The user is allergic to peanuts.", MemoryKind::Fact)
        .await
        .unwrap();

    let hits = memory
        .recall("ana", "any peanut allergies?", 5)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].memory.content.contains("peanuts"));

    // Equal scores: the more important memory first, though it is older.
    let hits = memory
        .recall("ana", "what does the user play", 5)
        .await
        .unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].score, hits[1].score);
    assert_eq!(hits[0].memory.id, important.id);
    assert!(hits[1].memory.content.contains("chess"));

    // Empty query: newest first, score 0.
    let recent = memory.recall("ana", "", 0).await.unwrap();
    assert_eq!(recent.len(), 3);
    assert!(recent.iter().all(|h| h.score == 0.0));

    assert!(memory.recall("ana", "volcano", 5).await.unwrap().is_empty());
    let block = memory.context("ana", "peanut", 3).await.unwrap().unwrap();
    assert!(block.ends_with("\n- The user is allergic to peanuts.\n</memories>"));
    assert!(memory.context("ana", "volcano", 3).await.unwrap().is_none());
}

#[tokio::test]
async fn namespaces_are_isolated_per_user_and_prefix() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
    let model = Arc::new(ScriptedModel::new([]));
    let a = MemoryManager::new(store.clone(), model.clone());
    let b = MemoryManager::new(store.clone(), model).with_namespace_prefix(["app2", "mem"]);
    a.add("ana", "The user likes tea.", MemoryKind::Preference)
        .await
        .unwrap();
    a.add("ana/x", "Nested user id.", MemoryKind::Fact)
        .await
        .unwrap();
    b.add("ana", "The user likes coffee.", MemoryKind::Preference)
        .await
        .unwrap();

    assert_eq!(a.namespace("ana"), vec!["memories", "ana"]);
    assert_eq!(a.list("ana").await.unwrap().len(), 1);
    assert!(a.recall("bob", "tea", 5).await.unwrap().is_empty());
    let b_hits = b.recall("ana", "likes", 5).await.unwrap();
    assert_eq!(b_hits.len(), 1);
    assert!(b_hits[0].memory.content.contains("coffee"));
    let item = store
        .get(&["app2", "mem", "ana"], &b_hits[0].memory.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(item.value["kind"], "preference");
}

#[tokio::test]
async fn works_on_sqlite_store_and_reembeds_on_model_change() {
    let store: Arc<dyn Store> = Arc::new(SqliteStore::in_memory().await.unwrap());
    let model = Arc::new(ScriptedModel::new([]));
    let plain = MemoryManager::new(store.clone(), model.clone());
    plain
        .add("ana", "The user prefers tea.", MemoryKind::Preference)
        .await
        .unwrap();
    plain
        .add("ana", "The user has a dog.", MemoryKind::Fact)
        .await
        .unwrap();

    // Saved without vectors; embedded (and written back) on first semantic recall.
    let semantic = MemoryManager::new(store.clone(), model).with_embeddings(TopicEmbeddings);
    let hits = semantic.recall("ana", "dog", 1).await.unwrap();
    assert_eq!(hits[0].memory.content, "The user has a dog.");
    let ns = ["memories", "ana"];
    let item = store.get(&ns, &hits[0].memory.id).await.unwrap().unwrap();
    assert_eq!(item.value["embedding_model"], "topics-v1");
    assert!(item.value["embedding"].is_array());
}

#[tokio::test]
async fn react_agent_recalls_before_and_saves_after_each_turn() {
    let model = Arc::new(ScriptedModel::new([
        // Turn 1: answer, then extraction.
        Message::assistant("Enjoy your coffee!"),
        extract(json!([{"content": "The user drinks coffee.", "kind": "preference"}])),
        // Turn 2: answer, then extraction with nothing new.
        Message::assistant("You like coffee."),
        extract(json!([])),
    ]));
    let memory = manager(model.clone()).with_embeddings(TopicEmbeddings);
    let agent = MemoryAgent::new(
        ReactAgent::new(model.clone(), ToolBox::new()).system_prompt("You are helpful."),
        memory.clone(),
    )
    .unwrap();

    let out = agent
        .invoke(
            "ana",
            AgentState::new(vec![Message::user("I always drink coffee")]),
        )
        .await
        .unwrap();
    assert_eq!(out.last().unwrap().content, "Enjoy your coffee!");
    let saved = agent.flush().await;
    assert_eq!(saved.len(), 1);
    assert_eq!(saved[0].as_ref().unwrap().added.len(), 1);

    let out = agent
        .invoke(
            "ana",
            AgentState::new(vec![Message::user("What coffee do I like?")]),
        )
        .await
        .unwrap();
    assert!(out.messages.iter().all(|m| m.role != Role::System));
    agent.flush().await;

    let reqs = model.requests();
    assert_eq!(reqs.len(), 4);
    // Turn 1 had no memories: the system prompt is unchanged.
    assert_eq!(reqs[0].messages[0].content, "You are helpful.");
    // Turn 2 saw the memory appended to the system prompt.
    let system = &reqs[2].messages[0].content;
    assert!(system.starts_with("You are helpful.\n\n"));
    assert!(system.contains("- The user drinks coffee."));
    // The extraction of turn 2 got only that turn.
    let transcript = &reqs[3].messages.last().unwrap().content;
    assert!(transcript.contains("What coffee do I like?"));
    assert!(!transcript.contains("I always drink coffee"));

    // Another user sees nothing.
    assert!(memory.recall("bob", "coffee", 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn react_agent_with_checkpointer_and_tools_recalls_once_per_turn() {
    let model = Arc::new(ScriptedModel::new([
        call("lookup", json!({})),
        Message::assistant("Done."),
        extract(json!([])),
    ]));
    let tools = ToolBox::new().with(sinew_core::FnTool::new(
        "lookup",
        "look something up",
        json!({"type": "object"}),
        |_| Ok(json!("ok")),
    ));
    let embeddings = Arc::new(sinew_rag::FakeEmbeddings::new(32));
    let memory = manager(model.clone()).with_embeddings_arc(embeddings.clone());
    memory
        .add("ana", "The user likes lookups.", MemoryKind::Preference)
        .await
        .unwrap();
    let calls_before = embeddings.calls();
    let agent = MemoryAgent::new(ReactAgent::new(model.clone(), tools), memory)
        .unwrap()
        .map_graph(|g| g.with_checkpointer(MemoryCheckpointer::new()))
        .with_options(MemoryOptions {
            background: false,
            scope: RememberScope::AllMessages,
            recall_limit: 3,
        });
    let outcome = agent
        .run(
            "ana",
            "thread-1",
            Some(AgentState::new(vec![Message::user("look it up")])),
        )
        .await
        .unwrap();
    assert!(matches!(outcome, sinew_graph::RunOutcome::Completed(_)));
    let reqs = model.requests();
    // Both model calls of the turn carry the memory; one query embedding.
    assert!(reqs[0].messages[0].content.contains("likes lookups"));
    assert!(reqs[1].messages[0].content.contains("likes lookups"));
    assert_eq!(embeddings.calls() - calls_before, 1);
    // Synchronous save already happened, with the whole conversation.
    assert_eq!(reqs.len(), 3);
    assert!(agent.flush().await.is_empty());
}
