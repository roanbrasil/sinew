//! End-to-end: an in-process server on 127.0.0.1:0 and the client against it.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{json, Value};
use sinew_a2a::{
    jsonrpc::error_code, A2aAgentTool, A2aClient, A2aError, A2aServer, AgentCard, AgentExecutor,
    AgentSkill, BoxError, ChatModelExecutor, EventQueue, GraphExecutor, Message, MessageSendParams,
    Part, RequestContext, SendMessageResult, ServerHandle, StreamResponse, TaskState,
};
use sinew_core::{mock::ScriptedModel, FnTool, Tool, ToolBox, ToolCall};
use sinew_graph::{
    prebuilt::{react_agent, AgentState},
    MemoryCheckpointer,
};
use tokio::{sync::Notify, time::timeout};

const LIMIT: Duration = Duration::from_secs(10);

fn card() -> AgentCard {
    AgentCard::new("Test Agent", "An agent for tests", "")
        .with_skill(AgentSkill::new("echo", "Echo", "Repeats things").with_tags(["test"]))
}

async fn spawn(executor: impl AgentExecutor) -> (ServerHandle, A2aClient) {
    let server = A2aServer::new(card(), executor)
        .spawn("127.0.0.1:0")
        .await
        .unwrap();
    let client = A2aClient::discover(&server.base_url()).await.unwrap();
    (server, client)
}

fn scripted(replies: &[&str]) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new(
        replies.iter().map(|r| sinew_core::Message::assistant(*r)),
    ))
}

async fn collect(mut s: sinew_a2a::A2aStream) -> Vec<StreamResponse> {
    let mut out = Vec::new();
    while let Some(ev) = timeout(LIMIT, s.next()).await.expect("stream stalled") {
        out.push(ev.unwrap());
    }
    out
}

fn rpc_code(e: A2aError) -> i64 {
    match e {
        A2aError::Rpc(r) => r.code,
        other => panic!("expected an RPC error, got {other}"),
    }
}

/// Emits `first`, waits for `gate`, emits `second`, completes.
struct Gated {
    gate: Arc<Notify>,
}

#[async_trait]
impl AgentExecutor for Gated {
    async fn execute(&self, _ctx: RequestContext, events: EventQueue) -> Result<(), BoxError> {
        events.working();
        let mut w = events.text_artifact("out");
        w.push("first");
        self.gate.notified().await;
        w.push(" second");
        w.finish();
        events.complete(None);
        Ok(())
    }
}

/// Works forever (until canceled).
struct Forever;

#[async_trait]
impl AgentExecutor for Forever {
    async fn execute(&self, _ctx: RequestContext, events: EventQueue) -> Result<(), BoxError> {
        events.progress("thinking hard");
        std::future::pending::<()>().await;
        Ok(())
    }
}

/// Replies with a message, no task.
struct Replier;

#[async_trait]
impl AgentExecutor for Replier {
    async fn execute(&self, ctx: RequestContext, events: EventQueue) -> Result<(), BoxError> {
        events.reply(vec![Part::text(format!("you said: {}", ctx.text()))]);
        Ok(())
    }
}

struct Failing;

#[async_trait]
impl AgentExecutor for Failing {
    async fn execute(&self, _ctx: RequestContext, events: EventQueue) -> Result<(), BoxError> {
        events.working();
        Err("database unavailable".into())
    }
}

#[tokio::test]
async fn serves_the_agent_card() {
    let (server, client) = spawn(ChatModelExecutor::new(scripted(&[]))).await;
    let card = client.card().unwrap();
    assert_eq!(card.name, "Test Agent");
    assert_eq!(card.url, server.url(), "spawn fills in the url");
    assert_eq!(card.protocol_version, "0.3.0");
    // Legacy path too.
    let legacy: AgentCard = reqwest::get(format!("{}/.well-known/agent.json", server.base_url()))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(legacy.name, "Test Agent");
}

#[tokio::test]
async fn send_returns_completed_task_with_artifact() {
    let (_server, client) =
        spawn(ChatModelExecutor::new(scripted(&["Hello there, friend."]))).await;
    let r = client.send_text("hi").await.unwrap();
    let task = r.as_task().expect("a task").clone();
    assert_eq!(task.status.state, TaskState::Completed);
    assert_eq!(r.text(), "Hello there, friend.");
    assert_eq!(task.artifacts.len(), 1);
    assert_eq!(task.artifacts[0].name.as_deref(), Some("response"));
    assert_eq!(task.artifacts[0].parts.len(), 1, "chunks were merged");
    assert_eq!(task.history[0].text(), "hi");
    assert_eq!(task.history[0].task_id.as_deref(), Some(task.id.as_str()));

    let got = client.get_task(&task.id, None).await.unwrap();
    assert_eq!(got, task);
    let trimmed = client.get_task(&task.id, Some(0)).await.unwrap();
    assert!(trimmed.history.is_empty());
}

#[tokio::test]
async fn stream_yields_task_status_and_artifact_chunks_in_order() {
    let (_server, client) = spawn(ChatModelExecutor::new(scripted(&["one two three four"]))).await;
    let events = collect(client.stream_text("count").await.unwrap()).await;

    let StreamResponse::Task(t) = &events[0] else {
        panic!("first event should be the task, got {:?}", events[0])
    };
    assert_eq!(t.status.state, TaskState::Submitted);
    let StreamResponse::StatusUpdate(s) = &events[1] else {
        panic!("expected working")
    };
    assert_eq!(s.status.state, TaskState::Working);
    assert!(!s.is_final);

    let chunks: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            StreamResponse::ArtifactUpdate(a) => Some(a),
            _ => None,
        })
        .collect();
    assert_eq!(chunks.len(), 5, "4 word chunks + closing chunk");
    assert_eq!(chunks[0].append, Some(false));
    assert!(chunks[1..].iter().all(|c| c.append == Some(true)));
    assert_eq!(chunks.last().unwrap().last_chunk, Some(true));
    let text: String = chunks
        .iter()
        .map(|c| sinew_a2a::types::parts_text(&c.artifact.parts))
        .collect();
    assert_eq!(text, "one two three four");

    let StreamResponse::StatusUpdate(last) = events.last().unwrap() else {
        panic!("expected final status")
    };
    assert_eq!(last.status.state, TaskState::Completed);
    assert!(last.is_final);

    let stored = client.get_task(&t.id, None).await.unwrap();
    assert_eq!(
        stored.artifacts[0].parts[0].as_text(),
        Some("one two three four")
    );
}

/// The first chunk must reach the client while the agent is still running:
/// the agent only continues after the test has seen it. Any buffering of the
/// response would deadlock (and time out).
#[tokio::test]
async fn chunks_are_delivered_without_buffering() {
    let gate = Arc::new(Notify::new());
    let (_server, client) = spawn(Gated { gate: gate.clone() }).await;
    let mut s = client.stream_text("go").await.unwrap();
    loop {
        let ev = timeout(LIMIT, s.next())
            .await
            .expect("first chunk was buffered")
            .unwrap()
            .unwrap();
        if let StreamResponse::ArtifactUpdate(a) = &ev {
            assert_eq!(a.artifact.parts[0].as_text(), Some("first"));
            break;
        }
    }
    gate.notify_one();
    let rest = collect(s).await;
    assert!(matches!(
        rest.last(),
        Some(StreamResponse::StatusUpdate(s)) if s.status.state == TaskState::Completed
    ));
}

#[tokio::test]
async fn cancel_stops_a_running_task() {
    let (_server, client) = spawn(Forever).await;
    let mut s = client.stream_text("work").await.unwrap();
    let task_id = loop {
        match timeout(LIMIT, s.next()).await.unwrap().unwrap().unwrap() {
            StreamResponse::StatusUpdate(u) => {
                assert_eq!(u.status.message.unwrap().text(), "thinking hard");
                break u.task_id;
            }
            _ => continue,
        }
    };
    let canceled = client.cancel_task(&task_id).await.unwrap();
    assert_eq!(canceled.status.state, TaskState::Canceled);
    // The progress note moved into the history.
    assert!(canceled.history.iter().any(|m| m.text() == "thinking hard"));

    let rest = collect(s).await;
    assert!(matches!(
        rest.last(),
        Some(StreamResponse::StatusUpdate(u)) if u.status.state == TaskState::Canceled && u.is_final
    ));
    assert_eq!(
        client.get_task(&task_id, None).await.unwrap().status.state,
        TaskState::Canceled
    );
    let again = client.cancel_task(&task_id).await.unwrap_err();
    assert_eq!(rpc_code(again), error_code::TASK_NOT_CANCELABLE);
}

#[tokio::test]
async fn resubscribe_backfills_then_streams_live() {
    let gate = Arc::new(Notify::new());
    let (_server, client) = spawn(Gated { gate: gate.clone() }).await;
    let mut s = client.stream_text("go").await.unwrap();
    let task_id = loop {
        if let StreamResponse::ArtifactUpdate(a) =
            timeout(LIMIT, s.next()).await.unwrap().unwrap().unwrap()
        {
            break a.task_id;
        }
    };
    drop(s); // client disconnects; the task keeps running

    let mut s = client.resubscribe(&task_id).await.unwrap();
    let StreamResponse::Task(snapshot) = timeout(LIMIT, s.next()).await.unwrap().unwrap().unwrap()
    else {
        panic!("resubscribe starts with a snapshot")
    };
    assert_eq!(snapshot.status.state, TaskState::Working);
    assert_eq!(snapshot.artifacts[0].parts[0].as_text(), Some("first"));

    gate.notify_one();
    let rest = collect(s).await;
    let live_text: String = rest.iter().filter_map(StreamResponse::text).collect();
    assert_eq!(live_text, " second");
    assert!(rest.last().unwrap().is_final());
    let done = client.get_task(&task_id, None).await.unwrap();
    assert_eq!(done.artifacts[0].parts[0].as_text(), Some("first second"));

    // Resubscribing to a finished task returns its final snapshot.
    let after = collect(client.resubscribe(&task_id).await.unwrap()).await;
    assert!(
        matches!(&after[..], [StreamResponse::Task(t)] if t.status.state == TaskState::Completed)
    );
}

#[tokio::test]
async fn non_blocking_send_returns_early() {
    let gate = Arc::new(Notify::new());
    let (_server, client) = spawn(Gated { gate: gate.clone() }).await;
    let r = client
        .send_message(MessageSendParams::new(Message::user_text("go")).non_blocking())
        .await
        .unwrap();
    let task = r.as_task().unwrap();
    assert!(!task.status.state.is_terminal());
    gate.notify_one();
    let mut state = task.status.state;
    for _ in 0..200 {
        state = client.get_task(&task.id, None).await.unwrap().status.state;
        if state == TaskState::Completed {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(state, TaskState::Completed);
}

#[tokio::test]
async fn direct_message_replies() {
    let (_server, client) = spawn(Replier).await;
    let r = client.send_text("ping").await.unwrap();
    let SendMessageResult::Message(m) = r else {
        panic!("expected a message")
    };
    assert_eq!(m.text(), "you said: ping");
    // No task was persisted for a direct reply.
    let missing = client
        .get_task(m.task_id.as_deref().unwrap(), None)
        .await
        .unwrap_err();
    assert_eq!(rpc_code(missing), error_code::TASK_NOT_FOUND);

    let events = collect(client.stream_text("pong").await.unwrap()).await;
    assert!(matches!(&events[..], [StreamResponse::Message(m)] if m.text() == "you said: pong"));
}

#[tokio::test]
async fn executor_errors_fail_the_task() {
    let (_server, client) = spawn(Failing).await;
    let r = client.send_text("x").await.unwrap();
    let t = r.as_task().unwrap();
    assert_eq!(t.status.state, TaskState::Failed);
    assert_eq!(r.text(), "database unavailable");
}

#[tokio::test]
async fn protocol_errors() {
    let (server, client) = spawn(ChatModelExecutor::new(scripted(&["done"]))).await;

    let e = client
        .request::<Value>("tasks/frobnicate", &json!({}))
        .await
        .unwrap_err();
    assert_eq!(rpc_code(e), error_code::METHOD_NOT_FOUND);

    let e = client.get_task("nope", None).await.unwrap_err();
    assert_eq!(rpc_code(e), error_code::TASK_NOT_FOUND);

    let e = client
        .request::<Value>("message/send", &json!({"message": 5}))
        .await
        .unwrap_err();
    assert_eq!(rpc_code(e), error_code::INVALID_PARAMS);

    let e = client
        .request::<Value>(
            "tasks/pushNotificationConfig/set",
            &json!({"taskId": "t", "pushNotificationConfig": {"url": "http://x"}}),
        )
        .await
        .unwrap_err();
    assert_eq!(rpc_code(e), error_code::PUSH_NOTIFICATION_NOT_SUPPORTED);

    // A finished task can't be continued.
    let t = client.send_text("a").await.unwrap();
    let id = t.as_task().unwrap().id.clone();
    let e = client
        .send_message(MessageSendParams::new(
            Message::user_text("more").with_task_id(id),
        ))
        .await
        .unwrap_err();
    assert_eq!(rpc_code(e), error_code::INVALID_PARAMS);

    // Raw HTTP: malformed JSON and a batch.
    let http = reqwest::Client::new();
    for (body, code) in [
        ("{not json", error_code::PARSE_ERROR),
        ("[]", error_code::INVALID_REQUEST),
        (
            r#"{"jsonrpc":"1.0","id":1,"method":"tasks/get"}"#,
            error_code::INVALID_REQUEST,
        ),
    ] {
        let v: Value = http
            .post(server.url())
            .body(body)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(v["error"]["code"], code, "{body}");
    }

    // An error answer to a streaming method is plain JSON.
    let e = match client.resubscribe("nope").await {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert_eq!(rpc_code(e), error_code::TASK_NOT_FOUND);
}

#[tokio::test]
async fn many_concurrent_requests() {
    let replies: Vec<String> = (0..20).map(|_| "same answer".to_string()).collect();
    let refs: Vec<&str> = replies.iter().map(String::as_str).collect();
    let (_server, client) = spawn(ChatModelExecutor::new(scripted(&refs))).await;
    let calls = (0..20).map(|i| {
        let c = client.clone();
        async move { c.send_text(format!("q{i}")).await.unwrap().text() }
    });
    let answers = futures_util::future::join_all(calls).await;
    assert!(answers.iter().all(|a| a == "same answer"));
}

fn adder() -> ToolBox {
    ToolBox::new().with(FnTool::new(
        "add",
        "Adds a and b",
        json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}),
        |a: Value| {
            Ok(json!(
                a["a"].as_i64().unwrap_or(0) + a["b"].as_i64().unwrap_or(0)
            ))
        },
    ))
}

fn add_call() -> sinew_core::Message {
    sinew_core::Message::assistant_with_tools(
        "",
        vec![ToolCall {
            id: "c1".into(),
            name: "add".into(),
            arguments: json!({"a": 2, "b": 40}),
        }],
    )
}

#[tokio::test]
async fn react_agent_streams_tokens_through_a2a() {
    let model = Arc::new(ScriptedModel::new([
        add_call(),
        sinew_core::Message::assistant("The answer is 42."),
    ]));
    let graph = react_agent(model, adder()).unwrap();
    let (_server, client) = spawn(GraphExecutor::agent(graph)).await;
    let events = collect(client.stream_text("2 + 40?").await.unwrap()).await;
    let chunks: Vec<String> = events
        .iter()
        .filter(|e| matches!(e, StreamResponse::ArtifactUpdate(_)))
        .filter_map(StreamResponse::text)
        .filter(|t| !t.is_empty())
        .collect();
    assert_eq!(chunks, vec!["The", " answer", " is", " 42."]);
    let task_id = match &events[0] {
        StreamResponse::Task(t) => t.id.clone(),
        other => panic!("{other:?}"),
    };
    let t = client.get_task(&task_id, None).await.unwrap();
    assert_eq!(t.status.state, TaskState::Completed);
    assert_eq!(sinew_a2a::types::task_text(&t), "The answer is 42.");
}

#[tokio::test]
async fn graph_interrupt_maps_to_input_required() {
    let model = Arc::new(ScriptedModel::new([
        add_call(),
        sinew_core::Message::assistant("2 + 40 = 42."),
        add_call(),
    ]));
    let graph = react_agent(model.clone(), adder())
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new())
        .interrupt_before(["tools"]);
    let (_server, client) = spawn(GraphExecutor::agent(graph)).await;

    let r = client.send_text("2 + 40?").await.unwrap();
    let t = r.as_task().unwrap().clone();
    assert_eq!(t.status.state, TaskState::InputRequired);
    let question = t.status.message.as_ref().unwrap();
    assert!(
        question.text().contains(r#"add({"a":2,"b":40})"#),
        "{}",
        question.text()
    );
    assert!(
        matches!(&question.parts[1], Part::Data { data, .. } if data["interrupt"]["next"] == json!(["tools"]))
    );

    // Approve: the same task continues and completes.
    let r = client
        .send_message(MessageSendParams::new(
            Message::user_text("yes").with_task_id(&t.id),
        ))
        .await
        .unwrap();
    let done = r.as_task().unwrap();
    assert_eq!(done.id, t.id);
    assert_eq!(done.status.state, TaskState::Completed);
    assert_eq!(r.text(), "2 + 40 = 42.");
    // History: question, approval and the earlier messages.
    let texts: Vec<String> = done.history.iter().map(Message::text).collect();
    assert!(texts.contains(&"yes".to_string()), "{texts:?}");
    assert!(
        texts.iter().any(|t| t.starts_with("Approval needed")),
        "{texts:?}"
    );

    // A second task in the same context continues the conversation and a
    // refusal cancels it.
    let r = client
        .send_message(MessageSendParams::new(
            Message::user_text("and again?").with_context_id(&t.context_id),
        ))
        .await
        .unwrap();
    let t2 = r.as_task().unwrap().clone();
    assert_eq!(t2.status.state, TaskState::InputRequired);
    let last_request = model.requests().last().unwrap().clone();
    assert_eq!(last_request.messages.len(), 5, "previous turns were kept");
    let r = client
        .send_message(MessageSendParams::new(
            Message::user_text("no").with_task_id(&t2.id),
        ))
        .await
        .unwrap();
    assert_eq!(r.as_task().unwrap().status.state, TaskState::Canceled);
}

#[tokio::test]
async fn local_agent_delegates_to_remote_agent() {
    let remote_model = scripted(&["Bonjour, le monde !"]);
    let (server, _) = spawn(ChatModelExecutor::new(remote_model.clone())).await;

    let tool = A2aAgentTool::discover(&server.base_url()).await.unwrap();
    assert_eq!(tool.spec().name, "test_agent");
    assert!(tool.spec().description.contains("Echo (Repeats things)"));

    let local_model = Arc::new(ScriptedModel::new([
        sinew_core::Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "d1".into(),
                name: "test_agent".into(),
                arguments: json!({"input": "Translate 'Hello, world!' to French"}),
            }],
        ),
        sinew_core::Message::assistant("In French: Bonjour, le monde !"),
    ]));
    let agent = react_agent(local_model.clone(), ToolBox::new().with(tool)).unwrap();
    let out = agent
        .invoke(AgentState::new(vec![sinew_core::Message::user(
            "How do you say hello world in French?",
        )]))
        .await
        .unwrap();
    assert_eq!(out.messages[2].content, "Bonjour, le monde !");
    assert_eq!(
        out.last().unwrap().content,
        "In French: Bonjour, le monde !"
    );
    assert_eq!(
        remote_model.requests()[0].messages[0].content,
        "Translate 'Hello, world!' to French"
    );
}

#[tokio::test]
async fn tool_reports_input_required_and_continues() {
    let model = Arc::new(ScriptedModel::new([
        add_call(),
        sinew_core::Message::assistant("42"),
    ]));
    let graph = react_agent(model, adder())
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new())
        .interrupt_before(["tools"]);
    let (server, _) = spawn(GraphExecutor::agent(graph)).await;
    let tool = A2aAgentTool::discover(&server.base_url()).await.unwrap();

    let first = tool.call(json!({"input": "2 + 40?"})).await.unwrap();
    let first = first.as_str().unwrap();
    assert!(
        first.starts_with("The remote agent needs more input (task_id: "),
        "{first}"
    );
    let task_id = first
        .split("task_id: ")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let second = tool
        .call(json!({"input": "yes", "task_id": task_id}))
        .await
        .unwrap();
    assert_eq!(second, json!("42"));
}

#[tokio::test]
async fn failed_remote_task_is_a_tool_error() {
    let (server, _) = spawn(Failing).await;
    let tool = A2aAgentTool::discover(&server.base_url()).await.unwrap();
    let e = tool.call(json!({"input": "x"})).await.unwrap_err();
    assert!(e.to_string().contains("database unavailable"), "{e}");
}

#[tokio::test]
async fn graceful_shutdown() {
    let (server, client) = spawn(ChatModelExecutor::new(scripted(&["ok"]))).await;
    client.send_text("x").await.unwrap();
    server.shutdown().await.unwrap();
    assert!(client.send_text("y").await.is_err());
}

/// Live check against a local Ollama: `cargo test -p sinew-a2a -- --ignored`.
#[tokio::test]
#[ignore = "needs Ollama on 127.0.0.1:11434 with qwen3:4b-instruct-2507-q4_K_M"]
async fn live_ollama_streams_over_a2a() {
    let model = sinew_core::providers::from_spec("ollama:qwen3:4b-instruct-2507-q4_K_M").unwrap();
    let (_server, client) = spawn(ChatModelExecutor::new(model).with_max_tokens(64)).await;
    let events = collect(
        client
            .stream_text("Say hello in five words.")
            .await
            .unwrap(),
    )
    .await;
    let chunks = events
        .iter()
        .filter(|e| matches!(e, StreamResponse::ArtifactUpdate(_)))
        .count();
    assert!(chunks > 2, "expected several streamed chunks, got {chunks}");
    assert!(matches!(
        events.last(),
        Some(StreamResponse::StatusUpdate(s)) if s.status.state == TaskState::Completed
    ));
}
