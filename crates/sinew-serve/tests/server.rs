//! End-to-end tests: real servers on 127.0.0.1:0, scripted models, raw HTTP
//! and the typed client.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, FnTool, Message, ToolBox, ToolCall};
use sinew_graph::{prebuilt::react_agent, StateGraph, END};
use sinew_serve::{
    Assistant, CorsConfig, MultitaskStrategy, OnDisconnect, RunCreate, RunStatus, RunningServer,
    ServeClient, Server, StreamMode, StreamPart, ThreadCreate, ThreadQuery, ThreadStatus,
    WebhookPolicy,
};
use sinew_trace::{MemorySink, Tracer};

fn add_tool() -> ToolBox {
    ToolBox::new().with(FnTool::new(
        "add",
        "Adds two integers",
        json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}),
        |a: Value| Ok(json!(a["a"].as_i64().unwrap_or(0) + a["b"].as_i64().unwrap_or(0))),
    ))
}

fn tool_then_answer() -> Vec<Message> {
    vec![
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "add".into(),
                arguments: json!({"a": 2, "b": 40}),
            }],
        ),
        Message::assistant("The answer is 42."),
    ]
}

fn agent(script: Vec<Message>) -> Assistant {
    let model = Arc::new(ScriptedModel::new(script));
    Assistant::agent("agent", react_agent(model, add_tool()).unwrap())
        .description("calculator agent")
        .config(json!({"temperature": 0}))
        .into()
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct Email {
    to: String,
    body: String,
    sent: bool,
}

/// draft -> send, paused before `send`.
fn email() -> Assistant {
    let mut g = StateGraph::<Email>::new();
    g.add_node("draft", |mut s: Email| async move {
        s.body = format!("Hi {}, the release moves to Friday.", s.to);
        Ok(s)
    });
    g.add_node("send", |mut s: Email| async move {
        s.sent = true;
        Ok(s)
    });
    g.add_edge("draft", "send");
    g.add_edge("send", END);
    g.set_entry_point("draft");
    Assistant::graph("email", g.compile().unwrap().interrupt_before(["send"])).into()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct Counter {
    n: u32,
    log: Vec<String>,
}

/// One node that sleeps `ms` milliseconds, then increments `n`.
fn slow(ms: u64) -> Assistant {
    let mut g = StateGraph::<Counter>::new();
    g.add_node_with_context("work", move |mut s: Counter, ctx| async move {
        ctx.emit(json!({"progress": "started"}));
        tokio::time::sleep(Duration::from_millis(ms)).await;
        s.n += 1;
        s.log.push(format!("run {}", s.n));
        Ok(s)
    });
    g.add_edge("work", END);
    g.set_entry_point("work");
    Assistant::graph("slow", g.compile().unwrap()).into()
}

async fn spawn(b: sinew_serve::ServerBuilder) -> (RunningServer, ServeClient) {
    let s = b.build().spawn("127.0.0.1:0").await.unwrap();
    let c = ServeClient::new(s.base_url());
    (s, c)
}

async fn collect(mut s: sinew_serve::RunStream) -> Vec<StreamPart> {
    let mut out = Vec::new();
    while let Some(p) = s.next().await {
        out.push(p.unwrap());
    }
    out
}

#[tokio::test]
async fn health_info_and_assistants() {
    let (srv, c) = spawn(
        Server::builder()
            .assistant(agent(vec![]))
            .assistant(email()),
    )
    .await;
    assert!(c.health().await.unwrap());
    let http = reqwest::Client::new();
    let ready: Value = http
        .get(format!("{}/ready", srv.base_url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ready["ok"], true);

    let list = c.assistants().await.unwrap();
    let ids: Vec<_> = list.iter().map(|a| a.assistant_id.as_str()).collect();
    assert_eq!(ids, ["agent", "email"]);
    let a = c.assistant("agent").await.unwrap();
    assert_eq!(a.description, "calculator agent");
    assert_eq!(a.config, json!({"temperature": 0}));
    assert!(a.chat);
    assert!(!c.assistant("email").await.unwrap().chat);
    let err = c.assistant("nope").await.unwrap_err();
    assert!(matches!(
        err,
        sinew_serve::ClientError::Api { status: 404, .. }
    ));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn threads_crud_state_and_history() {
    let (srv, c) = spawn(Server::builder().assistant(email())).await;
    let t = c
        .create_thread(&ThreadCreate {
            thread_id: Some("t-1".into()),
            metadata: Some(json!({"user": "ana"})),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(t.thread_id, "t-1");
    assert_eq!(t.status, ThreadStatus::Idle);
    // Duplicate id.
    let dup = c
        .create_thread(&ThreadCreate {
            thread_id: Some("t-1".into()),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(matches!(
        dup,
        sinew_serve::ClientError::Api { status: 409, .. }
    ));

    // Empty state before any run.
    let st = c.state("t-1").await.unwrap();
    assert_eq!(st.values, Value::Null);

    let run = c
        .wait(Some("t-1"), &RunCreate::new("email", json!({"to": "Ana"})))
        .await
        .unwrap();
    assert_eq!(run.status, RunStatus::Interrupted);
    assert_eq!(run.interrupted_before, ["send"]);
    let st = c.state("t-1").await.unwrap();
    assert_eq!(st.next, ["send"]);
    let email: Email = st.values_as().unwrap();
    assert!(email.body.contains("Friday"));
    assert_eq!(
        c.thread("t-1").await.unwrap().status,
        ThreadStatus::Interrupted
    );

    // Edit the paused state; invalid edits are rejected by the graph type.
    let st = c
        .update_state("t-1", json!({"body": "Edited."}))
        .await
        .unwrap();
    assert_eq!(st.values["body"], "Edited.");
    assert_eq!(st.next, ["send"]);
    let bad = c
        .update_state("t-1", json!({"sent": "not a bool"}))
        .await
        .unwrap_err();
    assert!(matches!(
        bad,
        sinew_serve::ClientError::Api { status: 400, .. }
    ));

    let hist = c.history("t-1", 10).await.unwrap();
    assert!(hist.len() >= 2);
    assert_eq!(hist[0].values["body"], "Edited.");

    let threads = c.threads(&ThreadQuery::default()).await.unwrap();
    assert_eq!(threads.len(), 1);
    assert_eq!(threads[0].assistant_id.as_deref(), Some("email"));
    assert_eq!(threads[0].metadata, json!({"user": "ana"}));

    c.delete_thread("t-1").await.unwrap();
    assert!(c.thread("t-1").await.is_err());
    assert!(c.delete_thread("t-1").await.is_err());
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn interrupt_then_resume_with_command_update() {
    let (srv, c) = spawn(Server::builder().assistant(email())).await;
    let first = collect(
        c.stream(
            Some("th"),
            &RunCreate::new("email", json!({"to": "Bob"})).with_stream_mode([StreamMode::Updates]),
        )
        .await
        .unwrap(),
    )
    .await;
    assert!(matches!(first[0], StreamPart::Metadata { .. }));
    assert!(first.iter().any(|p| matches!(p,
        StreamPart::Updates(u) if u["draft"]["body"].as_str().is_some_and(|b| b.contains("Bob")))));
    assert!(first.iter().any(|p| matches!(p,
        StreamPart::Interrupt { next, .. } if next == &["send".to_string()])));
    assert_eq!(
        first.last(),
        Some(&StreamPart::End {
            status: RunStatus::Interrupted
        })
    );

    // Resume, editing the draft on the way.
    let done = c
        .wait(
            Some("th"),
            &RunCreate::resume("email").with_update(json!({"body": "Approved text."})),
        )
        .await
        .unwrap();
    assert_eq!(done.status, RunStatus::Success);
    let out: Email = done.output_as().unwrap().unwrap();
    assert!(out.sent);
    assert_eq!(out.body, "Approved text.");
    assert_eq!(c.thread("th").await.unwrap().status, ThreadStatus::Idle);

    // Resuming a thread that never ran is an error run, not a crash.
    let none = c
        .wait(Some("fresh"), &RunCreate::resume("email"))
        .await
        .unwrap();
    assert_eq!(none.status, RunStatus::Error);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn sse_tokens_values_and_events_over_raw_http() {
    let (srv, _c) = spawn(Server::builder().assistant(agent(tool_then_answer()))).await;
    let body = reqwest::Client::new()
        .post(format!("{}/threads/chat-1/runs/stream", srv.base_url()))
        .json(&json!({
            "assistant_id": "agent",
            "input": {"messages": [{"role": "user", "content": "2 + 40?"}]},
            "stream_mode": ["tokens", "values", "events"]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        body.headers()["content-type"].to_str().unwrap(),
        "text/event-stream"
    );
    let text = body.text().await.unwrap();
    let events: Vec<(String, Value)> = text
        .split("\n\n")
        .filter(|f| !f.trim().is_empty() && !f.starts_with(':'))
        .map(|f| {
            let mut name = String::new();
            let mut data = String::new();
            for line in f.lines() {
                if let Some(v) = line.strip_prefix("event: ") {
                    name = v.into();
                } else if let Some(v) = line.strip_prefix("data: ") {
                    data = v.into();
                }
            }
            (name, serde_json::from_str(&data).unwrap())
        })
        .collect();
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.first(), Some(&"metadata"));
    assert_eq!(names.last(), Some(&"end"));
    let tokens: String = events
        .iter()
        .filter(|(n, _)| n == "tokens")
        .map(|(_, d)| d["delta"].as_str().unwrap())
        .collect();
    assert_eq!(tokens, "The answer is 42.");
    let starts: Vec<&str> = events
        .iter()
        .filter(|(n, d)| n == "events" && d["event"] == "node_start")
        .map(|(_, d)| d["node"].as_str().unwrap())
        .collect();
    assert_eq!(starts, ["agent", "tools", "agent"]);
    let last_values = events.iter().rev().find(|(n, _)| n == "values").unwrap();
    assert_eq!(last_values.1["messages"].as_array().unwrap().len(), 4);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn conversation_continues_on_a_thread() {
    let (srv, c) = spawn(Server::builder().assistant(agent(vec![
        Message::assistant("Hi Ana."),
        Message::assistant("You are Ana."),
    ])))
    .await;
    c.wait(Some("conv"), &RunCreate::new("agent", json!("I am Ana")))
        .await
        .unwrap();
    let run = c
        .wait(
            Some("conv"),
            &RunCreate::new(
                "agent",
                json!({"messages": [{"role": "user", "content": "Who am I?"}]}),
            ),
        )
        .await
        .unwrap();
    let msgs = run.output.unwrap()["messages"].as_array().unwrap().clone();
    assert_eq!(msgs.len(), 4, "history is appended, not replaced");
    assert_eq!(msgs[3]["content"], "You are Ana.");
    // Another assistant cannot hijack the thread.
    let srv2_err = c
        .wait(Some("conv"), &RunCreate::new("missing", json!("x")))
        .await
        .unwrap_err();
    assert!(matches!(
        srv2_err,
        sinew_serve::ClientError::Api { status: 404, .. }
    ));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn background_run_poll_join_and_cancel() {
    let (srv, c) = spawn(Server::builder().assistant(slow(60_000))).await;
    let run = c
        .create_run(Some("bg"), &RunCreate::new("slow", json!({"n": 0})))
        .await
        .unwrap();
    assert!(matches!(
        run.status,
        RunStatus::Pending | RunStatus::Running
    ));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(c.run(&run.run_id).await.unwrap().status, RunStatus::Running);
    assert_eq!(c.thread("bg").await.unwrap().status, ThreadStatus::Busy);

    let cancelled = c.cancel_run(&run.run_id, true).await.unwrap();
    assert_eq!(cancelled.status, RunStatus::Cancelled);
    assert_eq!(
        c.join_run(&run.run_id).await.unwrap().status,
        RunStatus::Cancelled
    );
    assert_eq!(c.thread("bg").await.unwrap().status, ThreadStatus::Idle);
    let runs = c.thread_runs("bg").await.unwrap();
    assert_eq!(runs.len(), 1);
    srv.shutdown().await.unwrap();

    // A quick one completes and can be joined.
    let (srv, c) = spawn(Server::builder().assistant(slow(20))).await;
    let run = c
        .create_run(Some("bg2"), &RunCreate::new("slow", json!({"n": 5})))
        .await
        .unwrap();
    let joined = c.join_run(&run.run_id).await.unwrap();
    assert_eq!(joined.status, RunStatus::Success);
    assert_eq!(joined.output.unwrap()["n"], 6);
    // Joining the stream of a finished run replays its end.
    let parts = collect(c.join_stream(&run.run_id).await.unwrap()).await;
    assert_eq!(
        parts.last(),
        Some(&StreamPart::End {
            status: RunStatus::Success
        })
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn join_stream_of_a_running_run_sees_custom_events() {
    let (srv, c) = spawn(Server::builder().assistant(slow(300))).await;
    let req = RunCreate::new("slow", json!({"n": 0}))
        .with_stream_mode([StreamMode::Custom, StreamMode::Values]);
    let run = c.create_run(None, &req).await.unwrap();
    let parts = collect(c.join_stream(&run.run_id).await.unwrap()).await;
    assert!(parts
        .iter()
        .any(|p| matches!(p, StreamPart::Values(v) if v["n"] == 1)));
    assert_eq!(
        parts.last(),
        Some(&StreamPart::End {
            status: RunStatus::Success
        })
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn multitask_reject_enqueue_interrupt() {
    let (srv, c) = spawn(Server::builder().assistant(slow(300))).await;
    let first = c
        .create_run(Some("mt"), &RunCreate::new("slow", json!({"n": 0})))
        .await
        .unwrap();
    // reject (default)
    let err = c
        .create_run(Some("mt"), &RunCreate::new("slow", json!({})))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        sinew_serve::ClientError::Api { status: 409, .. }
    ));
    // enqueue: runs after the first, on its output
    let second = c
        .create_run(
            Some("mt"),
            &RunCreate::new("slow", json!({})).with_multitask(MultitaskStrategy::Enqueue),
        )
        .await
        .unwrap();
    assert_eq!(
        c.run(&second.run_id).await.unwrap().status,
        RunStatus::Pending
    );
    let a = c.join_run(&first.run_id).await.unwrap();
    let b = c.join_run(&second.run_id).await.unwrap();
    assert_eq!(a.output.unwrap()["n"], 1);
    assert_eq!(b.status, RunStatus::Success);
    assert_eq!(
        b.output.unwrap()["n"],
        2,
        "enqueued run saw the first result"
    );

    // interrupt: cancels the running one
    let third = c
        .create_run(Some("mt"), &RunCreate::new("slow", json!({})))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    let fourth = c
        .wait(
            Some("mt"),
            &RunCreate::new("slow", json!({})).with_multitask(MultitaskStrategy::Interrupt),
        )
        .await
        .unwrap();
    assert_eq!(
        c.run(&third.run_id).await.unwrap().status,
        RunStatus::Cancelled
    );
    assert_eq!(fourth.status, RunStatus::Success);
    assert_eq!(fourth.output.unwrap()["n"], 3);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn disconnecting_a_stream_cancels_the_run() {
    let (srv, c) = spawn(
        Server::builder()
            .assistant(slow(60_000))
            .keep_alive(Some(Duration::from_millis(50))),
    )
    .await;
    // A raw socket, so dropping it really closes the connection.
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut sock = tokio::net::TcpStream::connect(srv.addr).await.unwrap();
    let body = r#"{"assistant_id":"slow","input":{"n":0}}"#;
    sock.write_all(
        format!(
            "POST /threads/dc/runs/stream HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let mut buf = vec![0u8; 4096];
    let mut got = String::new();
    while !got.contains("event: metadata") {
        let n = sock.read(&mut buf).await.unwrap();
        got.push_str(&String::from_utf8_lossy(&buf[..n]));
    }
    let start = got.find("\"run_id\":\"").unwrap() + 10;
    let run_id = got[start..start + 36].to_string();
    drop(sock);
    let run = tokio::time::timeout(Duration::from_secs(10), c.join_run(&run_id))
        .await
        .expect("disconnect noticed")
        .unwrap();
    assert_eq!(run.status, RunStatus::Cancelled);

    // With on_disconnect=continue it keeps going.
    let (srv2, c2) = spawn(Server::builder().assistant(slow(100))).await;
    let mut s = c2
        .stream(
            Some("dc"),
            &RunCreate::new("slow", json!({"n": 0})).with_on_disconnect(OnDisconnect::Continue),
        )
        .await
        .unwrap();
    let StreamPart::Metadata { run_id, .. } = s.next().await.unwrap().unwrap() else {
        panic!("metadata first")
    };
    drop(s);
    assert_eq!(
        c2.join_run(&run_id).await.unwrap().status,
        RunStatus::Success
    );
    srv.shutdown().await.unwrap();
    srv2.shutdown().await.unwrap();
}

#[tokio::test]
async fn stateless_runs() {
    let (srv, c) = spawn(Server::builder().assistant(agent(tool_then_answer()))).await;
    let parts = collect(
        c.stream(
            None,
            &RunCreate::new("agent", json!("2 + 40?")).with_stream_mode([StreamMode::Tokens]),
        )
        .await
        .unwrap(),
    )
    .await;
    let text: String = parts
        .iter()
        .filter_map(|p| match p {
            StreamPart::Token { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "The answer is 42.");
    assert!(c.threads(&ThreadQuery::default()).await.unwrap().is_empty());
    // A stateless run without input cannot resume anything.
    let r = c.wait(None, &RunCreate::resume("agent")).await.unwrap();
    assert_eq!(r.status, RunStatus::Error);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn openai_facade_with_raw_http() {
    let model = Arc::new(ScriptedModel::new([
        Message::assistant("Plain model reply."),
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "call_1".into(),
                name: "lookup".into(),
                arguments: json!({"q": "rust"}),
            }],
        ),
    ]));
    let mut script = tool_then_answer();
    script.extend(tool_then_answer());
    let (srv, _c) = spawn(
        Server::builder()
            .assistant(agent(script))
            .chat_model("plain", model)
            .api_key("sk-test"),
    )
    .await;
    let base = srv.base_url();
    let http = reqwest::Client::new();

    let unauth = http.get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(unauth.status(), 401);
    let models: Value = http
        .get(format!("{base}/v1/models"))
        .bearer_auth("sk-test")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["agent", "plain"]);

    // Non-streaming through the agent (runs the tool loop).
    let resp: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .header("x-api-key", "sk-test")
        .json(&json!({"model": "agent", "messages": [{"role": "user", "content": "2+40?"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp["object"], "chat.completion");
    assert_eq!(
        resp["choices"][0]["message"]["content"],
        "The answer is 42."
    );
    assert_eq!(resp["choices"][0]["finish_reason"], "stop");

    // Streaming through the agent.
    let text = http
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("sk-test")
        .json(
            &json!({"model": "agent", "stream": true, "stream_options": {"include_usage": true},
                      "messages": [{"role": "user", "content": "2+40?"}]}),
        )
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let datas: Vec<&str> = text
        .split("\n\n")
        .filter_map(|f| f.strip_prefix("data: "))
        .collect();
    assert_eq!(datas.last(), Some(&"[DONE]"));
    let chunks: Vec<Value> = datas[..datas.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let content: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "The answer is 42.");
    assert!(chunks
        .iter()
        .any(|c| c["choices"][0]["finish_reason"] == "stop"));
    assert!(chunks.last().unwrap()["usage"].is_object());

    // Plain model, non-streaming then streaming with a tool call.
    let resp: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("sk-test")
        .json(&json!({"model": "plain", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        resp["choices"][0]["message"]["content"],
        "Plain model reply."
    );
    assert!(resp["usage"]["total_tokens"].as_u64().unwrap() > 0);
    let text = http
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("sk-test")
        .json(&json!({"model": "plain", "stream": true,
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            "messages": [{"role": "user", "content": "search rust"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("\"tool_calls\""));
    assert!(text.contains("\"finish_reason\":\"tool_calls\""));
    assert!(text.ends_with("data: [DONE]\n\n"));

    // Unknown model.
    let r = http
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("sk-test")
        .json(&json!({"model": "gpt-9", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
    let v: Value = r.json().await.unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("gpt-9"));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn openai_client_from_sinew_core_talks_to_the_agent() {
    use sinew_core::{providers::OpenAiCompat, ChatModel, ChatRequest};
    let (srv, _c) = spawn(Server::builder().assistant(agent(tool_then_answer()))).await;
    let client = OpenAiCompat::new(format!("{}/v1", srv.base_url()), "agent");
    let mut s = client
        .chat_stream(ChatRequest::new(vec![Message::user("2 + 40?")]))
        .await
        .unwrap();
    let mut acc = sinew_core::StreamAccumulator::new();
    while let Some(ev) = s.next().await {
        acc.push(&ev.unwrap());
    }
    assert_eq!(acc.finish().message.content, "The answer is 42.");
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn auth_cors_and_public_probes() {
    let (srv, c) = spawn(
        Server::builder()
            .assistant(email())
            .api_key("k1")
            .api_key("k2")
            .cors(CorsConfig::origins(["https://app.example.com"])),
    )
    .await;
    let base = srv.base_url();
    let http = reqwest::Client::new();
    assert!(c.health().await.unwrap(), "/ok is public");
    let r = http.get(format!("{base}/assistants")).send().await.unwrap();
    assert_eq!(r.status(), 401);
    assert_eq!(r.headers()["www-authenticate"], "Bearer");
    assert!(c.assistants().await.is_err());
    let c2 = c.clone().with_api_key("k2");
    assert_eq!(c2.assistants().await.unwrap().len(), 1);
    let wrong = c.clone().with_api_key("k3");
    assert!(wrong.assistants().await.is_err());

    let pre = http
        .request(reqwest::Method::OPTIONS, format!("{base}/threads"))
        .header("origin", "https://app.example.com")
        .header("access-control-request-method", "POST")
        .send()
        .await
        .unwrap();
    assert_eq!(
        pre.headers()["access-control-allow-origin"],
        "https://app.example.com"
    );
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn webhook_fires_on_completion() {
    use axum::{routing::post, Json, Router};
    let got: Arc<Mutex<Vec<Value>>> = Arc::default();
    let sink = got.clone();
    let hook = Router::new().route(
        "/hook",
        post(move |Json(v): Json<Value>| {
            let sink = sink.clone();
            async move {
                sink.lock().unwrap().push(v);
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hook_url = format!("http://{}/hook", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, hook).await });

    // Disabled by default.
    let (srv, c) = spawn(Server::builder().assistant(slow(1))).await;
    let err = c
        .wait(
            None,
            &RunCreate::new("slow", json!({})).with_webhook(&hook_url),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        sinew_serve::ClientError::Api { status: 400, .. }
    ));
    srv.shutdown().await.unwrap();

    let (srv, c) = spawn(
        Server::builder()
            .assistant(slow(1))
            .webhooks(WebhookPolicy::Prefixes(vec![hook_url.clone()])),
    )
    .await;
    let run = c
        .wait(
            Some("wh"),
            &RunCreate::new("slow", json!({})).with_webhook(&hook_url),
        )
        .await
        .unwrap();
    for _ in 0..100 {
        if !got.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let calls = got.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["run_id"], json!(run.run_id));
    assert_eq!(calls[0]["status"], "success");
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn sqlite_threads_survive_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}", dir.path().join("serve.db").display());
    let store = sinew_serve::SqliteStore::connect(&url).await.unwrap();
    let (srv, c) = spawn(Server::builder().store(store).assistant(email())).await;
    let run = c
        .wait(
            Some("durable"),
            &RunCreate::new("email", json!({"to": "Eve"})),
        )
        .await
        .unwrap();
    assert_eq!(run.status, RunStatus::Interrupted);
    srv.shutdown().await.unwrap();

    let store = sinew_serve::SqliteStore::connect(&url).await.unwrap();
    let (srv, c) = spawn(Server::builder().store(store).assistant(email())).await;
    let t = c.thread("durable").await.unwrap();
    assert_eq!(t.status, ThreadStatus::Interrupted);
    assert_eq!(c.state("durable").await.unwrap().next, ["send"]);
    let done = c
        .wait(Some("durable"), &RunCreate::resume("email"))
        .await
        .unwrap();
    assert_eq!(done.status, RunStatus::Success);
    assert_eq!(done.output.unwrap()["sent"], true);
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn graceful_shutdown_cancels_after_grace() {
    let srv = Server::builder()
        .assistant(slow(60_000))
        .shutdown_grace(Duration::from_millis(100))
        .build()
        .spawn("127.0.0.1:0")
        .await
        .unwrap();
    let c = ServeClient::new(srv.base_url());
    let run = c
        .create_run(Some("sd"), &RunCreate::new("slow", json!({})))
        .await
        .unwrap();
    // A streaming client is connected while we shut down.
    let mut joined = c.join_stream(&run.run_id).await.unwrap();
    let t = std::time::Instant::now();
    srv.shutdown().await.unwrap();
    assert!(t.elapsed() < Duration::from_secs(10));
    let mut last = None;
    while let Some(p) = joined.next().await {
        last = Some(p.unwrap());
    }
    assert_eq!(
        last,
        Some(StreamPart::End {
            status: RunStatus::Cancelled
        })
    );
}

#[tokio::test]
async fn requests_and_graphs_are_traced() {
    let sink = Arc::new(MemorySink::new());
    let tracer = Tracer::new().with_shared_sink(sink.clone());
    let (srv, c) = spawn(
        Server::builder()
            .assistant(agent(vec![Message::assistant("hi")]))
            .tracer(tracer),
    )
    .await;
    c.wait(None, &RunCreate::new("agent", json!("hello")))
        .await
        .unwrap();
    c.health().await.unwrap();
    srv.shutdown().await.unwrap();
    let runs = sink.runs();
    assert!(runs.iter().any(|r| r.name == "POST /runs/wait"
        && r.metadata["http.status"] == 200
        && r.kind == sinew_trace::RunKind::Chain));
    assert!(runs.iter().any(|r| r.kind == sinew_trace::RunKind::Llm));
    assert!(!runs.iter().any(|r| r.name.contains("/ok")));
}

/// Live test against a local Ollama. Run with
/// `cargo test -p sinew-serve --test server -- --ignored live_ollama`.
#[tokio::test]
#[ignore]
async fn live_ollama_agent_streams_through_the_server() {
    use sinew_core::providers::OpenAiCompat;
    let model = Arc::new(OpenAiCompat::new(
        "http://127.0.0.1:11434/v1",
        "qwen3:4b-instruct-2507-q4_K_M",
    ));
    let (srv, c) = spawn(
        Server::builder()
            .assistant(Assistant::agent(
                "agent",
                react_agent(model.clone(), add_tool()).unwrap(),
            ))
            .chat_model("qwen", model),
    )
    .await;
    let t = std::time::Instant::now();
    let mut s = c
        .stream(
            Some("live"),
            &RunCreate::new(
                "agent",
                json!("Use the add tool to compute 1234 + 4321, then answer with the number."),
            )
            .with_stream_mode([StreamMode::Tokens]),
        )
        .await
        .unwrap();
    let mut first = None;
    let mut text = String::new();
    let mut end = None;
    while let Some(p) = s.next().await {
        match p.unwrap() {
            StreamPart::Token { delta, .. } => {
                first.get_or_insert(t.elapsed());
                text.push_str(&delta);
            }
            StreamPart::End { status } => end = Some(status),
            _ => {}
        }
    }
    eprintln!(
        "live: first token after {:?}, total {:?}: {text}",
        first,
        t.elapsed()
    );
    assert_eq!(end, Some(RunStatus::Success));
    assert!(text.contains("5555"), "{text}");
    let reply = c
        .chat_completion("qwen", &[Message::user("Say just the word: pong")])
        .await
        .unwrap();
    eprintln!("live facade: {reply}");
    assert!(reply.to_lowercase().contains("pong"));
    srv.shutdown().await.unwrap();
}
