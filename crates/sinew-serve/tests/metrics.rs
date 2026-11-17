//! Prometheus metrics: a traced react agent feeding a `MetricsSink`, and the
//! `GET /metrics` endpoint (content, gauges, auth).

use std::{sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, FnTool, Message, ToolBox, ToolCall};
use sinew_graph::{
    prebuilt::{react_agent, AgentState},
    StateGraph, END,
};
use sinew_serve::{Assistant, RunCreate, ServeClient, Server};
use sinew_trace::{metrics::CONTENT_TYPE, MetricsSink, Tracer};
use tokio::sync::Notify;

fn tools() -> ToolBox {
    ToolBox::new()
        .with(FnTool::new(
            "add",
            "Adds two integers",
            json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}),
            |a: Value| Ok(json!(a["a"].as_i64().unwrap_or(0) + a["b"].as_i64().unwrap_or(0))),
        ))
        .with(FnTool::new(
            "fail",
            "Always fails",
            json!({"type": "object"}),
            |_: Value| Err(sinew_core::Error::Tool { name: "fail".into(), message: "boom".into() }),
        ))
}

fn script() -> Vec<Message> {
    vec![
        Message::assistant_with_tools(
            "",
            vec![
                ToolCall {
                    id: "c1".into(),
                    name: "add".into(),
                    arguments: json!({"a": 2, "b": 40}),
                },
                ToolCall {
                    id: "c2".into(),
                    name: "fail".into(),
                    arguments: json!({}),
                },
            ],
        ),
        Message::assistant("The answer is 42."),
    ]
}

/// The value of the sample line that starts with `series` (name and labels).
fn value(text: &str, series: &str) -> Option<f64> {
    text.lines()
        .find_map(|l| l.strip_prefix(series)?.strip_prefix(' ')?.parse().ok())
}

#[tokio::test]
async fn traced_react_agent_feeds_the_sink() {
    let metrics = Arc::new(MetricsSink::new().project("calc"));
    let tracer = Tracer::new().with_shared_sink(metrics.clone());
    let agent = react_agent(Arc::new(ScriptedModel::new(script())), tools())
        .unwrap()
        .with_tracer(tracer);
    agent
        .invoke(AgentState {
            messages: vec![Message::user("what is 2 + 40")],
            ..Default::default()
        })
        .await
        .unwrap();

    let text = metrics.render();
    let llm = r#"{project="calc",kind="llm",name="scripted",status="ok"}"#;
    assert_eq!(value(&text, &format!("sinew_runs_total{llm}")), Some(2.0));
    assert_eq!(
        value(
            &text,
            r#"sinew_tool_calls_total{project="calc",tool="add",status="ok"}"#
        ),
        Some(1.0)
    );
    assert_eq!(
        value(
            &text,
            r#"sinew_tool_calls_total{project="calc",tool="fail",status="error"}"#
        ),
        Some(1.0)
    );
    assert_eq!(
        value(
            &text,
            r#"sinew_errors_total{project="calc",kind="tool",name="fail"}"#
        ),
        Some(1.0)
    );
    // ScriptedModel counts whitespace-separated words: 4 output tokens.
    assert_eq!(
        value(
            &text,
            r#"sinew_llm_tokens_total{project="calc",model="scripted",direction="output"}"#
        ),
        Some(4.0)
    );
    assert!(
        value(
            &text,
            r#"sinew_llm_tokens_total{project="calc",model="scripted",direction="input"}"#
        )
        .unwrap()
            > 0.0
    );
    // Graph and node runs are counted too.
    assert!(text.contains(r#"sinew_runs_total{project="calc",kind="graph","#));
    assert_eq!(
        value(
            &text,
            r#"sinew_runs_total{project="calc",kind="node",name="tools",status="ok"}"#
        ),
        Some(1.0)
    );
    assert_eq!(
        value(
            &text,
            r#"sinew_run_duration_seconds_count{project="calc",kind="llm",name="scripted"}"#
        ),
        Some(2.0)
    );
    // "scripted" is not in the price table: no cost series.
    assert!(!text.contains("sinew_llm_cost_usd_total{"));
}

#[tokio::test]
async fn metrics_endpoint_reports_runs_requests_and_ttft() {
    let model = Arc::new(ScriptedModel::new(script()));
    let srv = Server::builder()
        .assistant(Assistant::agent(
            "agent",
            react_agent(model, tools()).unwrap(),
        ))
        .metrics(MetricsSink::new())
        .api_key("secret")
        .build()
        .spawn("127.0.0.1:0")
        .await
        .unwrap();
    let base = srv.base_url();
    let http = reqwest::Client::new();
    // Streaming tokens records time to first token.
    let streamed = http
        .post(format!("{base}/runs/stream"))
        .bearer_auth("secret")
        .json(&json!({
            "assistant_id": "agent",
            "input": {"messages": [{"role": "user", "content": "2 + 40?"}]},
            "stream_mode": ["tokens"]
        }))
        .send()
        .await
        .unwrap();
    assert!(streamed.text().await.unwrap().contains("event: end"));

    // Open by default, even with API keys.
    let resp = http.get(format!("{base}/metrics")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["content-type"], CONTENT_TYPE);
    let text = resp.text().await.unwrap();
    assert!(text.contains(r#"sinew_runs_total{project="default",kind="chain",name="POST /runs/stream",status="ok"} 1"#), "{text}");
    assert!(
        value(
            &text,
            r#"sinew_llm_time_to_first_token_seconds_count{project="default",model="scripted"}"#
        ) >= Some(1.0),
        "{text}"
    );
    assert_eq!(value(&text, "sinew_serve_running_runs"), Some(0.0));
    assert_eq!(value(&text, "sinew_serve_queued_runs"), Some(0.0));
    assert_eq!(value(&text, "sinew_serve_active_streams"), Some(0.0));
    assert_eq!(value(&text, "sinew_serve_assistants"), Some(1.0));
    assert!(text.contains("sinew_serve_build_info{version="));
    // Scrapes are not traced.
    assert!(!text.contains("GET /metrics"));
    srv.shutdown().await.unwrap();
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct Gate {
    done: bool,
}

#[tokio::test]
async fn gauges_track_running_runs_and_open_streams() {
    let release = Arc::new(Notify::new());
    let r = release.clone();
    let mut g = StateGraph::<Gate>::new();
    g.add_node("wait", move |mut s: Gate| {
        let r = r.clone();
        async move {
            r.notified().await;
            s.done = true;
            Ok(s)
        }
    });
    g.add_edge("wait", END);
    g.set_entry_point("wait");
    // No sink: `/metrics` still serves the server gauges.
    let srv = Server::builder()
        .assistant(Assistant::graph("gate", g.compile().unwrap()))
        .build()
        .spawn("127.0.0.1:0")
        .await
        .unwrap();
    let base = srv.base_url();
    let http = reqwest::Client::new();
    let stream = http
        .post(format!("{base}/runs/stream"))
        .json(&json!({"assistant_id": "gate", "input": {}, "stream_mode": ["values"]}))
        .send()
        .await
        .unwrap();
    let mut text = String::new();
    for _ in 0..100 {
        text = http
            .get(format!("{base}/metrics"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        if value(&text, "sinew_serve_running_runs") == Some(1.0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        value(&text, "sinew_serve_running_runs"),
        Some(1.0),
        "{text}"
    );
    assert_eq!(value(&text, "sinew_serve_active_streams"), Some(1.0));
    assert!(!text.contains("sinew_runs_total"));
    release.notify_waiters();
    assert!(stream.text().await.unwrap().contains("event: end"));
    let text = http
        .get(format!("{base}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(value(&text, "sinew_serve_active_streams"), Some(0.0));
    srv.shutdown().await.unwrap();
}

#[tokio::test]
async fn metrics_auth_requires_the_api_key() {
    let srv = Server::builder()
        .assistant(Assistant::agent(
            "agent",
            react_agent(
                Arc::new(ScriptedModel::new([Message::assistant("hi")])),
                ToolBox::new(),
            )
            .unwrap(),
        ))
        .metrics(MetricsSink::new())
        .api_key("secret")
        .metrics_auth(true)
        .build()
        .spawn("127.0.0.1:0")
        .await
        .unwrap();
    let base = srv.base_url();
    let http = reqwest::Client::new();
    let denied = http.get(format!("{base}/metrics")).send().await.unwrap();
    assert_eq!(denied.status(), 401);
    let c = ServeClient::new(base.clone()).with_api_key("secret");
    c.wait(None, &RunCreate::new("agent", json!("hello")))
        .await
        .unwrap();
    let ok = http
        .get(format!("{base}/metrics"))
        .bearer_auth("secret")
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), 200);
    let text = ok.text().await.unwrap();
    assert!(
        text.contains(r#"kind="llm",name="scripted",status="ok"} 1"#),
        "{text}"
    );
    srv.shutdown().await.unwrap();
}
