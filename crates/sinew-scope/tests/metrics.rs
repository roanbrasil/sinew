//! `GET /metrics`: run metrics of ingested runs, ingest counters, store
//! gauges and the auth switch.

use serde_json::json;
use sinew_scope::{router_with, ServerConfig, Store};
use sinew_trace::metrics::CONTENT_TYPE;
use tokio::sync::oneshot;

struct Server {
    url: String,
    _stop: oneshot::Sender<()>,
}

async fn spawn(config: ServerConfig) -> Server {
    let store = Store::memory().await.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = oneshot::channel();
    tokio::spawn(sinew_scope::server::serve_with_shutdown(
        listener,
        router_with(store, config),
        async {
            let _ = rx.await;
        },
    ));
    Server { url, _stop: tx }
}

fn value(text: &str, series: &str) -> Option<f64> {
    text.lines()
        .find_map(|l| l.strip_prefix(series)?.strip_prefix(' ')?.parse().ok())
}

fn runs() -> serde_json::Value {
    json!({
        "project": "prod",
        "runs": [
            {"id": "g1", "trace_id": "g1", "parent_id": null, "name": "agent", "kind": "graph",
             "start_ms": 1000, "end_ms": 3000, "inputs": {}, "outputs": {},
             "error": null, "input_tokens": null, "output_tokens": null, "metadata": {}},
            {"id": "l1", "trace_id": "g1", "parent_id": "g1", "name": "openai", "kind": "llm",
             "start_ms": 1000, "end_ms": 2000, "inputs": {}, "outputs": {},
             "error": null, "input_tokens": 1000, "output_tokens": 200,
             "metadata": {"model": "gpt-4o-mini", "ttft_ms": 120}},
            {"id": "t1", "trace_id": "g1", "parent_id": "g1", "name": "search", "kind": "tool",
             "start_ms": 2000, "end_ms": 2100, "inputs": {}, "outputs": null,
             "error": "timeout", "input_tokens": null, "output_tokens": null, "metadata": {}}
        ]
    })
}

#[tokio::test]
async fn metrics_aggregate_ingested_runs() {
    let s = spawn(ServerConfig {
        token: Some("t0k".into()),
        disable_engine: true,
        ..Default::default()
    })
    .await;
    let http = reqwest::Client::new();
    let r = http
        .post(format!("{}/api/runs", s.url))
        .bearer_auth("t0k")
        .json(&runs())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    // Open by default even though the API requires a token.
    let resp = http.get(format!("{}/metrics", s.url)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["content-type"], CONTENT_TYPE);
    let text = resp.text().await.unwrap();
    assert_eq!(
        value(
            &text,
            r#"sinew_runs_total{project="prod",kind="llm",name="gpt-4o-mini",status="ok"}"#
        ),
        Some(1.0)
    );
    assert_eq!(
        value(
            &text,
            r#"sinew_tool_calls_total{project="prod",tool="search",status="error"}"#
        ),
        Some(1.0)
    );
    assert_eq!(
        value(
            &text,
            r#"sinew_run_duration_seconds_sum{project="prod",kind="graph",name="agent"}"#
        ),
        Some(2.0)
    );
    assert_eq!(
        value(
            &text,
            r#"sinew_llm_tokens_total{project="prod",model="gpt-4o-mini",direction="input"}"#
        ),
        Some(1000.0)
    );
    // Priced with the scope table: 1000 * 0.15 + 200 * 0.60 per million.
    let cost = value(
        &text,
        r#"sinew_llm_cost_usd_total{project="prod",model="gpt-4o-mini"}"#,
    )
    .unwrap();
    assert!((cost - 0.00027).abs() < 1e-12, "{cost}");
    assert_eq!(
        value(
            &text,
            r#"sinew_llm_time_to_first_token_seconds_bucket{project="prod",model="gpt-4o-mini",le="0.25"}"#
        ),
        Some(1.0)
    );
    assert_eq!(
        value(&text, r#"sinew_scope_ingested_runs_total{endpoint="runs"}"#),
        Some(3.0)
    );
    assert_eq!(
        value(&text, r#"sinew_scope_ingested_runs_total{endpoint="otlp"}"#),
        Some(0.0)
    );
    assert_eq!(value(&text, "sinew_scope_store_runs"), Some(3.0));
    assert_eq!(value(&text, "sinew_scope_store_traces"), Some(1.0));
    assert!(value(&text, "sinew_scope_store_size_bytes").unwrap() > 0.0);
    assert!(text.contains("sinew_scope_build_info{version="));
}

#[tokio::test]
async fn metrics_auth_requires_the_token() {
    let s = spawn(ServerConfig {
        token: Some("t0k".into()),
        disable_engine: true,
        metrics_auth: true,
        ..Default::default()
    })
    .await;
    let http = reqwest::Client::new();
    let denied = http.get(format!("{}/metrics", s.url)).send().await.unwrap();
    assert_eq!(denied.status(), 401);
    let ok = http
        .get(format!("{}/metrics", s.url))
        .bearer_auth("t0k")
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), 200);
    assert!(ok
        .text()
        .await
        .unwrap()
        .contains("sinew_scope_store_runs 0"));
}
