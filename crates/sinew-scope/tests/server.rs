//! End-to-end tests: the axum server runs in-process on an ephemeral port.

use std::{sync::Arc, time::Duration};

use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, FnTool, Message, ToolBox, ToolCall};
use sinew_eval::{evaluate, Contains, Dataset, Evaluator, ExactMatch, ScopeClient};
use sinew_graph::prebuilt::{react_agent, AgentState};
use sinew_scope::{router, Store};
use sinew_trace::{HttpSink, HttpSinkConfig, Run, RunKind, Tracer};
use tokio::sync::oneshot;

struct Server {
    url: String,
    stop: Option<oneshot::Sender<()>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
    }
}

async fn spawn(token: Option<&str>) -> Server {
    let store = Store::memory().await.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = oneshot::channel();
    tokio::spawn(sinew_scope::server::serve_with_shutdown(
        listener,
        router(store, token.map(str::to_string)),
        async {
            let _ = rx.await;
        },
    ));
    Server {
        url,
        stop: Some(tx),
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    id: &str,
    trace: &str,
    parent: Option<&str>,
    name: &str,
    kind: RunKind,
    start: u64,
    end: u64,
    tokens: Option<(u32, u32)>,
    error: Option<&str>,
) -> Run {
    Run {
        id: id.into(),
        trace_id: trace.into(),
        parent_id: parent.map(str::to_string),
        name: name.into(),
        kind,
        start_ms: start,
        end_ms: Some(end),
        inputs: json!({"q": name}),
        outputs: Some(json!("ok")),
        error: error.map(str::to_string),
        input_tokens: tokens.map(|t| t.0),
        output_tokens: tokens.map(|t| t.1),
        metadata: json!({}),
    }
}

/// Three traces: t1 (graph, 100ms, llm tokens 10→5), t2 (chain, 300ms, tool
/// error), t3 (graph, 200ms) in project "other".
fn sample_runs() -> (Vec<Run>, Vec<Run>) {
    let main = vec![
        run(
            "t1-llm",
            "t1",
            Some("t1"),
            "gpt",
            RunKind::Llm,
            1010,
            1090,
            Some((10, 5)),
            None,
        ),
        run(
            "t1",
            "t1",
            None,
            "agent",
            RunKind::Graph,
            1000,
            1100,
            None,
            None,
        ),
        run(
            "t2",
            "t2",
            None,
            "pipeline",
            RunKind::Chain,
            2000,
            2300,
            None,
            None,
        ),
        run(
            "t2-tool",
            "t2",
            Some("t2"),
            "search",
            RunKind::Tool,
            2100,
            2200,
            None,
            Some("timeout"),
        ),
    ];
    let other = vec![run(
        "t3",
        "t3",
        None,
        "agent-b",
        RunKind::Graph,
        3000,
        3200,
        Some((7, 3)),
        None,
    )];
    (main, other)
}

async fn get(c: &reqwest::Client, url: String) -> Value {
    let r = c.get(&url).send().await.unwrap();
    assert!(r.status().is_success(), "GET {url}: {}", r.status());
    r.json().await.unwrap()
}

#[tokio::test]
async fn ingest_query_filters_and_aggregates() {
    let s = spawn(None).await;
    let c = reqwest::Client::new();
    let (main, other) = sample_runs();

    // Array body + project header.
    let r = c
        .post(format!("{}/api/runs", s.url))
        .header("x-sinew-project", "main")
        .json(&main)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(
        r.json::<Value>().await.unwrap(),
        json!({"accepted": 4, "traces": 2})
    );
    // Object body with project field; sent twice to check idempotency.
    for _ in 0..2 {
        let r = c
            .post(format!("{}/api/runs", s.url))
            .json(&json!({"project": "other", "runs": other}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
    }
    // Re-sending a run with changes updates it in place.
    let mut updated = main[2].clone();
    updated.end_ms = Some(2400);
    c.post(format!("{}/api/runs", s.url))
        .header("x-sinew-project", "main")
        .json(&vec![updated])
        .send()
        .await
        .unwrap();

    let all = get(&c, format!("{}/api/traces", s.url)).await;
    assert_eq!(all["total"], 3);
    let st = &all["stats"];
    assert_eq!(st["trace_count"], 3);
    assert_eq!(st["run_count"], 5, "duplicates must not be stored twice");
    assert_eq!(st["error_count"], 1);
    assert!((st["error_rate"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-9);
    assert_eq!(st["input_tokens"], 17);
    assert_eq!(st["output_tokens"], 8);
    // Latencies 100, 200, 400 (t2 was updated).
    assert_eq!(st["p50_latency_ms"], 200);
    assert_eq!(st["p95_latency_ms"], 400);
    assert_eq!(st["max_latency_ms"], 400);
    assert!((st["avg_latency_ms"].as_f64().unwrap() - 700.0 / 3.0).abs() < 1e-6);
    // Newest first.
    let ids: Vec<&str> = all["traces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["trace_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["t3", "t2", "t1"]);
    let t1 = &all["traces"][2];
    assert_eq!(t1["name"], "agent");
    assert_eq!(t1["kind"], "graph");
    assert_eq!(t1["project"], "main");
    assert_eq!(t1["run_count"], 2);

    let q = |query: &str| get(&c, format!("{}/api/traces?{query}", s.url));
    assert_eq!(q("project=main").await["total"], 2);
    assert_eq!(q("project=other").await["stats"]["input_tokens"], 7);
    assert_eq!(q("errors=true").await["traces"][0]["trace_id"], "t2");
    assert_eq!(q("errors=true").await["traces"][0]["error"], "timeout");
    assert_eq!(q("kind=llm").await["traces"][0]["trace_id"], "t1");
    assert_eq!(q("kind=tool").await["total"], 1);
    assert_eq!(q("name=agent").await["total"], 2);
    assert_eq!(
        q("name=%25").await["total"],
        0,
        "LIKE wildcards are escaped"
    );
    assert_eq!(
        q("from_ms=1500&to_ms=2500").await["traces"][0]["trace_id"],
        "t2"
    );
    let page = q("limit=1&offset=1").await;
    assert_eq!(page["traces"].as_array().unwrap().len(), 1);
    assert_eq!(page["traces"][0]["trace_id"], "t2");
    assert_eq!(page["total"], 3);

    // Trace tree and single run.
    let t = get(&c, format!("{}/api/traces/t1", s.url)).await;
    assert_eq!(t["summary"]["latency_ms"], 100);
    let runs = t["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0]["id"], "t1", "ordered by start time");
    assert_eq!(runs[1]["parent_id"], "t1");
    let r = get(&c, format!("{}/api/runs/t1-llm", s.url)).await;
    assert_eq!(r["input_tokens"], 10);
    assert_eq!(r["project"], "main");
    assert_eq!(r["kind"], "llm");
    let r: Run = serde_json::from_value(r).unwrap();
    assert_eq!(r, main[0]);

    let projects = get(&c, format!("{}/api/projects", s.url)).await;
    assert_eq!(projects.as_array().unwrap().len(), 2);

    // Errors.
    let r = c
        .get(format!("{}/api/traces/nope", s.url))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
    let r = c
        .post(format!("{}/api/runs", s.url))
        .body("not json")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let r = c
        .post(format!("{}/api/runs", s.url))
        .json(&json!([{"id": "x"}]))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
}

#[tokio::test]
async fn otlp_ingest_is_queryable() {
    let s = spawn(None).await;
    let c = reqwest::Client::new();
    let body = json!({"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "py-app"}}]},
        "scopeSpans": [{"scope": {"name": "test"}, "spans": [
            {"traceId": "0af7651916cd43dd8448eb211c80319c", "spanId": "b7ad6b7169203331",
             "name": "agent", "startTimeUnixNano": "1700000000000000000", "endTimeUnixNano": "1700000000500000000",
             "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "invoke_agent"}}]},
            {"traceId": "0af7651916cd43dd8448eb211c80319c", "spanId": "00f067aa0ba902b7", "parentSpanId": "b7ad6b7169203331",
             "name": "chat llama3", "startTimeUnixNano": "1700000000100000000", "endTimeUnixNano": "1700000000400000000",
             "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.request.model", "value": {"stringValue": "llama3"}},
                {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "31"}},
                {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "9"}}]}
        ]}]
    }]});
    let r = c
        .post(format!("{}/v1/traces", s.url))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(
        r.json::<Value>().await.unwrap(),
        json!({"partialSuccess": {}})
    );

    let page = get(&c, format!("{}/api/traces?project=py-app", s.url)).await;
    assert_eq!(page["total"], 1);
    let t = &page["traces"][0];
    assert_eq!(t["name"], "agent");
    assert_eq!(t["latency_ms"], 500);
    assert_eq!(t["input_tokens"], 31);
    assert_eq!(t["output_tokens"], 9);
    let tree = get(
        &c,
        format!("{}/api/traces/0af7651916cd43dd8448eb211c80319c", s.url),
    )
    .await;
    assert_eq!(tree["runs"][1]["kind"], "llm");
    assert_eq!(tree["runs"][1]["name"], "llama3");

    // Header overrides the resource's service name; protobuf is refused.
    c.post(format!("{}/v1/traces", s.url))
        .header("x-sinew-project", "hdr")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(
        get(&c, format!("{}/api/traces?project=hdr", s.url)).await["total"],
        1
    );
    let r = c
        .post(format!("{}/v1/traces", s.url))
        .header("content-type", "application/x-protobuf")
        .body(vec![0x0a, 0xff])
        .send()
        .await
        .unwrap();
    // Malformed protobuf is a 400 with `otlp-proto`; without it protobuf is 415.
    let expected = if cfg!(feature = "otlp-proto") {
        400
    } else {
        415
    };
    assert_eq!(r.status(), expected);
}

#[tokio::test]
async fn bearer_token_auth() {
    let s = spawn(Some("s3cret")).await;
    let c = reqwest::Client::new();
    // Public: UI and health.
    let r = c.get(format!("{}/", s.url)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let health = get(&c, format!("{}/api/health", s.url)).await;
    assert_eq!(health["auth"], true);

    for (method, path) in [
        ("GET", "/api/traces"),
        ("POST", "/api/runs"),
        ("POST", "/v1/traces"),
        ("GET", "/api/datasets"),
    ] {
        let req = if method == "GET" {
            c.get(format!("{}{path}", s.url))
        } else {
            c.post(format!("{}{path}", s.url)).json(&json!([]))
        };
        assert_eq!(req.send().await.unwrap().status(), 401, "{method} {path}");
    }
    let r = c
        .get(format!("{}/api/traces", s.url))
        .bearer_auth("wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
    let r = c
        .get(format!("{}/api/traces", s.url))
        .bearer_auth("s3cret")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    // HttpSink with the token gets through; without it, runs are dropped.
    let good = Arc::new(HttpSink::new(HttpSinkConfig::new(&s.url).token("s3cret")));
    let bad = Arc::new(HttpSink::new(HttpSinkConfig::new(&s.url)));
    let tracer = Tracer::new()
        .with_shared_sink(good.clone())
        .with_shared_sink(bad.clone());
    tracer
        .start("x", RunKind::Chain, None, json!({}))
        .finish(json!(1));
    tokio::task::spawn_blocking(move || tracer.flush())
        .await
        .unwrap();
    assert_eq!((good.sent(), good.dropped()), (1, 0));
    assert_eq!((bad.sent(), bad.dropped()), (0, 1));
}

fn calculator() -> impl sinew_core::Tool {
    FnTool::new(
        "multiply",
        "Multiply two numbers",
        json!({"type": "object"}),
        |a: Value| {
            Ok(json!(
                a["a"].as_f64().unwrap_or(0.0) * a["b"].as_f64().unwrap_or(0.0)
            ))
        },
    )
}

#[tokio::test]
async fn react_agent_traces_arrive_through_http_sink() {
    let s = spawn(None).await;
    let model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "multiply".into(),
                arguments: json!({"a": 37, "b": 3}),
            }],
        ),
        Message::assistant("37 × 3 = 111."),
    ]));
    let sink = Arc::new(HttpSink::new(
        HttpSinkConfig::new(&s.url)
            .project("agents")
            .flush_interval(Duration::from_millis(50)),
    ));
    let tracer = Tracer::new().with_shared_sink(sink.clone());
    let agent = react_agent(model, ToolBox::new().with(calculator()))
        .unwrap()
        .with_tracer(tracer.clone());
    agent
        .invoke(AgentState::new(vec![Message::user("What is 37 times 3?")]))
        .await
        .unwrap();
    tokio::task::spawn_blocking(move || tracer.flush())
        .await
        .unwrap();
    assert_eq!(sink.dropped(), 0);

    let c = reqwest::Client::new();
    let page = get(&c, format!("{}/api/traces?project=agents", s.url)).await;
    assert_eq!(page["total"], 1);
    let t = &page["traces"][0];
    assert_eq!(t["kind"], "graph");
    // graph + agent, tools, agent nodes + 2 llm + 1 tool.
    assert_eq!(t["run_count"], 7);
    assert_eq!(t["error_count"], 0);
    assert!(t["input_tokens"].as_i64().unwrap() > 0);
    let tree = get(
        &c,
        format!("{}/api/traces/{}", s.url, t["trace_id"].as_str().unwrap()),
    )
    .await;
    let kinds: Vec<&str> = tree["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds.iter().filter(|k| **k == "llm").count(), 2);
    assert_eq!(kinds.iter().filter(|k| **k == "tool").count(), 1);
}

#[tokio::test]
async fn datasets_experiments_and_comparison() {
    let s = spawn(Some("tok")).await;
    let mut ds = Dataset::new("capitals").with_description("country → capital");
    ds.push(json!({"country": "France"}), Some(json!("Paris")))
        .push(json!({"country": "Japan"}), Some(json!("Tokyo")))
        .push(json!({"country": "Brazil"}), Some(json!("Brasília")));
    let evaluators: Vec<Box<dyn Evaluator>> =
        vec![Box::new(ExactMatch::new()), Box::new(Contains::reference())];

    let baseline = evaluate(
        |i: Value| async move {
            Ok::<_, String>(json!(match i["country"].as_str() {
                Some("France") => "Paris",
                Some("Japan") => "Kyoto",
                _ => "Rio de Janeiro",
            }))
        },
        &ds,
        &evaluators,
        2,
    )
    .await
    .with_name("baseline");
    let candidate = evaluate(
        |i: Value| async move {
            match i["country"].as_str() {
                Some("France") => Ok(json!("Paris")),
                Some("Japan") => Ok(json!("Tokyo")),
                _ => Err("model unavailable".to_string()),
            }
        },
        &ds,
        &evaluators,
        2,
    )
    .await
    .with_name("candidate");

    let client = ScopeClient::new(&s.url).with_token("tok");
    assert!(ScopeClient::new(&s.url).upload_dataset(&ds).await.is_err());
    client.upload_experiment(&ds, &baseline).await.unwrap();
    client.upload_report(&candidate).await.unwrap();
    // Idempotent on report id.
    client.upload_report(&candidate).await.unwrap();

    let c = reqwest::Client::new();
    let g = |path: String| {
        let c = c.clone();
        async move {
            let r = c.get(path).bearer_auth("tok").send().await.unwrap();
            assert!(r.status().is_success());
            r.json::<Value>().await.unwrap()
        }
    };
    let list = g(format!("{}/api/datasets", s.url)).await;
    assert_eq!(list[0]["name"], "capitals");
    assert_eq!(list[0]["example_count"], 3);
    assert_eq!(list[0]["experiment_count"], 2);
    let d = g(format!("{}/api/datasets/capitals", s.url)).await;
    assert_eq!(d["dataset"]["examples"][2]["reference"], "Brasília");
    assert_eq!(d["experiments"].as_array().unwrap().len(), 2);

    let exps = g(format!("{}/api/experiments?dataset=capitals", s.url)).await;
    assert_eq!(exps.as_array().unwrap().len(), 2);
    let e = g(format!("{}/api/experiments/{}", s.url, baseline.id)).await;
    assert_eq!(e["name"], "baseline");
    assert!(
        (e["aggregates"]["scores"]["exact_match"]["mean"]
            .as_f64()
            .unwrap()
            - 1.0 / 3.0)
            .abs()
            < 1e-9
    );

    let cmp = g(format!(
        "{}/api/compare?a={}&b={}",
        s.url, baseline.id, candidate.id
    ))
    .await;
    assert_eq!(cmp["rows"].as_array().unwrap().len(), 3);
    assert_eq!(cmp["rows"][1]["deltas"]["exact_match"], 1.0, "Japan fixed");
    assert_eq!(cmp["rows"][2]["b"]["error"], "model unavailable");
    let em = cmp["summary"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["key"] == "exact_match")
        .unwrap();
    assert!((em["delta"].as_f64().unwrap() - (1.0 - 1.0 / 3.0)).abs() < 1e-9);

    // An experiment on an unknown dataset creates the dataset from its rows.
    let mut orphan = baseline.clone();
    orphan.id = "orphan".into();
    orphan.dataset = "adhoc".into();
    client.upload_report(&orphan).await.unwrap();
    let d = g(format!("{}/api/datasets/adhoc", s.url)).await;
    assert_eq!(d["dataset"]["examples"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn ui_assets_are_embedded() {
    let s = spawn(None).await;
    let c = reqwest::Client::new();
    let r = c.get(format!("{}/", s.url)).send().await.unwrap();
    assert!(r.headers()["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/html"));
    let html = r.text().await.unwrap();
    assert!(html.contains("Sinew Scope") && html.contains("app.js"));
    assert!(!html.contains("https://"), "no external resources");
    let js = c.get(format!("{}/app.js", s.url)).send().await.unwrap();
    assert!(js.headers()["content-type"]
        .to_str()
        .unwrap()
        .contains("javascript"));
    assert!(js.text().await.unwrap().contains("tracesPage"));
    let css = c.get(format!("{}/style.css", s.url)).send().await.unwrap();
    assert!(css
        .text()
        .await
        .unwrap()
        .contains("prefers-color-scheme: dark"));
}

#[tokio::test]
async fn file_store_persists_across_reopen() {
    let dir = std::env::temp_dir().join(format!("sinew-scope-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("scope.db");
    let path = path.to_str().unwrap();
    let (main, _) = sample_runs();
    {
        let store = Store::open(path).await.unwrap();
        let pairs: Vec<_> = main.iter().cloned().map(|r| ("p".to_string(), r)).collect();
        store.ingest(&pairs).await.unwrap();
    }
    let store = Store::open(path).await.unwrap();
    let page = store
        .list_traces(&sinew_scope::TraceQuery::default())
        .await
        .unwrap();
    assert_eq!(page.total, 2);
    drop(store);
    std::fs::remove_dir_all(dir).unwrap();
}
