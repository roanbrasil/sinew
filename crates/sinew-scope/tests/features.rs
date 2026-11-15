//! End-to-end tests for the prompt hub, feedback, annotation queues, cost,
//! search, metrics, rules, alerts, retention, OTLP export and migrations.
//! Everything runs offline against an in-process server on 127.0.0.1:0.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{routing, Json, Router};
use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, ChatModel, Message};
use sinew_eval::{
    evaluate_pairwise, Dataset, EvalError, ExactMatch, Experiment, Feedback, FnPairwise,
    PairwiseEvaluator, PairwiseInput, Preference, PromptHub, PromptPush, ScopeClient,
};
use sinew_scope::{router_with, EngineConfig, ServerConfig, Store};
use sinew_trace::{OtlpSink, OtlpSinkConfig, Run, RunKind, TraceSink};
use tokio::sync::oneshot;

struct Server {
    url: String,
    store: Store,
    stop: Option<oneshot::Sender<()>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
    }
}

async fn spawn_with(config: ServerConfig) -> Server {
    let store = Store::memory().await.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = oneshot::channel();
    tokio::spawn(sinew_scope::server::serve_with_shutdown(
        listener,
        router_with(store.clone(), config),
        async {
            let _ = rx.await;
        },
    ));
    Server {
        url,
        store,
        stop: Some(tx),
    }
}

async fn spawn() -> Server {
    spawn_with(ServerConfig::default()).await
}

#[allow(clippy::too_many_arguments)]
fn run(
    id: &str,
    trace: &str,
    parent: Option<&str>,
    name: &str,
    kind: RunKind,
    start: u64,
    dur: u64,
    tokens: Option<(u32, u32)>,
    metadata: Value,
) -> Run {
    Run {
        id: id.into(),
        trace_id: trace.into(),
        parent_id: parent.map(str::to_string),
        name: name.into(),
        kind,
        start_ms: start,
        end_ms: Some(start + dur),
        inputs: json!({"question": format!("question for {id}")}),
        outputs: Some(json!(format!("answer from {id}"))),
        error: None,
        input_tokens: tokens.map(|t| t.0),
        output_tokens: tokens.map(|t| t.1),
        metadata,
    }
}

async fn post(c: &reqwest::Client, url: String, body: Value) -> (u16, Value) {
    let r = c.post(&url).json(&body).send().await.unwrap();
    let status = r.status().as_u16();
    (status, r.json().await.unwrap_or(Value::Null))
}

async fn get(c: &reqwest::Client, url: String) -> Value {
    let r = c.get(&url).send().await.unwrap();
    assert!(r.status().is_success(), "GET {url}: {}", r.status());
    r.json().await.unwrap()
}

async fn ingest(c: &reqwest::Client, url: &str, project: &str, runs: &[Run]) {
    let (status, body) = post(
        c,
        format!("{url}/api/runs"),
        json!({"project": project, "runs": runs}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Polls `f` until it returns `Some` (max ~5 s).
async fn eventually<F, Fut, T>(mut f: F) -> T
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Option<T>>,
{
    for _ in 0..100 {
        if let Some(v) = f().await {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("condition not met in time");
}

#[tokio::test]
async fn prompt_hub_versions_tags_diff_and_cache() {
    let s = spawn().await;
    let hub = PromptHub::new(&s.url).with_ttl(Duration::from_secs(0));

    let v1 = hub
        .push(
            PromptPush::new("support-answer", "Answer {question} politely.")
                .description("support bot")
                .model_config(json!({"model": "openai:gpt-4.1-mini", "temperature": 0.2}))
                .message("first")
                .tag("prod"),
        )
        .await
        .unwrap();
    assert!(v1.created);
    assert_eq!(v1.version, 1);
    assert_eq!(v1.commit.len(), 12);

    // Same content: no new version.
    let again = hub
        .push(
            PromptPush::new("support-answer", "Answer {question} politely.")
                .model_config(json!({"model": "openai:gpt-4.1-mini", "temperature": 0.2})),
        )
        .await
        .unwrap();
    assert!(!again.created);
    assert_eq!(again.commit, v1.commit);

    let v2 = hub
        .push(
            PromptPush::new(
                "support-answer",
                "Answer {question} politely.\nUse the {tone} tone.",
            )
            .model_config(json!({"model": "openai:gpt-4.1-mini", "temperature": 0.5}))
            .tag("staging"),
        )
        .await
        .unwrap();
    assert_eq!(v2.version, 2);
    assert_ne!(v2.commit, v1.commit);

    let prod = hub.pull("support-answer@prod").await.unwrap();
    assert_eq!(prod.variables(), vec!["question"]);
    let latest = hub.pull_version("support-answer").await.unwrap();
    assert_eq!(latest.commit, v2.commit);
    assert_eq!(latest.input_variables, vec!["question", "tone"]);
    assert_eq!(latest.tags, vec!["staging"]);
    let by_number = hub.pull_version("support-answer@v1").await.unwrap();
    assert_eq!(by_number.commit, v1.commit);
    let by_prefix = hub
        .pull_version(&format!("support-answer@{}", &v2.commit[..6]))
        .await
        .unwrap();
    assert_eq!(by_prefix.version, 2);
    let rendered = latest
        .render_messages(&HashMap::from([
            ("question", "where is my order?".to_string()),
            ("tone", "friendly".to_string()),
        ]))
        .unwrap();
    assert!(rendered[0].content.contains("friendly"));

    // Promote v2 to prod.
    hub.tag("support-answer", "prod", &v2.commit).await.unwrap();
    assert_eq!(
        hub.pull_version("support-answer@prod")
            .await
            .unwrap()
            .version,
        2
    );

    // Diff and listing over REST.
    let c = reqwest::Client::new();
    let d = get(
        &c,
        format!("{}/api/prompts/support-answer/diff?from=v1&to=v2", s.url),
    )
    .await;
    assert!(d["template"]
        .as_array()
        .unwrap()
        .iter()
        .any(|l| l["op"] == "add" && l["text"] == "Use the {tone} tone."));
    assert_eq!(d["variables_added"], json!(["tone"]));
    assert_eq!(d["model_config"][0]["key"], "temperature");
    let list = get(&c, format!("{}/api/prompts", s.url)).await;
    assert_eq!(list[0]["version_count"], 2);
    assert_eq!(list[0]["tags"]["prod"], v2.commit);
    let at = get(&c, format!("{}/api/prompts/support-answer@v1", s.url)).await;
    assert_eq!(at["version"], 1);
    assert_eq!(hub.versions("support-answer").await.unwrap().len(), 2);

    // Chat prompts and invalid templates.
    hub.push(PromptPush::chat(
        "qa",
        [("system", "Reply in {lang}."), ("user", "{q}")],
    ))
    .await
    .unwrap();
    let qa = hub.pull_version("qa").await.unwrap();
    assert_eq!(qa.messages.as_ref().unwrap().len(), 2);
    let bad = hub.push(PromptPush::new("broken", "oops {name")).await;
    assert!(matches!(bad, Err(EvalError::Http(m)) if m.contains("400")));
    let missing = hub.pull("nope@prod").await;
    assert!(matches!(missing, Err(EvalError::NotFound(_))));

    // Stale-on-error: the server goes away, the cached copy is served.
    let cached = PromptHub::new(&s.url).with_ttl(Duration::from_millis(1));
    cached.pull("support-answer@prod").await.unwrap();
    drop(s);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let t = cached.pull("support-answer@prod").await.unwrap();
    assert_eq!(t.variables(), vec!["question", "tone"]);
}

#[tokio::test]
async fn feedback_queues_and_dataset_from_runs() {
    let s = spawn().await;
    let c = reqwest::Client::new();
    let t0 = now_ms();
    ingest(
        &c,
        &s.url,
        "support",
        &[
            run(
                "r1",
                "r1",
                None,
                "agent",
                RunKind::Graph,
                t0,
                100,
                None,
                json!({}),
            ),
            run(
                "r2",
                "r2",
                None,
                "agent",
                RunKind::Graph,
                t0 + 1,
                100,
                None,
                json!({}),
            ),
        ],
    )
    .await;
    let client = ScopeClient::new(&s.url);
    let id = client
        .create_feedback(
            &Feedback::score("r1", "thumbs", 1.0)
                .with_comment("great")
                .with_source("user"),
        )
        .await
        .unwrap();
    assert!(!id.is_empty());
    let list = client.feedback_for_run("r1").await.unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["comment"], "great");
    assert_eq!(list[0]["trace_id"], "r1");
    // Unknown run and empty feedback are rejected.
    let (st, _) = post(
        &c,
        format!("{}/api/feedback", s.url),
        json!({"run_id": "nope", "key": "k", "score": 1}),
    )
    .await;
    assert_eq!(st, 400);
    let (st, _) = post(
        &c,
        format!("{}/api/feedback", s.url),
        json!({"run_id": "r1", "key": "k"}),
    )
    .await;
    assert_eq!(st, 400);

    // Annotation queue.
    let q = client
        .create_queue("review", "check answers", &["correctness", "tone"])
        .await
        .unwrap();
    client
        .add_to_queue(&q, &["r1", "r2", "ghost"])
        .await
        .unwrap();
    let detail = get(&c, format!("{}/api/queues/{q}", s.url)).await;
    assert_eq!(detail["pending"], 2);
    let next = get(&c, format!("{}/api/queues/{q}/next", s.url)).await;
    assert_eq!(next["item"]["run_id"], "r1");
    assert_eq!(next["item"]["run"]["name"], "agent");
    assert_eq!(next["queue"]["keys"], json!(["correctness", "tone"]));
    let (st, _) = post(
        &c,
        format!("{}/api/queues/{q}/items/r1/done", s.url),
        json!({"feedback": [{"key": "correctness", "score": 0.5, "comment": "partly"}, {"key": "tone"}]}),
    )
    .await;
    assert_eq!(st, 200);
    let next = get(&c, format!("{}/api/queues/{q}/next", s.url)).await;
    assert_eq!(next["item"]["run_id"], "r2");
    let fb = client.feedback_for_run("r1").await.unwrap();
    assert_eq!(fb.len(), 2, "empty `tone` entry is skipped");
    assert_eq!(fb[1]["source"], "annotation:review");
    let queues = get(&c, format!("{}/api/queues", s.url)).await;
    assert_eq!(
        (queues[0]["pending"].clone(), queues[0]["done"].clone()),
        (json!(1), json!(1))
    );
    let stats = get(&c, format!("{}/api/feedback/stats?project=support", s.url)).await;
    assert_eq!(stats.as_array().unwrap().len(), 2);

    // Add a run to a dataset.
    let (st, body) = post(
        &c,
        format!("{}/api/datasets/golden/examples", s.url),
        json!({"run_id": "r2"}),
    )
    .await;
    assert_eq!(st, 200, "{body}");
    let ds = get(&c, format!("{}/api/datasets/golden", s.url)).await;
    let ex = &ds["dataset"]["examples"][0];
    assert_eq!(ex["inputs"]["question"], "question for r2");
    assert_eq!(ex["reference"], "answer from r2");
    assert_eq!(ex["metadata"]["source_run_id"], "r2");
}

#[tokio::test]
async fn cost_search_and_metrics() {
    let s = spawn().await;
    let c = reqwest::Client::new();
    let t0 = now_ms() - 10_000;
    let mut runs = Vec::new();
    for i in 0..20u64 {
        let tr = format!("t{i:02}");
        let model = if i % 2 == 0 {
            "gpt-4o-mini"
        } else {
            "claude-sonnet-5-5"
        };
        let mut root = run(
            &tr,
            &tr,
            None,
            "qa-agent",
            RunKind::Graph,
            t0 + i * 100,
            50 + i * 10,
            None,
            json!({"tags": [if i < 10 { "prod" } else { "dev" }], "user": format!("u{}", i % 3)}),
        );
        if i == 3 {
            root.error = Some("timeout".into());
            root.inputs = json!({"question": "zebra crossing rules"});
        }
        runs.push(root);
        runs.push(run(
            &format!("{tr}-llm"),
            &tr,
            Some(&tr),
            model,
            RunKind::Llm,
            t0 + i * 100 + 1,
            40,
            Some((1_000_000, 100_000)),
            json!({"model": model, "ttft_ms": 20 + i}),
        ));
    }
    ingest(&c, &s.url, "demo", &runs).await;

    // Trace cost = LLM cost: gpt-4o-mini $0.15 + 0.1 × $0.60 = $0.21;
    // claude-sonnet-5-5 (Sonnet 5 pricing) $2 + 0.1 × $10 = $3.
    let t = get(&c, format!("{}/api/traces/t00", s.url)).await;
    assert!((t["summary"]["cost_usd"].as_f64().unwrap() - 0.21).abs() < 1e-9);
    let llm = t["runs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["kind"] == "llm")
        .unwrap();
    assert_eq!(llm["model"], "gpt-4o-mini");
    assert_eq!(llm["ttft_ms"], 20);
    let page = get(&c, format!("{}/api/traces?project=demo", s.url)).await;
    assert!(
        (page["stats"]["cost_usd"].as_f64().unwrap() - (10.0 * 0.21 + 10.0 * 3.0)).abs() < 1e-6
    );
    assert!(page["stats"]["p99_latency_ms"].as_i64().is_some());

    // Trace filters: tag, metadata, full text.
    let tagged = get(&c, format!("{}/api/traces?tag=prod", s.url)).await;
    assert_eq!(tagged["total"], 10);
    let by_user = get(&c, format!("{}/api/traces?meta.user=u1", s.url)).await;
    assert_eq!(by_user["total"], 7);
    let found = get(&c, format!("{}/api/traces?q=zebra", s.url)).await;
    assert_eq!(found["total"], 1);
    assert_eq!(found["traces"][0]["trace_id"], "t03");

    // Run search with cursor pagination.
    let mut cursor: Option<String> = None;
    let mut seen = Vec::new();
    loop {
        let url = match &cursor {
            Some(cur) => format!("{}/api/runs?kind=llm&limit=7&cursor={cur}", s.url),
            None => format!("{}/api/runs?kind=llm&limit=7", s.url),
        };
        let p = get(&c, url).await;
        for r in p["runs"].as_array().unwrap() {
            seen.push(r["id"].as_str().unwrap().to_string());
        }
        match p["next_cursor"].as_str() {
            Some(n) => cursor = Some(n.to_string()),
            None => break,
        }
    }
    assert_eq!(seen.len(), 20);
    assert_eq!(seen[0], "t19-llm", "newest first");
    let mut dedup = seen.clone();
    dedup.sort();
    dedup.dedup();
    assert_eq!(dedup.len(), 20);
    let claude = get(&c, format!("{}/api/runs?model=claude&limit=100", s.url)).await;
    assert_eq!(claude["runs"].as_array().unwrap().len(), 10);
    let text = get(&c, format!("{}/api/runs?q=answer%20t05", s.url)).await;
    assert!(text["runs"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r["id"] == "t05"));
    let errs = get(&c, format!("{}/api/runs?errors=true&root=true", s.url)).await;
    assert_eq!(errs["runs"].as_array().unwrap().len(), 1);

    // Metrics: traces by name, LLM runs by model.
    let m = get(&c, format!("{}/api/metrics?project=demo", s.url)).await;
    assert_eq!(m["totals"]["count"], 20);
    assert_eq!(m["totals"]["errors"], 1);
    assert!((m["totals"]["error_rate"].as_f64().unwrap() - 0.05).abs() < 1e-9);
    assert_eq!(m["totals"]["p50_latency_ms"], 150);
    assert_eq!(m["totals"]["p99_latency_ms"], 240);
    assert!(!m["buckets"].as_array().unwrap().is_empty());
    let llm = get(
        &c,
        format!("{}/api/metrics?level=llm&group_by=model&tag=prod", s.url),
    )
    .await;
    // Run-level tag filter: LLM runs carry no tags.
    assert_eq!(llm["totals"]["count"], 0);
    let llm = get(
        &c,
        format!("{}/api/metrics?level=llm&group_by=model", s.url),
    )
    .await;
    let groups = llm["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 2);
    let mini = groups.iter().find(|g| g["key"] == "gpt-4o-mini").unwrap();
    assert_eq!(mini["count"], 10);
    assert!((mini["cost_usd"].as_f64().unwrap() - 2.1).abs() < 1e-6);
    assert_eq!(mini["ttft_p50_ms"], 30);

    // Pricing override recomputes stored costs.
    let r = c
        .put(format!("{}/api/pricing", s.url))
        .json(&json!({"models": [{"model": "gpt-4o-mini", "input_per_mtok": 1.0, "output_per_mtok": 0.0}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let t = get(&c, format!("{}/api/traces/t00", s.url)).await;
    assert!((t["summary"]["cost_usd"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    let t1 = get(&c, format!("{}/api/traces/t01", s.url)).await;
    assert_eq!(t1["summary"]["cost_usd"], 0.0, "claude no longer priced");
    let r = c
        .delete(format!("{}/api/pricing", s.url))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let pricing = get(&c, format!("{}/api/pricing", s.url)).await;
    assert!(pricing["models"].as_array().unwrap().len() > 30);
    let bad = c
        .put(format!("{}/api/pricing", s.url))
        .json(&json!({"models": [{"model": "x", "input_per_mtok": -1.0, "output_per_mtok": 0.0}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);
}

#[tokio::test]
async fn online_rules_score_incoming_runs() {
    // The LLM judge resolves "mock:judge" to a scripted model.
    let judge: Arc<dyn ChatModel> = Arc::new(ScriptedModel::new([Message::assistant(
        "{\"score\": 0.9, \"reason\": \"polite\"}",
    )]));
    let factory: sinew_scope::ModelFactory = Arc::new(move |spec: &str| {
        if spec == "mock:judge" {
            Ok(judge.clone())
        } else {
            Err(format!("unknown model {spec}"))
        }
    });
    let s = spawn_with(ServerConfig {
        engine: Some(EngineConfig {
            model_factory: factory,
            ..Default::default()
        }),
        ..Default::default()
    })
    .await;
    let c = reqwest::Client::new();
    let (st, rule) = post(
        &c,
        format!("{}/api/rules", s.url),
        json!({
            "name": "answers",
            "filter": {"project": "prod", "kind": "graph"},
            "sample_rate": 1.0,
            "evaluators": [
                {"type": "contains", "needle": "answer", "key": "mentions_answer"},
                {"type": "llm_judge", "model": "mock:judge", "rubric": "Is it polite?", "key": "polite"}
            ]
        }),
    )
    .await;
    assert_eq!(st, 200, "{rule}");
    // Invalid rules are rejected.
    let (st, _) = post(
        &c,
        format!("{}/api/rules", s.url),
        json!({"name": "x", "evaluators": [{"type": "regex", "pattern": "("}]}),
    )
    .await;
    assert_eq!(st, 400);

    let t0 = now_ms();
    ingest(
        &c,
        &s.url,
        "prod",
        &[
            run(
                "g1",
                "g1",
                None,
                "agent",
                RunKind::Graph,
                t0,
                10,
                None,
                json!({}),
            ),
            run(
                "g1-tool",
                "g1",
                Some("g1"),
                "calc",
                RunKind::Tool,
                t0,
                5,
                None,
                json!({}),
            ),
        ],
    )
    .await;
    ingest(
        &c,
        &s.url,
        "other",
        &[run(
            "g2",
            "g2",
            None,
            "agent",
            RunKind::Graph,
            t0,
            10,
            None,
            json!({}),
        )],
    )
    .await;
    let fb = eventually(|| {
        let c = c.clone();
        let url = format!("{}/api/feedback?run_id=g1", s.url);
        async move {
            let v: Value = c.get(url).send().await.ok()?.json().await.ok()?;
            (v.as_array()?.len() == 2).then_some(v)
        }
    })
    .await;
    let polite = fb
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["key"] == "polite")
        .unwrap();
    assert_eq!(polite["score"], 0.9);
    assert_eq!(polite["source"], "rule:answers");
    assert_eq!(polite["comment"], "polite");
    let other = get(&c, format!("{}/api/feedback?run_id=g2", s.url)).await;
    assert!(
        other.as_array().unwrap().is_empty(),
        "filtered out by project"
    );
    let tool = get(&c, format!("{}/api/feedback?run_id=g1-tool", s.url)).await;
    assert!(tool.as_array().unwrap().is_empty(), "filtered out by kind");
    let rules = get(&c, format!("{}/api/rules", s.url)).await;
    assert_eq!(rules[0]["runs_evaluated"], 2);

    // Disabled rules do nothing; "run now" backfills.
    let id = rule["id"].as_str().unwrap();
    let mut input = rule.clone();
    input["enabled"] = json!(false);
    input["evaluators"] = json!([{"type": "not_empty"}]);
    let r = c
        .put(format!("{}/api/rules/{id}", s.url))
        .json(&input)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let (st, n) = post(
        &c,
        format!("{}/api/rules/{id}/run?limit=10", s.url),
        json!({}),
    )
    .await;
    assert_eq!(st, 200);
    assert_eq!(n["feedback"], 1, "one prod graph run");
}

#[tokio::test]
async fn alerts_post_to_webhook_and_retention_purges() {
    // Webhook receiver.
    let received: Arc<Mutex<Vec<Value>>> = Arc::default();
    let sink = received.clone();
    let hook = Router::new().route(
        "/hook",
        routing::post(move |Json(v): Json<Value>| {
            let sink = sink.clone();
            async move {
                sink.lock().unwrap().push(v);
                "ok"
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hook_url = format!("http://{}/hook", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, hook).await });

    let s = spawn().await;
    let c = reqwest::Client::new();
    let t0 = now_ms() - 1000;
    let mut runs = Vec::new();
    for i in 0..4u64 {
        let mut r = run(
            &format!("e{i}"),
            &format!("e{i}"),
            None,
            "flaky",
            RunKind::Chain,
            t0 + i,
            10,
            None,
            json!({}),
        );
        if i < 2 {
            r.error = Some("boom".into());
        }
        runs.push(r);
    }
    // An old trace for retention.
    runs.push(run(
        "old",
        "old",
        None,
        "ancient",
        RunKind::Chain,
        1_000,
        10,
        None,
        json!({}),
    ));
    ingest(&c, &s.url, "svc", &runs).await;

    let (st, alert) = post(
        &c,
        format!("{}/api/alerts", s.url),
        json!({
            "name": "errors",
            "metric": "error_rate",
            "threshold": 0.25,
            "window_ms": 60_000,
            "filter": {"project": "svc"},
            "webhook_url": hook_url,
            "cooldown_ms": 3_600_000
        }),
    )
    .await;
    assert_eq!(st, 200, "{alert}");
    let id = alert["id"].as_str().unwrap();
    let (_, check) = post(&c, format!("{}/api/alerts/{id}/check", s.url), json!({})).await;
    assert_eq!(check["status"], "firing");
    assert_eq!(check["fired"], true);
    assert!((check["value"].as_f64().unwrap() - 0.5).abs() < 1e-9);
    {
        let got = received.lock().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["alert"]["name"], "errors");
        assert!(got[0]["text"].as_str().unwrap().contains("error_rate"));
    }
    // Cooldown: firing again does not notify.
    let (_, check) = post(&c, format!("{}/api/alerts/{id}/check", s.url), json!({})).await;
    assert_eq!(
        (check["status"].clone(), check["fired"].clone()),
        (json!("firing"), json!(false))
    );
    assert_eq!(received.lock().unwrap().len(), 1);
    let a = get(&c, format!("{}/api/alerts/{id}", s.url)).await;
    assert_eq!(a["last_status"], "firing");
    let (st, _) = post(
        &c,
        format!("{}/api/alerts", s.url),
        json!({"name": "x", "metric": "nope", "webhook_url": hook_url}),
    )
    .await;
    assert_eq!(st, 400);

    // Retention.
    let r = c
        .put(format!("{}/api/settings", s.url))
        .json(&json!({"retention_days": 30}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let (_, purged) = post(&c, format!("{}/api/retention/run", s.url), json!({})).await;
    assert_eq!(purged["traces"], 1);
    assert_eq!(purged["runs"], 1);
    let page = get(&c, format!("{}/api/traces?project=svc", s.url)).await;
    assert_eq!(page["total"], 4);
    assert_eq!(
        get(&c, format!("{}/api/settings", s.url)).await["retention_days"],
        30
    );
    assert!(s.store.has_fts());
}

#[tokio::test]
async fn otlp_sink_round_trips_into_scope() {
    let s = spawn().await;
    let t0 = now_ms();
    let runs = vec![
        run(
            "8f1d2c3b-0000-4000-8000-000000000001",
            "8f1d2c3b-0000-4000-8000-000000000001",
            None,
            "planner",
            RunKind::Graph,
            t0,
            30,
            None,
            json!({"tags": ["otel"]}),
        ),
        run(
            "8f1d2c3b-0000-4000-8000-000000000002",
            "8f1d2c3b-0000-4000-8000-000000000001",
            Some("8f1d2c3b-0000-4000-8000-000000000001"),
            "gpt-4o-mini",
            RunKind::Llm,
            t0 + 1,
            20,
            Some((100, 10)),
            json!({"model": "gpt-4o-mini-2024-07-18", "ttft_ms": 7}),
        ),
    ];
    let json_sink = OtlpSink::new(
        OtlpSinkConfig::new(format!("{}/v1/traces", s.url))
            .service_name("otel-json")
            .project("otlp-json"),
    );
    for r in &runs {
        json_sink.record(r);
    }
    tokio::task::spawn_blocking(move || {
        json_sink.flush();
        assert_eq!(json_sink.sent(), 2);
    })
    .await
    .unwrap();

    let c = reqwest::Client::new();
    let page = get(&c, format!("{}/api/traces?project=otlp-json", s.url)).await;
    assert_eq!(page["total"], 1);
    let tid = page["traces"][0]["trace_id"].as_str().unwrap().to_string();
    assert_eq!(tid, "8f1d2c3b000040008000000000000001");
    let t = get(&c, format!("{}/api/traces/{tid}", s.url)).await;
    assert_eq!(t["summary"]["name"], "planner");
    assert_eq!(t["summary"]["kind"], "graph");
    let llm = t["runs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["kind"] == "llm")
        .unwrap();
    assert_eq!(llm["name"], "gpt-4o-mini");
    assert_eq!(llm["model"], "gpt-4o-mini-2024-07-18");
    assert_eq!(llm["ttft_ms"], 7);
    assert_eq!(llm["input_tokens"], 100);
    assert!(llm["cost_usd"].as_f64().unwrap() > 0.0);
    let root = t["runs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["kind"] == "graph")
        .unwrap();
    assert_eq!(root["metadata"]["tags"], json!(["otel"]));
    let tagged = get(&c, format!("{}/api/traces?tag=otel", s.url)).await;
    assert_eq!(tagged["total"], 1);

    #[cfg(feature = "otlp-proto")]
    {
        let proto_sink = OtlpSink::new(
            OtlpSinkConfig::new(format!("{}/v1/traces", s.url))
                .project("otlp-proto")
                .protocol(sinew_trace::OtlpProtocol::HttpProtobuf),
        );
        let mut r = runs[0].clone();
        r.id = "proto-root".into();
        r.trace_id = "proto-trace".into();
        proto_sink.record(&r);
        tokio::task::spawn_blocking(move || {
            proto_sink.flush();
            assert_eq!(proto_sink.sent(), 1);
        })
        .await
        .unwrap();
        let page = get(&c, format!("{}/api/traces?project=otlp-proto", s.url)).await;
        assert_eq!(page["total"], 1);
        assert_eq!(page["traces"][0]["name"], "planner");
    }
}

#[tokio::test]
async fn pairwise_upload_and_repetition_compare() {
    let s = spawn().await;
    let mut ds = Dataset::new("caps");
    ds.push(json!("France"), Some(json!("Paris")));
    ds.push(json!("Japan"), Some(json!("Tokyo")));
    let a = Experiment::new(&ds)
        .evaluator(ExactMatch::new())
        .repetitions(2)
        .name("always-paris")
        .run(|_: Value| async { Ok::<_, String>(json!("Paris")) })
        .await;
    let b = Experiment::new(&ds)
        .evaluator(ExactMatch::new())
        .repetitions(2)
        .name("lookup")
        .run(|i: Value| async move {
            Ok::<_, String>(json!(if i == "France" { "Paris" } else { "Tokyo" }))
        })
        .await;
    let client = ScopeClient::new(&s.url);
    client.upload_experiment(&ds, &a).await.unwrap();
    client.upload_report(&b).await.unwrap();
    let evaluators: Vec<Box<dyn PairwiseEvaluator>> = vec![Box::new(FnPairwise::new(
        "correct",
        |i: PairwiseInput<'_>| {
            let r = i.reference.cloned();
            match (Some(i.a.clone()) == r, Some(i.b.clone()) == r) {
                (true, false) => Some(Preference::A),
                (false, true) => Some(Preference::B),
                _ => Some(Preference::Tie),
            }
        },
    ))];
    let pw = evaluate_pairwise(&a, &b, &evaluators, 2).await;
    client.upload_pairwise(&pw).await.unwrap();

    let c = reqwest::Client::new();
    let list = get(&c, format!("{}/api/pairwise?dataset=caps", s.url)).await;
    assert_eq!(list[0]["summary"]["correct"]["wins_b"], 2);
    let back = get(&c, format!("{}/api/pairwise/{}", s.url, pw.id)).await;
    assert_eq!(back["results"].as_array().unwrap().len(), 4);
    let cmp = get(&c, format!("{}/api/compare?a={}&b={}", s.url, a.id, b.id)).await;
    let rows = cmp["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 4, "aligned by example and repetition");
    assert!(rows.iter().all(|r| !r["b"].is_null()));
}

#[tokio::test]
async fn migrates_a_v1_database() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.db");
    let path = path.to_str().unwrap().to_string();
    {
        // Build a database with the original (v1) schema and data.
        use sqlx::{sqlite::SqliteConnectOptions, ConnectOptions, Connection};
        use std::str::FromStr;
        let mut conn = SqliteConnectOptions::from_str(&format!("sqlite://{path}"))
            .unwrap()
            .create_if_missing(true)
            .connect()
            .await
            .unwrap();
        sqlx::raw_sql(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, parent_id TEXT, \
             project TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL, start_ms INTEGER NOT NULL, \
             end_ms INTEGER, inputs TEXT NOT NULL, outputs TEXT, error TEXT, input_tokens INTEGER, \
             output_tokens INTEGER, metadata TEXT NOT NULL); \
             CREATE TABLE traces (trace_id TEXT PRIMARY KEY, project TEXT NOT NULL, root_id TEXT NOT NULL, \
             name TEXT NOT NULL, kind TEXT NOT NULL, start_ms INTEGER NOT NULL, end_ms INTEGER, \
             latency_ms INTEGER, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, \
             run_count INTEGER NOT NULL, error_count INTEGER NOT NULL, error TEXT); \
             INSERT INTO runs VALUES ('a', 'a', NULL, 'p', 'agent', 'graph', 1, 50, '{\"q\":\"legacy kiwi\"}', '\"ok\"', NULL, NULL, NULL, '{}'); \
             INSERT INTO runs VALUES ('b', 'a', 'a', 'p', 'gpt-4o', 'llm', 2, 40, '{}', '\"hi\"', NULL, 1000000, 0, '{\"model\":\"gpt-4o\"}'); \
             INSERT INTO traces VALUES ('a', 'p', 'a', 'agent', 'graph', 1, 50, 49, 1000000, 0, 2, 0, NULL);",
        )
        .execute(&mut conn)
        .await
        .unwrap();
        conn.close().await.unwrap();
    }
    let store = Store::open(&path).await.unwrap();
    assert_eq!(
        store.schema_version().await.unwrap(),
        sinew_scope::store::SCHEMA_VERSION
    );
    let t = store.trace("a").await.unwrap().unwrap();
    assert!(
        (t.summary.cost_usd - 2.5).abs() < 1e-9,
        "gpt-4o: 1M input tokens = $2.50"
    );
    let b = t.runs.iter().find(|r| r.run.id == "b").unwrap();
    assert_eq!(b.model.as_deref(), Some("gpt-4o"));
    let hits = store
        .search_runs(&sinew_scope::RunQuery {
            q: Some("kiwi".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(hits.runs.len(), 1, "existing rows are indexed");
    // New tables work on the migrated database.
    store
        .push_prompt(&sinew_scope::store::PromptPushInput {
            name: "p".into(),
            template: "hi {x}".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    drop(store);
    // Reopening is a no-op.
    let store = Store::open(&path).await.unwrap();
    assert_eq!(store.prompts().await.unwrap().len(), 1);
}

#[tokio::test]
async fn ui_pages_are_served() {
    let s = spawn().await;
    let c = reqwest::Client::new();
    let html = c
        .get(format!("{}/", s.url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    for nav in [
        "#/monitor",
        "#/runs",
        "#/prompts",
        "#/queues",
        "#/rules",
        "#/settings",
    ] {
        assert!(html.contains(nav), "nav link {nav}");
    }
    assert!(html.contains("pages.js") && html.contains("logo.png"));
    assert!(!html.contains("https://"), "no external resources");
    let js = c.get(format!("{}/pages.js", s.url)).send().await.unwrap();
    assert!(js.headers()["content-type"]
        .to_str()
        .unwrap()
        .contains("javascript"));
    let js = js.text().await.unwrap();
    for f in [
        "function monitorPage",
        "function runsPage",
        "function promptsPage",
        "function promptPage",
        "function queuesPage",
        "function queuePage",
        "function rulesPage",
        "function settingsPage",
        "function pairwisePage",
        "function feedbackPanel",
        "function lineChart",
    ] {
        assert!(js.contains(f), "{f}");
    }
    let app = c
        .get(format!("{}/app.js", s.url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(app.contains("monitorPage") && app.contains("cost"));
    let css = c
        .get(format!("{}/style.css", s.url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(css.contains("#EA7A3A") || css.contains("#ea7a3a"));
    let logo = c.get(format!("{}/logo.png", s.url)).send().await.unwrap();
    assert_eq!(logo.headers()["content-type"], "image/png");
    assert!(logo.bytes().await.unwrap().starts_with(b"\x89PNG"));
}
