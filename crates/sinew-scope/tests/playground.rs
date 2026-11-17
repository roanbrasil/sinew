//! Playground API tests: scripted models registered on the server, the
//! allowed-provider policy, prompt-hub resolution, prefill from stored runs,
//! streaming and auth. Everything runs offline on 127.0.0.1:0.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, ChatModel, ChatRequest, ChatResponse, Message, Usage};
use sinew_scope::{router_with, PlaygroundConfig, ServerConfig, Store};
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

async fn spawn(playground: PlaygroundConfig, token: Option<&str>) -> Server {
    let store = Store::memory().await.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = oneshot::channel();
    tokio::spawn(sinew_scope::server::serve_with_shutdown(
        listener,
        router_with(
            store.clone(),
            ServerConfig {
                token: token.map(str::to_string),
                disable_engine: true,
                playground: playground.discover_models(false),
                ..Default::default()
            },
        ),
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

fn scripted(replies: &[&str]) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new(
        replies.iter().map(|r| Message::assistant(*r)),
    ))
}

async fn post(url: &str, body: Value) -> (u16, Value) {
    let r = reqwest::Client::new()
        .post(url)
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = r.status().as_u16();
    (status, r.json().await.unwrap_or(Value::Null))
}

async fn get(url: &str) -> (u16, Value) {
    let r = reqwest::get(url).await.unwrap();
    let status = r.status().as_u16();
    (status, r.json().await.unwrap_or(Value::Null))
}

/// A model that reports no capabilities, so the playground drops
/// `temperature` and `tools` for it.
struct Plain;

#[async_trait]
impl ChatModel for Plain {
    fn name(&self) -> &str {
        "plain"
    }
    async fn chat(&self, request: ChatRequest) -> sinew_core::Result<ChatResponse> {
        assert!(request.temperature.is_none() && request.tools.is_empty());
        Ok(ChatResponse {
            message: Message::assistant("plain answer"),
            usage: Usage::new(3, 2),
            model: "plain".into(),
        })
    }
}

#[tokio::test]
async fn runs_models_side_by_side_and_traces_them() {
    let a = scripted(&["Bonjour Ada"]);
    let b = scripted(&["Hello Ada, nice to meet you"]);
    let s = spawn(
        PlaygroundConfig::default()
            .with_model("test:a", a.clone())
            .with_model("ollama:fake", b.clone())
            .with_model("test:plain", Arc::new(Plain)),
        None,
    )
    .await;
    let (status, body) = post(
        &format!("{}/api/playground/run", s.url),
        json!({
            "messages": [
                {"role": "system", "content": "Reply in {lang}."},
                {"role": "user", "content": "Greet {name}."}
            ],
            "variables": {"lang": "French", "name": "Ada"},
            "models": ["test:a", "ollama:fake", "test:plain"],
            "temperature": 0.3,
            "max_tokens": 64
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["spec"], "test:a");
    assert_eq!(results[0]["output"]["content"], "Bonjour Ada");
    assert_eq!(
        results[1]["output"]["content"],
        "Hello Ada, nice to meet you"
    );
    assert_eq!(results[0]["output_tokens"], 2);
    assert!(results[0]["latency_ms"].is_u64());
    assert!(results[0]["ttft_ms"].is_u64());
    assert!(results[0]["error"].is_null());
    assert_eq!(results[2]["ignored"], json!(["temperature"]));

    // The model saw the rendered messages and the options.
    let req = &a.requests()[0];
    assert_eq!(req.messages[0].content, "Reply in French.");
    assert_eq!(req.messages[1].content, "Greet Ada.");
    assert_eq!(req.temperature, Some(0.3));
    assert_eq!(req.max_tokens, Some(64));

    // Traced under the playground project: a root plus one LLM run per model.
    let trace_id = body["trace_id"].as_str().unwrap();
    assert_eq!(body["project"], "playground");
    let t = s.store.trace(trace_id).await.unwrap().unwrap();
    assert_eq!(t.summary.project, "playground");
    assert_eq!(t.runs.len(), 4);
    let llm: Vec<_> = t
        .runs
        .iter()
        .filter(|r| r.run.kind == sinew_trace::RunKind::Llm)
        .collect();
    assert_eq!(llm.len(), 3);
    assert!(llm
        .iter()
        .all(|r| r.run.parent_id.as_deref() == Some(trace_id)));
    let run_a = llm.iter().find(|r| r.run.name == "test:a").unwrap();
    assert_eq!(run_a.run.metadata["spec"], "test:a");
    assert_eq!(
        run_a.run.metadata["invocation_params"]["temperature"],
        json!(0.3)
    );

    // That run opens back in the playground with the same inputs.
    let (status, p) = get(&format!(
        "{}/api/runs/{}/playground",
        s.url,
        results[0]["run_id"].as_str().unwrap()
    ))
    .await;
    assert_eq!(status, 200, "{p}");
    assert_eq!(p["spec"], "test:a");
    assert_eq!(p["spec_allowed"], true);
    assert_eq!(p["messages"][0]["content"], "Reply in French.");
    assert_eq!(p["max_tokens"], 64);
    assert_eq!(p["temperature"], 0.3);
}

#[tokio::test]
async fn model_errors_are_reported_per_model() {
    let s = spawn(
        PlaygroundConfig::default()
            .with_model("test:ok", scripted(&["fine"]))
            .with_model("test:empty", scripted(&[])),
        None,
    )
    .await;
    let (status, body) = post(
        &format!("{}/api/playground/run", s.url),
        json!({"messages": [{"role": "user", "content": "hi"}], "models": ["test:ok", "test:empty"]}),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(body["results"][0]["output"]["content"], "fine");
    assert!(body["results"][1]["error"].as_str().is_some());
    let t = s
        .store
        .trace(body["trace_id"].as_str().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(t.summary.error_count, 1);
}

#[tokio::test]
async fn streams_server_sent_events() {
    let s = spawn(
        PlaygroundConfig::default()
            .with_model("test:a", scripted(&["one two three"]))
            .with_model("test:b", scripted(&["four"])),
        None,
    )
    .await;
    let r = reqwest::Client::new()
        .post(format!("{}/api/playground/run", s.url))
        .json(&json!({
            "messages": [{"role": "user", "content": "count"}],
            "models": ["test:a", "test:b"],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let text = r.text().await.unwrap();
    let mut events: Vec<(String, Value)> = Vec::new();
    for block in text.split("\n\n") {
        let mut name = String::new();
        let mut data = String::new();
        for line in block.lines() {
            if let Some(v) = line.strip_prefix("event:") {
                name = v.trim().to_string();
            } else if let Some(v) = line.strip_prefix("data:") {
                data.push_str(v.trim_start());
            }
        }
        if !name.is_empty() {
            events.push((name, serde_json::from_str(&data).unwrap()));
        }
    }
    assert_eq!(events.first().unwrap().0, "start");
    assert_eq!(events.last().unwrap().0, "done");
    let deltas: String = events
        .iter()
        .filter(|(n, d)| n == "delta" && d["index"] == 0)
        .map(|(_, d)| d["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(deltas, "one two three");
    assert_eq!(events.iter().filter(|(n, _)| n == "result").count(), 2);
    let done = &events.last().unwrap().1;
    assert_eq!(done["results"][1]["output"]["content"], "four");
    assert!(s
        .store
        .trace(done["trace_id"].as_str().unwrap())
        .await
        .unwrap()
        .is_some());
}

#[tokio::test]
async fn enforces_allowed_providers() {
    // Default: local providers only.
    let s = spawn(PlaygroundConfig::default(), None).await;
    let url = format!("{}/api/playground/run", s.url);
    let msgs = json!([{"role": "user", "content": "hi"}]);
    for spec in [
        "openai:gpt-4.1-mini",
        "claude:claude-sonnet-4-5",
        "http://10.0.0.5:8000/v1|m",
    ] {
        let (status, body) = post(&url, json!({"messages": msgs, "models": [spec]})).await;
        assert_eq!(status, 403, "{spec}: {body}");
    }
    let (status, _) = post(&url, json!({"messages": msgs, "models": ["nope:m"]})).await;
    assert_eq!(status, 400);
    let (status, _) = post(&url, json!({"messages": msgs, "models": []})).await;
    assert_eq!(status, 400);
    let (status, _) = post(
        &url,
        json!({"messages": msgs, "models": vec!["ollama:x"; 9]}),
    )
    .await;
    assert_eq!(status, 400, "more than max_models");

    let (status, m) = get(&format!("{}/api/playground/models", s.url)).await;
    assert_eq!(status, 200);
    let allowed: Vec<&str> = m["allowed_providers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(allowed.contains(&"ollama") && allowed.contains(&"lmstudio"));
    assert!(!allowed.contains(&"openai"));
    assert!(m["providers"]
        .as_array()
        .unwrap()
        .iter()
        .all(|p| p["kind"] == "local"));

    // Allowing a hosted provider lets the factory build it (scripted here).
    let s = spawn(
        PlaygroundConfig::default()
            .allow_providers(["openai"])
            .with_model_factory(Arc::new(|_spec: &str| {
                Ok(scripted(&["from factory"]) as Arc<dyn ChatModel>)
            })),
        None,
    )
    .await;
    let (status, body) = post(
        &format!("{}/api/playground/run", s.url),
        json!({"messages": msgs, "models": ["openai:gpt-4.1-mini"]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["results"][0]["output"]["content"], "from factory");
    let (status, _) = post(
        &format!("{}/api/playground/run", s.url),
        json!({"messages": msgs, "models": ["ollama:x"]}),
    )
    .await;
    assert_eq!(status, 403, "ollama is no longer in the list");
}

#[tokio::test]
async fn resolves_prompt_hub_versions() {
    let model = scripted(&["4", "5"]);
    let s = spawn(
        PlaygroundConfig::default().with_model("test:a", model.clone()),
        None,
    )
    .await;
    let (status, body) = post(
        &format!("{}/api/prompts", s.url),
        json!({
            "name": "math",
            "template": "",
            "messages": [
                {"role": "system", "content": "Answer with a number."},
                {"role": "user", "content": "{a} + {b}?"}
            ],
            "model_config": {"model": "test:a", "temperature": 0.1, "max_tokens": 5},
            "tags": ["prod"]
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let commit = body["commit"].as_str().unwrap().to_string();
    let url = format!("{}/api/playground/run", s.url);
    let (status, body) = post(
        &url,
        json!({"prompt": "math@prod", "variables": {"a": 2, "b": "2"}}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["prompt"], format!("math@{commit}"));
    assert_eq!(body["results"][0]["spec"], "test:a");
    let req = &model.requests()[0];
    assert_eq!(req.messages[1].content, "2 + 2?");
    assert_eq!(req.temperature, Some(0.1));
    assert_eq!(req.max_tokens, Some(5));

    // Explicit options win over the version's model config.
    let (status, _) = post(
        &url,
        json!({"prompt": "math@v1", "variables": {"a": 2, "b": 3}, "models": ["test:a"], "temperature": 0.9}),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(model.requests()[1].temperature, Some(0.9));

    let (status, _) = post(&url, json!({"prompt": "math@prod", "variables": {"a": 1}})).await;
    assert_eq!(status, 400, "missing variable");
    let (status, _) = post(&url, json!({"prompt": "nope@prod"})).await;
    assert_eq!(status, 404);
    let (status, _) = post(
        &url,
        json!({"prompt": "math", "messages": [{"role": "user", "content": "x"}], "models": ["test:a"]}),
    )
    .await;
    assert_eq!(status, 400, "messages and prompt together");
}

#[tokio::test]
async fn prefills_from_ingested_runs() {
    let s = spawn(PlaygroundConfig::default(), None).await;
    let c = reqwest::Client::new();
    let runs = json!([
        {
            "id": "llm-1", "trace_id": "t1", "parent_id": "root-1", "name": "gpt-4o-mini",
            "kind": "llm", "start_ms": 1000, "end_ms": 1500,
            "inputs": {"messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Capital of France?"}
            ], "tools": ["search"]},
            "outputs": {"role": "assistant", "content": "Paris"},
            "error": null, "input_tokens": 12, "output_tokens": 1,
            "metadata": {"model": "gpt-4o-mini"}
        },
        {
            "id": "root-1", "trace_id": "t1", "parent_id": null, "name": "agent",
            "kind": "graph", "start_ms": 900, "end_ms": 1600, "inputs": {},
            "outputs": null, "error": null, "input_tokens": null, "output_tokens": null,
            "metadata": {}
        },
        {
            "id": "otel-1", "trace_id": "t2", "parent_id": null, "name": "llama3.2",
            "kind": "llm", "start_ms": 2000, "end_ms": 2100,
            "inputs": [{"role": "user", "parts": [{"type": "text", "content": "Hi there"}]}],
            "outputs": null, "error": null, "input_tokens": 2, "output_tokens": 3,
            "metadata": {"model": "llama3.2", "attributes": {
                "gen_ai.provider.name": "ollama",
                "gen_ai.request.temperature": 0.7,
                "gen_ai.request.max_tokens": 128
            }}
        }
    ]);
    let r = c
        .post(format!("{}/api/runs", s.url))
        .json(&runs)
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());

    let (status, p) = get(&format!("{}/api/runs/llm-1/playground", s.url)).await;
    assert_eq!(status, 200, "{p}");
    assert_eq!(p["model"], "gpt-4o-mini");
    assert_eq!(p["spec"], "openai:gpt-4o-mini");
    assert_eq!(p["spec_allowed"], false, "openai is not allowed by default");
    assert_eq!(p["messages"].as_array().unwrap().len(), 2);
    assert_eq!(p["messages"][1]["content"], "Capital of France?");
    assert_eq!(p["tool_names"], json!(["search"]));
    assert_eq!(p["output"]["content"], "Paris");

    let (status, p) = get(&format!("{}/api/runs/otel-1/playground", s.url)).await;
    assert_eq!(status, 200, "{p}");
    assert_eq!(p["spec"], "ollama:llama3.2");
    assert_eq!(p["spec_allowed"], true);
    assert_eq!(p["messages"][0]["content"], "Hi there");
    assert_eq!(p["temperature"], 0.7);
    assert_eq!(p["max_tokens"], 128);

    let (status, _) = get(&format!("{}/api/runs/root-1/playground", s.url)).await;
    assert_eq!(status, 400, "not an LLM run");
    let (status, _) = get(&format!("{}/api/runs/missing/playground", s.url)).await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn playground_requires_the_token() {
    let s = spawn(
        PlaygroundConfig::default().with_model("test:a", scripted(&["ok"])),
        Some("s3cret"),
    )
    .await;
    let c = reqwest::Client::new();
    let body = json!({"messages": [{"role": "user", "content": "hi"}], "models": ["test:a"]});
    for (method, path) in [
        ("POST", "/api/playground/run"),
        ("GET", "/api/playground/models"),
        ("GET", "/api/runs/x/playground"),
    ] {
        let req = if method == "POST" {
            c.post(format!("{}{path}", s.url)).json(&body)
        } else {
            c.get(format!("{}{path}", s.url))
        };
        assert_eq!(req.send().await.unwrap().status(), 401, "{path}");
    }
    let r = c
        .post(format!("{}/api/playground/run", s.url))
        .bearer_auth("s3cret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let r = c
        .get(format!("{}/api/playground/models", s.url))
        .bearer_auth("s3cret")
        .send()
        .await
        .unwrap();
    let m: Value = r.json().await.unwrap();
    assert_eq!(m["models"][0]["spec"], "test:a");
    assert_eq!(m["models"][0]["source"], "registered");
}

#[tokio::test]
async fn playground_page_is_served() {
    let s = spawn(PlaygroundConfig::default(), None).await;
    let html = reqwest::get(format!("{}/", s.url))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(html.contains("#/playground"));
    let js = reqwest::get(format!("{}/pages.js", s.url))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(js.contains("async function playgroundPage"));
    assert!(js.contains("/api/playground/run"));
}
