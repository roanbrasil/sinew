//! Error paths and less common endpoints of the sinew-serve HTTP API:
//! 404s, 409s, validation, pagination, thread-scoped run routes and
//! builder options. Offline, scripted graphs on 127.0.0.1:0.

use std::{sync::Arc, time::Duration};

use reqwest::{Client, Method, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_graph::{StateGraph, END};
use sinew_serve::{
    Assistant, CorsConfig, MemoryStore, RunCreate, RunStatus, RunningServer, ServeClient,
    ServeConfig, Server, ThreadStatus,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct Counter {
    n: u32,
}

/// One node that sleeps `ms`, then increments `n`.
fn counter(id: &str, ms: u64) -> Assistant {
    let mut g = StateGraph::<Counter>::new();
    g.add_node("work", move |mut s: Counter| async move {
        tokio::time::sleep(Duration::from_millis(ms)).await;
        s.n += 1;
        Ok(s)
    });
    g.add_edge("work", END);
    g.set_entry_point("work");
    Assistant::graph(id, g.compile().unwrap())
        .description(format!("{id} counter"))
        .into()
}

struct Http {
    base: String,
    c: Client,
}

impl Http {
    async fn call(&self, m: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
        let mut r = self.c.request(m, format!("{}{path}", self.base));
        if let Some(b) = body {
            r = r.json(&b);
        }
        let resp = r.send().await.unwrap();
        let st = resp.status();
        let text = resp.text().await.unwrap();
        (
            st,
            serde_json::from_str(&text).unwrap_or(Value::String(text)),
        )
    }
    async fn get(&self, path: &str) -> (StatusCode, Value) {
        self.call(Method::GET, path, None).await
    }
    async fn post(&self, path: &str, body: Value) -> (StatusCode, Value) {
        self.call(Method::POST, path, Some(body)).await
    }
}

async fn spawn(b: sinew_serve::ServerBuilder) -> (RunningServer, ServeClient, Http) {
    let s = b.build().spawn("127.0.0.1:0").await.unwrap();
    let c = ServeClient::new(s.base_url());
    let h = Http {
        base: s.base_url(),
        c: Client::new(),
    };
    (s, c, h)
}

#[tokio::test]
async fn info_assistant_search_and_thread_pagination() {
    let server = Server::builder()
        .assistant(counter("a", 0))
        .assistant(counter("b", 0))
        .assistant(counter("c", 0))
        .shared_store(Arc::new(MemoryStore::new()))
        .cors(CorsConfig::permissive())
        .config(ServeConfig::default())
        .build();
    let ids: Vec<String> = server
        .assistants()
        .into_iter()
        .map(|a| a.assistant_id)
        .collect();
    assert_eq!(ids, ["a", "b", "c"]);
    let s = server.spawn("127.0.0.1:0").await.unwrap();
    let h = Http {
        base: s.base_url(),
        c: Client::new(),
    };

    let (st, info) = h.get("/info").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(info["name"], "sinew-serve");
    assert_eq!(info["assistants"], json!(["a", "b", "c"]));

    let (_, page) = h
        .post("/assistants/search", json!({"limit": 1, "offset": 1}))
        .await;
    assert_eq!(page.as_array().unwrap().len(), 1);
    assert_eq!(page[0]["assistant_id"], "b");
    let (st, all) = h.call(Method::POST, "/assistants/search", None).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(all.as_array().unwrap().len(), 3);
    let (st, _) = h
        .call(Method::POST, "/assistants/search", Some(json!("x")))
        .await;
    assert_eq!(st, StatusCode::BAD_REQUEST);

    // Threads: create, `do_nothing`, unknown assistant, list with paging.
    for i in 0..3 {
        let (st, _) = h
            .post(
                "/threads",
                json!({"thread_id": format!("t{i}"), "assistant_id": "a"}),
            )
            .await;
        assert_eq!(st, StatusCode::OK);
    }
    let (st, t) = h
        .post(
            "/threads",
            json!({"thread_id": "t0", "if_exists": "do_nothing", "metadata": {"x": 1}}),
        )
        .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(t["assistant_id"], "a", "the existing thread is returned");
    let (st, e) = h.post("/threads", json!({"assistant_id": "ghost"})).await;
    assert_eq!(st, StatusCode::NOT_FOUND, "{e}");
    let (st, list) = h.get("/threads?limit=2").await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(list.as_array().unwrap().len(), 2);
    let (_, rest) = h.get("/threads?limit=2&offset=2").await;
    assert_eq!(rest.as_array().unwrap().len(), 1);
    let (st, _) = h.get("/threads?limit=many").await;
    assert_eq!(st, StatusCode::BAD_REQUEST);

    // PATCH merges metadata; unknown threads are 404.
    let (st, t) = h
        .call(
            Method::PATCH,
            "/threads/t1",
            Some(json!({"metadata": {"user": "ana"}})),
        )
        .await;
    assert_eq!(st, StatusCode::OK, "{t}");
    assert_eq!(t["metadata"]["user"], "ana");
    let (_, t) = h
        .call(
            Method::PATCH,
            "/threads/t1",
            Some(json!({"metadata": {"plan": "pro"}})),
        )
        .await;
    assert_eq!(t["metadata"], json!({"user": "ana", "plan": "pro"}));
    let (st, _) = h
        .call(
            Method::PATCH,
            "/threads/ghost",
            Some(json!({"metadata": {}})),
        )
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    s.shutdown().await.unwrap();
}

#[tokio::test]
async fn run_validation_and_thread_ownership() {
    let (s, c, h) = spawn(
        Server::builder()
            .assistant(counter("a", 0))
            .assistant(counter("b", 0)),
    )
    .await;
    // Unknown assistant.
    let (st, _) = h
        .post("/runs/wait", json!({"assistant_id": "ghost", "input": {}}))
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    // `if_not_exists: reject` on a missing thread.
    let (st, e) = h
        .post(
            "/threads/nope/runs/wait",
            json!({"assistant_id": "a", "input": {}, "if_not_exists": "reject"}),
        )
        .await;
    assert_eq!(st, StatusCode::NOT_FOUND, "{e}");
    // Webhooks are refused unless the policy allows them.
    let (st, e) = h
        .post(
            "/runs/wait",
            json!({"assistant_id": "a", "input": {}, "webhook": "http://127.0.0.1:9/x"}),
        )
        .await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "{e}");
    // A thread is bound to the assistant of its first run.
    let run = c
        .wait(Some("t"), &RunCreate::new("a", json!({"n": 1})))
        .await
        .unwrap();
    assert_eq!(run.status, RunStatus::Success);
    let (st, e) = h
        .post(
            "/threads/t/runs/wait",
            json!({"assistant_id": "b", "input": {}}),
        )
        .await;
    assert_eq!(st, StatusCode::CONFLICT);
    assert!(e["detail"]
        .as_str()
        .or(e["error"].as_str())
        .unwrap_or_default()
        .contains("belongs to assistant"));
    // Malformed run bodies.
    let (st, _) = h.post("/runs/wait", json!({"input": {}})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);

    // Thread-scoped run routes, and the same run under the wrong thread.
    let rid = run.run_id.clone();
    let (st, r) = h.get(&format!("/threads/t/runs/{rid}")).await;
    assert_eq!(
        (st, r["status"].clone()),
        (StatusCode::OK, json!("success"))
    );
    let (st, r) = h.get(&format!("/threads/t/runs/{rid}/join")).await;
    assert_eq!((st, r["output"]["n"].clone()), (StatusCode::OK, json!(2)));
    let (st, text) = h.get(&format!("/threads/t/runs/{rid}/stream")).await;
    assert_eq!(st, StatusCode::OK);
    assert!(text.as_str().unwrap().contains("event: end"), "{text}");
    let (st, r) = h
        .post(&format!("/threads/t/runs/{rid}/cancel"), json!({}))
        .await;
    assert_eq!(
        (st, r["status"].clone()),
        (StatusCode::OK, json!("success"))
    );
    for path in [
        format!("/threads/other/runs/{rid}"),
        format!("/threads/other/runs/{rid}/join"),
        format!("/threads/other/runs/{rid}/stream"),
        "/runs/ghost".to_string(),
        "/runs/ghost/join".to_string(),
    ] {
        assert_eq!(h.get(&path).await.0, StatusCode::NOT_FOUND, "{path}");
    }
    assert_eq!(
        h.post("/runs/ghost/cancel", json!({})).await.0,
        StatusCode::NOT_FOUND
    );
    s.shutdown().await.unwrap();
}

#[tokio::test]
async fn state_updates_on_fresh_busy_and_unknown_threads() {
    let (s, c, h) = spawn(Server::builder().assistant(counter("slow", 60_000))).await;
    // Seeding a fresh thread bound to an assistant: validated by its graph.
    h.post(
        "/threads",
        json!({"thread_id": "fresh", "assistant_id": "slow"}),
    )
    .await;
    let st = c.update_state("fresh", json!({"n": 41})).await.unwrap();
    assert_eq!(st.values, json!({"n": 41}));
    assert_eq!(st.step, 0);
    assert!(st.next.is_empty());
    let (code, _) = h
        .post("/threads/fresh/state", json!({"values": {"n": "x"}}))
        .await;
    assert_eq!(code, StatusCode::BAD_REQUEST);
    let (code, _) = h.post("/threads/fresh/state", json!({"nope": 1})).await;
    assert_eq!(code, StatusCode::BAD_REQUEST);
    for path in ["/threads/ghost/state", "/threads/ghost/history"] {
        assert_eq!(h.get(path).await.0, StatusCode::NOT_FOUND, "{path}");
    }
    assert_eq!(
        h.post("/threads/ghost/state", json!({"values": {}}))
            .await
            .0,
        StatusCode::NOT_FOUND
    );

    // A thread with a run in flight cannot be edited.
    let run = c
        .create_run(Some("fresh"), &RunCreate::new("slow", json!({})))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(c.thread("fresh").await.unwrap().status, ThreadStatus::Busy);
    let (code, _) = h
        .post("/threads/fresh/state", json!({"values": {"n": 1}}))
        .await;
    assert_eq!(code, StatusCode::CONFLICT);
    // Regression: `busy` is computed from the runs in flight, not stored,
    // so `?status=busy` found nothing and `?status=idle` listed busy
    // threads (shown as busy).
    h.post("/threads", json!({"thread_id": "idle-1"})).await;
    let ids = |v: &Value| -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|t| t["thread_id"].as_str().unwrap().to_string())
            .collect()
    };
    let (_, busy) = h.get("/threads?status=busy").await;
    assert_eq!(ids(&busy), ["fresh"]);
    assert_eq!(busy[0]["status"], "busy");
    let (_, idle) = h.get("/threads?status=idle").await;
    assert_eq!(ids(&idle), ["idle-1"]);
    let (_, idle) = h
        .post("/threads/search", json!({"status": "idle", "offset": 1}))
        .await;
    assert!(idle.as_array().unwrap().is_empty());
    let (code, r) = h
        .post(
            &format!("/threads/fresh/runs/{}/cancel?wait=true", run.run_id),
            json!({}),
        )
        .await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(r["status"], "cancelled");
    s.shutdown().await.unwrap();
}

#[test]
fn runtime_builder() {
    let rt = sinew_serve::runtime(Some(0)).unwrap();
    assert_eq!(rt.block_on(async { 1 + 1 }), 2);
    let rt = sinew_serve::runtime(None).unwrap();
    rt.block_on(async {});
}
