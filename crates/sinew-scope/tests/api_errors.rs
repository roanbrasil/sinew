//! Error paths and delete/update endpoints of the sinew-scope REST API:
//! auth failures, 404s, validation errors and the engine being disabled.
//! Offline, against an in-process server on 127.0.0.1:0.

use reqwest::{Client, Method, StatusCode};
use serde_json::{json, Value};
use sinew_scope::{router_with, ServerConfig, Store};
use tokio::sync::oneshot;

struct Server {
    url: String,
    c: Client,
    token: Option<String>,
    _stop: oneshot::Sender<()>,
}

async fn spawn(config: ServerConfig) -> Server {
    let token = config.token.clone();
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
    Server {
        url,
        c: Client::new(),
        token,
        _stop: tx,
    }
}

impl Server {
    async fn call(&self, method: Method, path: &str, body: Option<&str>) -> (StatusCode, Value) {
        let mut r = self.c.request(method, format!("{}{path}", self.url));
        if let Some(t) = &self.token {
            r = r.bearer_auth(t);
        }
        if let Some(b) = body {
            r = r
                .header("content-type", "application/json")
                .body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let status = resp.status();
        let text = resp.text().await.unwrap();
        (
            status,
            serde_json::from_str(&text).unwrap_or(Value::String(text)),
        )
    }
    async fn get(&self, path: &str) -> (StatusCode, Value) {
        self.call(Method::GET, path, None).await
    }
    async fn post(&self, path: &str, body: Value) -> (StatusCode, Value) {
        self.call(Method::POST, path, Some(&body.to_string())).await
    }
    async fn put(&self, path: &str, body: Value) -> (StatusCode, Value) {
        self.call(Method::PUT, path, Some(&body.to_string())).await
    }
    async fn delete(&self, path: &str) -> (StatusCode, Value) {
        self.call(Method::DELETE, path, None).await
    }
}

fn run(id: &str, trace: &str, parent: Option<&str>) -> Value {
    json!({
        "id": id, "trace_id": trace, "parent_id": parent, "name": "agent", "kind": "graph",
        "start_ms": 1_000, "end_ms": 1_500, "inputs": {"q": "hi"}, "outputs": "hello",
        "error": null, "input_tokens": null, "output_tokens": null
    })
}

fn no_engine() -> ServerConfig {
    ServerConfig {
        disable_engine: true,
        ..Default::default()
    }
}

#[tokio::test]
async fn auth_is_required_on_api_but_not_on_health() {
    let s = spawn(ServerConfig {
        token: Some("s3cret".into()),
        metrics_auth: true,
        ..no_engine()
    })
    .await;
    let anon = Server {
        token: None,
        url: s.url.clone(),
        c: s.c.clone(),
        _stop: oneshot::channel().0,
    };
    let wrong = Server {
        token: Some("nope".into()),
        url: s.url.clone(),
        c: s.c.clone(),
        _stop: oneshot::channel().0,
    };
    for path in ["/api/traces", "/api/prompts", "/metrics"] {
        let (st, body) = anon.get(path).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED, "{path}");
        assert_eq!(body["error"], "missing or invalid bearer token");
        assert_eq!(wrong.get(path).await.0, StatusCode::UNAUTHORIZED, "{path}");
        assert_eq!(s.get(path).await.0, StatusCode::OK, "{path}");
    }
    let resp =
        s.c.get(format!("{}/api/traces", s.url))
            .send()
            .await
            .unwrap();
    assert_eq!(
        resp.headers()["www-authenticate"],
        "Bearer realm=\"sinew-scope\""
    );
    // Lowercase scheme is accepted too.
    let resp =
        s.c.get(format!("{}/api/traces", s.url))
            .header("authorization", "bearer s3cret")
            .send()
            .await
            .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(anon.get("/api/health").await.0, StatusCode::OK);
}

#[tokio::test]
async fn ingest_and_lookup_validation() {
    let s = spawn(no_engine()).await;
    let (st, body) = s.call(Method::POST, "/api/runs", Some("{not json")).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        s.post("/api/runs", json!("a string")).await.0,
        StatusCode::BAD_REQUEST
    );
    let (st, body) = s.post("/api/runs", json!({"runs": "nope"})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "expected `runs` to be an array");
    let (st, body) = s.post("/api/runs", json!([{"id": "x"}])).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().starts_with("run 0:"));
    let (st, body) = s
        .post(
            "/api/runs",
            json!({"project": "p", "runs": [run("r1", "t1", None)]}),
        )
        .await;
    assert_eq!(st, StatusCode::OK, "{body}");

    for path in [
        "/api/runs/missing",
        "/api/traces/missing",
        "/api/traces/missing/cassette",
        "/api/datasets/missing",
        "/api/experiments/missing",
        "/api/compare?a=x&b=y",
        "/api/pairwise/missing",
        "/api/prompts/missing",
        "/api/prompts/missing@v1",
        "/api/prompts/missing/latest",
        "/api/prompts/missing/diff?from=v1",
        "/api/queues/missing",
        "/api/queues/missing/next",
        "/api/rules/missing",
        "/api/alerts/missing",
    ] {
        let (st, body) = s.get(path).await;
        assert_eq!(st, StatusCode::NOT_FOUND, "{path}: {body}");
        assert!(body["error"].is_string(), "{path}: {body}");
    }
    // Bad query strings and bodies.
    assert_eq!(
        s.get("/api/runs?limit=lots").await.0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        s.get("/api/metrics?from_ms=yesterday").await.0,
        StatusCode::BAD_REQUEST
    );
    for (path, body) in [
        ("/api/datasets", json!({"examples": 3})),
        ("/api/experiments", json!({"nope": true})),
        ("/api/pairwise", json!([])),
        ("/api/prompts", json!({"template": 1})),
        ("/api/feedback", json!([])),
        ("/api/queues", json!([])),
        ("/api/rules", json!({"name": 3})),
        ("/api/alerts", json!({"threshold": "high"})),
        ("/api/datasets/d/examples", json!({"inputs": 1, "id": []})),
    ] {
        let (st, b) = s.post(path, body).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "{path}: {b}");
    }
    assert_eq!(
        s.put("/api/pricing", json!({"models": "x"})).await.0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        s.put("/api/settings", json!({"retention_days": "x"}))
            .await
            .0,
        StatusCode::BAD_REQUEST
    );
    // Copying a missing run into a dataset.
    assert_eq!(
        s.post("/api/datasets/d/examples", json!({"run_id": "ghost"}))
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    // Copying a run keeps an explicit reference.
    let (st, b) = s
        .post(
            "/api/datasets/d/examples",
            json!({"run_id": "r1", "reference": "gold"}),
        )
        .await;
    assert_eq!(st, StatusCode::OK, "{b}");
    let (_, ds) = s.get("/api/datasets/d").await;
    let ex = &ds["dataset"]["examples"][0];
    assert_eq!(ex["reference"], "gold", "{ds}");
    assert_eq!(ex["inputs"], json!({"q": "hi"}));
    assert_eq!(ex["metadata"]["source_run_id"], "r1");

    // Pricing recompute and run feedback listing.
    let (st, b) = s.post("/api/pricing/recompute", json!({})).await;
    assert_eq!((st, b["ok"].clone()), (StatusCode::OK, json!(true)));
    let (st, b) = s.get("/api/runs/r1/feedback").await;
    assert_eq!((st, b), (StatusCode::OK, json!([])));
}

#[tokio::test]
async fn delete_endpoints_return_404_the_second_time() {
    let s = spawn(no_engine()).await;
    s.post(
        "/api/runs",
        json!([run("r1", "t1", None), run("r2", "t2", None)]),
    )
    .await;

    // Feedback.
    let (st, fb) = s
        .post(
            "/api/feedback",
            json!({"run_id": "r1", "key": "ok", "score": 1}),
        )
        .await;
    assert_eq!(st, StatusCode::OK, "{fb}");
    let fid = fb["id"].as_str().unwrap();
    assert_eq!(s.get("/api/runs/r1/feedback").await.1[0]["key"], "ok");
    assert_eq!(
        s.delete(&format!("/api/feedback/{fid}")).await.1,
        json!({"deleted": true})
    );
    assert_eq!(
        s.delete(&format!("/api/feedback/{fid}")).await.0,
        StatusCode::NOT_FOUND
    );

    // Prompts and tags.
    let (st, p) = s
        .post(
            "/api/prompts",
            json!({"name": "greet", "template": "Hi {name}", "tags": ["prod"]}),
        )
        .await;
    assert_eq!(st, StatusCode::OK, "{p}");
    let commit = p["commit"].as_str().unwrap();
    let (st, b) = s.put("/api/prompts/greet/tags/staging", json!({})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    assert!(b["error"].as_str().unwrap().contains("commit"));
    assert_eq!(
        s.put("/api/prompts/greet/tags/staging", json!({"ref": commit}))
            .await
            .0,
        StatusCode::OK
    );
    assert_eq!(
        s.put("/api/prompts/greet/tags/x", json!({"commit": "deadbeef"}))
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        s.get("/api/prompts/greet@staging").await.1["commit"],
        commit
    );
    assert_eq!(
        s.delete("/api/prompts/greet/tags/staging").await.0,
        StatusCode::OK
    );
    assert_eq!(
        s.delete("/api/prompts/greet/tags/staging").await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(s.delete("/api/prompts/greet").await.0, StatusCode::OK);
    assert_eq!(
        s.delete("/api/prompts/greet").await.0,
        StatusCode::NOT_FOUND
    );

    // Queues and items.
    let (_, q) = s
        .post("/api/queues", json!({"name": "review", "keys": ["ok"]}))
        .await;
    let qid = q["id"].as_str().unwrap().to_string();
    let (st, b) = s.post(&format!("/api/queues/{qid}/items"), json!({})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "{b}");
    let (_, b) = s
        .post(&format!("/api/queues/{qid}/items"), json!({"run_id": "r1"}))
        .await;
    assert_eq!(b["added"], 1);
    s.post(&format!("/api/queues/{qid}/items"), json!({"runs": ["r2"]}))
        .await;
    // Completing with an empty body, then an unknown item.
    let (st, _) = s
        .call(
            Method::POST,
            &format!("/api/queues/{qid}/items/r1/done"),
            Some(""),
        )
        .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(
        s.post(&format!("/api/queues/{qid}/items/ghost/done"), json!({}))
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        s.delete(&format!("/api/queues/{qid}/items/r2")).await.0,
        StatusCode::OK
    );
    assert_eq!(
        s.delete(&format!("/api/queues/{qid}/items/r2")).await.0,
        StatusCode::NOT_FOUND
    );
    // Nothing pending: `item` is null.
    let (_, next) = s.get(&format!("/api/queues/{qid}/next")).await;
    assert_eq!(next["item"], Value::Null);
    assert_eq!(next["queue"]["name"], "review");
    assert_eq!(
        s.delete(&format!("/api/queues/{qid}")).await.0,
        StatusCode::OK
    );
    assert_eq!(
        s.delete(&format!("/api/queues/{qid}")).await.0,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn rules_and_alerts_crud_and_engine_off() {
    let s = spawn(no_engine()).await;
    let (st, rule) = s
        .post(
            "/api/rules",
            json!({"name": "r", "evaluators": [{"type": "contains", "needle": "x", "key": "k"}]}),
        )
        .await;
    assert_eq!(st, StatusCode::OK, "{rule}");
    let rid = rule["id"].as_str().unwrap().to_string();
    assert_eq!(s.get(&format!("/api/rules/{rid}")).await.1["name"], "r");
    let (st, b) = s
        .put(
            &format!("/api/rules/{rid}"),
            json!({"name": "renamed", "evaluators": [{"type": "contains", "needle": "y", "key": "k"}]}),
        )
        .await;
    assert_eq!((st, b["name"].clone()), (StatusCode::OK, json!("renamed")));
    assert_eq!(
        s.put("/api/rules/ghost", json!({"name": "x"})).await.0,
        StatusCode::NOT_FOUND
    );
    // Running a rule needs the background engine.
    let (st, b) = s.post(&format!("/api/rules/{rid}/run"), json!({})).await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE, "{b}");
    assert_eq!(
        s.post("/api/rules/ghost/run", json!({})).await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        s.delete(&format!("/api/rules/{rid}")).await.0,
        StatusCode::OK
    );
    assert_eq!(
        s.delete(&format!("/api/rules/{rid}")).await.0,
        StatusCode::NOT_FOUND
    );

    let alert = json!({
        "name": "errors", "metric": "error_rate", "threshold": 0.5,
        "window_ms": 60_000, "webhook_url": "http://127.0.0.1:9/hook"
    });
    let (st, a) = s.post("/api/alerts", alert.clone()).await;
    assert_eq!(st, StatusCode::OK, "{a}");
    let aid = a["id"].as_str().unwrap().to_string();
    assert_eq!(s.get("/api/alerts").await.1.as_array().unwrap().len(), 1);
    let mut changed = alert.clone();
    changed["threshold"] = json!(0.9);
    let (st, b) = s.put(&format!("/api/alerts/{aid}"), changed).await;
    assert_eq!((st, b["threshold"].clone()), (StatusCode::OK, json!(0.9)));
    assert_eq!(
        s.put("/api/alerts/ghost", alert).await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        s.post(&format!("/api/alerts/{aid}/check"), json!({}))
            .await
            .0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        s.post("/api/alerts/ghost/check", json!({})).await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        s.delete(&format!("/api/alerts/{aid}")).await.0,
        StatusCode::OK
    );
    assert_eq!(
        s.delete(&format!("/api/alerts/{aid}")).await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(s.get("/api/alerts").await.1, json!([]));
}
