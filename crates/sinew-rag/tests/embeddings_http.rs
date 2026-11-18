//! Embedding clients against a local mock server: request shape, batching,
//! ordering and error mapping. Offline.

use std::sync::{Arc, Mutex};

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use serde_json::{json, Value};
use sinew_core::context::TextEmbedder;
use sinew_rag::{
    embeddings::ContextEmbedder, Embeddings, Error, GeminiEmbeddings, OpenAiEmbeddings,
};

#[derive(Clone, Default)]
struct Mock {
    /// (path, headers, body) of every request.
    seen: Arc<Mutex<Vec<(String, HeaderMap, Value)>>>,
}

/// Echo-style OpenAI `/embeddings`: the vector of input `i` is
/// `[len(text), i]`, returned in reverse order with `index` set. Special
/// inputs trigger errors.
async fn openai(
    State(m): State<Mock>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    m.seen
        .lock()
        .unwrap()
        .push(("/v1/embeddings".into(), headers, body.clone()));
    let inputs: Vec<String> = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    if inputs.iter().any(|t| t == "fail") {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"message": "bad input"}})),
        );
    }
    if inputs.iter().any(|t| t == "short") {
        return (StatusCode::OK, Json(json!({"data": []})));
    }
    let mut data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .map(|(i, t)| json!({"index": i, "embedding": [t.len() as f32, i as f32]}))
        .collect();
    data.reverse();
    (StatusCode::OK, Json(json!({ "data": data })))
}

async fn gemini(
    State(m): State<Mock>,
    Path(model): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    m.seen
        .lock()
        .unwrap()
        .push((format!("/v1beta/models/{model}"), headers, body.clone()));
    let reqs = body["requests"].as_array().unwrap();
    if reqs
        .iter()
        .any(|r| r["content"]["parts"][0]["text"] == "fail")
    {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "denied"})));
    }
    let n = if reqs
        .iter()
        .any(|r| r["content"]["parts"][0]["text"] == "short")
    {
        0
    } else {
        reqs.len()
    };
    let embeddings: Vec<Value> = reqs[..n]
        .iter()
        .map(|r| {
            let t = r["content"]["parts"][0]["text"].as_str().unwrap();
            json!({"values": [t.len() as f32]})
        })
        .collect();
    (StatusCode::OK, Json(json!({ "embeddings": embeddings })))
}

async fn serve() -> (String, Mock) {
    let mock = Mock::default();
    let app = Router::new()
        .route("/v1/embeddings", post(openai))
        .route("/v1beta/models/{model}", post(gemini))
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, mock)
}

fn texts(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[tokio::test]
async fn openai_batches_in_order_with_auth_and_dimensions() {
    let (url, mock) = serve().await;
    let e = OpenAiEmbeddings::new(format!("{url}/v1/"), "emb")
        .with_api_key("k")
        .with_dimensions(2)
        .with_batch_size(2)
        .with_concurrency(2)
        .with_client(reqwest::Client::new());
    assert_eq!(e.model(), "emb");
    let out = e
        .embed_documents(&texts(&["a", "bb", "", "dddd", "eeeee"]))
        .await
        .unwrap();
    // Order is restored from `index` within each batch and across batches.
    let lens: Vec<f32> = out.iter().map(|v| v[0]).collect();
    // The empty string is sent as " " (most servers reject empty input).
    assert_eq!(lens, vec![1.0, 2.0, 1.0, 4.0, 5.0]);
    assert!(e.embed_documents(&[]).await.unwrap().is_empty());
    assert_eq!(e.embed_query("xyz").await.unwrap()[0], 3.0);

    let seen = mock.seen.lock().unwrap();
    assert_eq!(seen.len(), 4, "3 batches + 1 query");
    assert_eq!(seen[0].1["authorization"], "Bearer k");
    assert_eq!(seen[0].2["model"], "emb");
    assert_eq!(seen[0].2["dimensions"], 2);
    assert!(seen
        .iter()
        .any(|(_, _, b)| b["input"] == json!([" ", "dddd"])));
}

#[tokio::test]
async fn openai_errors_are_mapped() {
    let (url, _) = serve().await;
    let e = OpenAiEmbeddings::new(format!("{url}/v1"), "emb");
    match e.embed_documents(&texts(&["ok", "fail"])).await {
        Err(Error::Provider {
            provider,
            status,
            body,
        }) => {
            assert_eq!(provider, "openai-compatible");
            assert_eq!(status, 400);
            assert!(body.contains("bad input"));
        }
        other => panic!("{other:?}"),
    }
    let err = e.embed_documents(&texts(&["short"])).await.unwrap_err();
    assert!(
        matches!(&err, Error::InvalidResponse(s) if s.contains("0 embeddings for 1 inputs")),
        "{err:?}"
    );
    // A failing batch fails the whole call.
    let batched = OpenAiEmbeddings::new(format!("{url}/v1"), "emb").with_batch_size(1);
    assert!(batched
        .embed_documents(&texts(&["a", "fail", "c"]))
        .await
        .is_err());
    // Through the core `TextEmbedder` adapter, errors become `Other`.
    let ctx = ContextEmbedder(e.clone());
    assert_eq!(ctx.embed(&texts(&["ab"])).await.unwrap()[0][0], 2.0);
    let err = ctx.embed(&texts(&["fail"])).await.unwrap_err();
    assert!(matches!(err, sinew_core::Error::Other(ref s) if s.contains("HTTP 400")));
    // Unreachable server.
    let down = OpenAiEmbeddings::new("http://127.0.0.1:9/v1", "emb");
    assert!(matches!(down.embed_query("x").await, Err(Error::Http(_))));
}

#[tokio::test]
async fn gemini_task_types_auth_batches_and_errors() {
    let (url, mock) = serve().await;
    let e = GeminiEmbeddings::new("gk", "text-embedding-004")
        .with_base_url(format!("{url}/v1beta/"))
        .with_dimensions(8)
        .with_batch_size(2)
        .with_concurrency(1)
        .with_client(reqwest::Client::new());
    let out = e
        .embed_documents(&texts(&["a", "bb", "ccc"]))
        .await
        .unwrap();
    assert_eq!(out, vec![vec![1.0], vec![2.0], vec![3.0]]);
    assert_eq!(e.embed_query("dddd").await.unwrap(), vec![4.0]);
    assert!(e.embed_documents(&[]).await.unwrap().is_empty());

    let bearer = GeminiEmbeddings::new("unused", "models/custom")
        .with_base_url(format!("{url}/v1beta"))
        .with_bearer_token("tok");
    bearer.embed_query("x").await.unwrap();

    {
        let seen = mock.seen.lock().unwrap();
        assert_eq!(
            seen[0].0,
            "/v1beta/models/text-embedding-004:batchEmbedContents"
        );
        assert_eq!(seen[0].1["x-goog-api-key"], "gk");
        let r = &seen[0].2["requests"][0];
        assert_eq!(r["model"], "models/text-embedding-004");
        assert_eq!(r["taskType"], "RETRIEVAL_DOCUMENT");
        assert_eq!(r["outputDimensionality"], 8);
        assert_eq!(seen[2].2["requests"][0]["taskType"], "RETRIEVAL_QUERY");
        let last = seen.last().unwrap();
        // A `models/` prefix is not doubled.
        assert_eq!(last.0, "/v1beta/models/custom:batchEmbedContents");
        assert_eq!(last.1["authorization"], "Bearer tok");
        assert!(last.1.get("x-goog-api-key").is_none());
    }

    let err = e.embed_query("fail").await.unwrap_err();
    assert!(
        matches!(&err, Error::Provider { provider, status: 403, .. } if provider == "gemini"),
        "{err:?}"
    );
    let err = e.embed_documents(&texts(&["short"])).await.unwrap_err();
    assert!(matches!(err, Error::InvalidResponse(_)), "{err:?}");
    let err = e.embed_query("short").await.unwrap_err();
    assert!(matches!(err, Error::InvalidResponse(_)), "{err:?}");
}
