//! `QdrantVectorStore` against a small in-process fake of the Qdrant REST
//! API (unnamed vectors, cosine search, filters recorded but not applied).
//! The real-server suite lives in `stores.rs` behind `--ignored`.
#![cfg(feature = "qdrant")]

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post, put},
    Json, Router,
};
use serde_json::{json, Value};
use sinew_rag::*;

/// A stored point: (vector, payload).
type Point = (Vec<f32>, Value);

#[derive(Default)]
struct Inner {
    /// collection -> point id -> (vector, payload)
    collections: BTreeMap<String, BTreeMap<String, Point>>,
    /// Every request: (method path, body).
    log: Vec<(String, Value)>,
    api_keys: Vec<Option<String>>,
    /// Answer the next collection PUT with 409 (but create it), like a
    /// concurrent writer winning the race.
    race_on_create: bool,
    /// Answer every request with 500.
    broken: bool,
}

type Shared = Arc<Mutex<Inner>>;
type Reply = (StatusCode, Json<Value>);

fn ok(result: Value) -> Reply {
    (
        StatusCode::OK,
        Json(json!({ "result": result, "status": "ok" })),
    )
}

fn not_found() -> Reply {
    (
        StatusCode::NOT_FOUND,
        Json(json!({"status": {"error": "Not found: Collection doesn't exist"}})),
    )
}

fn record(s: &Shared, what: String, headers: &HeaderMap, body: Value) -> Option<Reply> {
    let mut g = s.lock().unwrap();
    g.log.push((what, body));
    g.api_keys.push(
        headers
            .get("api-key")
            .map(|v| v.to_str().unwrap().to_string()),
    );
    g.broken
        .then(|| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!("boom"))))
}

async fn get_collection(State(s): State<Shared>, Path(c): Path<String>, h: HeaderMap) -> Reply {
    if let Some(r) = record(&s, format!("GET {c}"), &h, Value::Null) {
        return r;
    }
    if s.lock().unwrap().collections.contains_key(&c) {
        ok(json!({"status": "green"}))
    } else {
        not_found()
    }
}

async fn put_collection(
    State(s): State<Shared>,
    Path(c): Path<String>,
    h: HeaderMap,
    Json(body): Json<Value>,
) -> Reply {
    if let Some(r) = record(&s, format!("PUT {c}"), &h, body) {
        return r;
    }
    let mut g = s.lock().unwrap();
    g.collections.entry(c).or_default();
    if std::mem::take(&mut g.race_on_create) {
        return (StatusCode::CONFLICT, Json(json!("already exists")));
    }
    ok(json!(true))
}

async fn delete_collection(State(s): State<Shared>, Path(c): Path<String>, h: HeaderMap) -> Reply {
    if let Some(r) = record(&s, format!("DELETE {c}"), &h, Value::Null) {
        return r;
    }
    ok(json!(s.lock().unwrap().collections.remove(&c).is_some()))
}

async fn upsert(
    State(s): State<Shared>,
    Path(c): Path<String>,
    h: HeaderMap,
    Json(body): Json<Value>,
) -> Reply {
    if let Some(r) = record(&s, format!("UPSERT {c}"), &h, body.clone()) {
        return r;
    }
    let mut g = s.lock().unwrap();
    let Some(col) = g.collections.get_mut(&c) else {
        return not_found();
    };
    for p in body["points"].as_array().unwrap() {
        let v: Vec<f32> = serde_json::from_value(p["vector"].clone()).unwrap();
        col.insert(p["id"].as_str().unwrap().into(), (v, p["payload"].clone()));
    }
    ok(json!({"status": "completed"}))
}

async fn retrieve(
    State(s): State<Shared>,
    Path(c): Path<String>,
    h: HeaderMap,
    Json(body): Json<Value>,
) -> Reply {
    if let Some(r) = record(&s, format!("RETRIEVE {c}"), &h, body.clone()) {
        return r;
    }
    let g = s.lock().unwrap();
    let Some(col) = g.collections.get(&c) else {
        return not_found();
    };
    let points: Vec<Value> = body["ids"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|id| col.get(id.as_str().unwrap()).map(|p| (id, p)))
        .map(|(id, (_, payload))| json!({"id": id, "payload": payload}))
        .collect();
    ok(json!(points))
}

async fn delete_points(
    State(s): State<Shared>,
    Path(c): Path<String>,
    h: HeaderMap,
    Json(body): Json<Value>,
) -> Reply {
    if let Some(r) = record(&s, format!("DELETE_POINTS {c}"), &h, body.clone()) {
        return r;
    }
    let mut g = s.lock().unwrap();
    let Some(col) = g.collections.get_mut(&c) else {
        return not_found();
    };
    for id in body["points"].as_array().unwrap() {
        col.remove(id.as_str().unwrap());
    }
    ok(json!({"status": "completed"}))
}

async fn count(
    State(s): State<Shared>,
    Path(c): Path<String>,
    h: HeaderMap,
    Json(body): Json<Value>,
) -> Reply {
    if let Some(r) = record(&s, format!("COUNT {c}"), &h, body) {
        return r;
    }
    match s.lock().unwrap().collections.get(&c) {
        Some(col) => ok(json!({"count": col.len()})),
        None => not_found(),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

async fn search(
    State(s): State<Shared>,
    Path(c): Path<String>,
    h: HeaderMap,
    Json(body): Json<Value>,
) -> Reply {
    if let Some(r) = record(&s, format!("SEARCH {c}"), &h, body.clone()) {
        return r;
    }
    let g = s.lock().unwrap();
    let Some(col) = g.collections.get(&c) else {
        return not_found();
    };
    let q: Vec<f32> = serde_json::from_value(body["vector"].clone()).unwrap();
    let mut hits: Vec<(f32, &String, &Point)> = col
        .iter()
        .map(|(id, p)| (cosine(&q, &p.0), id, p))
        .collect();
    hits.sort_by(|a, b| b.0.total_cmp(&a.0));
    let with_vector = body["with_vector"].as_bool().unwrap_or(false);
    let limit = body["limit"].as_u64().unwrap() as usize;
    let out: Vec<Value> = hits
        .into_iter()
        .take(limit)
        .map(|(score, id, (v, payload))| {
            let mut p = json!({"id": id, "score": score, "payload": payload});
            if with_vector {
                p["vector"] = json!(v);
            }
            p
        })
        .collect();
    ok(json!(out))
}

async fn fake_qdrant() -> (String, Shared) {
    let state: Shared = Arc::default();
    let app = Router::new()
        .route(
            "/collections/{c}",
            get(get_collection)
                .put(put_collection)
                .delete(delete_collection),
        )
        .route("/collections/{c}/points", put(upsert).post(retrieve))
        .route("/collections/{c}/points/delete", post(delete_points))
        .route("/collections/{c}/points/count", post(count))
        .route("/collections/{c}/points/search", post(search))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, state)
}

fn ops(s: &Shared) -> Vec<String> {
    s.lock()
        .unwrap()
        .log
        .iter()
        .map(|(w, _)| w.clone())
        .collect()
}

#[tokio::test]
async fn full_lifecycle_against_the_fake() {
    let (url, state) = fake_qdrant().await;
    let store = QdrantVectorStore::new(format!("{url}/"), "docs", FakeEmbeddings::new(64))
        .with_api_key("qk")
        .with_batch_size(2)
        .with_client(reqwest::Client::new());

    // Before the collection exists: empty, not an error.
    assert_eq!(store.len().await.unwrap(), 0);
    assert!(store.is_empty().await.unwrap());
    assert!(store.get(&["a".into()]).await.unwrap().is_empty());
    assert!(store
        .similarity_search("x", 3, None)
        .await
        .unwrap()
        .is_empty());

    let ids = store
        .add_documents(vec![
            Document::new("rust ownership and borrowing")
                .with_id("rust-1")
                .with_metadata("lang", "rust"),
            Document::new("python asyncio loop")
                .with_id("py-1")
                .with_metadata("lang", "python"),
            Document::new("baking sourdough bread").with_id("bread"),
            // A UUID id is used as the point id unchanged.
            Document::new("uuid doc").with_id("6f1c2a8e-3b7d-4c55-9a0e-1d2b3c4d5e6f"),
            // No id: one is generated.
            Document::new("anonymous"),
        ])
        .await
        .unwrap();
    assert_eq!(ids.len(), 5);
    assert_eq!(store.len().await.unwrap(), 5);
    {
        let g = state.lock().unwrap();
        let col = &g.collections["docs"];
        assert!(col.contains_key("6f1c2a8e-3b7d-4c55-9a0e-1d2b3c4d5e6f"));
        assert!(!col.contains_key("rust-1"), "non-UUID ids are hashed");
        let create = g.log.iter().find(|(w, _)| w == "PUT docs").unwrap();
        assert_eq!(
            create.1["vectors"],
            json!({"size": 64, "distance": "Cosine"})
        );
        assert!(g.api_keys.iter().all(|k| k.as_deref() == Some("qk")));
    }
    // 5 points in batches of 2.
    assert_eq!(
        ops(&state).iter().filter(|o| *o == "UPSERT docs").count(),
        3
    );

    // Search: best hit first, payload decoded, filter forwarded.
    let hits = store
        .similarity_search_with_score("rust ownership and borrowing", 2, None)
        .await
        .unwrap();
    assert_eq!(hits[0].0.id.as_deref(), Some("rust-1"));
    assert_eq!(hits[0].0.metadata["lang"], "rust");
    assert!(hits[0].1 > 0.99);
    let f = Filter::eq("lang", "python");
    store.similarity_search("x", 1, Some(&f)).await.unwrap();
    {
        let g = state.lock().unwrap();
        let (_, body) = g
            .log
            .iter()
            .rev()
            .find(|(w, _)| w == "SEARCH docs")
            .unwrap();
        assert_eq!(
            body["filter"],
            json!({"must": [{"key": "metadata.lang", "match": {"value": "python"}}]})
        );
        assert_eq!(body["limit"], 1);
    }
    // k = 0 never calls the server.
    let before = ops(&state).len();
    assert!(store
        .search_by_vector(&[0.0; 64], 0, None, false)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(ops(&state).len(), before);

    // MMR needs the stored vectors.
    let mmr = store
        .max_marginal_relevance_search("rust", 2, 4, 0.5, None)
        .await
        .unwrap();
    assert_eq!(mmr.len(), 2);
    let with_vectors = store
        .search_by_vector(&[1.0; 64], 1, None, true)
        .await
        .unwrap();
    assert_eq!(with_vectors[0].vector.as_ref().map(Vec::len), Some(64));

    // Upsert replaces by id; get keeps the requested order and skips misses.
    store
        .add_documents(vec![Document::new("rye bread").with_id("bread")])
        .await
        .unwrap();
    assert_eq!(store.len().await.unwrap(), 5);
    let got = store
        .get(&["bread".into(), "missing".into(), "py-1".into()])
        .await
        .unwrap();
    let got: Vec<_> = got.iter().map(|d| d.content.as_str()).collect();
    assert_eq!(got, vec!["rye bread", "python asyncio loop"]);

    // Delete counts only ids that existed.
    assert_eq!(
        store
            .delete(&["rust-1".into(), "missing".into()])
            .await
            .unwrap(),
        1
    );
    assert_eq!(store.delete(&[]).await.unwrap(), 0);
    assert_eq!(store.len().await.unwrap(), 4);

    // The collection is created once, then remembered.
    assert_eq!(ops(&state).iter().filter(|o| *o == "PUT docs").count(), 1);

    store.delete_collection().await.unwrap();
    assert_eq!(store.len().await.unwrap(), 0);
    // After dropping it, the next insert creates it again.
    store.add_texts(vec!["again".into()]).await.unwrap();
    assert_eq!(store.len().await.unwrap(), 1);
    assert_eq!(ops(&state).iter().filter(|o| *o == "PUT docs").count(), 2);
}

#[tokio::test]
async fn concurrent_creation_and_server_errors() {
    let (url, state) = fake_qdrant().await;
    state.lock().unwrap().race_on_create = true;
    let store = QdrantVectorStore::new(&url, "raced", FakeEmbeddings::new(8));
    // The PUT fails with 409, but the collection now exists: not an error.
    store.ensure_collection(8).await.unwrap();
    store.add_texts(vec!["x".into()]).await.unwrap();
    assert_eq!(store.len().await.unwrap(), 1);

    // Empty inputs never reach the server.
    assert!(store.add_vectors(vec![], vec![]).await.unwrap().is_empty());
    assert!(store.get(&[]).await.unwrap().is_empty());
    assert!(matches!(
        store.add_vectors(vec![Document::new("a")], vec![]).await,
        Err(Error::Invalid(_))
    ));

    // 500s propagate as provider errors (404s are only "empty" for reads).
    state.lock().unwrap().broken = true;
    let fresh = QdrantVectorStore::new(&url, "other", FakeEmbeddings::new(8));
    for res in [
        fresh.len().await.map(|_| ()),
        fresh.get(&["a".into()]).await.map(|_| ()),
        fresh
            .search_by_vector(&[1.0; 8], 1, None, false)
            .await
            .map(|_| ()),
        fresh.delete_collection().await,
        fresh.add_texts(vec!["a".into()]).await.map(|_| ()),
    ] {
        match res {
            Err(Error::Provider {
                provider, status, ..
            }) => {
                assert_eq!(provider, "qdrant");
                assert_eq!(status, 500);
            }
            other => panic!("{other:?}"),
        }
    }
    // Unreachable server.
    let down = QdrantVectorStore::new("http://127.0.0.1:9", "c", FakeEmbeddings::new(8));
    assert!(matches!(down.len().await, Err(Error::Http(_))));
}
