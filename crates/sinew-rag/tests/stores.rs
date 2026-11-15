//! The same behaviour checks against every vector store.
//!
//! In-memory and SQLite run offline. Qdrant and pgvector need a server and
//! are ignored by default:
//!
//! ```bash
//! docker run -d --rm --name sinew-qdrant-test -p 6333:6333 qdrant/qdrant
//! SINEW_QDRANT_URL=http://127.0.0.1:6333 cargo test -p sinew-rag --features qdrant -- --ignored qdrant
//!
//! docker run -d --rm --name sinew-pgvector-test -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg16
//! SINEW_PGVECTOR_URL=postgres://postgres:pw@127.0.0.1:5432/postgres \
//!   cargo test -p sinew-rag --features pgvector -- --ignored pgvector
//! ```

use std::sync::Arc;

use serde_json::json;
use sinew_rag::*;

fn corpus() -> Vec<Document> {
    vec![
        Document::new("rust ownership and the borrow checker")
            .with_id("rust-1")
            .with_metadata("lang", "rust")
            .with_metadata("year", 2015),
        Document::new("rust async await and tokio runtimes")
            .with_id("rust-2")
            .with_metadata("lang", "rust")
            .with_metadata("year", 2019),
        Document::new("python asyncio event loop")
            .with_id("py-1")
            .with_metadata("lang", "python")
            .with_metadata("year", 2014),
        Document::new("baking sourdough bread at home")
            .with_id("bread")
            .with_metadata("lang", "none")
            .with_metadata("tags", json!(["food"])),
    ]
}

async fn exercise(store: Arc<dyn VectorStore>) {
    let ids = store.add_documents(corpus()).await.unwrap();
    assert_eq!(ids, ["rust-1", "rust-2", "py-1", "bread"]);
    assert_eq!(store.len().await.unwrap(), 4);

    // Similarity, best first, scores are cosine similarities.
    let hits = store
        .similarity_search_with_score("rust borrow checker ownership", 2, None)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].0.id.as_deref(), Some("rust-1"));
    assert!(hits[0].1 >= hits[1].1);
    assert!(hits[0].1 <= 1.001 && hits[0].1 > 0.3, "{}", hits[0].1);
    assert_eq!(hits[0].0.metadata["lang"], "rust");

    // Filters.
    let f = Filter::eq("lang", "python");
    let got = store.similarity_search("rust", 4, Some(&f)).await.unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id.as_deref(), Some("py-1"));
    let f = Filter::eq("lang", "rust").and(Filter::gte("year", 2016.0));
    let got = store.similarity_search("rust", 4, Some(&f)).await.unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id.as_deref(), Some("rust-2"));
    let f = Filter::is_in("lang", ["python", "none"]).not();
    assert_eq!(
        store
            .similarity_search("x", 4, Some(&f))
            .await
            .unwrap()
            .len(),
        2
    );
    let f = Filter::exists("year").not();
    let got = store.similarity_search("bread", 4, Some(&f)).await.unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id.as_deref(), Some("bread"));

    // MMR returns distinct documents.
    let mmr = store
        .max_marginal_relevance_search("rust", 2, 4, 0.3, None)
        .await
        .unwrap();
    assert_eq!(mmr.len(), 2);
    assert_ne!(mmr[0].id, mmr[1].id);

    // Upsert by id.
    store
        .add_documents(vec![Document::new("baking rye bread").with_id("bread")])
        .await
        .unwrap();
    assert_eq!(store.len().await.unwrap(), 4);
    let got = store
        .get(&["bread".into(), "missing".into()])
        .await
        .unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].content, "baking rye bread");

    // Delete.
    assert_eq!(
        store
            .delete(&["rust-1".into(), "missing".into()])
            .await
            .unwrap(),
        1
    );
    assert_eq!(store.len().await.unwrap(), 3);
    let hits = store
        .similarity_search("borrow checker", 4, None)
        .await
        .unwrap();
    assert!(hits.iter().all(|d| d.id.as_deref() != Some("rust-1")));

    // Works behind a retriever, a hybrid ensemble and the RAG tool.
    let r = VectorStoreRetriever::new(store.clone()).with_k(1);
    assert_eq!(
        r.retrieve("tokio").await.unwrap()[0].id.as_deref(),
        Some("rust-2")
    );
}

#[tokio::test]
async fn in_memory_store() {
    exercise(Arc::new(InMemoryVectorStore::new(FakeEmbeddings::new(256)))).await;
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_store_file_persists() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}", dir.path().join("rag.db").display());
    let store = SqliteVectorStore::connect(&url, FakeEmbeddings::new(256))
        .await
        .unwrap();
    exercise(Arc::new(store.clone())).await;

    // Collections are isolated.
    let other = store.clone().with_collection("other");
    assert_eq!(other.len().await.unwrap(), 0);
    other.add_texts(vec!["hello".into()]).await.unwrap();
    assert_eq!(other.len().await.unwrap(), 1);

    // Reopen: data survived; dimension mismatches are rejected.
    drop(store);
    let reopened = SqliteVectorStore::connect(&url, FakeEmbeddings::new(256))
        .await
        .unwrap();
    assert_eq!(reopened.len().await.unwrap(), 3);
    assert!(matches!(
        reopened
            .add_vectors(vec![Document::new("x")], vec![vec![1.0; 3]])
            .await,
        Err(Error::Dimensions { .. })
    ));
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_in_memory_store() {
    let store = SqliteVectorStore::in_memory(FakeEmbeddings::new(256))
        .await
        .unwrap();
    exercise(Arc::new(store)).await;
}

#[cfg(feature = "qdrant")]
#[tokio::test]
#[ignore = "needs a Qdrant server (SINEW_QDRANT_URL)"]
async fn qdrant_store() {
    let url = std::env::var("SINEW_QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".into());
    let name = format!("sinew_test_{}", std::process::id());
    let store = QdrantVectorStore::new(&url, &name, FakeEmbeddings::new(256));
    assert_eq!(store.len().await.unwrap(), 0);
    assert!(store
        .similarity_search("x", 3, None)
        .await
        .unwrap()
        .is_empty());
    exercise(Arc::new(store.clone())).await;
    store.delete_collection().await.unwrap();
}

#[cfg(feature = "pgvector")]
#[tokio::test]
#[ignore = "needs Postgres with pgvector (SINEW_PGVECTOR_URL)"]
async fn pgvector_store() {
    let url = std::env::var("SINEW_PGVECTOR_URL")
        .unwrap_or_else(|_| "postgres://postgres:pw@127.0.0.1:5432/postgres".into());
    let name = format!("sinew_test_{}", std::process::id());
    let store = PgVectorStore::connect(&url, FakeEmbeddings::new(256))
        .await
        .unwrap()
        .with_collection(&name);
    store.clear().await.unwrap();
    exercise(Arc::new(store.clone())).await;

    // Typed dimension + HNSW index path.
    let typed = store
        .clone()
        .with_collection(format!("{name}_typed"))
        .with_dimensions(256);
    typed.add_documents(corpus()).await.unwrap();
    typed.create_hnsw_index().await.unwrap();
    let hits = typed
        .similarity_search("sourdough bread", 1, None)
        .await
        .unwrap();
    assert_eq!(hits[0].id.as_deref(), Some("bread"));
    typed.clear().await.unwrap();
    store.clear().await.unwrap();
}
