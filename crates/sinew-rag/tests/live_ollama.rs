//! Live embeddings against a local Ollama server.
//!
//! ```bash
//! ollama pull bge-m3      # or any embedding model; set SINEW_EMBED_MODEL
//! cargo test -p sinew-rag --test live_ollama -- --ignored
//! ```

use sinew_rag::*;

fn model() -> String {
    std::env::var("SINEW_EMBED_MODEL").unwrap_or_else(|_| "bge-m3".into())
}

#[tokio::test]
#[ignore = "needs Ollama with an embedding model (SINEW_EMBED_MODEL, default bge-m3)"]
async fn ollama_embeddings_rank_semantically() {
    // Batch size 2 forces several concurrent requests.
    let e = OpenAiEmbeddings::ollama(model()).with_batch_size(2);
    let texts: Vec<String> = [
        "The cat sat on the mat.",
        "Kubernetes schedules containers onto nodes.",
        "A kitten is resting on a rug.",
        "Postgres is a relational database.",
        "",
    ]
    .map(String::from)
    .to_vec();
    let vs = e.embed_documents(&texts).await.unwrap();
    assert_eq!(vs.len(), texts.len());
    let dim = vs[0].len();
    assert!(dim >= 64);
    assert!(vs.iter().all(|v| v.len() == dim));

    let store = InMemoryVectorStore::new(e);
    store
        .add_vectors(
            texts[..4]
                .iter()
                .map(|t| Document::new(t.as_str()))
                .collect(),
            vs[..4].to_vec(),
        )
        .await
        .unwrap();
    let hits = store
        .similarity_search("a small cat lying on a carpet", 2, None)
        .await
        .unwrap();
    let top: Vec<&str> = hits.iter().map(|d| d.content.as_str()).collect();
    assert!(top.contains(&"The cat sat on the mat."), "{top:?}");
    assert!(top.contains(&"A kitten is resting on a rug."), "{top:?}");
}

#[tokio::test]
#[ignore = "needs Ollama with an embedding model (SINEW_EMBED_MODEL, default bge-m3)"]
async fn spec_and_cache_against_ollama() {
    let e = embeddings_from_spec(&format!("ollama:{}", model())).unwrap();
    let cached = CachedEmbeddings::new(e);
    let a = cached.embed_query("hello world").await.unwrap();
    let b = cached.embed_query("hello world").await.unwrap();
    assert_eq!(a, b);
}
