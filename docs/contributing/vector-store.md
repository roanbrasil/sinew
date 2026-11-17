# Adding a vector store

Vector stores live in `crates/sinew-rag/src/vectorstore/`, one module per backend. The in-memory store
(`memory.rs`) is the reference for the trait semantics; `qdrant.rs` (REST) and `pgvector.rs` (SQL) are the
references for remote backends.

## The trait

From `crates/sinew-rag/src/vectorstore/mod.rs`. Implement the six required methods; text search, batch indexing
and MMR are provided:

```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// The model used to embed documents and queries.
    fn embeddings(&self) -> &Arc<dyn Embeddings>;

    /// Inserts or replaces documents with precomputed vectors (one per
    /// document). Documents without an id get a UUID. Returns the ids.
    async fn add_vectors(&self, docs: Vec<Document>, vectors: Vec<Vec<f32>>)
        -> Result<Vec<String>>;

    /// Removes documents by id; returns how many existed.
    async fn delete(&self, ids: &[String]) -> Result<usize>;

    /// Fetches documents by id (missing ids are skipped).
    async fn get(&self, ids: &[String]) -> Result<Vec<Document>>;

    /// Number of stored documents.
    async fn len(&self) -> Result<usize>;

    /// The `k` most similar documents to `query`, best first, optionally
    /// restricted by a metadata filter. `with_vectors` also returns the
    /// stored vectors (needed for MMR).
    async fn search_by_vector(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        with_vectors: bool,
    ) -> Result<Vec<ScoredDocument>>;

    // Provided: is_empty, add_documents, add_texts, similarity_search_with_score,
    // similarity_search, max_marginal_relevance_search.
}
```

Contract every store must keep (the shared test suite checks it):

- **Scores are cosine similarity**, higher is more similar, at most 1. Translate distances (`1 - distance` for
  cosine distance) before returning `ScoredDocument::score`.
- **Upsert by id.** Re-adding a document with the same `Document::id` replaces it. Documents without an id get a
  UUID (`crate::document::ensure_ids` does this).
- **Filters.** Every `Filter` must work. Translate it to a native filter when the backend has one (see
  `Filter::to_qdrant` and the SQL builder in `pgvector.rs`), or fetch candidates and evaluate
  `Filter::matches(&metadata)` in Rust.
- **Lengths.** Call `check_lengths(docs.len(), vectors.len())` in `add_vectors`; return `Error::Dimensions` when a
  vector does not match the stored dimension.
- **`with_vectors`** fills `ScoredDocument::vector` so MMR can work.

## Steps

1. **Module.** Create `crates/sinew-rag/src/vectorstore/acme.rs` with a doc comment that names the feature, the
   distance used, the storage layout (what goes in the payload or columns) and how ids are mapped.

2. **Constructor and options.** Follow the existing stores: `new(url, collection, embeddings)` or
   `connect(url, embeddings).await`, `with_collection(name)`, `with_api_key(...)` where relevant, and a reused
   `reqwest::Client` or connection pool. Accept `impl Embeddings + 'static` and store `Arc<dyn Embeddings>`.

3. **Feature flag.** In `crates/sinew-rag/Cargo.toml`:

   ```toml
   [features]
   # `AcmeVectorStore`: Acme over its REST API.
   acme = []                       # or ["dep:acme-client"] for a new dependency
   ```

   New dependencies go in `[workspace.dependencies]` of the root `Cargo.toml` and are referenced with
   `workspace = true, optional = true`. Then gate the module and exports:

   ```rust
   // crates/sinew-rag/src/vectorstore/mod.rs
   #[cfg(feature = "acme")]
   mod acme;
   #[cfg(feature = "acme")]
   pub use acme::AcmeVectorStore;

   // crates/sinew-rag/src/lib.rs
   #[cfg(feature = "acme")]
   pub use vectorstore::AcmeVectorStore;
   ```

   Forward it from the umbrella crate, `crates/sinew/Cargo.toml`:

   ```toml
   acme = ["rag", "sinew-rag/acme"]
   ```

   and add it to `full` if it needs no system library or running service to compile.

4. **Errors.** Use the existing `sinew_rag::Error` variants: `Provider { provider, status, body }` for HTTP errors,
   `Http`, `InvalidResponse`, `Dimensions`, `Config`, `Invalid`. Only SQL backends use `Sql` (it is gated on the
   `sqlite` / `pgvector` features).

5. **Tests.**
   - **Unit tests** in the module for pure functions: filter translation, id mapping, vector encoding (see
     `point_ids_are_stable_uuids` in `qdrant.rs`, `filter_sql` and `vector_text_roundtrip` in `pgvector.rs`).
     For REST backends, test request bodies and the decoding of **recorded** responses as JSON constants.
   - **Behavior suite.** `crates/sinew-rag/tests/stores.rs` has `exercise(store)`, which checks scores, filters,
     MMR, upserts, `get`, `delete` and the retriever, ensemble and RAG tool on top. Add a test that runs it
     against your store, ignored by default and configured from the environment:

     ```rust
     #[cfg(feature = "acme")]
     #[tokio::test]
     #[ignore = "needs an Acme server (SINEW_ACME_URL)"]
     async fn acme_store() {
         let url = std::env::var("SINEW_ACME_URL").unwrap_or_else(|_| "http://127.0.0.1:7000".into());
         let name = format!("sinew_test_{}", std::process::id());
         let store = AcmeVectorStore::new(&url, &name, FakeEmbeddings::new(256));
         exercise(Arc::new(store.clone())).await;
         // clean up the collection
     }
     ```

     Add the `docker run` and `cargo test` lines to the header comment of `stores.rs`, next to Qdrant and
     pgvector.
   - `FakeEmbeddings::new(dim)` gives deterministic vectors, so the suite needs no embedding model.

6. **Benchmarks** (optional). If the store searches in-process, add a case to
   `crates/sinew-rag/benches/search.rs`.

## Docs to update

- `docs/rag.md`: a row in the vector store table (constructor, feature, best for) and a short paragraph on the
  storage layout, like the Qdrant and pgvector ones; the `cargo test` line in **Testing**.
- The table in the module docs of `crates/sinew-rag/src/vectorstore/mod.rs` and the list in
  `crates/sinew-rag/src/lib.rs`.
- `README.md`: the `sinew-rag` row of the crate table, and the `rag` feature row.
- `crates/sinew/src/lib.rs`: the feature table in the crate docs.

## CHANGELOG

```markdown
- `AcmeVectorStore` (feature `acme` in `sinew-rag` and `sinew`): Acme over its REST API, filters translated to Acme filters (#123).
```
