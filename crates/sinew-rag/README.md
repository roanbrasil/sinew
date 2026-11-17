# sinew-rag

Retrieval for [Sinew](https://github.com/roanbrasil/sinew): the LangChain retrieval stack in Rust.
Embeddings, document loaders, text splitters, vector stores, retrievers, a retriever tool for agents
and a RAG chain that answers with sources.

```toml
[dependencies]
sinew-rag = "0.1"                                                   # in-memory + SQLite
sinew-rag = { version = "0.1", features = ["qdrant", "pgvector", "pdf"] }
```

| Feature | Default | Adds |
|---|---|---|
| `sqlite` | yes | `SqliteVectorStore` (sqlx) |
| `qdrant` | no | `QdrantVectorStore` (REST, no extra dependency) |
| `pgvector` | no | `PgVectorStore` (sqlx Postgres) |
| `pdf` | no | `PdfLoader` (pure-Rust `pdf-extract`) |

## Quick start

```rust
use std::sync::Arc;
use sinew_rag::*;

// Load and split.
let docs = DirectoryLoader::new("docs").glob("**/*.md").load()?;
let chunks = RecursiveCharacterTextSplitter::new(1000, 150).split_documents(&docs);

// Embed and index (any OpenAI-compatible server, Gemini, or FakeEmbeddings offline).
let embeddings = CachedEmbeddings::new(OpenAiEmbeddings::ollama("bge-m3"));
let store = SqliteVectorStore::connect("sqlite://rag.db", embeddings).await?;
store.add_documents(chunks.clone()).await?;
let store: Arc<dyn VectorStore> = Arc::new(store);

// Hybrid retrieval: BM25 + vectors with reciprocal rank fusion.
let retriever: Arc<dyn Retriever> = Arc::new(
    EnsembleRetriever::new(vec![
        Arc::new(Bm25Retriever::from_documents_with_tokenizer(chunks, english_tokenize)),
        Arc::new(VectorStoreRetriever::new(store).with_k(6)),
    ])
    .with_k(4),
);

// Answer with sources.
let model = sinew_core::providers::from_spec("ollama:qwen3:4b")?;
let out = RagChain::new(model, retriever.clone()).invoke("How do I resume a thread?").await?;
println!("{}\n{:?}", out.answer, out.source_names());

// Or give the retriever to an agent.
let tool = retriever_tool(retriever, "search_docs", "Searches the project documentation");
```

## What is inside

**Embeddings** (`Embeddings` trait: `embed_documents` in batches, `embed_query`)

- `OpenAiEmbeddings`: OpenAI-compatible `/embeddings`. Constructors and `from_spec` use the same
  names, base URLs and key variables as the chat presets in `sinew-core`: `openai`, `mistral`,
  `together`, `fireworks`, `nvidia`, `openrouter`, `huggingface`, and local `ollama`, `lmstudio`,
  `llamacpp`, `vllm`, `sglang`, `localai`, `jan`, `llamafile`, `tgi`/`tei`. Requests are batched
  (`with_batch_size`, default 256) and sent concurrently (`with_concurrency`, default 4) over one
  pooled `reqwest::Client`.
- `GeminiEmbeddings`: native `batchEmbedContents` (100 texts per request, `RETRIEVAL_DOCUMENT` /
  `RETRIEVAL_QUERY` task types, `outputDimensionality`), AI Studio key or Vertex bearer token.
- `embeddings_from_spec("ollama:bge-m3")` builds any of them.
- `FakeEmbeddings`: deterministic feature hashing, similarity-aware, for tests and offline demos.
- `CachedEmbeddings`: only cache misses reach the model, de-duplicated, in one batch; pluggable
  `EmbeddingCache` (bounded `InMemoryEmbeddingCache` included).

**Documents and loaders**: `Document { id, content, metadata }`; `TextLoader`, `MarkdownLoader`
(front matter to metadata), `HtmlLoader` / `html_to_text` (no DOM dependency), `JsonLoader`
(JSON Pointer for records and content, JSONL), `CsvLoader`, `DirectoryLoader` (glob include and
exclude, loader per extension, custom loaders), `PdfLoader` (feature `pdf`, one document per page).

**Splitters**: `RecursiveCharacterTextSplitter` with LangChain semantics (checked against
LangChain's reference output), `KeepSeparator`, `start_index`, code presets (`Language::Rust`,
`Python`, `JavaScript`, `TypeScript`, `Go`, `Java`, `Cpp`, `Markdown`, `Html`) and a length hook
(`approx_token_len` or your tokenizer); `MarkdownHeaderTextSplitter`.

**Vector stores** (`VectorStore` trait: upsert by id, delete, get, search with scores, metadata
`Filter`s, MMR). All scores are cosine similarity.

| Store | Search | Filters |
|---|---|---|
| `InMemoryVectorStore` | contiguous `f32` matrix, normalised rows, unrolled dot product, heap top-k, scoped-thread scan above a size threshold | evaluated in Rust |
| `SqliteVectorStore` | streams `(id, blob)` rows, dot product on the blob bytes, fetches only the winners | evaluated in Rust |
| `QdrantVectorStore` | Qdrant HNSW | translated to Qdrant filters |
| `PgVectorStore` | `<=>` cosine; exact, or HNSW with `with_dimensions` + `create_hnsw_index` | translated to SQL on `jsonb` |

`Filter` has builders (`Filter::eq("lang", "en").and(Filter::gte("year", 2024.0))`) and parses
LangChain / Chroma style JSON (`{"lang": "en", "year": {"$gte": 2024}}`).

**Retrievers** (`Retriever` trait): `VectorStoreRetriever` (similarity, score threshold, MMR),
`Bm25Retriever`, `EnsembleRetriever` (weighted reciprocal rank fusion), `MultiQueryRetriever`
(`QueryExpander` hook, `LlmQueryExpander`), `ContextualCompressionRetriever` with the
`DocumentCompressor` hook (`LlmReranker` using any `ChatModel`, `EmbeddingsFilter`),
`ParentDocumentRetriever` (`DocStore` hook, `InMemoryDocStore`).

**Agents and chains**: `retriever_tool` turns any retriever into a `sinew_core::Tool`;
`RagChain` runs question → retrieve → prompt with numbered passages → answer, returning
`RagAnswer { answer, sources, usage }` with `source_names()` and `cited()`.

## Performance

`cargo bench -p sinew-rag --bench search` measures in-memory top-10 search over 384-dimension
vectors (1k, 10k, 100k rows; one thread, eight threads, filtered). The scan allocates nothing per
row; the only per-query allocations are the normalised query copy and the k-sized heap.

## Tests

```bash
cargo test -p sinew-rag --all-features             # offline: fake embeddings, temp SQLite
cargo test -p sinew-rag --test live_ollama -- --ignored                 # Ollama embeddings
SINEW_QDRANT_URL=http://127.0.0.1:6333 cargo test -p sinew-rag --features qdrant -- --ignored qdrant
SINEW_PGVECTOR_URL=postgres://postgres:pw@127.0.0.1:5432/postgres \
  cargo test -p sinew-rag --features pgvector -- --ignored pgvector
```

More in [docs/rag.md](https://github.com/roanbrasil/sinew/blob/main/docs/rag.md) and the
`samples/15-rag` program.

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

MIT OR Apache-2.0
