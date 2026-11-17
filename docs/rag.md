# Retrieval (RAG)

`sinew-rag` is the retrieval half of LangChain in Rust: load documents, split them, embed them,
store them, retrieve them and answer questions with sources. Every piece is a small trait, so you
can swap a provider, a store or a reranker without touching the rest.

```text
Loader ──► TextSplitter ──► Embeddings ──► VectorStore ──► Retriever ──► RagChain / retriever_tool
 (files)     (chunks)        (vectors)      (index)       (top-k docs)     (answer + sources)
```

```toml
[dependencies]
sinew-rag = "0.1"                                                  # in-memory + SQLite stores
sinew-rag = { version = "0.1", features = ["qdrant", "pgvector", "pdf"] }
```

## Documents and loaders

A `Document` is `{ id: Option<String>, content: String, metadata: Map<String, Value> }`. Stores
give documents without an id a UUID; giving stable ids yourself makes re-indexing an upsert.

| Loader | Input | Metadata it adds |
|---|---|---|
| `TextLoader::new(path)` | any text file (invalid UTF-8 replaced) | `source` |
| `MarkdownLoader::new(path)` | Markdown; YAML-style front matter is removed and becomes metadata | `source`, front-matter keys, `title` (first `# ` heading) |
| `HtmlLoader::new(path)` | HTML converted with `html_to_text` (drops `script`/`style`/`head`, keeps paragraphs and list items, decodes entities) | `source`, `title` |
| `JsonLoader::new(path)` / `::json_lines(path)` | JSON or JSONL; `with_records_pointer("/items")`, `with_content_pointer("/text")`, `with_metadata_fields([...])` | `source`, `seq_num` |
| `CsvLoader::new(path)` | one document per row, `column: value` lines; `with_content_columns`, `with_metadata_columns`, `with_delimiter` | `source`, `row` |
| `PdfLoader::new(path)` (feature `pdf`) | text per page (pure Rust, `pdf-extract`); `.single()` for one document | `source`, `page` |
| `DirectoryLoader::new(root)` | walks a tree; `glob("**/*.md")`, `exclude("**/target/**")`, `non_recursive()`, `skip_errors()`, `with_loader("ext", f)` | `relative_path` plus the per-file loader's fields |

`DirectoryLoader` picks the loader from the extension (`md`, `html`, `json`, `jsonl`, `csv`, `tsv`,
`pdf`, anything else as text), skips hidden files and visits files in sorted order.

Loaders are synchronous. In an async server, run them inside `tokio::task::spawn_blocking`.

## Splitting

```rust
use sinew_rag::*;

let splitter = RecursiveCharacterTextSplitter::new(1000, 150);   // chunk_size, chunk_overlap
let chunks = splitter.split_documents(&docs);
```

`RecursiveCharacterTextSplitter` follows LangChain exactly: try the separators in order (default
`"\n\n"`, `"\n"`, `" "`, `""`), split on the first one present, merge small pieces back into chunks
of at most `chunk_size` with up to `chunk_overlap` of trailing context, and recurse into pieces that
are still too long. The unit test reproduces LangChain's reference output.

| Option | Meaning |
|---|---|
| `with_separators([...])` | literal separators, tried in order |
| `for_language(Language::Rust, size, overlap)` | code-aware presets: Rust, Python, JavaScript, TypeScript, Go, Java, C++, Markdown, HTML |
| `with_keep_separator(KeepSeparator::Start / End / None)` | where the separator goes (`Start` is the default) |
| `with_length_fn(approx_token_len)` | measure in approximate tokens (chars / 4), or pass a real tokenizer closure |
| `with_start_index(true)` | `start_index` metadata (byte offset in the parent) |

Chunks keep the parent's metadata plus `chunk_index`; if the parent has an id, chunk ids are
`"{parent}:{n}"`.

`MarkdownHeaderTextSplitter::new([("#", "h1"), ("##", "h2")])` cuts a Markdown document into its
sections and records the enclosing headers as metadata (headers inside code fences are ignored).
A common pipeline is headers first, then `RecursiveCharacterTextSplitter` to bound the size.

## Embeddings

```rust
#[async_trait]
pub trait Embeddings: Send + Sync {
    fn model(&self) -> &str;
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}
```

| Type | Endpoint | Configuration |
|---|---|---|
| `OpenAiEmbeddings::openai("text-embedding-3-small")` | OpenAI `/v1/embeddings` | `OPENAI_API_KEY`; `with_dimensions(n)` |
| `OpenAiEmbeddings::ollama("bge-m3")` | Ollama `/v1/embeddings` | `OLLAMA_HOST` (default 127.0.0.1:11434) |
| `OpenAiEmbeddings::lm_studio`, `vllm`, `llama_cpp` | local OpenAI-compatible servers | `LMSTUDIO_HOST`, `VLLM_HOST`, `LLAMACPP_HOST` |
| `OpenAiEmbeddings::new(base_url, model)` | anything else | `with_api_key` |
| `GeminiEmbeddings::from_env("gemini-embedding-001")` | Gemini `batchEmbedContents` | `GEMINI_API_KEY`; Vertex via `with_base_url` + `with_bearer_token` |
| `FakeEmbeddings::new(dim)` | none (feature hashing) | deterministic, offline |

`embeddings_from_spec("provider:model")` accepts the same provider names and environment
variables as the chat `from_spec` in `sinew-core` (`openai`, `mistral`, `together`, `fireworks`,
`nvidia`, `openrouter`, `huggingface`, `gemini`, `ollama`, `lmstudio`, `llamacpp`, `vllm`,
`sglang`, `localai`, `jan`, `llamafile`, `tgi`/`tei`) plus `"http://host:port/v1|model"`.

Throughput: inputs go out in batches (`with_batch_size`, default 256 for OpenAI-compatible
servers, 100 for Gemini) with several requests in flight (`with_concurrency`, default 4), over
one pooled HTTP client that clones share.

`CachedEmbeddings::new(inner)` wraps any model: cached vectors are reused, misses are
de-duplicated and embedded in one batch. Keys are `model + text`. Implement `EmbeddingCache` to
put the cache in Redis or a database; `InMemoryEmbeddingCache::with_capacity(n)` bounds memory.

## Vector stores

```rust
let store = InMemoryVectorStore::new(OpenAiEmbeddings::ollama("bge-m3"));
store.add_documents(chunks).await?;                          // embeds in batches, upserts by id
let hits = store.similarity_search_with_score("query", 4, None).await?;
let diverse = store.max_marginal_relevance_search("query", 4, 20, 0.5, None).await?;
store.delete(&["docs/a.md#3".into()]).await?;
```

| Store | Feature | Best for |
|---|---|---|
| `InMemoryVectorStore` | always | tests, CLIs, up to ~1M vectors in RAM |
| `SqliteVectorStore::connect("sqlite://rag.db", e)` | `sqlite` (default) | a persistent index on one machine, up to a few hundred thousand chunks |
| `QdrantVectorStore::new("http://127.0.0.1:6333", "docs", e)` | `qdrant` | large collections, approximate search, filtering at scale |
| `PgVectorStore::connect("postgres://...", e)` | `pgvector` | keeping vectors next to your relational data |

Scores are always cosine similarity (1 is identical). To add your own store, implement
`embeddings`, `add_vectors`, `delete`, `get`, `len` and `search_by_vector`; text search, batch
indexing and MMR come from the trait.

**In-memory internals.** Vectors are normalised on insert and stored in one contiguous row-major
`Vec<f32>`. A query is normalised once, then each row costs one dot product (8-lane unrolled so
LLVM vectorises it) and one comparison against the worst hit of a k-sized min-heap. No allocation
happens inside the scan. Above `with_parallel_threshold` floats (default about 5,000 × 384), the
rows are split across scoped threads (`with_threads`, default up to 8 cores) and the partial
heaps are merged. Run `cargo bench -p sinew-rag --bench search` to measure on your hardware.

**SQLite internals.** One `sinew_vectors` table keyed by `(collection, id)`, vectors as
little-endian `f32` blobs. A search streams `(id, embedding)` rows, computes the dot product
straight from the blob bytes, keeps a top-k heap, and only then loads the content and metadata
of the winners. With a filter, the metadata column is read and the filter is evaluated in Rust.
`with_collection("name")` keeps several indexes in one file.

**Qdrant.** Collections use cosine distance and are created on the first insert. The payload is
`page_content`, `metadata` and `doc_id`; point ids are the document id when it is a UUID, else a
UUIDv5 of it, so re-adding a document replaces it.

**pgvector.** One `sinew_pgvector` table with an untyped `vector` column, `jsonb` metadata and a
`collection` column. Searches are exact by default; for approximate search declare the size with
`with_dimensions(n)` and call `create_hnsw_index()`, which indexes `embedding::vector(n)` with
`vector_cosine_ops`. Creating the extension needs a role that may do so.

### Metadata filters

```rust
let f = Filter::eq("lang", "en")
    .and(Filter::gte("year", 2024.0))
    .and(Filter::is_in("tag", ["rust", "llm"]).not());
let same = Filter::from_json(&serde_json::json!({
    "lang": "en", "year": {"$gte": 2024}, "tag": {"$nin": ["rust", "llm"]}
}))?;
store.similarity_search("query", 4, Some(&f)).await?;
```

Operators: `Eq`, `Ne`, `In`, `Nin`, `Gt`, `Gte`, `Lt`, `Lte`, `Exists`, `And`, `Or`, `Not`, on
top-level metadata fields. Numbers compare numerically (`2024` equals `2024.0`). In-memory and
SQLite evaluate filters in Rust; Qdrant and pgvector translate them into native filters
(`Filter::to_qdrant`, SQL on `jsonb`).

## Retrievers

```rust
#[async_trait]
pub trait Retriever: Send + Sync {
    async fn retrieve(&self, query: &str) -> Result<Vec<Document>>;
}
```

| Retriever | What it does |
|---|---|
| `VectorStoreRetriever::new(store)` | `with_k`, `with_filter`, `mmr()`, `with_score_threshold(t)`, `with_scores()` (adds `score` metadata) |
| `Bm25Retriever::from_documents(docs)` | Okapi BM25 on an inverted index, `with_k`, `with_params(k1, b)`; `from_documents_with_tokenizer(docs, english_tokenize)` adds stop words and light stemming |
| `EnsembleRetriever::new(vec![bm25, vector])` | hybrid search: runs retrievers concurrently, fuses with weighted reciprocal rank fusion (`with_weights`, `with_c`, default 60), adds `rrf_score` |
| `MultiQueryRetriever::from_llm(base, model)` | a chat model writes alternative phrasings; results of all queries are merged without duplicates. Implement `QueryExpander` for your own rewriting |
| `ContextualCompressionRetriever::new(base, compressor)` | post-process results with a `DocumentCompressor` |
| `LlmReranker::new(model)` | compressor: one listwise call scores every passage 0–10, keeps `with_top_n` (adds `relevance_score`); an unparseable reply keeps the original order |
| `EmbeddingsFilter::new(embeddings, threshold)` | compressor: drop passages below a cosine similarity to the query |
| `ParentDocumentRetriever::new(store, docstore, child_splitter)` | index small chunks, return their parent documents; `with_parent_splitter` for medium-sized parents |

A typical high-quality setup is hybrid retrieval of 20 candidates, then an LLM rerank down to 4:

```rust
let candidates = EnsembleRetriever::new(vec![bm25, vector]).with_k(20);
let retriever = ContextualCompressionRetriever::new(
    Arc::new(candidates),
    Arc::new(LlmReranker::new(model.clone()).with_top_n(4)),
);
```

## Agents and chains

`retriever_tool(retriever, name, description)` is a `sinew_core::Tool` taking `{"query": "..."}`
and returning numbered passages (`[1] (source: docs/a.md)\n...`). Register it in a `ToolBox` and
pass it to `react_agent` like any other tool. `with_max_chars(n)` cuts long passages.

`RagChain::new(model, retriever)` is the "stuff" chain: retrieve, put the numbered passages in the
prompt, ask the model to cite them, return `RagAnswer { answer, sources, usage }`. `source_names()`
lists unique sources; `cited()` returns the passages whose `[n]` marker appears in the answer.
`with_system_prompt`, `with_template` (`{context}`, `{question}`), `with_temperature` and
`with_max_chars` tune it.

## Evaluation

`sinew-eval` with the `rag` feature measures a RAG pipeline: faithfulness, answer relevance,
context precision and recall, citation accuracy (judged by any `ChatModel`), answer similarity
(embeddings), hit rate, recall, precision, MRR and nDCG at `k` (no model), and writes synthetic
test sets from your chunks. `rag_output(&answer)` turns a `RagAnswer` into an experiment output.
See [Evaluating RAG](evaluation.md#evaluating-rag).

## Sample

`samples/15-rag` indexes this repository's `docs/*.md` (header split, then size split, section
path prepended to each chunk), builds a hybrid BM25 + vector retriever, answers a question with
`RagChain` and lists the sources, then runs a ReAct agent with the retriever as a tool. Last, it
writes a two-question synthetic test set from `docs/checkpointing.md` and prints the RAG metrics
of the chain's answers.

```bash
cargo run -p sample-15-rag                         # offline: FakeEmbeddings + scripted model
cargo run -p sample-15-rag -- --ollama             # Ollama chat + Ollama bge-m3 embeddings
cargo run -p sample-15-rag -- --ollama "How do I trace a run?"
cargo run -p sample-15-rag -- --model gemini:gemini-2.5-flash --embed gemini:gemini-embedding-001
```

Put `--ollama` / `--model` first; `--embed provider:model` picks the embedding model
(`SINEW_EMBED_MODEL` changes the Ollama default).

## Testing

Unit and integration tests run offline with `FakeEmbeddings` and temporary SQLite files. The same
behaviour suite (`tests/stores.rs`) runs against every store; the Qdrant, pgvector and Ollama
variants are `#[ignore]` and need a server:

```bash
cargo test -p sinew-rag --all-features
cargo test -p sinew-rag --test live_ollama -- --ignored
SINEW_QDRANT_URL=http://127.0.0.1:6333 cargo test -p sinew-rag --features qdrant -- --ignored qdrant
SINEW_PGVECTOR_URL=postgres://postgres:pw@127.0.0.1:5432/postgres \
  cargo test -p sinew-rag --features pgvector -- --ignored pgvector
```
