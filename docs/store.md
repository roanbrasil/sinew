# Long-term memory: `Store`

Checkpoints belong to one thread. A **store** holds what must outlive a
thread: user preferences, facts learned in earlier conversations, shared
notes. It plays the role of LangGraph's `BaseStore`.

For extraction, deduplication and recall of user memories on top of a store,
see [long-term memory](memory.md) (`sinew-memory`).

Items are JSON values addressed by a hierarchical **namespace** and a
**key**:

```rust
use serde_json::json;
use sinew_graph::{store::SearchQuery, InMemoryStore, Store};

let store = InMemoryStore::new();
store.put(&["memories", "ana"], "lang", json!({ "text": "prefers Portuguese", "kind": "pref" })).await?;
let item = store.get(&["memories", "ana"], "lang").await?;      // Option<Item>
store.delete(&["memories", "ana"], "lang").await?;
```

| Method | Does |
|---|---|
| `put(ns, key, value)` / `put_with(ns, key, value, PutOptions { index, ttl })` | insert or replace; `ttl` expires the item, `index: Some(false)` skips embedding |
| `get(ns, key)` | one item (`namespace`, `key`, `value`, `created_at`, `updated_at`) |
| `delete(ns, key)` | remove |
| `search(prefix, SearchQuery)` | items under a namespace prefix (whole segments), filtered and ranked |
| `list_namespaces(prefix, max_depth)` | distinct namespaces, optionally truncated |

## Search

```rust
let hits = store
    .search(&["memories"], SearchQuery::new()
        .filter("kind", json!("pref"))     // exact match on top-level fields
        .query("what language?")           // semantic ranking (needs an index)
        .limit(5))
    .await?;
for h in hits { println!("{:?} {}", h.score, h.item.value); }
```

Without a query (or without an index) results are newest first. With a query
and an index they are sorted by cosine similarity, and `score` is set.

## Semantic search with your embeddings

The store does not depend on any embedding provider. Implement `Embedder`
over whatever you use (a local model, an HTTP API, the embeddings trait of
another crate) and pass it in an `IndexConfig`:

```rust
use sinew_graph::store::{Embedder, IndexConfig};

struct MyEmbedder { /* client */ }

#[async_trait::async_trait]
impl Embedder for MyEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BoxError> {
        /* one vector per text */
    }
}

let store = InMemoryStore::new()
    .with_index(IndexConfig::new(MyEmbedder { .. }).with_fields(["text"]));
```

`fields` lists the top-level fields (or dotted paths such as
`"profile.bio"`) whose text is embedded on `put`. When empty, the whole value
is embedded (a string value as is, anything else as JSON). For tests,
`FnEmbedder(|text: &str| -> Vec<f32> { .. })` wraps a closure.

## Backends

| Type | Crate | Notes |
|---|---|---|
| `InMemoryStore` | `sinew-graph` | process-local |
| `SqliteStore` | `sinew-checkpoint` (feature `sqlite`) | table `sinew_store`; can share a file with `SqliteCheckpointer`; embeddings stored as JSON and ranked in process; `purge_expired()` deletes expired rows |

Implement the `Store` trait for anything else (five methods; `put` has a
default).

## From a node

Attach a store to the compiled graph; every node (and every subgraph) can
reach it through its context:

```rust
let graph = g.compile()?.with_store(SqliteStore::connect("sqlite://memory.db").await?);

g.add_node_with_context("recall", |mut s: Chat, ctx| async move {
    let store = ctx.store().expect("graph has a store");
    let facts = store.search(&["memories", &s.user], SearchQuery::new().query(&s.question)).await?;
    s.context = facts.into_iter().map(|h| h.item.value).collect();
    Ok(s)
});
```

`graph.store()` returns the same store for use outside the graph, and
`NodeContext::new(..).with_store(store)` lets tests call such nodes directly.
