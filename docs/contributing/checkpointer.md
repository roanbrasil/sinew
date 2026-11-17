# Adding a checkpointer or store

Two kinds of persistence back a graph:

| Trait | Stores | Defined in | Durable implementations in |
|---|---|---|---|
| `Checkpointer` (+ `CheckpointStore` for housekeeping) | The state of each thread after every superstep: resume, interrupts, time travel | `crates/sinew-graph/src/checkpoint.rs` | `crates/sinew-checkpoint/src/sqlite.rs`, `postgres.rs` |
| `Store` | Long-term memories shared across threads, with optional semantic search | `crates/sinew-graph/src/store.rs` | `crates/sinew-checkpoint/src/store.rs` (`SqliteStore`) |

New durable backends go in `sinew-checkpoint`, one module per backend, behind a feature. `sinew-graph` keeps only
the traits and the in-memory implementations.

## The `Checkpointer` trait

```rust
/// Storage for checkpoints.
///
/// Only `put`, `latest` and `history` are required. The namespace- and
/// id-aware methods have default implementations on top of `history`;
/// database backends override them with indexed queries.
#[async_trait]
pub trait Checkpointer: Send + Sync {
    async fn put(&self, checkpoint: Checkpoint) -> Result<(), String>;
    /// Most recently written checkpoint of the thread's root graph.
    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>, String>;
    /// Every checkpoint of the thread (all namespaces), oldest first.
    async fn history(&self, thread_id: &str) -> Result<Vec<Checkpoint>, String>;

    /// Most recently written checkpoint of a namespace (`""` = root graph).
    async fn latest_in(&self, thread_id: &str, ns: &str) -> Result<Option<Checkpoint>, String> { /* default */ }

    /// Checkpoints of one namespace, oldest first (in write order).
    async fn history_in(&self, thread_id: &str, ns: &str) -> Result<Vec<Checkpoint>, String> { /* default */ }

    /// A checkpoint by id.
    async fn get(&self, thread_id: &str, checkpoint_id: &str)
        -> Result<Option<Checkpoint>, String> { /* default */ }
}
```

and the housekeeping trait from `crates/sinew-checkpoint/src/lib.rs`:

```rust
#[async_trait]
pub trait CheckpointStore: Checkpointer {
    /// All thread ids, most recently written first.
    async fn list_threads(&self) -> Result<Vec<String>, CheckpointError>;
    /// Deletes every checkpoint of a thread. Returns the number of rows removed.
    async fn delete_thread(&self, thread_id: &str) -> Result<u64, CheckpointError>;
    /// Keeps only the `keep` most recently written checkpoints of a thread
    /// (`keep = 0` empties it). Returns the number of rows removed.
    async fn prune(&self, thread_id: &str, keep: usize) -> Result<u64, CheckpointError>;
    /// Applies `prune` to every thread at once.
    async fn prune_all(&self, keep: usize) -> Result<u64, CheckpointError>;
}
```

Errors from `Checkpointer` methods are `String`s; the graph wraps them in `GraphError::Checkpoint`. Include the
backend's own message so users can act on it.

### Semantics to keep

- **Order is write order.** `latest` returns the last checkpoint written for the root namespace (`""`), not the
  highest `step`: after a fork or replay, an older step can be the newest checkpoint. `history` is oldest first.
- **Store the whole `Checkpoint` as an opaque JSON blob** (`serde_json::to_string`), plus indexed columns for
  `thread_id`, `checkpoint_id`, `checkpoint_ns`, `step` and the write time. New fields added to `Checkpoint`
  later are then persisted without a schema change; this is how the SQLite and Postgres backends work (see
  `util::encode` and `util::decode_row` in `lib.rs`).
- **Override the defaults** (`latest_in`, `history_in`, `get`) with indexed queries: the defaults load the whole
  history.
- **Concurrency.** Several tasks write to the same thread during a fan-out and several processes may share the
  backend. `put` must be atomic; retention (`with_retention(n)`, if offered) must not delete the checkpoint being
  written.
- **Migrations.** `ensure_schema` creates the table and migrates older layouts in place. Do not break existing
  rows.

## The `Store` trait

```rust
/// Namespaced key-value storage with search. See the [module docs](self).
#[async_trait]
pub trait Store: Send + Sync {
    /// Inserts or replaces an item.
    async fn put_with(
        &self,
        namespace: &[&str],
        key: &str,
        value: Value,
        options: PutOptions,
    ) -> Result<(), StoreError>;

    async fn get(&self, namespace: &[&str], key: &str) -> Result<Option<Item>, StoreError>;

    async fn delete(&self, namespace: &[&str], key: &str) -> Result<(), StoreError>;

    /// Items whose namespace starts with `prefix`, filtered and (with a
    /// query and an index) ranked by similarity; otherwise newest first.
    async fn search(
        &self,
        prefix: &[&str],
        query: SearchQuery,
    ) -> Result<Vec<SearchItem>, StoreError>;

    /// Distinct namespaces starting with `prefix`, truncated to `max_depth`
    /// segments when given.
    async fn list_namespaces(
        &self,
        prefix: &[&str],
        max_depth: Option<usize>,
    ) -> Result<Vec<Vec<String>>, StoreError>;

    /// Inserts or replaces an item with default options.
    async fn put(&self, namespace: &[&str], key: &str, value: Value) -> Result<(), StoreError> { /* default */ }
}
```

Helpers in `sinew_graph::store` keep backends consistent: `validate_namespace` (call it in every method that
takes a namespace), `matches_filter`, `cosine`, and `rank`, which filters, scores, sorts and pages candidates. A
backend without native vector search can load candidates with their stored embeddings and call `rank`, as
`SqliteStore` does. Honor `PutOptions::ttl` (expired items are not returned) and `PutOptions::index`, and embed
with the store's `IndexConfig` (`with_index`).

## Steps

1. **Module.** Create `crates/sinew-checkpoint/src/acme.rs` with `AcmeCheckpointer` (and `AcmeStore` if you add a
   store). Offer the same constructors as the existing backends where they make sense: `connect(url)`,
   `from_pool(...)` or `from_client(...)`, `with_retention(keep)`, `ensure_schema()`.

2. **Feature flag** in `crates/sinew-checkpoint/Cargo.toml`:

   ```toml
   [features]
   # `AcmeCheckpointer`: checkpoints in Acme.
   acme = ["dep:acme-client"]
   ```

   Gate the module and the `pub use` in `lib.rs` with `#[cfg(feature = "acme")]`, as `postgres` is. Forward it
   from `crates/sinew/Cargo.toml` (`acme = ["dep:sinew-checkpoint", "sinew-checkpoint/acme"]`). Add it to `full`
   only if it needs no running service or system library (`full` leaves out the Postgres-backed features).

3. **Errors.** `CheckpointError` today has `Database(sqlx::Error)`, `Serde`, `StepOverflow` and `NegativeStep`. A
   backend that does not use `sqlx` needs a new variant; `CheckpointError` is not `#[non_exhaustive]`, so adding
   one is a breaking change. Propose it in the issue so it can be scheduled for a release.

4. **Tests.**
   - **Shared scenarios.** `crates/sinew-checkpoint/tests/common/mod.rs` has backend-independent scenarios for
     any `CheckpointStore`: `round_trip`, `ordering`, `multiple_threads`, `pruning`, `retention_on_put`,
     `concurrent_puts`, `concurrent_puts_with_retention`, `namespaces_and_ids`. Create
     `crates/sinew-checkpoint/tests/acme.rs` that runs every one against your backend, like `sqlite.rs` and
     `postgres.rs` do.
   - **Server-backed tests** skip themselves when their URL variable is unset, as `postgres.rs` does with
     `SINEW_TEST_PG_URL`, and use a throwaway namespace (schema, key prefix, database) per test. Put the
     `docker run` and `cargo test` commands in the file header.
   - **End-to-end.** `tests/resume.rs` and `tests/graph_features.rs` run real graphs (interrupts, subgraphs, time
     travel) across a simulated restart on SQLite; add the same for your backend if it is server-backed.
   - **Store.** Port the scenarios of `crud_prefix_filter_and_namespaces` and `semantic_search_and_ttl` from
     `crates/sinew-checkpoint/src/store.rs` (namespaces, prefix search, filters, semantic ranking with a
     deterministic `FnEmbedder`, TTL).

## Docs to update

- [docs/checkpointing.md](../checkpointing.md): the backend table, the **Constructors** table, **Storage layout**,
  and **Testing** (the command to run the server-backed tests).
- [docs/store.md](../store.md): the **Backends** table, for a `Store`.
- `README.md`: the `sinew-checkpoint` row and the feature table.
- The crate docs in `crates/sinew-checkpoint/src/lib.rs` (the backend list at the top).

## CHANGELOG

```markdown
- `AcmeCheckpointer` (feature `acme` in `sinew-checkpoint` and `sinew`): durable checkpoints in Acme, with `CheckpointStore` housekeeping and retention (#123).
```
