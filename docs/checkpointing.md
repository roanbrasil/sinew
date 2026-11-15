# Checkpointing

A compiled `StateGraph` writes a `Checkpoint` when a run starts and after
every step when it has a checkpointer. A checkpoint holds the thread id, the
step, the tasks that run next, the serialized state, its own id and its
parent's id, the subgraph namespace and some metadata. That is what makes
`run(thread, None)`, `get_state`, `update_state`, interrupts (static and
dynamic, see [commands and interrupts](commands-and-interrupts.md)) and
[time travel](time-travel.md) work.

`sinew-graph` ships `MemoryCheckpointer`, which loses everything when the
process exits. The `sinew-checkpoint` crate adds durable backends:

| Type | Cargo feature | Use it for |
|---|---|---|
| `SqliteCheckpointer` | `sqlite` (default) | one machine: a CLI, a desktop app, a single service; in-memory for tests |
| `PostgresCheckpointer` | `postgres` | several processes or hosts sharing threads |

```toml
[dependencies]
sinew-checkpoint = "0.1"                                     # SQLite only
sinew-checkpoint = { version = "0.1", features = ["postgres"] }  # SQLite + Postgres
sinew-checkpoint = { version = "0.1", default-features = false, features = ["postgres"] }
```

Both use `sqlx` with the tokio runtime and rustls. Queries are built at run
time, so building never needs a database.

## Quick start

```rust
use sinew_checkpoint::SqliteCheckpointer;

let store = SqliteCheckpointer::connect("sqlite://agent.db").await?;
let graph = g.compile()?
    .with_checkpointer(store.clone())
    .interrupt_before(["publish"]);

graph.run("thread-1", Some(input)).await?;   // pauses before `publish`
// ... the process can exit here ...

// Later, in a new process:
let store = SqliteCheckpointer::connect("sqlite://agent.db").await?;
let graph = build_graph().with_checkpointer(store).interrupt_before(["publish"]);
let (mut state, next) = graph.get_state("thread-1").await?;
state.approved = true;
graph.update_state("thread-1", state).await?;
graph.run("thread-1", None).await?;          // resumes from the database
```

A runnable version is in `crates/sinew-checkpoint/examples/durable_agent.rs`:

```bash
cargo run -p sinew-checkpoint --example durable_agent -- --step1
cargo run -p sinew-checkpoint --example durable_agent -- --step2   # or --step2 --reject
```

## Constructors

| Call | What it does |
|---|---|
| `SqliteCheckpointer::connect("sqlite://path.db")` | opens or creates the file (WAL mode, 10 s busy timeout), creates the schema |
| `SqliteCheckpointer::connect("sqlite::memory:")` / `in_memory()` | private in-memory database, one pooled connection that lives as long as the checkpointer and its clones |
| `PostgresCheckpointer::connect("postgres://user:pass@host/db")` | pool of 10 connections, creates the schema |
| `from_pool(pool)` (both) | wraps a pool you configured yourself, creates the schema |
| `ensure_schema()` (both) | idempotent DDL; `connect` and `from_pool` already call it |
| `.with_retention(n)` (both) | keep at most `n` checkpoints per thread, pruned inside each `put` |
| `.pool()` (both) | the underlying `sqlx` pool |

`sinew_checkpoint::sqlx` re-exports `sqlx`, so you can build a pool for
`from_pool` without matching versions by hand. The checkpointers are cheap to
clone: clones share the pool.

## Read semantics

- `latest(thread)` returns the **most recently written** checkpoint of the
  thread's root graph. This matches `MemoryCheckpointer`, and it stays correct
  when a thread is restarted with `run(thread, Some(input))` (the step counter
  goes back to 0) and after a fork (the fork is the newest write).
- `latest_in(thread, ns)` does the same for a subgraph namespace.
- `history(thread)` returns every checkpoint (all namespaces) ordered by
  step, then by insertion order. `update_state` writes a second checkpoint
  with the same step, so both appear, in the order they were written.
- `history_in(thread, ns)` returns one namespace in write order; this is what
  `get_state_history` uses.
- `get(thread, checkpoint_id)` returns one checkpoint by id.

## Housekeeping: `CheckpointStore`

Both types implement the `CheckpointStore` trait (bring it into scope with
`use sinew_checkpoint::CheckpointStore`):

| Method | Returns |
|---|---|
| `list_threads()` | thread ids, most recently written first |
| `delete_thread(id)` | number of rows deleted |
| `prune(id, keep)` | keeps the `keep` newest checkpoints of one thread (`0` empties it) |
| `prune_all(keep)` | the same for every thread, in one statement |

Retention set with `with_retention(n)` prunes on every write. Values below 1
are treated as 1, so a thread's latest checkpoint is never lost. `prune` and
`prune_all` are for scheduled cleanup instead.

## Storage layout

One table, `sinew_checkpoints`:

| Column | SQLite | Postgres |
|---|---|---|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | `BIGINT GENERATED ALWAYS AS IDENTITY` |
| `thread_id` | `TEXT` | `TEXT` |
| `step` | `INTEGER` | `BIGINT` |
| `created_at` | `INTEGER` (Unix milliseconds) | `TIMESTAMPTZ DEFAULT now()` |
| `checkpoint` | `TEXT` (JSON) | `JSONB` |
| `checkpoint_id` | `TEXT` (NULL for old rows) | `TEXT` (NULL for old rows) |
| `checkpoint_ns` | `TEXT NOT NULL DEFAULT ''` | `TEXT NOT NULL DEFAULT ''` |

Indexes are on `(thread_id, id)` for pruning and `list_threads`, on
`(thread_id, step, id)` for `history`, on `(thread_id, checkpoint_ns, id)`
for `latest`/`latest_in`/`history_in`, and on `(thread_id, checkpoint_id)`
for `get`.

The whole `Checkpoint` struct is stored as one JSON blob. Only `thread_id`,
`step`, `created_at`, `checkpoint_id` and `checkpoint_ns` are copied into
their own columns. Other fields added to `Checkpoint` are persisted without a
migration. Because the
Postgres column is `JSONB`, you can query it directly:

```sql
SELECT thread_id, checkpoint->>'next' AS next
FROM sinew_checkpoints
WHERE checkpoint->'state'->>'approved' = 'false';
```

`JSONB` normalizes the JSON text: it reorders object keys and drops duplicate
keys. This does not change the deserialized `Checkpoint`.

On Postgres the table is created unqualified, so it lands in the first schema
of the connection's `search_path`. To isolate tenants or tests, give each one a
schema and set `search_path` on the pool (see `tests/postgres.rs`).

## Migrations

Tables created by the first release lack `checkpoint_id` and
`checkpoint_ns`. `ensure_schema` (called by `connect` and `from_pool`) adds
them in place:

- **SQLite** reads `pragma_table_info` and runs `ALTER TABLE ... ADD COLUMN`
  for each missing column (a concurrent process that migrated first is
  tolerated), then creates the new indexes.
- **Postgres** runs `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` under the
  schema advisory lock.

Existing rows keep working: they belong to the root namespace and are read
with a synthetic id, `legacy-<row id>`, which `get` (and therefore replay and
forks) accepts. Nothing is rewritten.

## Long-term store

`SqliteStore` (feature `sqlite`) implements the graph's long-term `Store`
(namespaced items, filters, optional semantic search, TTLs) in a
`sinew_store` table, possibly in the same file as the checkpoints. See
[store](store.md).

## Concurrency

- **SQLite**: file databases use WAL mode, so readers never block the writer,
  and a 10 s busy timeout, so concurrent writers wait instead of failing. Each
  `put` and its retention pruning run in one transaction. The transaction
  starts with the `INSERT`, so it never has to upgrade a read lock. Several
  processes can share the file.
- **Postgres**: each `put`, `prune` and `delete_thread` takes a transaction-scoped
  advisory lock on a hash of the thread id. Writers to the same thread are
  serialized, so ids follow commit order and pruning never races an insert.
  Writers to different threads don't block each other. `ensure_schema` takes a
  global advisory lock, because `CREATE TABLE IF NOT EXISTS` alone is not safe
  when several processes start at once.

## Testing

```bash
cargo test -p sinew-checkpoint                     # SQLite: in-memory, temp files, restart test

# Postgres (skipped when SINEW_TEST_PG_URL is unset); each test uses its own schema
docker run -d --rm --name sinew-test-pg -e POSTGRES_PASSWORD=sinew \
  -p 127.0.0.1:55432:5432 postgres:16-alpine
SINEW_TEST_PG_URL=postgres://postgres:sinew@127.0.0.1:55432/postgres \
  cargo test -p sinew-checkpoint --features postgres
docker stop sinew-test-pg
```

The same scenarios (`tests/common/mod.rs`) run against both backends: round
trip, ordering, several threads, pruning, retention, and concurrent writers.
`tests/resume.rs` runs a graph with `interrupt_before` on a SQLite file, drops
everything, re-opens the file in a new graph instance and resumes the thread.
