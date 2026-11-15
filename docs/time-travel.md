# Time travel

With a checkpointer, every thread keeps a tree of checkpoints. You can list
them, inspect any of them, re-run the graph from one (**replay**), or write
an edited state after one (**fork**) and continue on that new branch.

## Checkpoint ids and parents

Every checkpoint the graph writes has:

| Field | Meaning |
|---|---|
| `checkpoint_id` | unique, time-ordered (UUID v7 layout) |
| `parent_id` | the checkpoint it follows (`None` for the first one) |
| `metadata.source` | `input` (run start), `loop` (after a superstep), `update` (`update_state` or a command input), `fork`, `interrupt` (a paused superstep) |
| `metadata.writes` | nodes that ran in that superstep |
| `created_at` | Unix milliseconds |

Starting a run writes an `input` checkpoint before the first superstep, so
you can replay a run from its very beginning.

Checkpoints stored by older versions have no id. The SQLite and Postgres
checkpointers give them a stable synthetic id, `legacy-<row id>`, which works
everywhere an id is accepted.

## History

```rust
let history = graph.get_state_history("t").await?;   // newest first
for h in &history {
    println!("{} step {} next {:?} {:?}", h.checkpoint_id, h.step, h.next, h.metadata["source"]);
}
let snap = graph.get_state_at("t", &history[2].checkpoint_id).await?;
```

`StateSnapshot` has the decoded `state`, `next` nodes, `step`,
`checkpoint_id`, `parent_id`, pending `interrupts`, `metadata` and
`created_at`. The history covers the root graph only (subgraph checkpoints
live in their own namespaces) and lists every branch in write order.

## Replay

Continue from any checkpoint instead of the latest:

```rust
use sinew_graph::{RunConfig, RunInput};

let out = graph
    .run_with(RunConfig::new("t").at_checkpoint(&id), RunInput::Continue)
    .await?;
```

Nodes after that checkpoint run again (they may produce different results,
for example with a model at non-zero temperature). The new checkpoints have
the replayed one as parent: a new branch. The thread's latest checkpoint is
now the end of that branch.

`run_with` also accepts a `Command` input at a checkpoint (resume an old
interrupt, or apply an update and continue).

## Fork

Write an edited state after an older checkpoint:

```rust
let mut edited = history[1].state.clone();
edited.plan = "a different plan".into();
let fork_id = graph
    .update_state_with(RunConfig::new("t").at_checkpoint(&history[1].checkpoint_id), edited, None)
    .await?;
graph.run("t", None).await?;   // continues the fork
```

`update_state_with(config, state, as_node)`:

- without `checkpoint_id` it edits the latest checkpoint (what
  `update_state(thread, state)` does);
- with `as_node: Some("node")` the update counts as that node's output, so the
  next tasks become that node's outgoing edge evaluated on the new state
  (LangGraph's `as_node`); otherwise the pending tasks are kept.

It returns the new checkpoint's id. The original branch is untouched and
still reachable through its ids.

## Storage

| Backend | How it is stored |
|---|---|
| `MemoryCheckpointer` | in a `Vec` per thread |
| `SqliteCheckpointer`, `PostgresCheckpointer` | `checkpoint_id` and `checkpoint_ns` columns, indexed with the thread id; added in place to tables created by older versions (see [checkpointing](checkpointing.md#migrations)) |

Custom checkpointers only have to implement `put`, `latest` and `history`.
`latest_in`, `history_in` and `get` have default implementations on top of
`history`; override them with indexed queries for large threads.

A runnable example is `samples/18-time-travel`.
