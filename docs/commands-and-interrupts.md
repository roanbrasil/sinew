# Commands and interrupts

Two LangGraph features that let a node steer the run instead of leaving
everything to edges:

- a **`Command`** combines a state update with navigation (`goto`);
- **`interrupt`** pauses the run from inside a node until a human (or another
  system) answers, and the node continues with that answer.

Static pauses (`interrupt_before` / `interrupt_after`) are covered at the end.

## `Command` from a node

Register the node with `add_command_node`. It returns
`Result<Command<S>, BoxError>`:

```rust
use sinew_graph::{Command, StateGraph, END};

g.add_command_node("triage", ["billing", "support", END], |s: Ticket, _ctx| async move {
    let next = if s.text.contains("invoice") { "billing" } else { "support" };
    Ok(Command::goto(next).with_update(Ticket { triaged: true, ..s }))
});
```

| Builder | Effect |
|---|---|
| `Command::goto(node)` / `.with_goto(node)` | run `node` next (`END` ends the branch) |
| `Command::goto_many([a, b])` | run several nodes in parallel next superstep (needs a reducer) |
| `.with_send(SendTo::new(node, input))` | map-style branch with its own input |
| `Command::end()` | same as `goto(END)` |
| `Command::update(state)` / `.with_update(state)` | replace the state, like a plain node's return value |
| `.with_json_update(json!({..}))` | partial update: top-level fields overwrite the state's fields (applied after `update`) |
| `.to_parent()` | apply update and `goto` to the parent graph (from a subgraph; see [subgraphs](subgraphs.md)) |

Rules:

- `destinations` (second argument) lists where the command may go. It is
  only used for the compile-time reachability check; pass an empty list to
  turn that check off for the graph. `goto` targets are validated at run time
  (`GraphError::UnknownRoute`).
- A command node does not need an outgoing edge. If it has one, the edge's
  targets run **in addition** to `goto` (LangGraph semantics).
- A command without `update` / `update_json` writes nothing. In a parallel
  superstep such outputs are left out of the reducer fold, so a node that only
  routes never duplicates state.
- `set_node_destinations(node, [...])` declares run-time destinations for any
  node (for example a subgraph whose nodes route the parent).

## Dynamic interrupts

Any node with a `NodeContext` can pause:

```rust
g.add_node_with_context("approve", |mut s: Email, ctx| async move {
    let ok: bool = ctx.interrupt(json!({ "question": "Send this?", "draft": s.body }))?;
    s.approved = ok;
    Ok(s)
});
let graph = g.compile()?.with_checkpointer(MemoryCheckpointer::new());
```

1. The first time, `ctx.interrupt(value)` returns an error (a
   `GraphInterrupt`) that the `?` propagates. The run stops and returns
   `RunOutcome::Interrupted { interrupts, .. }`. Each `Interrupt` has an `id`,
   the `value` you passed, the `node` and its subgraph `namespace`. Streams end
   with `GraphEvent::Interrupted { interrupts, .. }`, and
   `get_state_snapshot(thread).interrupts` lists them too.
2. Resume with `graph.resume(thread, answer)` (shorthand for
   `graph.run_with(thread, Command::resume(answer))`).
3. The node **runs again from the start**. This time `interrupt` returns the
   resume value, deserialized into the type you asked for.

A node can call `interrupt` several times. Answers are stored per task in the
checkpoint: on every re-execution the calls already answered return their
stored values in order, and the first unanswered call pauses again. Code that
runs before an `interrupt` therefore runs once per resume, so keep it
idempotent (or move side effects after the last interrupt, or into another
node).

Several pending interrupts (parallel branches, or a subgraph):

- `Command::resume(v)` answers the **first** pending interrupt (task order);
- `Command::resume_map([(id, v), ...])` answers specific ones by id.

When a superstep has several tasks and one interrupts, the others still run to
completion. Their results are stored in the checkpoint and are **not**
executed again on resume.

Without a checkpointer an interrupt cannot be resumed, so `invoke` and `run`
return `GraphError::Interrupted(node)`.

### Resuming with edits or a jump

`run_with(thread, command)` accepts a whole `Command` as input:

```rust
// Edit the paused state and continue.
graph.run_with(thread, Command::resume(true).with_json_update(json!({ "cc": "boss@x" }))).await?;
// Skip ahead: replace the pending tasks.
graph.run_with(thread, Command::goto("publish")).await?;
```

## Static pauses

- `.interrupt_before([...])` pauses before a superstep that contains one of
  the nodes; `run(thread, None)` continues.
- `.interrupt_after([...])` pauses after a superstep in which one of the
  nodes ran (if anything is left to run). On resume, `interrupt_before` still
  applies to the next superstep.

Both need a checkpointer. `RunOutcome::Interrupted.interrupts` is empty for
static pauses.

## Policies that interact with interrupts

- **Retries** never retry an interrupt.
- **Cached** nodes are not cached when they interrupt.
- **Timeouts** count each attempt separately.

See `set_node_retry`, `set_node_cache` and `set_node_timeout` on
`StateGraph`:

```rust
g.set_node_retry("fetch", RetryPolicy::new(4)
    .with_initial_interval(Duration::from_millis(200))
    .retry_on(|e| e.downcast_ref::<NodeTimeout>().is_some() || e.to_string().contains("429")));
g.set_node_timeout("fetch", Duration::from_secs(10));
g.set_node_cache("embed", CachePolicy::with_ttl(Duration::from_secs(3600)));
```

Retry waits are `min(initial * factor^(attempt-1), max_interval)` with up to
+50% jitter. Cache keys hash the node name with the JSON-serialized input (or
with your `with_key_fn`); entries live in the compiled graph and are shared by
its clones (`clear_cache()` drops them). The recursion limit counts supersteps
per run (default 25): change it with `with_recursion_limit` or per run with
`RunConfig::with_recursion_limit`.

A runnable example is `samples/18-time-travel`.
