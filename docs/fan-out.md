# Parallel fan-out / fan-in

A node's outgoing edge can schedule several nodes at once. Those nodes run
concurrently, their outputs are merged into the shared state by a reducer
you provide, and the branches join at the next node.

See `cargo run --example fan_out` for a map-reduce over documents.

## Three ways to fan out

```rust
// 1. Static: always run `a` and `b` after `start`.
g.add_fan_out("start", ["a", "b"]);

// 2. Conditional: the router picks any subset (empty list = this branch ends).
g.add_conditional_edges_multi("route", |s: &S| s.wanted.clone(), ["a", "b", "c", END]);

// 3. Map (`Send` in LangGraph): one branch per item, each with its own input.
g.add_send_edges("plan", |s: &S| {
    s.docs.iter().map(|d| SendTo::new("summarize", S::for_doc(d))).collect()
}, ["summarize"]);

// Required whenever the graph can fan out (compile() checks this):
g.with_reducer(|mut acc: S, branch: S| { acc.summaries.extend(branch.summaries); acc });
```

The single-target `add_conditional_edges` and `add_edge` behave exactly as
before.

## Execution model: supersteps

The engine follows the Pregel / LangGraph model and runs the graph in
**supersteps**:

1. A superstep has a set of **tasks**. A task is a node, plus an input for
   `SendTo` tasks. The first superstep is the entry point.
2. All tasks run **concurrently**. A single task runs inline. Several tasks
   run as separate tokio tasks, so CPU-heavy nodes can also use several
   threads. Each plain task receives a clone of the state as it was at the
   start of the superstep. A `SendTo` task receives its own input.
3. The outputs are **merged**:
   - A superstep with exactly one plain task replaces the state with that
     task's output. Graphs without fan-out therefore behave exactly as
     before.
   - In every other case (several tasks, or any `SendTo` task, even just
     one), the new state is a left fold:
     `reducer(...reducer(reducer(before, out_0), out_1)..., out_n)`.
     The fold starts from the state **before** the superstep, and outputs
     are taken in **task order**, not completion order. The result is
     deterministic.
4. A checkpoint is written (see below).
5. The **next superstep** is built by evaluating the outgoing edge of every
   node executed in this superstep, once per distinct node, **on the merged
   state**. Plain targets are **deduplicated**: when `a` and `b` both go to
   `join`, `join` runs once. This is how branches join. `SendTo` tasks are
   never deduplicated. `END` targets are dropped, and the run completes when
   no tasks are left.

### Joins with uneven branches

Deduplication only happens within one superstep. With `a -> join` and
`b -> b2 -> join`, `join` runs in superstep 2 together with `b2`, and again
in superstep 3. This is the same behavior as Pregel and LangGraph. If `join`
must wait for every branch, make the branches the same length, or have
`join` check a counter or flag in the state and route to itself or `END`.

## Deriving the reducer

Most states merge field by field. Derive it and pick a strategy per field:

```rust
use sinew::graph::State;   // also in sinew::prelude

#[derive(Clone, Default, Serialize, Deserialize, State)]
struct Research {
    #[state(append)]  notes: Vec<String>,           // what each branch added
    #[state(add)]     sources: u32,                 // counters summed across branches
    #[state(merge)]   scores: HashMap<String, f32>, // entries added or changed
                      answer: Option<String>,       // replace (default): the branch that changed it wins
    #[state(reducer = "pick_best")] draft: String,  // fn(&base, acc, branch) -> value
}

g.with_derived_reducer();
```

| Strategy | Field types | Result |
|---|---|---|
| `replace` (default) | any `PartialEq` | the branch value if the branch changed it, else the value merged so far |
| `overwrite` | any | always the branch value (the last branch in task order wins) |
| `keep` | any | ignore branch values |
| `append` | `Vec<T: PartialEq>` | the items a branch added; works both for branches that got a clone of the state and for `SendTo` branches with minimal inputs |
| `add` | numbers | `acc + (branch - base)` |
| `merge` | `HashMap`, `BTreeMap` | the entries a branch added or changed |
| `reducer = "path"` | any | `path(&base_field, acc_field, branch_field)` |

Every strategy compares against the state *before* the superstep, so it gives the same result whether a branch
received a full clone of the state or a minimal `SendTo` input. The derive needs the `macros` feature (on by
default in `sinew`; `features = ["macros"]` on `sinew-graph`).

## Writing reducers by hand

Nodes return **whole states**, not partial updates. A reducer receives
`acc`, the state merged so far, and `branch`, one branch's complete output,
and decides what to take from the branch.

- **`SendTo` with minimal inputs**: the branch input contains only what the
  branch needs, for example `S { doc: Some(d), ..Default::default() }`. Its
  output then contains only its own contribution, so appending is enough:
  `acc.summaries.extend(branch.summaries)`.
- **Disjoint fields**: each branch writes its own field, and the reducer
  copies that field over, e.g.
  `if branch.weather.is_some() { acc.weather = branch.weather }`.
- **Branches that got a clone of the shared state** return the old values
  too. Use `with_reducer_from_base`, which also passes the pre-superstep
  state, and keep only what each branch added:

  ```rust
  g.with_reducer_from_base(|base: &S, mut acc: S, branch: S| {
      acc.log.extend(branch.log.into_iter().skip(base.log.len()));
      acc
  });
  ```

There is one reducer per graph. To handle several fields, write one function
that merges each field the way it needs, or derive it (above).

## Recursion limit

`with_recursion_limit(n)` (default 25) limits the number of **supersteps**
per thread. Without fan-out, one superstep is one node execution, so the
meaning is unchanged. A superstep that runs ten branches counts once.

## Errors

If any branch fails, the run fails with `GraphError::Node` for that branch.
The other branches of the superstep are cancelled, and no checkpoint is
written for the failed superstep. A panicking branch is reported as a node
error for that node. A fan-out without a reducer is rejected by `compile()`
(and by `GraphError::NoReducer` at run time, as a safety net).

## Checkpoints, interrupts and resume

A checkpoint is written after every superstep. `Checkpoint` gained a
`tasks: Vec<PendingTask>` field holding the full next superstep, including
the serialized `SendTo` inputs. `next` is kept and holds the first task's
node, or `END` when the run is finished. Both changes are backwards
compatible. A checkpoint written before `tasks` existed deserializes with an
empty `tasks`, and resuming falls back to `next`. `Checkpointer` itself did
not change.

`interrupt_before([...])` pauses the **whole superstep** if any of its nodes
is listed. Nothing in that superstep runs until you resume.

- `RunOutcome::Interrupted` has a new `next_nodes: Vec<String>` field listing
  every paused node. `next` is still the first one.
- `get_state_snapshot(thread)` returns the state, every next node and the
  step. `get_state` still returns `(state, first_next_node)`.
- `update_state` replaces the shared state and keeps the pending tasks,
  including `SendTo` inputs.
- `run(thread, None)` (or `stream(thread, None)`) resumes and runs the whole
  superstep.

## Tracing

Every node execution in a superstep becomes a `node` run whose parent is the
`graph` run. Each run carries `step` metadata, and during a fan-out it also
carries `task` metadata (the branch index). Parallel runs therefore show up
as siblings with overlapping timings in `sinew-scope`.
