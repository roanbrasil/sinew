# 0002. Pregel supersteps with deterministic task-order merges

- Status: Accepted
- Date: 2026-11-16
- Related: [0001](0001-typed-whole-state-nodes.md), [docs/fan-out.md](../fan-out.md)

## Context

Agent graphs fan out (parallel tools, map-reduce over documents, supervisors) and must also be
checkpointed, interrupted, resumed and replayed. If branches were merged as they finished, the
final state would depend on network timing, and replaying a thread could give a different result.

## Considered options

- Free-running tasks: each node schedules its successors as soon as it finishes.
- Supersteps (Pregel / LangGraph): all tasks of a step run, then their outputs are merged, then
  the next step is scheduled.

## Decision

Execution runs in supersteps ([crates/sinew-graph/src/pregel.rs](../../crates/sinew-graph/src/pregel.rs)).
Tasks of a step run concurrently on a `tokio::task::JoinSet`; a single non-parallel task is
awaited inline. Outcomes are stored by task index, and `merge` folds them with the reducer in
task order, starting from the pre-step state, independent of completion order. Edges and `goto`s
are evaluated on the merged state; identical targets are deduplicated so branches join. When a
checkpointer is attached, a checkpoint is written after each superstep. A recursion limit
(default 25 supersteps) bounds cycles.

## Consequences

- Good: the same input gives the same state regardless of which branch is faster; replays and
  forks from a checkpoint are reproducible.
- Good: checkpoint boundaries are clear (one per superstep), which keeps interrupts, resume and
  time travel simple to reason about.
- Good: measured overhead is a few microseconds per superstep (`cargo bench -p sinew-graph`).
- Bad: a superstep is as slow as its slowest branch; a fast branch cannot start its successor
  early. Long independent pipelines should be subgraphs or single nodes.
- Bad: the first failing branch ends the superstep with its error and the other branches are
  aborted; their work is lost unless the node has a retry policy.
