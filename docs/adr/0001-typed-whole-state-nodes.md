# 0001. Typed whole-state nodes (`S -> S`) with reducers

- Status: Accepted
- Date: 2026-11-16
- Related: [0002](0002-pregel-supersteps.md), [docs/fan-out.md](../fan-out.md)

## Context

LangGraph nodes take the state and return a partial update (a dict); each state key can have a
reducer that merges updates. In Rust, a partial update needs either a second "update" type per
state, a `serde_json::Value` patch, or a derive macro that generates both. Each of these loses
type checking or adds generated code users have to read.

A graph state in Sinew is any type that is `Clone + Send + Sync + Serialize + DeserializeOwned`
(`State` in [crates/sinew-graph/src/graph.rs](../../crates/sinew-graph/src/graph.rs)).

## Considered options

- Partial updates as `serde_json::Value` patches with per-key reducers.
- A derive macro generating an update type and per-field reducers.
- Whole-state nodes: a node receives `S` and returns `S`; one optional graph-level reducer
  merges parallel branches.

## Decision

Nodes are async functions `S -> Result<S, BoxError>` (`NodeResult<S>`), or return a `Command<S>`
whose `update` is again a whole `S`. When a superstep has one plain node, its output replaces
the state. When it has several branches, the reducer set with `StateGraph::with_reducer`
(`Fn(S, S) -> S`) or `with_reducer_from_base` (`Fn(&S, S, S) -> S`) folds them. `compile()`
rejects a graph that can fan out without a reducer and reports every build problem at once
(`GraphError::Build(Vec<String>)`).

## Consequences

- Good: the compiler checks every node against the state type; no macro, no JSON patches, no
  string keys. Plain Rust structs and enums work as state.
- Good: sequential graphs need no reducer, and a single node can take the state by move instead
  of a clone (see [0002](0002-pregel-supersteps.md)).
- Bad: a reducer sees whole states, so it must know which fields each branch owns. When branches
  append to a shared list, the reducer needs the pre-step state (`with_reducer_from_base`) to
  keep only what each branch added.
- Bad: there is one reducer per graph, not one per field; large states with many parallel
  writers need a hand-written merge.
- Bad: each parallel branch gets a clone of the state, so a large state costs one copy per
  branch.
