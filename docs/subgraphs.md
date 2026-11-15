# Subgraphs

A compiled graph can be a node of another graph. Use it to package an agent,
reuse a workflow, or give a team of agents its own private state.

## Shared state

When both graphs use the same state type, the subgraph runs on the current
state and its final state replaces it:

```rust
let review: CompiledGraph<Doc> = review_graph.compile()?;

let mut g = StateGraph::<Doc>::new();
g.add_node("draft", draft);
g.add_subgraph("review", review);
g.add_edge("draft", "review");
g.add_edge("review", END);
```

## Different state (input/output mapping)

`add_subgraph_mapped` converts the parent state into the subgraph's input,
and folds the subgraph's final state back:

```rust
g.add_subgraph_mapped(
    "researcher",
    research_graph,                                   // CompiledGraph<Research>
    |s: &Chat| Research { question: s.last_question(), ..Default::default() },
    |mut s: Chat, r: Research| { s.notes.push(r.summary); s },
);
```

## What a subgraph inherits

| | Behaviour |
|---|---|
| Checkpointer | The parent's, unless the subgraph has its own. Subgraph checkpoints are written under the same thread with a **namespace** (`"node:task"`, nested ones joined by `|`), so they never mix with the parent's history. |
| Store | The parent's, unless the subgraph has its own. |
| Tracer | The subgraph's graph run is nested under the parent's node run. |
| Recursion limit | The subgraph's own. |

## Interrupts inside a subgraph

A node deep inside a subgraph can call `ctx.interrupt(...)` (or the subgraph
can have `interrupt_before`/`interrupt_after`). The whole run pauses: the
subgraph saves its checkpoint in its namespace, and the parent saves one
where the subgraph node is pending. The `Interrupt` carries the namespace of
the node that raised it:

```rust
let out = graph.run("t", Some(input)).await?;
assert_eq!(out.interrupts()[0].namespace, vec!["review:0"]);
let out = graph.resume("t", "approved").await?;   // continues inside the subgraph
```

On resume the parent re-runs the subgraph node, which continues from its own
checkpoint instead of starting over: nodes that already ran inside the
subgraph are not executed again.

## Routing the parent: `Command::to_parent`

A node of a subgraph can leave it and route the parent graph:

```rust
inner.add_command_node("decide", Vec::<String>::new(), |s: S, _ctx| async move {
    Ok(Command::goto("escalate").with_json_update(json!({ "priority": "high" })).to_parent())
});
// In the parent: the subgraph node may now route to `escalate`.
parent.add_subgraph("triage", inner.compile()?);
parent.set_node_destinations("triage", ["escalate"]);
```

The subgraph stops at once. Its update is applied to the parent's state (an
`update` is serialized and its top-level fields overwrite the parent's; with a
mapped subgraph only matching fields are copied), and the parent continues at
the `goto` targets (plus the subgraph node's own outgoing edge, if any).
Returning `to_parent()` from the root graph is a `GraphError::InvalidCommand`.

This is how [handoffs](multi-agent.md) between agents work.

## Streaming subgraph events

Subgraph events are off by default. Turn them on with `StreamOptions`:

```rust
let mut events = graph.stream_with(
    "t",
    input,
    StreamOptions::new([StreamMode::Updates, StreamMode::Messages]).with_subgraphs(true),
);
while let Some(ev) = events.next().await {
    if let GraphEvent::Subgraph { namespace, event } = ev {
        // namespace: ["review:0"], or ["team:0", "researcher:0"] when nested
        // event: GraphEvent<serde_json::Value> (the subgraph's state is serialized)
    }
}
```

The stream modes apply to subgraph events too. Terminal events of subgraphs
are not forwarded; the parent's run decides how the stream ends.
