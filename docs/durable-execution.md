# Durable execution

A checkpoint after every superstep lets a graph resume where it stopped. That is not enough for agents: a tool
node may call five tools, and if the process dies after the third, resuming re-runs the whole node. The e-mail
goes out twice, the card is charged twice.

Sinew records every tool call in the checkpointer's **tool ledger**:

```text
pending ── tool runs ──▶ done(result)
```

On resume:

| Ledger entry | Tool without side effects | Tool with side effects (`side_effects()`) |
|---|---|---|
| `done` | result replayed, tool not called | result replayed, tool not called |
| `pending` (started, outcome unknown) | runs again | follows the `UnknownOutcome` policy |
| none | runs | runs |

It works with no extra code in the prebuilt `ReactAgent` (and the supervisor and swarm agents built on it) and in
the harness, as long as the checkpointer keeps a ledger: `MemoryCheckpointer`, `SqliteCheckpointer` and
`PostgresCheckpointer` do (table `sinew_tool_ledger`, cleared by `delete_thread`).

## Marking side effects

```rust
/// Charge the customer's card.
#[tool(side_effects)]
async fn charge(#[state] stripe: Stripe, customer: String, cents: u64) -> Result<Receipt, StripeError> {
    // The key is stable across retries and resumes of this exact call.
    let key = sinew::core::tool::idempotency_key().unwrap_or_default();
    stripe.charge(&customer, cents).idempotency_key(&key).await
}
```

Hand-written tools override `fn side_effects(&self) -> bool { true }`.

`idempotency_key()` returns the call's ledger key inside durable runs (and `None` elsewhere). Pass it to APIs that
deduplicate requests (Stripe, many payment and messaging APIs accept an `Idempotency-Key` header) or use it as a
unique key in your own database. Then even a retried call cannot apply twice.

## When the outcome is unknown

A side-effecting call that started but never recorded a result may or may not have happened. Choose what the agent
does with `ReactAgent::unknown_tool_outcome(..)` or `AgentBuilder::unknown_tool_outcome(..)`:

| `UnknownOutcome` | Behavior |
|---|---|
| `Fail` (default) | The model gets `error: the outcome of `charge` is unknown ... Check its effect before calling it again.` It can call a read-only tool to check, or tell the user. |
| `Retry` | Run it again. Safe only when the tool uses `idempotency_key()` with a system that drops duplicates. |
| `Interrupt` | `ReactAgent` only: pause with `unknown_tool_outcome` in the interrupt payload; `resume(thread, true)` runs it again, `false` answers as `Fail`. (In the harness it behaves like `Fail`; use approvals for human review.) |

## Resuming

```rust
// Graphs: run again with no input; the interrupted superstep re-executes.
let out = agent.run("order-42", None).await?;

// Harness: continue the turn that was running when the process stopped.
let out = harness_agent.recover("order-42").await?;
```

Replayed calls are emitted as `GraphEvent::Custom` with a `tool_replayed` field.

## Keys and scope

A ledger key combines the graph namespace, the node, the position of the assistant message that requested the
call, the call id and a hash of the tool name and arguments. So it is stable when the same superstep re-runs, and
different for every new call, even when a local server reuses call ids across turns.

What the ledger does **not** cover: work inside a plain custom node (not a tool), and tools called outside the
prebuilt agents. For those, call `sinew_graph::durable::durable_tool_call` (with a `ToolBox`) or
`durable_run` (any closure) from your own node.

## Custom checkpointers

Implement `supports_tool_ledger`, `get_tool_entry` and `put_tool_entry` on your `Checkpointer`. Without them the
defaults keep today's behavior: no ledger, tools run on every re-execution.
