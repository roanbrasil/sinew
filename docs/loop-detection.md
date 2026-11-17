# Loop detection

The most common way a tool-calling agent fails in production is not a crash: it calls the same tool with the
same arguments again and again, or keeps calling tools that fail, until a recursion limit or a token budget
stops it. Every call costs time and money, and the final error tells you nothing about why.

`LoopGuard` catches both patterns right before tools run, per turn (the messages since the last user message):

| Pattern | Default threshold | Check |
|---|---|---|
| Repeated call | 2 identical calls; the 3rd is caught | same tool name and same arguments (object keys compared in any order) |
| Stuck turn | 3 failing tool results in a row | tool results that start with `error: ` |

## Actions

| `LoopAction` | Repeated call | Stuck turn |
|---|---|---|
| `Nudge` (default) | The call does not run. The model gets a tool result explaining it already made this call and should use the earlier result, change the arguments, or answer. | The run stops. |
| `Stop` | The run stops with `LOOP_STOP_REPLY`. | The run stops. |
| `Interrupt` | The run pauses for a human (`ReactAgent`), or the call goes to approval (harness). | Same, for every call of the turn. |

When the run stops, every pending tool call still gets a result (`skipped: ...`) so the thread stays valid for
providers that require one result per call, and the next turn can continue normally.

## ReAct agents

Off by default on the low-level `ReactAgent`, to keep its behavior predictable:

```rust
use sinew::graph::prebuilt::{LoopAction, LoopGuard, ReactAgent};

let agent = ReactAgent::new(model, tools)
    .loop_guard(LoopGuard::default())                       // nudge on the 3rd identical call
    // .loop_guard(LoopGuard::default().on_loop(LoopAction::Stop).max_repeats(1))
    .compile()?;
```

Each detection is emitted as a `GraphEvent::Custom` whose value has a `loop_detected` field (the `LoopCheck`:
repeated calls, consecutive errors, `stuck`), so it shows up in streams and traces.

With `LoopAction::Interrupt` and a checkpointer, the run returns `RunOutcome::Interrupted` with the
`loop_detected` payload. Resume with `true` to run the calls anyway or `false` to stop:

```rust
let out = agent.run("thread-1", Some(state)).await?;
if out.is_interrupted() {
    let out = agent.resume("thread-1", false).await?;      // stop
}
```

## Harness

On by default in `sinew-harness`, because harness agents run autonomously:

- repeated calls are denied with the nudge as the reason (the model sees `permission denied: loop detected ...`);
- a stuck turn ends with `StopReason::Guardrail("loop detected: ...")` in `RunResult::stop`;
- `LoopAction::Interrupt` turns repeated calls into approval requests.

```rust
let agent = sinew::harness::Agent::builder()
    .model_spec("ollama:qwen3:4b")
    .loop_guard(LoopGuard::default().max_repeats(3))   // or .loop_guard(None) to turn it off
    .build()?;
```

## When to change the defaults

- **Polling tools** (job status, queue length) legitimately repeat the same call. Raise `max_repeats`, include
  something that changes in the arguments (an attempt counter), or turn the guard off for that agent.
- **Flaky backends** produce error streaks that are not the model's fault. Raise `max_consecutive_errors`, or
  wrap the tool in retries so the model only sees persistent failures.
- **Cheap, fast models** loop more. `max_repeats(1)` with `Nudge` is a good default for 1–8B local models.

`LoopGuard::check` is public, so custom graphs can run the same check in their own tool nodes.
