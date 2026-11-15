# Multi-agent graphs

`sinew_graph::prebuilt` has three building blocks, modelled on LangGraph's
`create_react_agent`, `langgraph-supervisor` and `langgraph-swarm`:

| Type | What it builds |
|---|---|
| `ReactAgent` | one tool-calling agent, with options |
| `Supervisor` | a supervisor model that routes the conversation among worker agents |
| `Swarm` | agents that hand the conversation to each other |

All of them return ordinary `CompiledGraph`s: add checkpointers, stores,
tracers, stream them, or nest them as subgraphs.

## `ReactAgent`

`react_agent(model, tools)` is unchanged. The builder adds options:

```rust
use sinew_graph::prebuilt::{AgentState, Handoff, ReactAgent};

let agent = ReactAgent::new(model, tools)
    .system_prompt("You are a careful analyst.")
    .pre_model_hook(|s: &AgentState| {
        // What the model sees: e.g. only the last 20 messages.
        s.messages.iter().rev().take(20).rev().cloned().collect()
    })
    .post_model_hook(|s: AgentState, ctx| async move {
        // Runs after every model call, before tools: guardrails or a human
        // check with ctx.interrupt(...).
        Ok(s)
    })
    .max_iterations(6)
    .response_format(StructuredOutput::new("report", schema))
    .compile()?;
```

| Option | Effect |
|---|---|
| `system_prompt` | prepended to every model call (not stored in the state) |
| `pre_model_hook` | builds the model's input from the state (trimming, summaries); the state is unchanged |
| `post_model_hook` | a node between the model and the tools; can edit the state or interrupt |
| `max_iterations(n)` | at most `n` model calls per run (per turn, or per handoff inside a supervisor); then it answers `MAX_ITERATIONS_REPLY` |
| `response_format(StructuredOutput)` | after the loop, asks for a final answer in that schema and stores it in `AgentState::structured_response` |
| `handoff(Handoff)` | adds a `transfer_to_<agent>` tool (see swarms) |
| `graph()` | the uncompiled `StateGraph`, to add nodes or policies (retries, timeouts) |

## Supervisor

```rust
use sinew_graph::prebuilt::{OutputMode, Supervisor};

let graph = Supervisor::new(supervisor_model)
    .agent("researcher", "looks up facts", researcher)   // CompiledGraph<AgentState>
    .agent("math", "does arithmetic", math)
    .output_mode(OutputMode::FullHistory)                 // or LastMessage
    .compile()?
    .with_recursion_limit(20);
```

How it runs:

1. The `supervisor` node calls its model with one handoff tool per agent
   (`transfer_to_researcher`, ...) and a default system prompt listing the
   agents (replace it with `.prompt(..)`).
2. A handoff call is answered (`Successfully transferred to researcher`) and
   the agent runs as a subgraph on the shared conversation.
3. When the agent finishes, its messages are added (all of them, or only the
   last one with `OutputMode::LastMessage`), followed by a
   `transfer_back_to_supervisor` exchange, and control returns to the
   supervisor.
4. A supervisor reply without tool calls ends the run.

The supervisor makes one transfer at a time; extra transfer calls in the same
reply are answered as ignored. `create_supervisor(model, vec![NamedAgent::new(..)])`
is the function form.

## Swarm

In a swarm there is no central router: each agent decides when to hand off.

```rust
use sinew_graph::prebuilt::{Handoff, ReactAgent, Swarm, SwarmState};

let alice = ReactAgent::new(m1, sales_tools)
    .handoff(Handoff::to("bob", "Transfer to Bob for billing questions"))
    .compile()?;
let bob = ReactAgent::new(m2, billing_tools)
    .handoff(Handoff::to("alice", "Transfer to Alice for anything else"))
    .compile()?;

let swarm = Swarm::new()
    .agent("alice", alice)
    .agent("bob", bob)
    .default_agent("alice")
    .compile()?
    .with_checkpointer(MemoryCheckpointer::new());

let out = swarm.run("t", Some(SwarmState::new(vec![Message::user("my invoice is wrong")]))).await?;
// out.state().active_agent == Some("bob")
```

When an agent's model calls a handoff tool, the agent answers the call, stops,
and returns a `Command::to_parent()` that moves the conversation to the
target agent (see [subgraphs](subgraphs.md#routing-the-parent-commandto_parent)).
`SwarmState::active_agent` records who spoke last: pass the returned state back
in, with the next user message, and the conversation continues with that
agent.

Handoffs work in any parent graph, not only swarms: a `ReactAgent` with a
`Handoff::to("x", ..)` used as a subgraph routes its parent to node `x`. Used
on its own (as the root graph) a handoff is an error.

## Streaming agents

Worker agents are subgraphs, so their tokens and node events reach the stream
only with subgraph events on:

```rust
let mut events = graph.stream_with(
    "t",
    AgentState::new(vec![Message::user(q)]),
    StreamOptions::new([StreamMode::Messages, StreamMode::Tasks]).with_subgraphs(true),
);
// GraphEvent::Subgraph { namespace: ["researcher:0"], event: Token { .. } }
```

A runnable example is `samples/17-multi-agent`.
