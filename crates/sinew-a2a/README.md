# sinew-a2a

[Agent2Agent (A2A)](https://a2a-protocol.org) protocol support for [Sinew](https://github.com/roanbrasil/sinew):
serve any Sinew model or agent graph as an A2A agent, call remote A2A agents, and let a local agent
delegate to a remote one as if it were a tool.

It implements A2A **0.3.0** over the JSON-RPC 2.0 binding, with Server-Sent Events for streaming.

## Features

- **Server** (`A2aServer`, axum): Agent Card at `/.well-known/agent-card.json` (and the older
  `/.well-known/agent.json`), plus `message/send`, `message/stream`, `tasks/get`, `tasks/cancel` and
  `tasks/resubscribe`. Tasks go to a swappable `TaskStore` (`InMemoryTaskStore` by default).
- **Executors**: implement `AgentExecutor` yourself, or use an adapter:
  - `ChatModelExecutor` serves any `ChatModel` (OpenAI-compatible, Anthropic, Gemini, Ollama...).
  - `GraphExecutor` serves any `CompiledGraph`, with a ready-made mode for `react_agent`
    (`GraphExecutor::agent`). Graph interrupts become `input-required`, and the client's reply
    resumes the run.
- **Low latency streaming**: model tokens go out as `artifact-update` chunks the moment the model
  produces them. Each event is serialized once and shared by every subscriber. Connections use
  `TCP_NODELAY`. No response is buffered in full.
- **Client** (`A2aClient`): discover a card, `send_message`, `stream_message` (an async `Stream` of
  events), `get_task`, `cancel_task`, `resubscribe`. One pooled `reqwest::Client` per client, shared by
  its clones.
- **Delegation** (`A2aAgentTool`): a remote agent as a Sinew `Tool`. Its name and description come
  from the Agent Card, and it handles `input-required` round trips.
- Spec-exact types (`AgentCard`, `Message`, `Part`, `Task`, `Artifact`, status and artifact update
  events) and the spec's JSON-RPC error codes (`-32001` to `-32007`).

## Quick example

```rust
use std::sync::Arc;
use futures_util::StreamExt;
use sinew_core::{providers::OpenAiCompat, ToolBox};
use sinew_graph::prebuilt::react_agent;
use sinew_a2a::{A2aClient, A2aServer, AgentCard, AgentSkill, GraphExecutor, StreamResponse};

// Serve a react agent.
let model = Arc::new(OpenAiCompat::ollama("qwen3:4b-instruct-2507-q4_K_M"));
let agent = react_agent(model, ToolBox::new())?;
let card = AgentCard::new("Helper", "Answers questions", "")   // empty url: filled in on bind
    .with_skill(AgentSkill::new("qa", "Q&A", "Answers general questions"));
let server = A2aServer::new(card, GraphExecutor::agent(agent))
    .spawn("127.0.0.1:9100")
    .await?;

// Call it from anywhere.
let client = A2aClient::discover("http://127.0.0.1:9100").await?;
let mut events = client.stream_text("What is A2A?").await?;
while let Some(event) = events.next().await {
    if let StreamResponse::ArtifactUpdate(chunk) = event? {
        print!("{}", sinew_a2a::types::parts_text(&chunk.artifact.parts));
    }
}
```

Delegating from a local agent:

```rust
use sinew_a2a::A2aAgentTool;

let remote = A2aAgentTool::discover("http://127.0.0.1:9100").await?;   // tool `helper`
let local = react_agent(local_model, ToolBox::new().with(remote))?;
```

A runnable version is in [`samples/14-a2a`](../../samples/14-a2a/src/main.rs). It works offline,
with `--ollama`, or with `--model provider:model`.

## Not implemented yet

- Push notifications (`tasks/pushNotificationConfig/*` answer `-32003`) and the authenticated extended
  card (`-32007`).
- The gRPC and HTTP+JSON transports, and A2A 1.0, which renamed the methods and dropped the `kind`
  discriminator.

More detail is in [docs/a2a.md](../../docs/a2a.md).

## License

MIT OR Apache-2.0.
