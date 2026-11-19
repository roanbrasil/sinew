# Agent2Agent protocol (`sinew-a2a`)

`sinew-a2a` connects Sinew agents to the [Agent2Agent (A2A) protocol](https://a2a-protocol.org). MCP
gives a model tools. A2A lets one agent talk to another agent: it discovers what the other agent can
do, sends it a task, and follows the task's progress.

| Part | What it does |
|---|---|
| `A2aServer` | Serves an `AgentExecutor` over HTTP: the Agent Card plus the JSON-RPC methods, with SSE streaming. |
| `ChatModelExecutor`, `GraphExecutor` | Ready-made executors for any `ChatModel` and any compiled graph, including `react_agent`. |
| `A2aClient` | Calls any A2A agent: card discovery, send, stream, get, cancel, resubscribe. |
| `A2aAgentTool` | A remote agent as a Sinew `Tool`, so a local agent can delegate to it. |

The crate implements A2A **0.3.0** over the JSON-RPC 2.0 binding.

## Serving an agent

```rust
use sinew_a2a::{A2aServer, AgentCard, AgentSkill, GraphExecutor};

let card = AgentCard::new("Currency Agent", "Converts amounts between currencies", "")
    .with_skill(
        AgentSkill::new("convert", "Convert currency", "Converts an amount to other currencies")
            .with_tags(["finance"])
            .with_examples(["How much is 250 USD in EUR?"]),
    );
let executor = GraphExecutor::agent_with_system(react_agent(model, tools)?, Some("Use the tools."));

// In the background on a free port (handy in tests and samples)...
let handle = A2aServer::new(card.clone(), executor).spawn("127.0.0.1:0").await?;
println!("{}", handle.url());

// ...or in the foreground on your own listener.
let listener = tokio::net::TcpListener::bind("0.0.0.0:9100").await?;
A2aServer::new(card, executor2).serve(listener).await?;
```

Routes:

| Route | Purpose |
|---|---|
| `GET /.well-known/agent-card.json` | The Agent Card (also at the pre-0.3 `/.well-known/agent.json`) |
| `POST /` | JSON-RPC: `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`, `tasks/resubscribe` |

An empty card `url` is filled in with the listener address when the server binds. Behind a proxy, set
the public URL yourself. `with_rpc_path("/a2a")` moves the JSON-RPC endpoint, and `router()` returns
the axum `Router`, so you can nest it into a larger app or add auth middleware.

`with_store(Arc<dyn TaskStore>)` replaces the in-memory task store. The server writes a snapshot when
a task is first streamed, on every status change and when a turn ends. It never writes per token.

## Executors

### Any chat model

```rust
let executor = ChatModelExecutor::new(model)
    .with_system("You are terse.")
    .with_max_tokens(512);
```

Each message becomes one `chat_stream` call. Text deltas go out as chunks of one artifact named
`response`, and the task then completes. When a message continues a task, the task's history is sent
as the conversation.

### Graphs and `react_agent`

`GraphExecutor::agent(graph)` covers any `CompiledGraph<AgentState>`:

- **Thread id = A2A `contextId`.** With a checkpointer, a new task in the same context continues the
  previous conversation.
- **Tokens** (`GraphEvent::Token`, which `react_agent` emits while streaming) become artifact chunks.
  Text from successive agent turns is separated by a blank line. A graph that emits no tokens gets its
  final answer (the last assistant message) as the artifact.
- **Interrupts** (`interrupt_before(["tools"])`) become `input-required`. The status message lists the
  pending tool calls, and a data part carries `{"interrupt": {"next": [...]}}`. The client answers with
  a message carrying the same `taskId`. A reply of `yes` (or anything else) resumes the run. A reply of
  `no`, `reject`, `deny`, `cancel` or `stop` cancels the task. Resuming needs a checkpointer.

For any other state type, use `GraphExecutor::new(graph, input, output)`. `input(&ctx, previous_state)`
builds the run input and `output(&state)` renders the answer. `with_resume` decides what a reply to an
interrupt does: edit the state, resume as is, or refuse. `with_interrupt_message` sets the question text.

### Writing your own

```rust
use sinew_a2a::{AgentExecutor, BoxError, EventQueue, RequestContext};

struct Echo;

#[async_trait::async_trait]
impl AgentExecutor for Echo {
    async fn execute(&self, ctx: RequestContext, events: EventQueue) -> Result<(), BoxError> {
        events.working();
        let mut out = events.text_artifact("echo");
        for word in ctx.text().split_inclusive(' ') {
            out.push(word);                    // streamed to the client right away
        }
        out.finish();                          // lastChunk: true
        events.complete(None);
        Ok(())
    }
}
```

`EventQueue` methods never block. The main ones are `working`, `progress(text)`, `artifact(...)`,
`text_artifact(name)`, `input_required(message)`, `complete`, `fail(text)`, `reply(parts)` (a direct
message instead of a task) and `status(state, message)`. If `execute` returns `Ok` without a final
status, the task completes. If it returns `Err`, the task fails and the error text becomes the status
message. If it panics, the task fails with "agent panicked".

## Task lifecycle

```text
message/send ──► submitted ──► working ──► completed | failed | canceled | rejected
                                  │
                                  └──► input-required ──(message with taskId)──► working ...
```

- Each executor runs on its own tokio task. A per-task driver applies the executor's events in order,
  updates the task, and fans each event out to every open stream.
- A client that disconnects does not stop the task. `tasks/resubscribe` reattaches. The Sinew server
  first sends the current task snapshot, which holds every artifact chunk produced so far, and then the
  live events. No event is lost between the snapshot and the subscription.
- `tasks/cancel` aborts the executor future and records `canceled`. Cancelling a finished task returns
  `-32002 TaskNotCancelable`.
- `message/send` waits until the turn ends (a terminal state, `input-required` or `auth-required`).
  With `configuration.blocking: false` it returns as soon as the task exists. `historyLength` trims
  the returned history.
- A message that continues a finished task gets `-32602`. A message that continues a task still
  running gets `-32004`.

The server streams events in this order:

1. a `task` snapshot (`submitted`);
2. `status-update` events (`working`, with an optional progress message);
3. `artifact-update` chunks (`append: false` for the first chunk, `true` after it, and a closing
   `lastChunk: true`);
4. a final `status-update` with `final: true`, after which the server closes the stream.

A direct reply (`EventQueue::reply`) is streamed as a single `message` event, and no task is stored.

## Calling agents

```rust
let client = A2aClient::discover("http://127.0.0.1:9100").await?;   // fetches the card
let answer = client.send_text("Convert 10 USD to EUR").await?.text();

let mut events = client.stream_text("And to BRL?").await?;
while let Some(ev) = events.next().await {
    match ev? {
        StreamResponse::ArtifactUpdate(a) => print!("{}", sinew_a2a::types::parts_text(&a.artifact.parts)),
        StreamResponse::StatusUpdate(s) if s.is_final => println!("\n[{:?}]", s.status.state),
        _ => {}
    }
}

let task = client.get_task(&task_id, Some(10)).await?;   // last 10 history messages
client.cancel_task(&task_id).await?;
let stream = client.resubscribe(&task_id).await?;
```

Other constructors are `A2aClient::new(rpc_url)`, `from_card(card)` and
`with_http_client(reqwest::Client, url)`, which reuses your connection pool, TLS and proxy settings.
`with_bearer_token` and `with_header` add auth headers. `request(method, params)` sends any JSON-RPC
call. Errors are `A2aError::Rpc` (with the spec code), `Status` (HTTP), `Http`, `Json` or `Protocol`.

## Delegating from a local agent

```rust
let remote = A2aAgentTool::discover("http://127.0.0.1:9100").await?;
let agent = react_agent(local_model, ToolBox::new().with(remote))?;
```

- The tool name is the card name made tool-safe (`Currency Agent` becomes `currency_agent`). The
  description is the card description plus its skills. `with_name` and `with_description` override
  them.
- Arguments: `{"input": "...", "task_id"?: "..."}`. Calls share one A2A `contextId`, so the remote
  agent sees one conversation. `with_shared_context(false)` gives each call a fresh context.
- When the remote agent asks for more input, the tool returns
  `The remote agent needs more input (task_id: ...): <question>`, and the model answers by calling the
  tool again with that `task_id`.
- A remote task that fails, is canceled or is rejected becomes `Err(Error::Tool)`, so the model sees an
  error message and can recover.

## Signed cards and verified callers

An agent can publish its signed identity in its card and require one from its callers (see
[Agent identity](governance.md#agent-identity) for the documents, keys and trust store):

```rust
use sinew_core::identity::{AgentCredentials, IdentityVerifier, TrustStore};

// Server: the identity in the card, the card signed (again after the URL is filled in),
// callers verified.
let server = A2aServer::new(AgentCard::new("Billing", "Refunds", "").with_identity(&billing_identity), executor)
    .with_card_signing_key(billing_key)
    .with_identity_verifier(IdentityVerifier::new(trust.clone()))
    .spawn("0.0.0.0:9100")
    .await?;

// Client: only talk to agents whose card verifies, and prove who we are on every call.
let client = A2aClient::discover_verified("https://billing.internal", &trust)
    .await?
    .with_credentials(AgentCredentials::new(orchestrator_key, orchestrator_identity)?);
let tool = A2aAgentTool::from_client(client);
```

- The card signature is a JWS (`alg: EdDSA`, `kid` = key id) over the card's canonical JSON without
  `signatures`, in the A2A `signatures` field. The identity is the capability extension
  `IDENTITY_EXTENSION_URI` (`params.identity`).
- Each request carries a call assertion of the current `CallContext` in the params'
  `metadata["sinew.dev/identity"]`, signed by the caller's key for the receiving agent (the card's
  `agent_id()`, or `with_audience`). Missing, forged, expired, misdirected or replayed proofs are
  answered with `-32040`, on `message/*` and `tasks/*` alike.
- The executor runs in the caller's context with this agent appended to the chain
  (`current_context()`), and gets the caller in `RequestContext::caller`. An on-behalf-of credential
  sent by the caller (`metadata["sinew.dev/credential"]`) becomes `user_credential()` and the
  context's subject token; it is removed from the metadata the executor sees.
- A traced tool call into the remote agent and the remote graph's runs share one trace, and every
  tool run records the delegation chain.

## Protocol notes

- Version 0.3.0: `kind` discriminators, `message/send`-style method names, and the card at
  `/.well-known/agent-card.json`. Incoming objects that omit `kind` are accepted. Outgoing objects
  always carry it.
- A2A 1.0 (`SendMessage`-style names, no `kind`) and the gRPC and HTTP+JSON transports are not
  implemented.
- Push notifications are not supported. The card says `pushNotifications: false`, and the
  `tasks/pushNotificationConfig/*` methods answer `-32003`. `agent/getAuthenticatedExtendedCard`
  answers `-32007`.
- Batches are rejected with `-32600`. Malformed JSON gets `-32700`.
- If the card sets `streaming: false`, `message/stream` and `tasks/resubscribe` answer `-32004`.
