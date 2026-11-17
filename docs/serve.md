# sinew-serve: deploying graphs and agents

`sinew-serve` puts compiled Sinew graphs behind an HTTP API. It plays the role
LangGraph Platform / LangServe play in Python: assistants, durable threads,
runs you can wait for, stream or start in the background, plus an
OpenAI-compatible endpoint so existing OpenAI clients can talk to your agents.

It is a library. You build a `Server`, register graphs, and serve it on a
listener, nest its router inside your own axum app, or spawn it in the
background for tests. `samples/20-serve` is a complete example main.

```rust
use sinew_serve::{Assistant, Server, SqliteStore};

let server = Server::builder()
    .store(SqliteStore::connect("sqlite://serve.db").await?)
    .assistant(Assistant::agent("weather", react_agent(model.clone(), tools)?)
        .description("Answers weather questions"))
    .assistant(Assistant::graph("email", email_graph.interrupt_before(["send"])))
    .chat_model("qwen", model)          // a plain model on /v1/chat/completions
    .api_key(std::env::var("SINEW_SERVE_KEY")?)
    .build();
server.serve_addr("0.0.0.0:8123".parse()?).await?;
```

## Concepts

| Concept | What it is |
|---|---|
| Assistant | A registered graph with an id, a name, a description and free-form `config`/`metadata`. Its state type only has to be `Serialize + DeserializeOwned` (the `sinew_graph::State` bound). |
| Thread | A durable conversation: a record (status, metadata, owning assistant) plus the graph's checkpoints. |
| Run | One execution of an assistant, on a thread (stateful) or without one (stateless). |

### Registering graphs

- `Assistant::graph(id, compiled)` wraps any `CompiledGraph<S>`. A run's
  `input` is shallow-merged over the thread's current state (top-level keys
  replace) and decoded into `S`. Override with
  `.with_input(|prev: Option<S>, input: Value| -> Result<S, String>)`.
- `Assistant::agent(id, compiled)` is for `CompiledGraph<AgentState>`
  (for example `react_agent`). Input `{"messages": [...]}`, a bare array of
  messages or a plain string is **appended** to the conversation, and the
  agent shows up in `/v1/models`.
- `.with_chat(to_state, reply)` exposes any other graph through the OpenAI
  façade.

Register graphs **without** a checkpointer: the server attaches its own store
to the copy it uses for threads, and keeps a checkpoint-free copy for
stateless runs and the OpenAI façade. `interrupt_before` and the recursion
limit set on the compiled graph are kept; runs can override both.

### Storage

| Store | Feature | Notes |
|---|---|---|
| `MemoryStore` | always | Default. Lost on exit. |
| `SqliteStore::connect(url)` | `sqlite` (default) | Built on `SqliteCheckpointer`; adds table `sinew_serve_threads` next to `sinew_checkpoints`. |
| `PostgresStore::connect(url)` | `postgres` | Same schema in Postgres; several replicas can share it. |

Implement `ThreadStore` to plug in anything else.

Runs themselves live in memory (the most recent 10 000 finished runs are
kept for polling); their effects are durable through the checkpoints and the
thread status.

## HTTP API

All bodies are JSON. Timestamps are Unix epoch milliseconds. Errors are
`{"error": "message"}` with a meaningful status code (400, 401, 404, 409, 503).

### System

| Method and path | |
|---|---|
| `GET /ok`, `GET /health` | Liveness, always public. |
| `GET /ready` | Store reachable and not shutting down (503 otherwise). Public. |
| `GET /info` | Version, assistant ids, chat model ids. |
| `GET /metrics` | Prometheus metrics; public unless `.metrics_auth(true)`. See [monitoring.md](monitoring.md#prometheus-metrics). |

### Assistants

| Method and path | |
|---|---|
| `GET /assistants?limit&offset` | List. |
| `POST /assistants/search` | Same, body `{"limit", "offset"}`. |
| `GET /assistants/{id}` | One assistant with its `config`, `metadata` and whether it is `chat`-capable. |

### Threads

| Method and path | |
|---|---|
| `POST /threads` | Create. Body (all optional): `thread_id`, `assistant_id`, `metadata`, `if_exists: "raise" \| "do_nothing"`. |
| `GET /threads?limit&offset&assistant_id&status`, `POST /threads/search` | List, most recently updated first. |
| `GET /threads/{tid}` | Record with `status`: `idle`, `busy`, `interrupted`, `error`. |
| `PATCH /threads/{tid}` | Merge `{"metadata": {...}}`. |
| `DELETE /threads/{tid}` | Cancels its runs, deletes the record and every checkpoint (204). |
| `GET /threads/{tid}/state` | `{"thread_id", "values", "next", "step"}`. `values` is `null` before the first run. |
| `POST /threads/{tid}/state` | Body `{"values": {...}}`, shallow-merged into the latest checkpoint and validated against the graph's state type. On a fresh thread it seeds the state. 409 while a run is in flight. |
| `GET /threads/{tid}/history?limit=10` | Checkpoints, newest first, same shape as `state`. |

### Runs

The same body creates every kind of run:

```json
{
  "assistant_id": "email",
  "input": {"to": "Ana"},
  "command": {"update": {"body": "Edited by a human"}},
  "stream_mode": ["values", "updates", "tokens", "events", "custom"],
  "multitask_strategy": "reject",
  "on_disconnect": "cancel",
  "webhook": "https://hooks.example.com/sinew",
  "metadata": {"user": "ana"},
  "recursion_limit": 50,
  "interrupt_before": ["send"],
  "if_not_exists": "create"
}
```

Only `assistant_id` is required. Without `input`, a thread run **resumes** its
latest checkpoint (that is how interrupted runs continue); `command.update`
edits the paused state first. A run on an unknown thread creates it unless
`if_not_exists` is `"reject"`. A thread belongs to the assistant of its first
run; another assistant gets 409.

| Method and path | |
|---|---|
| `POST /threads/{tid}/runs` | Background run, returns the `Run` at once (`pending` or `running`). |
| `POST /threads/{tid}/runs/wait` | Returns the finished `Run`. |
| `POST /threads/{tid}/runs/stream` | SSE stream of the run. |
| `GET /threads/{tid}/runs` | Runs of the thread, newest first. |
| `GET /threads/{tid}/runs/{rid}` | Poll status. |
| `POST /threads/{tid}/runs/{rid}/cancel?wait=true` | Cancel (checkpoints written so far stay). |
| `GET /threads/{tid}/runs/{rid}/join` | Wait for it, return the `Run`. |
| `GET /threads/{tid}/runs/{rid}/stream` | Attach to its SSE stream from now on (a finished run replays its terminal events). |
| `POST /runs`, `/runs/wait`, `/runs/stream` | Stateless versions: no thread, no checkpoints, `input` required. |
| `GET /runs/{rid}`, `POST /runs/{rid}/cancel`, `GET /runs/{rid}/join`, `GET /runs/{rid}/stream` | Same as above, by run id only. |

A `Run` is:

```json
{"run_id": "…", "thread_id": "t-1", "assistant_id": "email",
 "status": "interrupted", "created_at": 1763220000000, "updated_at": 1763220000020,
 "multitask_strategy": "reject", "metadata": {},
 "output": {"to": "Ana", "body": "…", "sent": false},
 "error": null, "interrupted_before": ["send"]}
```

`status` is `pending`, `running`, `success`, `error`, `interrupted` or
`cancelled`. `/wait` answers 200 for all of them: look at `status`.

#### Concurrency on a thread (`multitask_strategy`)

| Strategy | When the thread already has a run |
|---|---|
| `reject` (default) | 409 Conflict. |
| `enqueue` | The run waits (`pending`) and starts when the previous ones finish, in order, on their final state. |
| `interrupt` | Running and queued runs are cancelled, then this one starts. |

Runs on one thread never execute at the same time.

#### SSE events

Every stream starts with `metadata` and ends with `end`:

| `event:` | `data:` | When |
|---|---|---|
| `metadata` | `{"run_id", "thread_id"}` | First. |
| `values` | the full state | After every node (`values`, the default mode). |
| `updates` | `{"<node>": {changed top-level keys}}` | After every node (`updates`). |
| `tokens` | `{"node", "delta", "step", "task"}` | Each model token (`tokens`, alias `messages`). |
| `events` | `{"event": "node_start" \| "node_end", "node", "step", "task"}` | `events`. |
| `custom` | `{"node", "value", "step", "task"}` | `NodeContext::emit` (`custom`). |
| `interrupt` | `{"next", "values"}` | The run paused. |
| `error` | `{"error"}` | The run failed. |
| `end` | `{"run_id", "status"}` | Last. |

Idle streams get a `: ping` comment every 15 s (`ServerBuilder::keep_alive`).
When a client disconnects from `/runs/stream`, the run is cancelled unless it
was created with `"on_disconnect": "continue"`. Joining clients never cancel.

With fan-out, `values` and `updates` carry each branch's own output (the
graph emits branch outputs before the reducer merges them).

#### Webhooks

When a run ends, the final `Run` is POSTed to its `webhook`. Webhooks are off
by default (a URL supplied by API callers is an SSRF risk); allow them with
`.webhooks(WebhookPolicy::Prefixes(vec!["https://hooks.example.com/".into()]))`
or `WebhookPolicy::Any`. Delivery is best effort, 10 s timeout, no retries.

### OpenAI-compatible façade

| Method and path | |
|---|---|
| `GET /v1/models` | Chat-capable assistants and registered chat models. |
| `POST /v1/chat/completions` | `model`, `messages` (string or text-part content, `system`/`developer`/`user`/`assistant` with `tool_calls`/`tool`), `stream`, `stream_options.include_usage`, `temperature`, `max_tokens`/`max_completion_tokens`, `tools`. |

- For an **agent**, the request messages become the initial state and the
  agent runs its whole loop (tools included) statelessly; the reply is the last
  assistant message. Streaming forwards every model token as a
  `chat.completion.chunk`. Usage is reported as zero.
- For a **chat model**, the request is forwarded as is (tools included) and
  tool calls come back in OpenAI format with real usage.

Any OpenAI client works, including Sinew's own provider:

```rust
let client = OpenAiCompat::new("http://127.0.0.1:8123/v1", "weather").with_api_key(key);
```

Errors use the OpenAI shape: `{"error": {"message", "type", "code"}}`.

## Typed client

```rust
use sinew_serve::{RunCreate, ServeClient, StreamMode, StreamPart};

let client = ServeClient::new("http://127.0.0.1:8123").with_api_key(key);
let run = client.wait(Some("t-1"), &RunCreate::new("email", json!({"to": "Ana"}))).await?;
if run.status == RunStatus::Interrupted {
    client.update_state("t-1", json!({"body": "Approved."})).await?;
    client.wait(Some("t-1"), &RunCreate::resume("email")).await?;
}

let mut s = client
    .stream(None, &RunCreate::new("weather", json!("Hi")).with_stream_mode([StreamMode::Tokens]))
    .await?;
while let Some(part) = s.next().await {
    if let StreamPart::Token { delta, .. } = part? { print!("{delta}"); }
}
```

Also: `assistants`, `assistant`, `create_thread`, `thread`, `threads`,
`delete_thread`, `state`, `update_state`, `history`, `create_run`, `run`,
`thread_runs`, `cancel_run`, `join_run`, `join_stream`, `models`,
`chat_completion`. `ThreadState::values_as::<S>()` and `Run::output_as::<S>()`
decode into your state type.

## Operations

- **Auth**: `.api_key(k)` (repeatable) requires `Authorization: Bearer k` or
  `x-api-key: k` on everything but `/ok`, `/health`, `/ready`, `/metrics`
  (unless `.metrics_auth(true)`) and CORS preflights. Keys are compared in
  constant time.
- **CORS**: `.cors(CorsConfig::permissive())` or `CorsConfig::origins([...])`.
- **Tracing**: `.tracer(tracer)` records one `chain` run per request
  (`POST /threads/{tid}/runs/wait`, with `http.status` and `latency_us`
  metadata) and attaches the tracer to every graph, so graph, node, model and
  tool runs are recorded too. Use an `HttpSink` to ship them to Sinew Scope.
- **Metrics**: `.metrics(MetricsSink::new())` aggregates the same runs into
  Prometheus counters and histograms at `GET /metrics`, next to queue and
  stream gauges; see [monitoring.md](monitoring.md#prometheus-metrics) and
  the Grafana dashboard in `deploy/grafana/sinew.json`.
- **Shutdown**: `serve(listener, signal)` / `serve_addr(addr)` (Ctrl-C). On the
  signal the server stops accepting, new runs get 503, running runs have
  `shutdown_grace` (10 s) to finish, the rest are cancelled, and the call
  returns once connections close.
- **Embedding**: `server.router()` is a plain `axum::Router`.

## Performance

- Each SSE frame is serialized once into a shared arena (`BytesMut`) and
  handed to every subscriber as the same reference-counted `Bytes`; no
  per-subscriber copies and no per-event `String`s. Frames already queued are
  coalesced into one write.
- State is serialized only for the stream modes that need it; a
  `tokens`-only stream never serializes the state until the end.
- The OpenAI chunk prefix (id, model, created) is rendered once per request;
  per token only the escaped delta is written.
- Sockets use `TCP_NODELAY`; HTTP/1.1 keep-alive is on.
- `sinew_serve::runtime(workers)` builds a multi-threaded runtime with named
  workers for a dedicated binary.

`cargo bench -p sinew-serve --bench ttft` measures time-to-first-token of a
react agent over an instant 32-token model: directly through
`CompiledGraph::stream` versus through the server (raw SSE, `ServeClient`,
OpenAI streaming), plus the full `/runs/wait` round trip. The difference is
the server's overhead; see `crates/sinew-serve/README.md` for numbers.

## Limitations

- Run records are in memory: polling a run after a restart returns 404 (its
  thread state survives).
- `values`/`updates` after a fan-out superstep carry branch outputs, not the
  merged state; read `/state` for the merged one.
- `command.resume` values are only honoured when they are objects (merged
  like `update`): Sinew graphs pause before nodes, they do not wait for a
  value inside a node.
- No assistant versioning or per-assistant configuration at run time.
