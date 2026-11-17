# sinew-serve

Low-latency HTTP server for [Sinew](https://github.com/roanbrasil/sinew)
graphs and agents: the Rust counterpart of LangGraph Platform / LangServe.

- **Assistants**: register any `CompiledGraph<S>` (`S: Serialize + DeserializeOwned`).
- **Threads**: durable state in SQLite (default), Postgres (`postgres` feature)
  or memory; read, edit and page through checkpoint history.
- **Runs**: wait, stream over SSE (`values`, `updates`, `tokens`, `events`,
  `custom`), or run in the background and poll, join or cancel. Interrupted
  threads resume with a new run, optionally editing the state first.
  Per-thread multitask strategy: `reject`, `enqueue`, `interrupt`. Optional
  completion webhooks.
- **OpenAI façade**: `POST /v1/chat/completions` (streaming or not) and
  `GET /v1/models` over agents and plain chat models.
- **Operations**: bearer API keys, CORS, `/ok` `/health` `/ready`, graceful
  shutdown, request tracing into `sinew-trace` / Sinew Scope.
- **`ServeClient`**: typed Rust client.

```rust
use std::sync::Arc;
use serde_json::json;
use sinew_core::{mock::ScriptedModel, Message, ToolBox};
use sinew_graph::prebuilt::react_agent;
use sinew_serve::{Assistant, RunCreate, ServeClient, Server};

let model = Arc::new(ScriptedModel::new([Message::assistant("Hello!")]));
let server = Server::builder()
    .assistant(Assistant::agent("agent", react_agent(model, ToolBox::new())?))
    .api_key("sk-local")
    .build()
    .spawn("127.0.0.1:0")          // or .serve_addr(addr) / .serve(listener, signal)
    .await?;

let client = ServeClient::new(server.base_url()).with_api_key("sk-local");
let run = client.wait(Some("thread-1"), &RunCreate::new("agent", json!("hi"))).await?;
println!("{:?} {:?}", run.status, run.output);
server.shutdown().await?;
```

Any OpenAI client can call the agent:

```bash
curl http://127.0.0.1:8123/v1/chat/completions \
  -H "Authorization: Bearer sk-local" -H "content-type: application/json" \
  -d '{"model": "agent", "stream": true, "messages": [{"role": "user", "content": "hi"}]}'
```

## Endpoints

| | |
|---|---|
| System | `GET /ok`, `/health`, `/ready`, `/info` |
| Assistants | `GET /assistants`, `POST /assistants/search`, `GET /assistants/{id}` |
| Threads | `POST /threads`, `GET /threads`, `POST /threads/search`, `GET/PATCH/DELETE /threads/{tid}`, `GET/POST /threads/{tid}/state`, `GET /threads/{tid}/history` |
| Thread runs | `POST /threads/{tid}/runs` (background), `/runs/wait`, `/runs/stream` (SSE), `GET /threads/{tid}/runs`, `GET /threads/{tid}/runs/{rid}`, `POST …/cancel`, `GET …/join`, `GET …/stream` |
| Stateless runs | `POST /runs`, `/runs/wait`, `/runs/stream`, `GET /runs/{rid}`, `POST /runs/{rid}/cancel`, `GET /runs/{rid}/join`, `GET /runs/{rid}/stream` |
| OpenAI | `GET /v1/models`, `POST /v1/chat/completions` |

Full reference (bodies, SSE events, statuses): [docs/serve.md](https://github.com/roanbrasil/sinew/blob/main/docs/serve.md).

## Features

| Feature | Default | Adds |
|---|---|---|
| `sqlite` | yes | `SqliteStore` |
| `postgres` | no | `PostgresStore` |

## Performance

SSE frames are serialized once into a shared arena and fanned out as
reference-counted `Bytes`; queued frames are coalesced into one write; state
is serialized only for the stream modes that need it; OpenAI chunks reuse a
pre-rendered prefix; sockets use `TCP_NODELAY`.

`cargo bench -p sinew-serve --bench ttft` (react agent over an instant model
streaming 32 tokens, 2000 iterations, loopback, release build, on a
shared 8-core machine under heavy load, so the tails are noisy):

| Path | TTFT p50 | Overhead vs direct |
|---|---|---|
| `CompiledGraph::stream` (direct) | 4.6 µs | |
| `POST /v1/chat/completions` (stream) | 142 µs | +138 µs |
| `POST /runs/stream`, raw reqwest | 231 µs | +226 µs |
| `ServeClient::stream` | 820 µs | +815 µs |
| `POST /runs/wait` (full reply, 32 tokens) | 149 µs | |

The server adds a fraction of a millisecond per request, well below the
time-to-first-token of any real model.

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

MIT OR Apache-2.0
