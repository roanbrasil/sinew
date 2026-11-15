# sinew-scope: trace server and web UI

`sinew-scope` is the LangSmith-style part of Sinew. It is one binary that:

- stores runs from Sinew apps (`HttpSink`) and from any OpenTelemetry GenAI app (OTLP/HTTP JSON);
- serves a query API with filters, pagination and aggregates;
- stores datasets and evaluation experiments from `sinew-eval`;
- embeds a web UI at `/` (plain HTML/CSS/JS, no CDN, works offline);
- still includes the terminal tree viewer for JSONL files.

Storage is a single SQLite file (sqlx, WAL mode).

## Command line

```text
sinew-scope serve  [--db scope.db] [--addr 127.0.0.1:7878] [--token T]
sinew-scope import <traces.jsonl> [--db scope.db] [--project default]
sinew-scope view   <traces.jsonl> [--trace <id-prefix>] [--io]
sinew-scope <traces.jsonl> [--trace <id-prefix>] [--io]     # same as `view`
```

- `serve` binds to `127.0.0.1:7878` by default. The token can also come from
  `SINEW_SCOPE_TOKEN`. Binding a non-loopback address without a token prints a warning.
- `import` loads a JSONL file written by `JsonlSink` into the database. Runs are upserted, so
  importing the same file twice is safe.
- `view` is the original terminal viewer; a bare path keeps working.

## Sending traces from Sinew

Enable the `http` feature of `sinew-trace`:

```toml
sinew-trace = { version = "0.1", features = ["http"] }
```

```rust
use sinew_trace::{HttpSink, HttpSinkConfig, Tracer};

let tracer = Tracer::new().with_sink(HttpSink::new(
    HttpSinkConfig::new("http://127.0.0.1:7878")
        .project("my-agent")
        .token("...")                         // only if the server uses --token
        .flush_interval(std::time::Duration::from_millis(500)),
));
let graph = graph.with_tracer(tracer.clone());
// ... run ...
tracer.flush(); // before exit: delivers the last batch
```

How `HttpSink` behaves:

- `record` only does a `try_send` on a bounded channel (`capacity`, default 10,000). It never
  blocks the app and never panics. When the channel is full, the run is dropped and counted.
- A background thread POSTs batches to `/api/runs` when `batch_size` runs (default 100) are
  queued or `flush_interval` (default 1 s) has passed.
- Network errors and non-2xx responses are counted as dropped (`sink.dropped()`); successful
  deliveries are counted in `sink.sent()`.
- `Tracer::flush()` (or `TraceSink::flush`) blocks the calling thread until the queue is
  delivered, up to `flush_timeout` (default 5 s). In async code, call it from
  `tokio::task::spawn_blocking`. Dropping the last `HttpSink` also flushes.
- `HttpSinkConfig::from_env()` reads `SINEW_SCOPE_URL`, `SINEW_SCOPE_TOKEN` and `SINEW_PROJECT`.

`TraceSink` gained a `flush(&self)` method with an empty default, so existing sinks keep
compiling. `JsonlSink` flushes its file.

## Sending traces with OpenTelemetry

Point any OTLP exporter at `http://127.0.0.1:7878/v1/traces` with protocol `http/json`
(protobuf gets `415`). For example, with the OpenTelemetry SDK environment variables:

```bash
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:7878/v1/traces
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/json
OTEL_EXPORTER_OTLP_HEADERS="x-sinew-project=my-app,authorization=Bearer%20<token>"
```

Spans map to runs using the OpenTelemetry GenAI semantic conventions:

| Run field | Source |
|---|---|
| `id`, `trace_id`, `parent_id` | `spanId`, `traceId`, `parentSpanId` |
| `start_ms`, `end_ms` | `startTimeUnixNano`, `endTimeUnixNano` |
| `kind` | `gen_ai.operation.name` = `chat` / `text_completion` / `generate_content` / `embeddings` → `llm`; `execute_tool` or a `gen_ai.tool.name` → `tool`; a model plus token usage → `llm`; otherwise `chain` |
| `name` | LLM: `gen_ai.response.model`, else `gen_ai.request.model`; tool: `gen_ai.tool.name`; otherwise the span name |
| `input_tokens`, `output_tokens` | `gen_ai.usage.input_tokens` / `output_tokens` (also the older `prompt_tokens` / `completion_tokens`) |
| `inputs` | `gen_ai.input.messages`, `gen_ai.prompt`, `gen_ai.tool.call.arguments` or `input.value` (JSON strings are parsed); older `gen_ai.*.message` span events |
| `outputs` | `gen_ai.output.messages`, `gen_ai.completion`, `gen_ai.tool.call.result` or `output.value`; a `gen_ai.choice` event |
| `error` | status code `ERROR`: the status message, else `error.type`, else `"error"` |
| `metadata` | all span attributes (`attributes`), resource attributes (`resource`), `scope`, `span_kind`, `span_name`, `model`, `otel: true` |

The project is taken from the `X-Sinew-Project` header, else the resource attribute
`sinew.project`, else `service.name`, else `default`.

## HTTP API

All `/api/*` routes except `/api/health`, and `/v1/traces`, require
`Authorization: Bearer <token>` when the server runs with a token. The UI files are public;
the UI asks for the token and keeps it in the browser's local storage.

Errors are JSON: `{"error": "..."}` with a 4xx/5xx status.

### Ingest

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST` | `/api/runs` | `[Run, ...]` or `{"project": "...", "runs": [Run, ...]}` | Upsert by run id. Project: body field, else `X-Sinew-Project` header, else `default`. Returns `{"accepted": n, "traces": m}`. |
| `POST` | `/v1/traces` | OTLP `ExportTraceServiceRequest` (JSON) | Returns `{"partialSuccess": {}}`. |

A `Run` is the JSON form of `sinew_trace::Run`. Children may arrive before their parents; trace
summaries are recomputed on every batch.

### Query

| Method | Path | Returns |
|---|---|---|
| `GET` | `/api/health` | `{"ok": true, "version": "...", "auth": bool}` |
| `GET` | `/api/projects` | `[{project, trace_count, last_start_ms}]` |
| `GET` | `/api/traces` | `{traces, total, limit, offset, stats}` |
| `GET` | `/api/traces/{trace_id}` | `{summary, runs}`; runs are flat, ordered by start time, each with `project` |
| `GET` | `/api/runs/{id}` | one run with `project` |

`/api/traces` query parameters (all optional):

| Parameter | Meaning |
|---|---|
| `project` | exact project |
| `name` | substring of the root run name |
| `kind` | trace contains at least one run of this kind (`graph`, `node`, `llm`, `tool`, `chain`) |
| `errors=true` | only traces with at least one failed run |
| `from_ms`, `to_ms` | trace start time range, Unix milliseconds, inclusive |
| `limit` (default 50, max 1000), `offset` | pagination; newest first |

`stats` covers every trace matching the filters, not just the page: `trace_count`,
`error_count`, `error_rate`, `run_count`, `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`,
`max_latency_ms`, `input_tokens`, `output_tokens`.

A trace summary has `trace_id`, `project`, `root_id`, `name` and `kind` (of the root run),
`start_ms`, `end_ms`, `latency_ms`, token totals, `run_count`, `error_count` and `error` (the root
error, or the first error in the trace). The root is the run without a parent, else the run whose
parent is unknown, else the earliest run.

### Datasets and experiments

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/datasets` | `[{name, description, created_ms, updated_ms, example_count, experiment_count}]` |
| `POST` | `/api/datasets` | a `sinew_eval::Dataset`; creates or replaces it (by name), including its examples |
| `GET` | `/api/datasets/{name}` | `{dataset, experiments}` |
| `GET` | `/api/experiments[?dataset=...]` | `[{id, name, dataset, created_ms, aggregates}]`, newest first |
| `POST` | `/api/experiments` | a `sinew_eval::ExperimentReport`; upsert by id. An unknown dataset is created from the report's examples. |
| `GET` | `/api/experiments/{id}` | the full report |
| `GET` | `/api/compare?a={id}&b={id}` | `{a, b, keys, summary, rows}`: rows aligned by example id with per-key score deltas (`b - a`) |

## Web UI

Open `http://127.0.0.1:7878/`. Pages (hash routes, so links can be shared):

- **Traces** (`#/traces`): filters (name, run kind, time range, errors only), project picker,
  stats cards (trace count, error rate, p50/p95/avg latency, runs, tokens) and a paginated table.
- **Trace** (`#/trace/<id>`): collapsible run tree with a waterfall bar per run on a shared
  timeline, and a detail panel for the selected run: ids, timing, tokens, model, error, inputs,
  outputs and metadata as pretty-printed JSON (chat messages also get a readable "Messages" view).
- **Datasets** (`#/datasets`, `#/dataset/<name>`): examples and the experiments run on them. Tick
  two experiments and press "Compare selected".
- **Experiments** (`#/experiments`, `#/experiment/<id>`): aggregate cards and a per-example table
  with inputs, reference, output, one score column per evaluator (with the reason) and latency;
  a filter shows only failures.
- **Compare** (`#/compare?a=<id>&b=<id>`): two experiments side by side: mean score per evaluator
  with deltas, then each example with both outputs, both scores and the delta; a filter keeps only
  rows that differ.

Light and dark themes follow the system setting; the moon button overrides it.

## Using the server from Rust

`sinew-scope` is also a library, so tests and examples can run the server in-process:

```rust
let store = sinew_scope::Store::memory().await?;           // or Store::open("scope.db")
let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
let url = format!("http://{}", listener.local_addr()?);
tokio::spawn(sinew_scope::server::serve_with_shutdown(
    listener,
    sinew_scope::router(store, None),                      // Some(token) enables auth
    std::future::pending(),
));
```

See `crates/sinew-scope/examples/traced_agent.rs` and `eval_agent.rs`, and the integration tests in
`crates/sinew-scope/tests/server.rs`.

## Security notes

- The server binds to loopback by default. If you expose it, use `--token` and put TLS in front.
- The token check is constant-time over equal-length inputs.
- Request bodies are limited to 64 MiB.
