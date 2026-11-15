# sinew-scope: trace server and web UI

`sinew-scope` is the LangSmith-style part of Sinew. It is one binary that:

- stores runs from Sinew apps (`HttpSink`) and from any OpenTelemetry GenAI app (OTLP/HTTP JSON,
  and protobuf with the `otlp-proto` feature);
- computes the cost of every LLM run from its token usage and a pricing table;
- serves a query API with filters, full-text search, cursor pagination and aggregates;
- stores datasets, evaluation experiments and pairwise comparisons from `sinew-eval`;
- hosts a **prompt hub** (versioned prompts with tags), see [prompt-hub.md](prompt-hub.md);
- collects **feedback** on runs and runs **annotation queues** for human review;
- runs **online evaluators** on incoming runs and **alerts** to webhooks, see
  [monitoring.md](monitoring.md);
- deletes old data after a configurable **retention** period;
- embeds a web UI at `/` (plain HTML/CSS/JS, no CDN, works offline);
- still includes the terminal tree viewer for JSONL files.

Storage is a single SQLite file (sqlx, WAL mode). Older databases are migrated forward on open
(see [Schema and migrations](#schema-and-migrations)).

## Command line

```text
sinew-scope serve  [--db scope.db] [--addr 127.0.0.1:7878] [--token T]
                   [--pricing prices.json] [--retention-days N] [--no-engine]
sinew-scope import <traces.jsonl> [--db scope.db] [--project default]
sinew-scope view   <traces.jsonl> [--trace <id-prefix>] [--io]
sinew-scope <traces.jsonl> [--trace <id-prefix>] [--io]     # same as `view`
```

- `serve` binds to `127.0.0.1:7878` by default. The token can also come from
  `SINEW_SCOPE_TOKEN`. Binding a non-loopback address without a token prints a warning.
- `--pricing` replaces the built-in model price table with a JSON file (`{"models": [...]}` or a
  bare array, see [Cost](#cost)); the table is stored in the database, so it survives restarts.
- `--retention-days` sets the retention period (also settable in the UI and API).
- `--no-engine` disables the background engine (online rules, alerts, retention).
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

Point any OTLP exporter at `http://127.0.0.1:7878/v1/traces` with protocol `http/json`. With the
`otlp-proto` feature (`cargo install sinew-scope --features otlp-proto`, or
`cargo run -p sinew-scope --features otlp-proto -- serve`) `http/protobuf` is accepted too;
without it protobuf gets `415`. For example, with the OpenTelemetry SDK environment variables:

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
| `kind` | `sinew.run.kind` if present (spans from `OtlpSink`); else `gen_ai.operation.name` = `chat` / `text_completion` / `generate_content` / `embeddings` → `llm`; `execute_tool` or a `gen_ai.tool.name` → `tool`; `invoke_agent` / `create_agent` → `graph`; a model plus token usage → `llm`; otherwise `chain` |
| `name` | `sinew.run.name` if present; LLM: `gen_ai.response.model`, else `gen_ai.request.model`; tool: `gen_ai.tool.name`; agent: `gen_ai.agent.name`; otherwise the span name |
| `input_tokens`, `output_tokens` | `gen_ai.usage.input_tokens` / `output_tokens` (also the older `prompt_tokens` / `completion_tokens`) |
| `inputs` | `gen_ai.input.messages`, `gen_ai.prompt`, `gen_ai.tool.call.arguments` or `input.value` (JSON strings are parsed); older `gen_ai.*.message` span events |
| `outputs` | `gen_ai.output.messages`, `gen_ai.completion`, `gen_ai.tool.call.result` or `output.value`; a `gen_ai.choice` event |
| `error` | status code `ERROR`: the status message, else `error.type`, else `"error"` |
| `metadata` | all span attributes (`attributes`), resource attributes (`resource`), `scope`, `span_kind`, `span_name`, `model`, `otel: true`; `sinew.metadata.<key>` attributes become top-level keys again (so `tags`, `ttft_ms`... survive an OTLP round trip) |

The project is taken from the `X-Sinew-Project` header, else the resource attribute
`sinew.project`, else `service.name`, else `default`.

To send Sinew traces to *another* OpenTelemetry backend (Jaeger, Tempo, Langfuse, Phoenix), use
`sinew_trace::OtlpSink`; see [otel.md](otel.md).

## HTTP API

All `/api/*` routes except `/api/health`, and `/v1/traces`, require
`Authorization: Bearer <token>` when the server runs with a token. The UI files are public;
the UI asks for the token and keeps it in the browser's local storage.

Errors are JSON: `{"error": "..."}` with a 4xx/5xx status.

### Ingest

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST` | `/api/runs` | `[Run, ...]` or `{"project": "...", "runs": [Run, ...]}` | Upsert by run id. Project: body field, else `X-Sinew-Project` header, else `default`. Returns `{"accepted": n, "traces": m}`. |
| `POST` | `/v1/traces` | OTLP `ExportTraceServiceRequest` (JSON; protobuf with `otlp-proto`) | Returns `{"partialSuccess": {}}`. |

A `Run` is the JSON form of `sinew_trace::Run`. Children may arrive before their parents; trace
summaries are recomputed on every batch.

### Query

| Method | Path | Returns |
|---|---|---|
| `GET` | `/api/health` | `{"ok": true, "version": "...", "auth": bool, "fts": bool, "engine": bool, "otlp_protobuf": bool}` |
| `GET` | `/api/projects` | `[{project, trace_count, last_start_ms}]` |
| `GET` | `/api/traces` | `{traces, total, limit, offset, stats}` |
| `GET` | `/api/traces/{trace_id}` | `{summary, runs}`; runs are flat, ordered by start time, each with `project` |
| `GET` | `/api/runs/{id}` | one run with `project`, `model`, `cost_usd`, `ttft_ms` |
| `GET` | `/api/runs` | run search: `{runs, next_cursor}` (see below) |
| `GET` | `/api/metrics` | monitoring aggregates, see [monitoring.md](monitoring.md) |

`/api/traces` query parameters (all optional):

| Parameter | Meaning |
|---|---|
| `project` | exact project |
| `name` | substring of the root run name |
| `kind` | trace contains at least one run of this kind (`graph`, `node`, `llm`, `tool`, `chain`) |
| `errors=true` | only traces with at least one failed run |
| `from_ms`, `to_ms` | trace start time range, Unix milliseconds, inclusive |
| `q` | full-text search: every word must appear (prefix match) in the name, inputs, outputs or error of at least one run of the trace |
| `tag` | the root run's `metadata.tags` contains this tag |
| `meta.<key>=<value>` | the root run's top-level `metadata.<key>` equals the value (repeatable; `true`/`false` match JSON booleans) |
| `limit` (default 50, max 1000), `offset` | pagination; newest first |

`stats` covers every trace matching the filters, not just the page: `trace_count`,
`error_count`, `error_rate`, `run_count`, `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`,
`p99_latency_ms`, `max_latency_ms`, `input_tokens`, `output_tokens`, `cost_usd`.

A trace summary has `trace_id`, `project`, `root_id`, `name` and `kind` (of the root run),
`start_ms`, `end_ms`, `latency_ms`, token totals, `cost_usd`, `run_count`, `error_count` and `error`
(the root error, or the first error in the trace).

#### Run search

`GET /api/runs` searches individual runs, newest first, with keyset pagination: pass the
`next_cursor` of a page as `cursor` to get the next one (deep pages cost the same as the first).

| Parameter | Meaning |
|---|---|
| `q` | full text over name, inputs, outputs and error (FTS5; `LIKE` fallback) |
| `project`, `kind`, `trace_id` | exact |
| `name` | substring of the run name |
| `model` | model prefix (`gpt-4o` matches `gpt-4o-mini`) |
| `tag`, `meta.<key>=<value>` | on the run's own metadata |
| `errors=true`, `root=true` | failed runs; root runs only |
| `from_ms`, `to_ms`, `min_latency_ms` | time range, minimum latency |
| `feedback_key` | runs that have feedback with this key |
| `limit` (default 50, max 500), `cursor` | pagination |

Each result has ids, `project`, `name`, `kind`, times, `latency_ms`, `error`, tokens, `model`,
`cost_usd`, `ttft_ms`, `tags` and 300-character `input_preview` / `output_preview`. The root is the run without a parent, else the run whose
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
| `GET` | `/api/compare?a={id}&b={id}` | `{a, b, keys, summary, rows}`: rows aligned by example id and repetition with per-key score deltas (`b - a`) |
| `POST` | `/api/datasets/{name}/examples` | append one example: an `Example`, or `{"run_id": "...", "reference"?: ...}` to copy a run (inputs, and outputs as the reference); creates the dataset if needed |
| `GET` | `/api/pairwise[?dataset=...]` | pairwise comparisons `[{id, name, dataset, a_id, b_id, created_ms, summary}]` |
| `POST` | `/api/pairwise` | a `sinew_eval::PairwiseReport` (upsert by id) |
| `GET` | `/api/pairwise/{id}` | the full report |

### Feedback and annotation queues

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/feedback` | `{run_id, key, score?, value?, comment?, source?}`; at least one of score/value/comment. Returns the stored entry (`id`, `trace_id`, `created_ms`, `source` defaults to `api`). |
| `GET` | `/api/feedback?run_id=&trace_id=&key=` | entries, oldest first |
| `GET` | `/api/runs/{id}/feedback` | entries of one run |
| `GET` | `/api/feedback/stats?project=&from_ms=&to_ms=` | `[{key, count, scored, mean}]` |
| `DELETE` | `/api/feedback/{id}` | |
| `GET`, `POST` | `/api/queues` | list `[{id, name, description, keys, pending, done}]`; create `{name, description, keys}` |
| `GET`, `DELETE` | `/api/queues/{id}` | `{queue, pending, done, items}` (`?status=pending` or `done`, `&limit=`) |
| `POST` | `/api/queues/{id}/items` | `{"run_ids": [...]}` (unknown runs are skipped); returns `{added}` |
| `GET` | `/api/queues/{id}/next` | oldest pending item with its run and existing feedback, or `item: null` |
| `POST` | `/api/queues/{id}/items/{run_id}/done` | `{"feedback": [{key, score?, comment?}]}`; stored with source `annotation:<queue name>` |
| `DELETE` | `/api/queues/{id}/items/{run_id}` | remove from the queue |

From Rust: `ScopeClient::create_feedback(&Feedback::score(run_id, "thumbs", 1.0))`,
`feedback_for_run`, `create_queue`, `add_to_queue`, `add_example`, `upload_pairwise`.

### Prompt hub, monitoring, rules, alerts, settings

| Method | Path | See |
|---|---|---|
| `GET`, `POST` | `/api/prompts` | [prompt-hub.md](prompt-hub.md) |
| `GET`, `DELETE` | `/api/prompts/{name}` (`{name}@{ref}` resolves a version) | [prompt-hub.md](prompt-hub.md) |
| `GET` | `/api/prompts/{name}/{ref}`, `/api/prompts/{name}/diff?from=&to=` | [prompt-hub.md](prompt-hub.md) |
| `PUT`, `DELETE` | `/api/prompts/{name}/tags/{tag}` | [prompt-hub.md](prompt-hub.md) |
| `GET` | `/api/metrics` | [monitoring.md](monitoring.md) |
| `GET`, `POST`, `GET`/`PUT`/`DELETE` `{id}`, `POST {id}/run` | `/api/rules` | [monitoring.md](monitoring.md) |
| `GET`, `POST`, `GET`/`PUT`/`DELETE` `{id}`, `POST {id}/check` | `/api/alerts` | [monitoring.md](monitoring.md) |
| `GET`, `PUT`, `DELETE` | `/api/pricing` | [Cost](#cost) |
| `POST` | `/api/pricing/recompute` | [Cost](#cost) |
| `GET`, `PUT` | `/api/settings` | `{"retention_days": N}` (or `null`) |
| `POST` | `/api/retention/run` | apply retention now; returns `{traces, runs, feedback}` deleted |

## Cost

Every run is priced at ingest and stored with `model`, `cost_usd` and `ttft_ms`; trace summaries
sum the cost of their runs.

- **Model**: `metadata.model` (set by `traced_chat`), else the OTLP `gen_ai.response.model` /
  `gen_ai.request.model` attribute, else the run name for `llm` runs.
- **Cost**: `metadata.cost_usd` if the app set one; else, for `llm` runs with token usage,
  `input_tokens × input price + output_tokens × output price` from the pricing table.
- **Time to first token**: `metadata.ttft_ms`, set by `traced_chat_stream`.

The built-in table (`sinew_scope::Pricing::builtin()`) lists list prices in USD per million tokens
for common OpenAI, Anthropic, Google Gemini, xAI, Mistral and DeepSeek models. Lookup lowercases
the model, strips a `provider:` or `provider/` prefix and takes the **longest matching prefix**,
so `gpt-4o-mini-2024-07-18` uses the `gpt-4o-mini` entry and `claude-sonnet-4-6` uses
`claude-sonnet-4`. Unknown models (local Ollama models, for example) get no cost.

Prices change: override the table with `--pricing file.json`, the Settings page, or
`PUT /api/pricing`:

```json
{"models": [
  {"model": "gpt-4.1-mini", "input_per_mtok": 0.40, "output_per_mtok": 1.60, "provider": "openai"},
  {"model": "qwen3", "input_per_mtok": 0, "output_per_mtok": 0}
]}
```

A new table is stored in the database and every stored run is re-priced. `DELETE /api/pricing`
restores the built-in table; `POST /api/pricing/recompute` re-prices without changing it.

## Web UI

Open `http://127.0.0.1:7878/`. Pages (hash routes, so links can be shared):

- **Monitor** (`#/monitor`): dashboards over a time range (1 h to 30 days), filterable by trace
  name, tag and root-run metadata: latency p50/p95/p99, error rate, trace count, tokens, cost and
  LLM time to first token over time; tables per trace name and per model; feedback means.
- **Traces** (`#/traces`): filters (full-text search, name, tag, run kind, time range, errors
  only), project picker, stats cards (trace count, error rate, p50/p95/p99 latency, runs, tokens,
  cost) and a paginated table with a cost column.
- **Trace** (`#/trace/<id>`): collapsible run tree with a waterfall bar per run on a shared
  timeline, and a detail panel for the selected run: ids, timing, tokens, model, cost, time to
  first token, tags, error, **feedback** (list, add a score/comment, delete), **actions** (add the
  run to a dataset, add it to an annotation queue), inputs, outputs and metadata as pretty-printed
  JSON (chat messages also get a readable "Messages" view).
- **Runs** (`#/runs`): run search (full text, kind, model, tag, metadata, has-feedback, errors,
  root only) with "load older" cursor paging; tick runs and add them to an annotation queue.
- **Datasets** (`#/datasets`, `#/dataset/<name>`): examples and the experiments run on them. Tick
  two experiments and press "Compare selected".
- **Experiments** (`#/experiments`, `#/experiment/<id>`): aggregate cards and a per-example table
  with inputs, reference, output, one score column per evaluator (with the reason) and latency;
  a filter shows only failures.
- **Compare** (`#/compare?a=<id>&b=<id>`): two experiments side by side: mean score per evaluator
  with deltas, then each example with both outputs, both scores and the delta; a filter keeps only
  rows that differ.
- **Pairwise** (`#/pairwise/<id>`): win/tie counts and win rates per pairwise evaluator, and every
  example with both outputs and the verdict with its reason (listed on the Experiments page).
- **Prompts** (`#/prompts`, `#/prompt/<name>`): the prompt hub: versions with tags, template,
  variables and model config, "edit as new version", tag management and a line diff between any
  two versions.
- **Annotate** (`#/queues`, `#/queue/<id>`): annotation queues; the review screen shows the run's
  inputs, outputs and existing feedback, a score (0 / 0.5 / 1 or any number) and a comment per
  queue key, "Submit & next" and "Remove from queue".
- **Rules & alerts** (`#/rules`): online evaluation rules (enable/disable, run on the latest 100
  runs) and threshold alerts with webhooks ("Check now").
- **Settings** (`#/settings`): retention period, pricing table (view, edit as JSON, reset,
  recompute) and server capabilities.

Light and dark themes follow the system setting; the moon button overrides it. The accent is the
Sinew orange (`#C2410C` light, `#EA7A3A` dark) and the logo is served at `/logo.png`.

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

`router` starts the background engine (rules, alerts, retention) when it runs inside a Tokio
runtime. `router_with` takes a `ServerConfig { token, engine: Option<EngineConfig>,
disable_engine }`; `EngineConfig` sets the alert and retention intervals, the webhook timeout and
the `model_factory` that turns an LLM-judge `provider:model` spec into a `ChatModel` (default:
`sinew_core::providers::from_spec`; tests inject a scripted model there).

The `Store` is usable directly as well: `search_runs`, `metrics`, `push_prompt`,
`resolve_prompt`, `add_feedback`, `set_pricing`, `purge_before`, ...

## Schema and migrations

The schema version is kept in `PRAGMA user_version` (`sinew_scope::store::SCHEMA_VERSION`, now 3).
`Store::open` upgrades older files in place, each step in a transaction and idempotent:

| Version | Changes |
|---|---|
| 1 | `runs`, `traces`, `datasets`, `examples`, `experiments` (databases created before versioning report 0) |
| 2 | `runs.model`, `runs.cost_usd`, `runs.ttft_ms`, `traces.cost_usd`; tables `settings`, `prompts`, `prompt_versions`, `prompt_tags`, `feedback`, `annotation_queues`, `queue_items`, `rules`, `alerts`, `pairwise`; indexes for search and monitoring (`runs(start_ms)`, `runs(project, start_ms)`, `runs(model, start_ms)`, `runs(name, start_ms)`, `runs(parent_id)`, `traces(name, start_ms)`, feedback and queue indexes). Model, cost and time to first token are backfilled for existing runs. |
| 3 | `runs_fts`: an FTS5 index over run name, inputs, outputs and error (external content, kept in sync by triggers, existing rows indexed). If the SQLite build has no FTS5, this step is skipped and search falls back to `LIKE`. |

## Retention

Set `retention_days` (CLI `--retention-days`, Settings page, or `PUT /api/settings`). Every hour
the engine deletes traces that started before the cutoff, with their runs, feedback and queue
items; `POST /api/retention/run` does it immediately. Without a value nothing is deleted. Deleted
space is reused by SQLite; run `VACUUM` yourself if you need the file to shrink (then
`INSERT INTO runs_fts(runs_fts) VALUES ('rebuild')`, because VACUUM may renumber rowids).

See `crates/sinew-scope/examples/traced_agent.rs` and `eval_agent.rs`, and the integration tests in
`crates/sinew-scope/tests/server.rs`.

## Security notes

- The server binds to loopback by default. If you expose it, use `--token` and put TLS in front.
- The token check is constant-time over equal-length inputs.
- Request bodies are limited to 64 MiB.
- Alert webhooks are called from the server; only `http(s)` URLs are accepted. Anyone with the
  token can create alerts, so treat the token as an admin credential.
- LLM-judge rules use the provider API keys found in the server's environment.
