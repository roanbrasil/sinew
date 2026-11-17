# OpenTelemetry

Sinew speaks OTLP in both directions:

- **export**: `sinew_trace::OtlpSink` sends runs as OpenTelemetry spans to any OTLP/HTTP collector
  (an OpenTelemetry Collector, Jaeger, Grafana Tempo, Langfuse, Arize Phoenix, `sinew-scope`...),
  following the OpenTelemetry **GenAI semantic conventions**;
- **ingest**: `sinew-scope` accepts OTLP/HTTP traces from any GenAI-instrumented app (see
  [scope.md](scope.md#sending-traces-with-opentelemetry)).

Spans carry the run's inputs, outputs and error messages. To mask API keys, tokens, e-mail
addresses and card numbers before they leave the process, set a `Redactor` on the `Tracer`;
`OtlpSink` then receives the redacted runs. See [redaction.md](redaction.md).

## Features

```toml
sinew-trace = { version = "0.1", features = ["otlp"] }               # OTLP/HTTP JSON
sinew-trace = { version = "0.1", features = ["otlp", "otlp-proto"] } # + protobuf
sinew-scope = { version = "0.1", features = ["otlp-proto"] }          # accept protobuf on ingest
```

`otlp` reuses the `reqwest` blocking client already used by `HttpSink`. `otlp-proto` is a small
hand-written protobuf codec for the trace messages (no `prost`, no code generation, no extra
dependencies): `sinew_trace::otlp::proto::{encode_request, decode_request}` convert between the
OTLP JSON form and protobuf bytes, so one span mapping serves both encodings. The mapping
functions (`sinew_trace::otlp::{run_to_span, export_request, trace_id_hex, span_id_hex}`) need no
feature.

## OtlpSink

```rust
use sinew_trace::{OtlpProtocol, OtlpSink, OtlpSinkConfig, Tracer};

let tracer = Tracer::new().with_sink(OtlpSink::new(
    OtlpSinkConfig::new("http://localhost:4318/v1/traces")   // full traces endpoint
        .service_name("support-bot")
        .resource_attribute("deployment.environment", "prod")
        .project("support-bot")                               // `sinew.project`, read by sinew-scope
        .protocol(OtlpProtocol::HttpProtobuf)                 // needs `otlp-proto`; default HttpJson
        .capture_content(true),                               // inputs/outputs/messages (default true)
));
let graph = graph.with_tracer(tracer.clone());
// ... run ...
tracer.flush(); // before exit (from async code: tokio::task::spawn_blocking)
```

It behaves like `HttpSink`: `record` never blocks (bounded queue, default 10,000 runs), a
background thread sends batches (default 256 spans or every 2 s), failures are counted
(`sink.sent()`, `sink.dropped()`) and never raised, `flush` waits up to `flush_timeout`. A tracer
can have several sinks, e.g. `HttpSink` to `sinew-scope` and `OtlpSink` to your company's
collector.

`OtlpSinkConfig::from_env()` reads the standard variables:

| Variable | Use |
|---|---|
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | full endpoint; else `OTEL_EXPORTER_OTLP_ENDPOINT` + `/v1/traces`; else `http://localhost:4318/v1/traces` |
| `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TRACES_HEADERS` | `key=value,key2=value2` (values URL-decoded) |
| `OTEL_SERVICE_NAME` | `service.name` |
| `OTEL_RESOURCE_ATTRIBUTES` | extra resource attributes |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` / `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` selects protobuf (with `otlp-proto`) |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false` leaves out inputs, outputs and messages |

Headers: `.header(k, v)`, `.bearer(token)`, `.basic_auth(user, password)`.

## Span mapping

| Run | Span |
|---|---|
| `trace_id` | `traceId`: the UUID's 32 hex digits (Sinew trace ids are UUIDs), else a 128-bit hash |
| `id`, `parent_id` | `spanId`, `parentSpanId`: the UUID folded to 64 bits (high XOR low), else a hash |
| `start_ms`, `end_ms` | `startTimeUnixNano`, `endTimeUnixNano` |
| `error` | `status = {code: ERROR, message}`, `error.type` (text before the first `:`) |
| every run | `sinew.run.kind`, `sinew.run.name`, `sinew.run.id`; each top-level metadata key as `sinew.metadata.<key>` (scalars as is, arrays/objects as JSON strings) |

Per kind, following the GenAI semantic conventions:

| Kind | Span name | Kind | Attributes |
|---|---|---|---|
| `llm` | `chat {model}` | `CLIENT` | `gen_ai.operation.name = chat`, `gen_ai.request.model` (run name), `gen_ai.response.model` (`metadata.model`), `gen_ai.provider.name` (`metadata.provider` if set), `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.input.messages`, `gen_ai.output.messages` |
| `tool` | `execute_tool {tool}` | `INTERNAL` | `gen_ai.operation.name = execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` |
| `graph` | `invoke_agent {name}` | `INTERNAL` | `gen_ai.operation.name = invoke_agent`, `gen_ai.agent.name`, `input.value` / `output.value` |
| `node`, `chain` | run name | `INTERNAL` | `input.value`, `output.value` (+ `*.mime_type = application/json`, as OpenInference / Phoenix expect) |

Messages use the GenAI message schema, serialized as a JSON string (accepted by collectors that do
not support structured attributes):

```json
[{"role": "user", "parts": [{"type": "text", "content": "What is 2 + 2?"}]},
 {"role": "assistant", "parts": [{"type": "tool_call", "id": "c1", "name": "add", "arguments": {"a": 2, "b": 2}}]},
 {"role": "tool", "parts": [{"type": "tool_call_response", "id": "c1", "response": "4"}]}]
```

Output messages carry `finish_reason` (`stop` or `tool_call`). Resource attributes:
`service.name`, `telemetry.sdk.name = sinew`, `telemetry.sdk.language = rust`,
`telemetry.sdk.version`, plus yours. Instrumentation scope: `sinew-trace`.

`sinew-scope` reads the `sinew.*` attributes back, so a run exported with `OtlpSink` and ingested
by `sinew-scope` keeps its kind, name, tags, `ttft_ms` and other metadata; ids become the hex span
ids.

## Backends

**OpenTelemetry Collector / Jaeger / Grafana Tempo**: OTLP/HTTP on port 4318.

```rust
OtlpSinkConfig::new("http://localhost:4318/v1/traces").service_name("my-agent")
```

```bash
docker run --rm -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one   # UI on :16686
```

**Langfuse** (OTLP endpoint, basic auth with the project keys):

```rust
OtlpSinkConfig::new("https://cloud.langfuse.com/api/public/otel/v1/traces")
    .basic_auth(std::env::var("LANGFUSE_PUBLIC_KEY")?, std::env::var("LANGFUSE_SECRET_KEY")?)
```

**Arize Phoenix** (local):

```rust
OtlpSinkConfig::new("http://localhost:6006/v1/traces").service_name("my-agent")
```

**sinew-scope** itself (useful to test the exporter, or behind a collector):

```rust
OtlpSinkConfig::new("http://127.0.0.1:7878/v1/traces").project("my-agent").bearer(token)
```

Check each vendor's documentation for the current endpoint, accepted protocol (JSON or protobuf)
and auth header.

## Ingesting protobuf in sinew-scope

Build or install `sinew-scope` with `--features otlp-proto` and point exporters at `/v1/traces`
with `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` (the SDK default in many languages).
Without the feature, protobuf requests get `415 Unsupported Media Type`; malformed protobuf gets
`400`. `GET /api/health` reports `otlp_protobuf: true` when enabled.
