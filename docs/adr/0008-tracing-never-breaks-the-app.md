# 0008. Tracing never breaks the application

- Status: Accepted
- Date: 2026-11-16
- Related: [docs/otel.md](../otel.md), [docs/scope.md](../scope.md)

## Context

Every graph, node, model call and tool call can be traced to files, Sinew Scope or an
OpenTelemetry collector. Those destinations fail: disks fill up, collectors go down, networks are
slow. An agent should not fail or stall because its observability backend is unavailable.

## Considered options

- Sinks return `Result` and the caller decides what to do.
- Sinks swallow their own errors; network delivery happens off the request path.

## Decision

- `TraceSink::record` returns `()` ([crates/sinew-trace/src/lib.rs](../../crates/sinew-trace/src/lib.rs)),
  so a sink cannot propagate an error into the application.
- `JsonlSink` drops the line on I/O error.
- `HttpSink` and `OtlpSink` only `try_send` into a bounded channel; a background thread delivers
  batches and counts runs as sent or dropped when the channel is full or delivery fails
  ([crates/sinew-trace/src/batch.rs](../../crates/sinew-trace/src/batch.rs)). Dropping the sink
  does not hang on a stuck worker.
- A `RunHandle` dropped without `finish` or `fail` is recorded as failed ("run dropped before
  finishing"), so cancelled work still shows up.

## Consequences

- Good: tracing can stay on in production; a broken collector costs dropped runs, not failed
  requests.
- Good: recording a run never waits on the network.
- Bad: trace loss is silent to the caller. Network sinks expose `sent()` and `dropped()`
  counters, but nothing fails loudly; watch those counters if completeness matters.
- Bad: under sustained overload, runs are dropped rather than buffered without bound.
