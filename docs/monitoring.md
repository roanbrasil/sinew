# Monitoring, online evaluation and alerts

`sinew-scope` turns stored runs into production monitoring:

- **dashboards**: latency percentiles, time to first token, error rate, tokens and cost over time,
  per trace name and per model, filterable by project, time range, tag and metadata;
- **feedback**: user and reviewer scores on runs (see [scope.md](scope.md#feedback-and-annotation-queues));
- **online evaluation rules**: `sinew-eval` evaluators (including an LLM judge) run on sampled
  incoming runs, in the background, with results stored as feedback;
- **alerts**: thresholds on error rate, latency, time to first token, cost or volume, notified
  through a webhook;
- **Prometheus**: `GET /metrics` on `sinew-serve` and `sinew-scope`, with a Grafana dashboard (see
  [Prometheus metrics](#prometheus-metrics)).

## Metrics API

`GET /api/metrics` aggregates either traces or runs:

| Parameter | Meaning |
|---|---|
| `level` | `trace` (default: one point per trace; latency is the trace's, tokens and cost are trace totals), a run kind (`llm`, `tool`, `node`, `graph`, `chain`) or `run` (every run) |
| `group_by` | `name` (default), `model`, `project` or `kind` (traces only have `name`, `project`, `kind`) |
| `project`, `from_ms`, `to_ms` | scope and time range (by start time) |
| `bucket_ms` | time bucket width; default: the range split in about 60 buckets (minimum 1 s) |
| `name` | substring of the trace or run name |
| `model` | model prefix (run levels) |
| `tag`, `meta.<key>=<value>` | root-run tag / metadata for `level=trace`, the run's own for run levels |

Response:

```json
{
  "level": "trace", "from_ms": 1731600000000, "to_ms": 1731686400000, "bucket_ms": 1440000,
  "totals":  {"key": "all", "count": 1200, "errors": 18, "error_rate": 0.015,
              "avg_latency_ms": 840.2, "p50_latency_ms": 610, "p95_latency_ms": 2100, "p99_latency_ms": 4800,
              "ttft_p50_ms": null, "ttft_p95_ms": null,
              "input_tokens": 1830000, "output_tokens": 240000, "cost_usd": 1.92},
  "buckets": [{"start_ms": 1731600000000, "count": 20, "...": "same fields"}],
  "groups":  [{"key": "support-bot", "count": 900, "...": "same fields"}],
  "truncated": false
}
```

Percentiles use the nearest-rank method over the matching points. At most 200,000 points (the
newest) are aggregated per query; `truncated` says when the cap was hit. Queries use the
`runs(start_ms)`, `runs(project, start_ms)`, `runs(model, start_ms)` and `traces(project, start_ms)`
indexes.

Time to first token comes from `metadata.ttft_ms`, which `sinew_trace::traced_chat_stream` records
(the time from the request to the first text or tool-call delta). Use `level=llm&group_by=model`
to compare models.

`GET /api/feedback/stats?project=&from_ms=&to_ms=` returns `[{key, count, scored, mean}]` for the
same window.

### Dashboard

`#/monitor` shows, for the chosen time range (last hour to 30 days, or all time) and filters:

- cards: traces, error rate, latency p50/p95/p99, LLM TTFT p50/p95, tokens, cost;
- charts (inline SVG, no external libraries): trace latency p50/p95/p99, error rate, traces,
  input/output tokens, cost, LLM time to first token;
- tables: per trace name and per model (count, error rate, p50/p95/p99, TTFT, tokens, cost);
- feedback means per key.

The project picker in the top bar scopes the dashboard like every other page.

## Prometheus metrics

`sinew-serve` and `sinew-scope` expose `GET /metrics` in the Prometheus text exposition format
(0.0.4), for alerting and dashboards in an existing Prometheus / Grafana stack. The `/api/metrics`
API above answers ad-hoc questions over stored runs; `/metrics` gives counters and histograms that
Prometheus scrapes and keeps.

Both endpoints are built on `sinew_trace::metrics::MetricsSink`, a `TraceSink` that aggregates
every finished run. It has no dependencies beyond the standard library and can be used in any
application:

```rust
use std::sync::Arc;
use sinew_trace::{MetricsSink, Tracer};

let metrics = Arc::new(MetricsSink::new().project("support-bot"));
let tracer = Tracer::new().with_shared_sink(metrics.clone());
// ... run graphs with `tracer`, then serve `metrics.render()` with
// Content-Type `sinew_trace::metrics::CONTENT_TYPE`.
```

### Run metrics

| Metric | Type | Labels |
|---|---|---|
| `sinew_runs_total` | counter | `project`, `kind`, `name`, `status` |
| `sinew_run_duration_seconds` | histogram | `project`, `kind`, `name` |
| `sinew_errors_total` | counter | `project`, `kind`, `name` |
| `sinew_tool_calls_total` | counter | `project`, `tool`, `status` |
| `sinew_llm_tokens_total` | counter | `project`, `model`, `direction` (`input`, `output`, `cached`) |
| `sinew_llm_cost_usd_total` | counter | `project`, `model` |
| `sinew_llm_time_to_first_token_seconds` | histogram | `project`, `model` |
| `sinew_metrics_label_values_folded_total` | counter | none |

- `kind` is the run kind (`graph`, `node`, `llm`, `tool`, `chain`); `status` is `ok` or `error`.
- `name` is the graph, node, tool or chain name; for `llm` runs it is the model
  (`metadata.model`, else the run name), the same value as `model` on the LLM metrics.
- `project` is the ingest project in `sinew-scope`, else `metadata.project`, else the sink's
  default (`MetricsSink::project`, `default` if unset).
- `cached` tokens are prompt tokens read from the provider's cache, a subset of `input`.
  `traced_chat` and `traced_chat_stream` record them as `metadata.cached_input_tokens`.
- Cost: `metadata.cost_usd` if present, else the token usage priced with the `sinew-core` price
  table (`sinew_core::pricing`). `sinew-scope` uses its own pricing table instead, so the value
  matches the UI. Unknown models produce no cost series.
- Time to first token comes from `metadata.ttft_ms`, recorded for streamed calls only.
- Duration buckets (seconds): 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120,
  300. Time-to-first-token buckets: 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10.

**Cardinality.** Each label (`project`, `model`, `tool`, and `name` per kind) keeps at most 100
distinct values (`MetricsSink::max_label_values`); later values are reported as `other` and counted
in `sinew_metrics_label_values_folded_total`. Values are truncated to 128 characters. Run names
should therefore be stable identifiers (node and tool names, routes), not user input.

### sinew-serve

```rust
use sinew_serve::Server;
use sinew_trace::MetricsSink;

let server = Server::builder()
    // .assistant(...)
    .metrics(MetricsSink::new())
    .build();
```

`ServerBuilder::metrics` adds the sink to the server's tracer (creating one if `.tracer()` was
not called), so it sees graph, node, LLM and tool runs plus one `chain` run per HTTP request
(`name` is `METHOD /route`; 5xx responses count as errors). `GET /metrics` is always routed and also
reports, with or without a sink:

| Metric | Type | Meaning |
|---|---|---|
| `sinew_serve_queued_runs` | gauge | runs waiting for an earlier run on their thread |
| `sinew_serve_running_runs` | gauge | runs executing |
| `sinew_serve_active_streams` | gauge | open SSE responses |
| `sinew_serve_assistants` | gauge | registered assistants |
| `sinew_serve_build_info` | gauge | `1`, with a `version` label |

Scrapes of `/metrics` are not traced.

### sinew-scope

`sinew-scope serve` aggregates every run accepted by `POST /api/runs` and `POST /v1/traces` with the
labels above (unfinished runs are skipped; a re-ingested run is counted again) and adds:

| Metric | Type | Meaning |
|---|---|---|
| `sinew_scope_ingested_runs_total` | counter | runs accepted, by `endpoint` (`runs`, `otlp`) |
| `sinew_scope_store_runs` | gauge | runs in the database |
| `sinew_scope_store_traces` | gauge | traces in the database |
| `sinew_scope_store_size_bytes` | gauge | SQLite page count times page size (WAL excluded) |
| `sinew_scope_build_info` | gauge | `1`, with a `version` label |

The counters start at zero when the server starts; the store gauges cover all stored data.

### Authentication

`/metrics` does not require the API key or bearer token by default, because scrapers usually run
inside the same network. It exposes counts, durations and the names of graphs, nodes, tools,
models and projects, but no inputs or outputs. To require the same credential as the rest of the
API, use `ServerBuilder::metrics_auth(true)` in `sinew-serve`, and `--metrics-auth` (or
`ServerConfig::metrics_auth`) in `sinew-scope`. Without configured keys / token the endpoint stays
open either way.

### Scrape configuration

```yaml
scrape_configs:
  - job_name: sinew-serve
    scrape_interval: 15s
    static_configs:
      - targets: ["agents.internal:8000"]
  - job_name: sinew-scope
    scrape_interval: 30s
    static_configs:
      - targets: ["scope.internal:7878"]
    # With --metrics-auth:
    # authorization:
    #   type: Bearer
    #   credentials_file: /etc/prometheus/sinew-scope-token
```

If an application traces to both a local `MetricsSink` and `sinew-scope` (through `HttpSink`),
the same runs appear in both jobs; filter by `job` in queries to avoid counting them twice.

Example queries:

```promql
# p95 graph latency over 5 minutes
histogram_quantile(0.95, sum by (le) (rate(sinew_run_duration_seconds_bucket{kind="graph"}[5m])))
# time to first token p95 by model
histogram_quantile(0.95, sum by (le, model) (rate(sinew_llm_time_to_first_token_seconds_bucket[5m])))
# cost per hour by model
sum by (model) (rate(sinew_llm_cost_usd_total[1h])) * 3600
# tool error ratio
sum by (tool) (rate(sinew_tool_calls_total{status="error"}[5m]))
  / sum by (tool) (rate(sinew_tool_calls_total[5m]))
```

### Grafana dashboard

[`deploy/grafana/sinew.json`](../deploy/grafana/sinew.json) charts run rate and error rate by kind,
graph and LLM latency p50/p95/p99, time to first token, tokens per second, cost per hour, error rate
by model and by tool, node and tool latency, HTTP latency per route, the serve gauges, and the
scope ingest and store gauges. It has `datasource`, `project`, `graph` and `model` variables.

Import it with **Dashboards → New → Import → Upload JSON file** and pick a Prometheus data source,
or provision it:

```yaml
# /etc/grafana/provisioning/dashboards/sinew.yaml
apiVersion: 1
providers:
  - name: sinew
    type: file
    options:
      path: /var/lib/grafana/dashboards/sinew   # copy sinew.json here
```

## Online evaluation rules

A rule watches incoming runs (`POST /api/runs` and `/v1/traces`). After each ingest batch is
stored, it is queued for the background engine (a bounded queue: if it is full, the batch is not
evaluated; ingest never waits). For each enabled rule:

1. the run must be finished and match the rule's filter: `project`, `kind`, exact `name`, `tag`,
   `errors_only`, `root_only`, `metadata` (top-level equality);
2. it is sampled at `sample_rate` (0..1), deterministically by rule and run id, so re-ingesting a
   run does not change the decision;
3. every evaluator scores the run: inputs = run inputs, output = run outputs (`null` for failed
   runs), reference = `metadata.reference` if the app set one;
4. each score is stored as feedback with key = evaluator key, the reason as comment and source
   `rule:<rule name>`.

```bash
curl -s localhost:7878/api/rules -H 'content-type: application/json' -d '{
  "name": "answer quality",
  "filter": {"project": "support-bot", "kind": "chain", "root_only": true},
  "sample_rate": 0.2,
  "evaluators": [
    {"type": "not_empty"},
    {"type": "contains", "needle": "sorry", "ignore_case": true, "key": "apologizes"},
    {"type": "llm_judge", "model": "openai:gpt-4.1-mini",
     "rubric": "Score 1 if the answer is correct, polite and specific.", "key": "quality"}
  ]
}'
```

| `type` | Fields | Scores |
|---|---|---|
| `exact_match` | `strict` | output equals `metadata.reference` |
| `contains` | `needle` (else the reference), `ignore_case` | output contains the text |
| `regex` | `pattern` | output matches |
| `json_field` | `path`, `expected` (else the reference's field) | field equality |
| `numeric` | `tolerance`, `relative`, `path` | number within tolerance of the reference |
| `llm_judge` | `model` (`provider:model`, as `sinew_core::providers::from_spec`), `rubric` | `sinew_eval::LlmJudge` |
| `not_empty` | | output is not empty |
| `is_json` | | output (or the message content) parses as a JSON object or array |

Every spec takes an optional `key` to rename the score. Specs are validated when the rule is saved
(unknown types, bad regexes and missing fields give `400`). LLM judges read the provider's API key
from the server environment (`OPENAI_API_KEY`, ...); embedded servers can pass their own
`EngineConfig::model_factory`.

| Method | Path | Notes |
|---|---|---|
| `GET`, `POST` | `/api/rules` | list; create `{name, enabled, filter, sample_rate, evaluators}` |
| `GET`, `PUT`, `DELETE` | `/api/rules/{id}` | |
| `POST` | `/api/rules/{id}/run?limit=100` | score the latest matching runs now (backfill; sampling still applies) |

Each rule tracks `runs_evaluated` (feedback entries written), `last_run_ms` and `last_error`.
Rule feedback appears in the trace view, the run search (`feedback_key=`), the dashboard's feedback
table and the annotation review screen.

## Alerts

An alert computes a metric over a sliding window of traces (or runs, with `filter.level`) every 30
seconds (`EngineConfig::alert_interval`). When the value is **greater than** the threshold and the
cooldown has passed, it POSTs to the webhook:

```json
{
  "text": "[sinew-scope] alert `errors`: error_rate = 0.5000 > 0.25 over the last 15 min (4 points)",
  "alert": {"id": "...", "name": "errors", "metric": "error_rate", "threshold": 0.25, "window_ms": 900000},
  "value": 0.5, "status": "firing", "count": 4,
  "filter": {"project": "svc"}, "fired_ms": 1731686400000
}
```

The `text` field makes it work with Slack-compatible incoming webhooks as is.

| Field | Meaning |
|---|---|
| `metric` | `error_rate`, `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, `avg_latency_ms`, `ttft_p95_ms`, `cost_usd`, `count` |
| `threshold` | fires when the metric is above it |
| `window_ms` | sliding window (default 15 min) |
| `filter` | `project`, `name` (substring), `tag`, `metadata`, `level` (`trace` default, or a run kind) |
| `webhook_url` | `http(s)` URL |
| `cooldown_ms` | minimum time between notifications (default 1 h) |

| Method | Path | Notes |
|---|---|---|
| `GET`, `POST` | `/api/alerts` | list; create |
| `GET`, `PUT`, `DELETE` | `/api/alerts/{id}` | |
| `POST` | `/api/alerts/{id}/check` | evaluate now; returns `{id, value, status, fired, error}` |

`status` is `ok`, `firing` or `no_data` (no points in the window). The last check is stored on the
alert (`last_checked_ms`, `last_value`, `last_status`, `last_fired_ms`, `last_error`, e.g. a webhook
that answered 500).

## Retention

See [scope.md](scope.md#retention): `retention_days` deletes old traces with their runs and
feedback every hour.

## In the UI

- `#/monitor`: dashboards;
- `#/rules`: rules (enable/disable, "Run on latest 100", delete) and alerts ("Check now"), with
  creation forms;
- `#/settings`: retention and pricing.
