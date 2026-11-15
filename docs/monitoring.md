# Monitoring, online evaluation and alerts

`sinew-scope` turns stored runs into production monitoring:

- **dashboards**: latency percentiles, time to first token, error rate, tokens and cost over time,
  per trace name and per model, filterable by project, time range, tag and metadata;
- **feedback**: user and reviewer scores on runs (see [scope.md](scope.md#feedback-and-annotation-queues));
- **online evaluation rules**: `sinew-eval` evaluators (including an LLM judge) run on sampled
  incoming runs, in the background, with results stored as feedback;
- **alerts**: thresholds on error rate, latency, time to first token, cost or volume, notified
  through a webhook.

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
