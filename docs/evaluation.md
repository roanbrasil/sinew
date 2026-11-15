# sinew-eval: datasets, evaluators and experiments

`sinew-eval` runs an async target (a model call, a chain, a whole graph) over a dataset, scores
every output with one or more evaluators, and produces an `ExperimentReport`. Reports can be
uploaded to `sinew-scope serve` to browse them and compare experiments side by side.

## Datasets

A `Dataset` is a name, a description and a list of `Example`s. An example has `inputs` (any JSON),
an optional `reference` (the expected output), optional `metadata`, and an `id` (generated when
missing).

```rust
use serde_json::json;
use sinew_eval::{Dataset, Example};

let mut ds = Dataset::new("capitals").with_description("country → capital");
ds.push(json!({"country": "France"}), Some(json!("Paris")));
let ds = ds.with_example(Example::new(json!({"country": "Japan"})).with_reference(json!("Tokyo")));

ds.save_jsonl("capitals.jsonl")?;
let ds = Dataset::load_jsonl("capitals.jsonl")?;   // named after the file stem
```

JSONL format, one example per line:

```json
{"id": "fr", "inputs": {"country": "France"}, "reference": "Paris"}
{"inputs": {"country": "Japan"}, "reference": "Tokyo"}
```

A malformed line fails with `EvalError::Parse { line, .. }`.

## Evaluators

```rust
#[async_trait]
pub trait Evaluator: Send + Sync {
    fn name(&self) -> &str;                                  // score key / column name
    async fn evaluate(&self, input: EvalInput<'_>) -> Score; // inputs, output, reference
}
```

A `Score` has a `key`, a `score` in `[0, 1]` (clamped; non-finite becomes 0) and an optional
`reason`. Evaluators should not fail: when they cannot score, they return 0 with a reason
("example has no reference output", "no number found in output", ...).

| Evaluator | Key | Passes when |
|---|---|---|
| `ExactMatch::new()` / `::strict()` | `exact_match` | output equals the reference (strings trimmed unless strict) |
| `Contains::new("x")` / `Contains::reference()` (+ `.ignore_case()`) | `contains` | output text contains the needle or the reference text |
| `RegexMatch::new(pattern)?` | `regex` | output text matches the regex |
| `JsonFieldEquals::new("a.b")` / `::expecting("/a/b", value)` | `json_field:<path>` | the field of the output equals the same field of the reference, or the given value. Paths are dotted (`items.0.name`) or JSON pointers. String outputs are parsed as JSON first. |
| `NumericTolerance::absolute(t)` / `::relative(t)` (+ `.at(path)`) | `numeric` | `|output − reference| ≤ t` (or `t·|reference|`). Numbers inside text are extracted (the last one wins). |
| `LlmJudge::new(model, rubric)` (+ `.named(key)`) | `llm_judge` | graded by any `ChatModel` |

"Output text" means the string itself for JSON strings, and compact JSON otherwise.

### LLM as judge

`LlmJudge` sends the rubric, the inputs, the reference and the response to the model at
temperature 0 and asks for `{"score": <0..1>, "reason": "..."}`. The reply parser tolerates prose
and code fences around the JSON, string scores, and 0–10 or 0–100 scales (scaled down). An
unparseable reply or a model error gives score 0 with the reason.

```rust
use std::sync::Arc;
use sinew_core::providers::OpenAiCompat;
use sinew_eval::LlmJudge;

let judge = LlmJudge::new(
    Arc::new(OpenAiCompat::ollama("qwen3:4b-instruct-2507-q4_K_M")),
    "Score 1 if the response is correct and concise, 0.5 if correct but verbose, 0 if wrong.",
)
.named("correctness");
```

## Running an experiment

```rust
pub async fn evaluate<F, Fut, E>(
    target: F,                          // Fn(Value) -> Future<Output = Result<Value, E>>, E: Display
    dataset: &Dataset,
    evaluators: &[Box<dyn Evaluator>],
    concurrency: usize,                 // max examples in flight
) -> ExperimentReport
```

- At most `concurrency` target calls run at once; results keep the dataset order.
- The target's latency is measured per example.
- A target error is stored on the example (`error`), its evaluators are skipped, and the run
  continues.

```rust
let evaluators: Vec<Box<dyn Evaluator>> = vec![
    Box::new(NumericTolerance::absolute(1e-6)),
    Box::new(Contains::reference()),
];
let report = evaluate(
    |inputs: Value| {
        let agent = agent.clone();
        async move { my_agent(agent, inputs).await.map_err(|e| e.to_string()) }
    },
    &dataset,
    &evaluators,
    4,
)
.await
.with_name("react-agent v2")
.with_metadata(json!({"model": "qwen3:4b"}));

println!("{}", report.summary());
assert!(report.mean("numeric").unwrap() > 0.9);
```

`ExperimentReport` fields: `id`, `name` (default `<dataset>-<timestamp>`), `dataset`, `created_ms`,
`metadata`, `results` (one `ExampleResult` per example: `example_id`, `index`, `inputs`,
`reference`, `output`, `error`, `latency_ms`, `scores`) and `aggregates` (`count`, `errors`,
`mean_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, `max_latency_ms`, and per score key `mean`,
`min`, `max`, `count`).

To trace the target too, give your graph a `Tracer` with an `HttpSink` (see
[scope.md](scope.md)); every example becomes a trace.

## Uploading to sinew-scope

```rust
use sinew_eval::ScopeClient;

let client = ScopeClient::new("http://127.0.0.1:7878");   // .with_token("...") if needed
client.upload_experiment(&dataset, &report).await?;       // dataset (replace by name) + report
client.upload_report(&other_report).await?;               // upsert by report id
```

`ScopeClient::from_env()` reads `SINEW_SCOPE_URL` and `SINEW_SCOPE_TOKEN`. Then open
`http://127.0.0.1:7878/#/datasets`, pick the dataset, tick two experiments and compare them.
Uploading a report whose dataset the server does not know creates that dataset from the report's
examples.

## Example

`crates/sinew-scope/examples/eval_agent.rs` evaluates two variants of an arithmetic agent (no tools
vs. `react_agent` with a `multiply` tool) over `examples/data/arithmetic.jsonl`, traces every call,
and uploads both experiments:

```bash
cargo run -p sinew-scope --example eval_agent                       # offline, in-process server
cargo run -p sinew-scope --example eval_agent -- --ollama --url http://127.0.0.1:7878
```

With `--ollama` an `LlmJudge` (key `judge`) is added, using the same local model.
