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

## Repetitions, summary evaluators and concurrency: `Experiment`

`evaluate` is the short form. `Experiment` adds every option:

```rust
use sinew_eval::{ClassificationF1, Contains, Experiment, PassRate};

let report = Experiment::new(&dataset)
    .evaluator(Contains::reference())
    .summary_evaluator(PassRate::new("contains", 1.0))   // key `pass_rate:contains`
    .summary_evaluator(ClassificationF1::new("positive")) // key `f1`
    .repetitions(3)      // run every example 3 times to see variance
    .concurrency(4)      // at most 4 target calls in flight, across repetitions
    .name("baseline")
    .metadata(json!({"model": "gpt-4.1-mini"}))
    .run(|inputs: Value| async move { my_target(inputs).await })
    .await;
println!("{:?}", report.summary_score("f1"));
```

- With repetitions, `results` has one entry per example and repetition (`repetition` is 0-based)
  and aggregates cover all of them. `report.repetitions` records the count.
- **Summary evaluators** score the experiment as a whole (`SummaryEvaluator::evaluate(&[ExampleResult])
  -> Score`); results land in `report.summary_scores`. Built-ins: `PassRate::new(key, threshold)`
  (fraction of examples whose `key` score is at least the threshold; errors count as failures),
  `ClassificationF1::new(positive_label)` (precision/recall in the reason) and
  `FnSummary::new(name, |results| Score)`.

Old reports without these fields still deserialize (`repetition` 0, no summary scores).

## Pairwise (A/B) experiments

Absolute scores are noisy for open-ended outputs; asking "which of the two is better?" is often
more reliable. `evaluate_pairwise` takes two experiments on the same dataset, aligns results by
example id and repetition, and runs `PairwiseEvaluator`s on each pair:

```rust
use sinew_eval::{evaluate_pairwise, LlmPairwiseJudge, PairwiseEvaluator};

let judges: Vec<Box<dyn PairwiseEvaluator>> = vec![Box::new(
    LlmPairwiseJudge::new(judge_model, "Which answer is more helpful and correct?")
        .named("helpfulness"),
)];
let pairwise = evaluate_pairwise(&report_a, &report_b, &judges, 4).await;
println!("{}", pairwise.summary_text());
ScopeClient::new(url).upload_pairwise(&pairwise).await?;
```

- `LlmPairwiseJudge` asks for `{"preference": "A" | "B" | "tie", "reason": "..."}` at temperature 0.
  By default it asks **twice with the responses swapped** and reports a tie when the verdict
  follows the position, which cancels position bias; `.swap_positions(false)` asks once.
  `parse_preference` accepts JSON (`preference`, `winner` or `choice`), code fences and bare
  `A` / `B` / `tie`.
- `FnPairwise::new(name, |input| Some(Preference::A))` wraps a closure (heuristics, tests).
- A `PairwiseScore` has `preference` (`a`, `b`, `tie` or `null` when undecided), `score_a` /
  `score_b` (1, 0.5 or 0) and a `reason`.
- `PairwiseReport.summary` has, per key, `wins_a`, `wins_b`, `ties`, `undecided` and
  `win_rate_a` / `win_rate_b` (ties count half; undecided pairs are excluded).
- Examples missing on one side or failed on either side keep an `error` and get no verdicts.

In the UI, pairwise reports are listed on the Experiments page and open at `#/pairwise/<id>`.

## Uploading to sinew-scope

```rust
use sinew_eval::ScopeClient;

let client = ScopeClient::new("http://127.0.0.1:7878");   // .with_token("...") if needed
client.upload_experiment(&dataset, &report).await?;       // dataset (replace by name) + report
client.upload_report(&other_report).await?;               // upsert by report id
```

`ScopeClient::from_env()` reads `SINEW_SCOPE_URL` and `SINEW_SCOPE_TOKEN`. The client also covers
pairwise reports (`upload_pairwise`), feedback (`create_feedback`, `feedback_for_run`), annotation
queues (`create_queue`, `add_to_queue`), appending examples (`add_example`) and generic
`get_json` / `post_json` / `put_json` / `delete` for the rest of the API. Prompts are pulled with
`PromptHub` ([prompt-hub.md](prompt-hub.md)). Then open
`http://127.0.0.1:7878/#/datasets`, pick the dataset, tick two experiments and compare them.
Uploading a report whose dataset the server does not know creates that dataset from the report's
examples.

## Online evaluation

The same evaluators can score production traffic: a `sinew-scope` **rule** samples incoming runs
and stores the scores as feedback. See [monitoring.md](monitoring.md#online-evaluation-rules).

## Examples

`samples/21-prompt-hub-and-feedback` runs a pairwise A/B between two prompt versions with an LLM
judge (offline by default, `--ollama` or `--model provider:model` for a real judge).

`crates/sinew-scope/examples/eval_agent.rs` evaluates two variants of an arithmetic agent (no tools
vs. `react_agent` with a `multiply` tool) over `examples/data/arithmetic.jsonl`, traces every call,
and uploads both experiments:

```bash
cargo run -p sinew-scope --example eval_agent                       # offline, in-process server
cargo run -p sinew-scope --example eval_agent -- --ollama --url http://127.0.0.1:7878
```

With `--ollama` an `LlmJudge` (key `judge`) is added, using the same local model.
