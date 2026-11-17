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

## Evaluating agent trajectories

For an agent, the answer is not the whole story: the right tools, called in the right order,
without wasted calls, matter too. The `trajectory` module scores that path.

### `Trajectory`

A `Trajectory` holds the ordered `tool_calls` (each a `ToolStep` with `name`, `arguments` and, when
known, `result` or `error`), `model_calls`, `input_tokens` / `output_tokens` and `cost_usd` (`None`
when unknown) and the `final_answer`. Build one:

- **From messages**: `Trajectory::from_messages(&state.messages)` for `AgentState.messages` or the
  harness thread messages. Every assistant message counts as one model call; tool results are paired
  by call id (results starting with `error: ` become `error`). The final answer is the last assistant
  message when it has no tool calls. `Trajectory::from_last_turn` starts after the last user message,
  for multi-turn threads. Messages carry no usage; add it with `.with_usage(input, output)` and
  `.with_cost(usd)`.
- **From a trace**: `Trajectory::from_runs(&runs)` over the runs of one trace (for example
  `MemorySink::runs()`). `tool` runs become steps, `llm` runs are model calls with their tokens. The
  cost is `metadata.cost_usd` or the built-in price table applied to the tokens and the model, and
  stays `None` if any model call has no known cost.
- **By hand**: `Trajectory::new().tool("search", json!({"q": "rust"})).with_final_answer("...")`.

```rust
use sinew_eval::Trajectory;

let state = agent.invoke(AgentState::new(vec![Message::user("What is 37 times 3?")])).await?;
let trajectory = Trajectory::from_messages(&state.messages);
assert_eq!(trajectory.tool_names(), vec!["multiply"]);
```

With the harness, `RunResult.usage` has the tokens and cost of the turn:

```rust
let AgentOutcome::Completed(r) = agent.run("t1", "When is the launch?").await? else { todo!() };
let state = agent.state("t1").await?.unwrap();
let trajectory = Trajectory::from_last_turn(&state.messages)
    .with_usage(r.usage.input_tokens, r.usage.output_tokens)
    .with_cost(r.usage.cost_usd);
```

### Expected trajectories in datasets

`Example` has an optional `expected_trajectory` (omitted from JSONL when absent, so existing files
load unchanged):

```json
{"inputs": {"city": "Paris"}, "reference": "sunny",
 "expected_trajectory": {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}}
```

In Rust: `Example::new(inputs).with_expected_trajectory(Trajectory::new().tool("get_weather", args))`.
In an expected step, `"arguments": null` (or omitted) matches any arguments.

### Trajectory evaluators

```rust
#[async_trait]
pub trait TrajectoryEvaluator: Send + Sync {
    fn name(&self) -> &str;
    async fn evaluate(&self, input: TrajectoryInput<'_>) -> Score; // trajectory, expected, inputs, output, reference
}
```

| Evaluator | Key | Passes when |
|---|---|---|
| `TrajectoryMatch::exact()` | `trajectory_exact` | same calls in the same order, nothing extra |
| `TrajectoryMatch::unordered()` | `trajectory_unordered` | same calls (as a multiset) in any order |
| `TrajectoryMatch::subsequence()` | `trajectory_subsequence` | the expected calls appear in order; extra calls before, between and after are allowed |
| `TrajectoryMatch::superset()` | `trajectory_superset` | every expected call was made, any order; a call listed twice must be made twice |
| `TrajectoryMatch::subset()` | `trajectory_subset` | every call made matches an expected call |
| `ToolUsage::new().require("search").forbid("shell")` | `tool_usage` | required tools were called, forbidden tools were not |
| `TrajectoryBudget::new().max_steps(n).max_tool_calls(n).max_tokens(n).max_cost(usd)` | `trajectory_budget` | every limit that is set holds; a token or cost limit fails when the value is unknown |
| `RelativeEfficiency::new()` (+ `.metric(EfficiencyMetric::Steps \| ToolCalls \| Tokens \| Cost)`) | `trajectory_efficiency` | score `min(1, reference / actual)` against the expected trajectory (default metric: tool calls) |
| `NoRepeatedCalls::new()` (+ `.allowed(n)`) | `no_repeated_calls` | no identical call (same name, equal arguments) is made more than once (or `n` times) |
| `TrajectoryJudge::new(model, rubric)` | `trajectory_judge` | graded by any `ChatModel` |

All evaluators except `RelativeEfficiency` and `TrajectoryJudge` give 1 or 0, with a reason on
failure (for example `missing calls: search; expected [search, answer], got [answer]`). Every
evaluator has `.named(key)`. `TrajectoryMatch` and `RelativeEfficiency` use the example's expected
trajectory, or a fixed one given with `.expecting(trajectory)`; without either they score 0 with
the reason "example has no expected trajectory".

**Arguments.** `TrajectoryMatch` compares tool names by default. `.arguments(ArgumentMatch::Subset)`
also requires the expected arguments to be a JSON subset of the actual ones: every expected key
must be present with a matching value, extra keys are allowed (recursively), arrays must have the
same length, and numbers compare by value (`1` matches `1.0`). `ArgumentMatch::Exact` requires
equal arguments. Arguments sent as a JSON string are parsed first. `json_subset(expected, actual)`
is public for your own evaluators.

**LLM judge.** `TrajectoryJudge` sends the rubric, the example inputs, the expected trajectory and
the actual one (rendered with `Trajectory::to_text`, tool results cut at 500 characters,
`.max_result_chars(n)`) at temperature 0, and reads `{"score": 0..1, "reasoning": "..."}` through
`StructuredOutput` (a forced tool call by default; `.mode(StructuredMode::Json)` for models without
forced tool choice). Scores on a 0–10 or 0–100 scale are scaled down. An invalid reply (after one
retry) or a model error scores 0 with the reason.

```rust
let judge = TrajectoryJudge::new(
    judge_model,
    "Score 1 if every tool call was needed and in a sensible order, 0.5 if one call was \
     redundant, 0 if the agent looped or used the wrong tools.",
)
.named("path_quality");
```

### In an experiment

The target returns `trajectory.into_output()`, an object `{"answer": ..., "trajectory": {...}}`, and
trajectory evaluators are added with `Experiment::trajectory_evaluator`. Their scores land in the
same `scores` list as the output evaluators, so aggregates, summary evaluators, pairwise
comparisons and the scope UI work unchanged. An output without a trajectory scores 0 with a reason.
The short form `evaluate` runs output evaluators only.

```rust
use sinew_eval::*;

let report = Experiment::new(&dataset)
    .trajectory_evaluator(TrajectoryMatch::subsequence().arguments(ArgumentMatch::Subset))
    .trajectory_evaluator(ToolUsage::new().require("multiply").forbid("shell"))
    .trajectory_evaluator(NoRepeatedCalls::new())
    .trajectory_evaluator(TrajectoryBudget::new().max_steps(4))
    .summary_evaluator(PassRate::new("trajectory_subsequence", 1.0))
    .run(|inputs: Value| async move {
        let state = run_agent(inputs).await?;
        Ok::<_, BoxError>(Trajectory::from_messages(&state.messages).into_output())
    })
    .await;
println!("{}", report.summary());
```

Output evaluators still see the whole output; read the answer with
`JsonFieldEquals::expecting("answer", value)` or a custom evaluator.

`crates/sinew-harness/tests/trajectory.rs` grades three scripted harness agents (direct, looping,
and one that searches before reading) with every trajectory evaluator, and checks that
`Trajectory::from_runs` on the harness trace agrees with the messages. Sample 13 runs a trajectory
experiment next to the answer experiment.

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
