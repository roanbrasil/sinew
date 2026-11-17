<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew-eval

[![crates.io](https://img.shields.io/crates/v/sinew-eval.svg)](https://crates.io/crates/sinew-eval)
[![docs.rs](https://docs.rs/sinew-eval/badge.svg)](https://docs.rs/sinew-eval)
[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

Datasets, evaluators and experiments: the LangSmith evaluation loop for Sinew.

Part of [Sinew](https://github.com/roanbrasil/sinew): LLM chains, agent graphs, tracing and evaluation in Rust.

## Quick example

```rust
use serde_json::{json, Value};
use sinew_eval::*;

let mut ds = Dataset::new("capitals");
ds.push(json!({"country": "France"}), Some(json!("Paris")));
let evaluators: Vec<Box<dyn Evaluator>> = vec![Box::new(Contains::reference())];
let report = evaluate(
    |inputs: Value| async move { Ok::<_, String>(json!(format!("{} → Paris", inputs["country"]))) },
    &ds,
    &evaluators,
    4,
)
.await;
println!("{}", report.summary());
ScopeClient::new("http://127.0.0.1:7878").upload_experiment(&ds, &report).await?;
```

## Evaluators

`ExactMatch`, `Contains`, `RegexMatch`, `JsonFieldEquals`, `NumericTolerance` and `LlmJudge` (any `ChatModel` as the judge). Implement `Evaluator` for your own.

For agents, trajectory evaluators score the tool calls as well as the answer: `TrajectoryMatch` (exact, unordered, in-order subsequence, superset, subset; optional JSON-subset argument matching), `ToolUsage`, `TrajectoryBudget`, `RelativeEfficiency`, `NoRepeatedCalls` and `TrajectoryJudge`. Build a `Trajectory` from messages or from trace runs.

Upload datasets and reports to [`sinew-scope`](../sinew-scope) with `ScopeClient` to compare experiments in the browser. See the [evaluation guide](../../docs/evaluation.md).

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/roanbrasil/sinew/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/roanbrasil/sinew/blob/main/LICENSE-MIT) at your option.
