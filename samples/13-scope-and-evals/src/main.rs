//! The LangSmith-style loop: traces go to a `sinew-scope` server over HTTP,
//! an evaluation scores the agent on a small dataset, and both show up in the
//! web UI (trace waterfall, experiment table). A second experiment grades
//! the agent's trajectory: the tool calls it made, against the expected ones.
//!
//! By default the sample starts an in-memory scope server on a free local
//! port. Pass `--url http://127.0.0.1:7878` to use one you started with
//! `cargo run -p sinew-scope -- serve`, and `--keep` to leave the UI running.
//!
//! ```bash
//! cargo run -p sample-13-scope-and-evals -- --keep
//! cargo run -p sample-13-scope-and-evals -- --ollama --keep
//! ```

use std::sync::Arc;

use sample_common::{is_live, pick_model, BoxError};
use serde_json::{json, Value};
use sinew::core::{mock::ScriptedModel, ChatModel};
use sinew::eval::{
    evaluate, ArgumentMatch, Contains, Dataset, Evaluator, Example, Experiment, NoRepeatedCalls,
    NumericTolerance, ScopeClient, Trajectory, TrajectoryBudget, TrajectoryMatch,
};
use sinew::prelude::*;
use sinew::trace::{HttpSink, HttpSinkConfig};

fn calculator() -> FnTool<impl Fn(Value) -> sinew::core::Result<Value> + Send + Sync> {
    FnTool::new(
        "multiply",
        "Multiply two numbers",
        json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
        |a: Value| {
            Ok(json!(
                a["a"].as_f64().unwrap_or(0.0) * a["b"].as_f64().unwrap_or(0.0)
            ))
        },
    )
}

/// Offline stand-in: a model that "calls" the tool and then states the product.
fn scripted_for(a: i64, b: i64) -> Arc<dyn ChatModel> {
    Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "1".into(),
                name: "multiply".into(),
                arguments: json!({"a": a, "b": b}),
            }],
        ),
        Message::assistant(format!("{a} times {b} is {}.", a * b)),
    ]))
}

/// Runs the prebuilt agent with the calculator on one example.
async fn run_agent(tracer: Tracer, inputs: Value) -> Result<AgentState, BoxError> {
    let (a, b) = (
        inputs["a"].as_i64().unwrap_or(0),
        inputs["b"].as_i64().unwrap_or(0),
    );
    let model = if is_live() {
        pick_model(vec![])?
    } else {
        scripted_for(a, b)
    };
    let agent = react_agent(model, ToolBox::new().with(calculator()))?.with_tracer(tracer);
    let q = inputs["question"].as_str().unwrap_or_default().to_string();
    Ok(agent
        .invoke(AgentState::new(vec![
            Message::system("Use the tool. Answer with the number."),
            Message::user(q),
        ]))
        .await?)
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let keep = args.iter().any(|a| a == "--keep");
    let url = match args
        .iter()
        .position(|a| a == "--url")
        .and_then(|i| args.get(i + 1))
    {
        Some(u) => u.trim_end_matches('/').to_string(),
        None => {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let url = format!("http://{}", listener.local_addr()?);
            let app = sinew_scope::router(sinew_scope::Store::memory().await?, None);
            tokio::spawn(sinew_scope::server::serve_with_shutdown(
                listener,
                app,
                std::future::pending(),
            ));
            url
        }
    };
    println!("scope UI: {url}\n");

    let tracer =
        Tracer::new().with_sink(HttpSink::new(HttpSinkConfig::new(&url).project("samples")));

    // A dataset: inputs, a reference answer and the expected tool call.
    let mut dataset = Dataset::new("multiplication");
    for (a, b) in [(37, 3), (12, 12), (250, 4), (19, 21)] {
        dataset = dataset.with_example(
            Example::new(json!({"a": a, "b": b, "question": format!("What is {a} times {b}?")}))
                .with_reference(json!(a * b))
                .with_expected_trajectory(
                    Trajectory::new().tool("multiply", json!({"a": a, "b": b})),
                ),
        );
    }

    let evaluators: Vec<Box<dyn Evaluator>> = vec![
        Box::new(NumericTolerance::absolute(1e-6)),
        Box::new(Contains::reference()),
    ];

    // The system under test: the prebuilt agent with a calculator tool.
    let report = evaluate(
        |inputs: Value| {
            let tracer = tracer.clone();
            async move {
                let out = run_agent(tracer, inputs).await?;
                Ok::<Value, BoxError>(json!(out
                    .last()
                    .map(|m| m.content.clone())
                    .unwrap_or_default()))
            }
        },
        &dataset,
        &evaluators,
        2,
    )
    .await
    .with_name(if is_live() {
        "react-agent (live)"
    } else {
        "react-agent (scripted)"
    });

    println!("{}\n", report.summary());
    ScopeClient::new(&url)
        .upload_experiment(&dataset, &report)
        .await?;

    // Trajectory evaluation: did the agent call `multiply` with the right
    // arguments, once, within two model calls?
    let trajectories = Experiment::new(&dataset)
        .trajectory_evaluator(TrajectoryMatch::exact().arguments(ArgumentMatch::Subset))
        .trajectory_evaluator(NoRepeatedCalls::new())
        .trajectory_evaluator(TrajectoryBudget::new().max_steps(2).max_tool_calls(1))
        .concurrency(2)
        .name("react-agent trajectory")
        .run(|inputs: Value| {
            let tracer = tracer.clone();
            async move {
                let out = run_agent(tracer, inputs).await?;
                Ok::<Value, BoxError>(Trajectory::from_messages(&out.messages).into_output())
            }
        })
        .await;
    println!("{}\n", trajectories.summary());
    ScopeClient::new(&url).upload_report(&trajectories).await?;
    tokio::task::spawn_blocking(move || tracer.flush()).await?;

    println!("traces:      {url}/");
    println!("experiment:  {url}/#/experiments/{}", report.id);
    if keep {
        println!("\nserver running — open the links above; Ctrl+C to stop");
        tokio::signal::ctrl_c().await?;
    }
    Ok(())
}
