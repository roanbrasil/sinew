//! Evaluates two agent variants over a small arithmetic dataset and uploads
//! both experiments to a scope server, where they can be compared.
//!
//! Offline (scripted models, in-process server):
//!     cargo run -p sinew-scope --example eval_agent
//! Against a running server, with local Ollama and an LLM judge:
//!     cargo run -p sinew-scope --example eval_agent -- --ollama --url http://127.0.0.1:7878
//!
//! Variant A answers without tools; variant B is the `react_agent` with a
//! `multiply` tool. Every target call is also traced to project `eval`.

use std::sync::Arc;

use serde_json::{json, Value};
use sinew_core::{
    mock::ScriptedModel, providers::OpenAiCompat, ChatModel, FnTool, Message, Tool, ToolBox,
    ToolCall,
};
use sinew_eval::{
    evaluate, Contains, Dataset, Evaluator, LlmJudge, NumericTolerance, RegexMatch, ScopeClient,
};
use sinew_graph::prebuilt::{react_agent, AgentState};
use sinew_scope::{router, Store};
use sinew_trace::{HttpSink, HttpSinkConfig, Tracer};

const DATASET: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/data/arithmetic.jsonl"
);

fn calculator() -> impl Tool {
    FnTool::new(
        "multiply",
        "Multiply two numbers",
        json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
        |args: Value| {
            Ok(json!(
                args["a"].as_f64().unwrap_or(0.0) * args["b"].as_f64().unwrap_or(0.0)
            ))
        },
    )
}

fn ollama_model() -> Arc<dyn ChatModel> {
    Arc::new(OpenAiCompat::ollama(
        std::env::var("SINEW_MODEL").unwrap_or_else(|_| "qwen3:4b-instruct-2507-q4_K_M".into()),
    ))
}

fn product(inputs: &Value) -> f64 {
    inputs["a"].as_f64().unwrap_or(0.0) * inputs["b"].as_f64().unwrap_or(0.0)
}

/// Scripted stand-in for "mental math": right for small products, rounded
/// (wrong) for large ones.
fn scripted_no_tools(inputs: &Value) -> Arc<dyn ChatModel> {
    let p = product(inputs);
    let answer = if p > 100.0 {
        format!("Roughly {}.", (p / 10.0).round() * 10.0)
    } else {
        format!("{p}")
    };
    Arc::new(ScriptedModel::new([Message::assistant(answer)]))
}

/// Scripted tool-using model: calls `multiply`, then reports the result.
fn scripted_with_tools(inputs: &Value) -> Arc<dyn ChatModel> {
    Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "call_1".into(),
                name: "multiply".into(),
                arguments: json!({"a": inputs["a"], "b": inputs["b"]}),
            }],
        ),
        Message::assistant(format!("The answer is {}.", product(inputs))),
    ]))
}

async fn run_agent(
    model: Arc<dyn ChatModel>,
    tools: ToolBox,
    question: String,
    tracer: Tracer,
) -> Result<Value, String> {
    let agent = react_agent(model, tools)
        .map_err(|e| e.to_string())?
        .with_tracer(tracer);
    let out = agent
        .invoke(AgentState::new(vec![
            Message::system("Answer with the number only, no explanation."),
            Message::user(question),
        ]))
        .await
        .map_err(|e| e.to_string())?;
    Ok(json!(out
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default()))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let ollama = argv.iter().any(|a| a == "--ollama");
    let url_arg = argv
        .iter()
        .position(|a| a == "--url")
        .and_then(|i| argv.get(i + 1).cloned());
    let token = std::env::var("SINEW_SCOPE_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());

    let url = match url_arg {
        Some(u) => u.trim_end_matches('/').to_string(),
        None => {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let url = format!("http://{}", listener.local_addr()?);
            tokio::spawn(sinew_scope::server::serve_with_shutdown(
                listener,
                router(Store::memory().await?, None),
                std::future::pending(),
            ));
            println!("spawned in-process scope server at {url}");
            url
        }
    };

    // Load the dataset and round-trip it through JSONL.
    let mut dataset = Dataset::load_jsonl(DATASET)?;
    dataset.description = "Two-number multiplication questions".into();
    let copy = std::env::temp_dir().join("sinew-arithmetic.jsonl");
    dataset.save_jsonl(&copy)?;
    assert_eq!(Dataset::load_jsonl(&copy)?.examples, dataset.examples);

    let mut evaluators: Vec<Box<dyn Evaluator>> = vec![
        Box::new(NumericTolerance::absolute(1e-6)),
        Box::new(Contains::reference()),
        Box::new(RegexMatch::new(r"(?s)^.{1,80}$")?),
    ];
    if ollama {
        evaluators.push(Box::new(
            LlmJudge::new(
                ollama_model(),
                "Score 1 if the RESPONSE states the correct product given in REFERENCE \
                 and is concise; 0.5 if correct but verbose; 0 if wrong.",
            )
            .named("judge"),
        ));
    }

    let mut cfg = HttpSinkConfig::new(&url).project("eval");
    if let Some(t) = &token {
        cfg = cfg.token(t);
    }
    let tracer = Tracer::new().with_sink(HttpSink::new(cfg));
    let label = if ollama { "ollama" } else { "scripted" };
    let concurrency = if ollama { 2 } else { 4 };

    let no_tools = evaluate(
        |inputs: Value| {
            let tracer = tracer.clone();
            async move {
                let model = if ollama {
                    ollama_model()
                } else {
                    scripted_no_tools(&inputs)
                };
                let q = inputs["question"].as_str().unwrap_or_default().to_string();
                run_agent(model, ToolBox::new(), q, tracer).await
            }
        },
        &dataset,
        &evaluators,
        concurrency,
    )
    .await
    .with_name(format!("no-tools ({label})"))
    .with_metadata(json!({"variant": "no-tools", "model": label}));

    let with_tools = evaluate(
        |inputs: Value| {
            let tracer = tracer.clone();
            async move {
                let model = if ollama {
                    ollama_model()
                } else {
                    scripted_with_tools(&inputs)
                };
                let q = inputs["question"].as_str().unwrap_or_default().to_string();
                run_agent(model, ToolBox::new().with(calculator()), q, tracer).await
            }
        },
        &dataset,
        &evaluators,
        concurrency,
    )
    .await
    .with_name(format!("react-agent ({label})"))
    .with_metadata(json!({"variant": "react-agent", "model": label}));

    println!("{}\n", no_tools.summary());
    println!("{}\n", with_tools.summary());

    let mut client = ScopeClient::new(&url);
    if let Some(t) = &token {
        client = client.with_token(t);
    }
    client.upload_experiment(&dataset, &no_tools).await?;
    client.upload_report(&with_tools).await?;
    tokio::task::spawn_blocking(move || tracer.flush()).await?;

    println!("uploaded to {url}");
    println!(
        "compare: {url}/#/compare?a={}&b={}",
        no_tools.id, with_tools.id
    );
    Ok(())
}
