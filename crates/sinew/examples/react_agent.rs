//! Tool-calling agent with tracing.
//!
//! Offline (scripted model):   cargo run --example react_agent
//! Local Ollama:               SINEW_MODEL=qwen3:8b cargo run --example react_agent -- --ollama
//! Anthropic:                  ANTHROPIC_API_KEY=... cargo run --example react_agent -- --anthropic
//!
//! Then inspect the run:       cargo run -p sinew-scope -- traces.jsonl --io

use std::sync::Arc;

use serde_json::{json, Value};
use sinew::core::mock::ScriptedModel;
use sinew::prelude::*;

fn calculator() -> impl Tool {
    FnTool::new(
        "multiply",
        "Multiply two numbers",
        json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
        |args: Value| {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!(a * b))
        },
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let model: Arc<dyn ChatModel> = match mode.as_str() {
        "--ollama" => Arc::new(OpenAiCompat::ollama(
            std::env::var("SINEW_MODEL").unwrap_or_else(|_| "qwen3:8b".into()),
        )),
        "--anthropic" => Arc::new(Anthropic::from_env(
            std::env::var("SINEW_MODEL").unwrap_or_else(|_| "claude-sonnet-5-5".into()),
        )?),
        _ => Arc::new(ScriptedModel::new([
            Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "call_1".into(),
                    name: "multiply".into(),
                    arguments: json!({"a": 37, "b": 3}),
                }],
            ),
            Message::assistant("37 × 3 = 111."),
        ])),
    };

    let tracer = Tracer::new().with_sink(JsonlSink::open("traces.jsonl")?);
    let agent = react_agent(model, ToolBox::new().with(calculator()))?.with_tracer(tracer);

    let out = agent
        .invoke(AgentState::new(vec![
            Message::system("Use the tools for arithmetic. Answer briefly."),
            Message::user("What is 37 times 3?"),
        ]))
        .await?;

    for m in &out.messages {
        let calls = if m.tool_calls.is_empty() {
            String::new()
        } else {
            format!(
                " {:?}",
                m.tool_calls.iter().map(|c| &c.name).collect::<Vec<_>>()
            )
        };
        println!("{:<9} {}{}", format!("{:?}", m.role), m.content, calls);
    }
    println!("\ntrace appended to traces.jsonl");
    Ok(())
}
