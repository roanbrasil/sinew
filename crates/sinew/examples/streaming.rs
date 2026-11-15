//! Live token streaming, from a raw model call and through an agent graph.
//!
//! Offline (scripted model):   cargo run --example streaming
//! Local Ollama:               SINEW_MODEL=qwen3:4b-instruct-2507-q4_K_M \
//!                               cargo run --example streaming -- --ollama
//! Anthropic:                  ANTHROPIC_API_KEY=... cargo run --example streaming -- --anthropic

use std::{io::Write, sync::Arc};

use futures_util::StreamExt;
use serde_json::{json, Value};
use sinew::core::{mock::ScriptedModel, StreamEvent};
use sinew::graph::GraphEvent;
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

fn scripted() -> Arc<dyn ChatModel> {
    let call = || {
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "call_1".into(),
                name: "multiply".into(),
                arguments: json!({"a": 37, "b": 3}),
            }],
        )
    };
    Arc::new(ScriptedModel::new([
        call(),
        call(),
        Message::assistant(
            "37 times 3 is 111. Multiplication is just repeated addition, done quickly.",
        ),
    ]))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let model: Arc<dyn ChatModel> = match mode.as_str() {
        "--ollama" => Arc::new(OpenAiCompat::ollama(
            std::env::var("SINEW_MODEL").unwrap_or_else(|_| "qwen3:4b-instruct-2507-q4_K_M".into()),
        )),
        "--anthropic" => Arc::new(Anthropic::from_env(
            std::env::var("SINEW_MODEL").unwrap_or_else(|_| "claude-sonnet-5-5".into()),
        )?),
        _ => scripted(),
    };
    let tools = ToolBox::new().with(calculator());
    let question = vec![
        Message::system("Use the tools for arithmetic. Answer briefly."),
        Message::user("What is 37 times 3?"),
    ];

    // 1. Raw model stream: watch the tool call arrive in pieces.
    println!("== raw chat_stream ==");
    // Capped: small local models occasionally loop on tool calls.
    let req = ChatRequest::new(question.clone())
        .with_tools(tools.specs())
        .with_max_tokens(256);
    let mut stream = model.chat_stream(req).await?;
    while let Some(ev) = stream.next().await {
        match ev? {
            StreamEvent::TextDelta { text } => print!("{text}"),
            StreamEvent::ToolCallDelta {
                index,
                name,
                arguments,
                ..
            } => {
                println!("[tool #{index} delta] name={name:?} args+={arguments:?}")
            }
            StreamEvent::ToolCall { call, .. } => {
                println!("[tool call] {}({})", call.name, call.arguments)
            }
            StreamEvent::Usage { usage } => {
                println!(
                    "[usage] {} in / {} out",
                    usage.input_tokens, usage.output_tokens
                )
            }
            StreamEvent::Done { model, stop_reason } => {
                println!("[done] model={model} stop={stop_reason:?}")
            }
        }
    }

    // 2. The same question through a react agent, tokens printed live.
    println!("\n== react_agent stream ==");
    let agent = react_agent(model, tools)?;
    let mut events = agent.stream("demo", Some(AgentState::new(question)));
    while let Some(ev) = events.next().await {
        match ev {
            GraphEvent::NodeStart { node, step, .. } => println!("\n-- {node} (step {step})"),
            GraphEvent::Token { delta, .. } => {
                print!("{delta}");
                std::io::stdout().flush()?;
            }
            GraphEvent::NodeEnd { node, state, .. } if node == "tools" => {
                if let Some(m) = state.last() {
                    println!("tool result: {}", m.content);
                }
            }
            GraphEvent::Completed { state } => {
                println!("\n\ncompleted with {} messages", state.messages.len())
            }
            GraphEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }
    Ok(())
}
