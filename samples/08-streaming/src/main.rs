//! Streaming, two ways:
//!
//! 1. `chat_stream` on a model: print text as it arrives, like a chat UI.
//! 2. `stream` on a graph: get node start/end events, tool results and model
//!    tokens from inside the agent, in order, while it runs.
//!
//! Every model supports `chat_stream`; providers without native streaming fall
//! back to one chunk. Dropping a graph stream cancels the run.
//!
//! ```bash
//! cargo run -p sample-08-streaming
//! cargo run -p sample-08-streaming -- --ollama
//! ```

use std::io::Write;

use futures_util::StreamExt;
use sample_common::{pick_model, BoxError};
use serde_json::{json, Value};
use sinew::prelude::*;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = pick_model(vec![
        Message::assistant("Ownership means every value has exactly one owner, and the value is freed when that owner goes away."),
        Message::assistant_with_tools("", vec![ToolCall { id: "1".into(), name: "celsius_to_fahrenheit".into(), arguments: json!({"celsius": -18}) }]),
        Message::assistant("-18 °C is about -0.4 °F, so bring a heavy coat."),
    ])?;

    // 1. A plain streamed reply.
    println!("== chat_stream ==");
    let req = ChatRequest::new(vec![Message::user(
        "Explain Rust ownership in one sentence.",
    )])
    .with_max_tokens(120);
    let mut stream = model.chat_stream(req).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text } => {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            StreamEvent::Usage { usage } => println!(
                "\n[{} tokens in, {} out]",
                usage.input_tokens, usage.output_tokens
            ),
            _ => {}
        }
    }

    // 2. An agent graph, streamed.
    println!("\n== agent stream ==");
    let tools = ToolBox::new().with(FnTool::new(
        "celsius_to_fahrenheit",
        "Converts a temperature from Celsius to Fahrenheit",
        json!({"type": "object", "properties": {"celsius": {"type": "number"}}, "required": ["celsius"]}),
        |a: Value| {
            let f = a["celsius"].as_f64().unwrap_or(0.0) * 9.0 / 5.0 + 32.0;
            Ok(json!((f * 10.0).round() / 10.0))
        },
    ));
    let agent = react_agent(model, tools)?;
    let input = AgentState::new(vec![
        Message::system("Use the tool for conversions. Answer in one short sentence."),
        Message::user("It is -18 °C in Montreal today. What is that in Fahrenheit?"),
    ]);

    let mut events = agent.stream("weather-chat", Some(input));
    while let Some(event) = events.next().await {
        match event {
            GraphEvent::NodeStart { node, step, .. } => println!("\n▶ {node} (step {step})"),
            GraphEvent::Token { delta, .. } => {
                print!("{delta}");
                std::io::stdout().flush()?;
            }
            GraphEvent::NodeEnd { node, state, .. } if node == "tools" => {
                if let Some(result) = state.last() {
                    println!("tool → {}", result.content);
                }
            }
            GraphEvent::Completed { state } => println!(
                "\n\n✔ done, {} messages in the thread",
                state.messages.len()
            ),
            GraphEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }
    Ok(())
}
