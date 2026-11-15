//! Deploying an agent: `sinew-serve` puts a react agent behind an HTTP API
//! (threads, runs, SSE streaming) and an OpenAI-compatible endpoint. The
//! sample starts the server in-process, then talks to it two ways:
//!
//! 1. `ServeClient`, the typed Rust client: a streamed run on a thread, a
//!    follow-up on the same thread, and its checkpoint history;
//! 2. the OpenAI façade, through Sinew's own OpenAI-compatible provider,
//!    exactly as any OpenAI SDK would call it.
//!
//! ```bash
//! cargo run -p sample-20-serve
//! cargo run -p sample-20-serve -- --ollama
//! cargo run -p sample-20-serve -- --model lmstudio:qwen3-8b
//! # keep serving on a fixed address until Ctrl-C:
//! SINEW_SERVE_ADDR=127.0.0.1:8123 cargo run -p sample-20-serve -- --ollama
//! ```

use std::io::Write;

use futures_util::StreamExt;
use sample_common::{pick_model, BoxError};
use serde_json::{json, Value};
use sinew::prelude::*;
use sinew_serve::{Assistant, RunCreate, ServeClient, Server, StreamMode, StreamPart};

const API_KEY: &str = "sk-sinew-demo";

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = pick_model(vec![
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "1".into(),
                name: "celsius_to_fahrenheit".into(),
                arguments: json!({"celsius": -18}),
            }],
        ),
        Message::assistant("-18 °C is about -0.4 °F, so bring a heavy coat."),
        Message::assistant("In Kelvin that is about 255 K."),
        Message::assistant("Montreal winters are cold: dress in layers."),
        Message::assistant("Wear a hat, gloves and boots rated for -30 °C."),
    ])?;
    let tools = ToolBox::new().with(FnTool::new(
        "celsius_to_fahrenheit",
        "Converts a temperature from Celsius to Fahrenheit",
        json!({"type": "object", "properties": {"celsius": {"type": "number"}}, "required": ["celsius"]}),
        |a: Value| {
            let f = a["celsius"].as_f64().unwrap_or(0.0) * 9.0 / 5.0 + 32.0;
            Ok(json!((f * 10.0).round() / 10.0))
        },
    ));
    let server = Server::builder()
        .assistant(
            Assistant::agent("weather", react_agent(model, tools)?)
                .description("Answers weather questions, converting units with a tool"),
        )
        .api_key(API_KEY)
        .build();

    if let Ok(addr) = std::env::var("SINEW_SERVE_ADDR") {
        println!("bearer key: {API_KEY}");
        return Ok(server.serve_addr(addr.parse()?).await?);
    }

    let running = server.spawn("127.0.0.1:0").await?;
    let base = running.base_url();
    println!("sinew-serve listening on {base}\n");

    // 1. The typed client.
    let client = ServeClient::new(&base).with_api_key(API_KEY);
    for a in client.assistants().await? {
        println!("assistant `{}`: {}", a.assistant_id, a.description);
    }

    println!("\n== streamed run on thread `montreal` ==");
    let req = RunCreate::new(
        "weather",
        json!({"messages": [
            {"role": "system", "content": "Use the tool for conversions. Answer in one short sentence."},
            {"role": "user", "content": "It is -18 °C in Montreal today. What is that in Fahrenheit?"}
        ]}),
    )
    .with_stream_mode([StreamMode::Tokens, StreamMode::Updates]);
    let mut stream = client.stream(Some("montreal"), &req).await?;
    while let Some(part) = stream.next().await {
        match part? {
            StreamPart::Metadata { run_id, .. } => println!("run {run_id}"),
            StreamPart::Token { delta, .. } => {
                print!("{delta}");
                std::io::stdout().flush()?;
            }
            StreamPart::Updates(u) => {
                if let Some(msgs) = u["tools"]["messages"].as_array() {
                    let result = msgs.last().map(|m| m["content"].clone());
                    println!("tool → {}", result.unwrap_or_default());
                }
            }
            StreamPart::End { status } => println!("\n[{status:?}]"),
            _ => {}
        }
    }

    println!("\n== follow-up on the same thread ==");
    let run = client
        .wait(
            Some("montreal"),
            &RunCreate::new("weather", json!("And in Kelvin?")),
        )
        .await?;
    let state: AgentState = run.output_as().expect("finished run has output")?;
    println!("{}", state.last().map_or("", |m| m.content.as_str()));
    let history = client.history("montreal", 100).await?;
    println!(
        "thread has {} messages and {} checkpoints",
        state.messages.len(),
        history.len()
    );

    // 2. The OpenAI façade, called with an OpenAI-compatible client.
    println!("\n== /v1/chat/completions ==");
    let openai = OpenAiCompat::new(format!("{base}/v1"), "weather").with_api_key(API_KEY);
    let reply = openai
        .chat(ChatRequest::new(vec![Message::user(
            "One tip for Montreal winters?",
        )]))
        .await?;
    println!("non-streaming: {}", reply.message.content);
    print!("streaming:     ");
    let mut s = openai
        .chat_stream(ChatRequest::new(vec![Message::user(
            "What should I wear at -18 °C?",
        )]))
        .await?;
    while let Some(ev) = s.next().await {
        if let StreamEvent::TextDelta { text } = ev? {
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }
    println!();

    running.shutdown().await?;
    Ok(())
}
