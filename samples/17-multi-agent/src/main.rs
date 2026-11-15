//! Multi-agent supervisor: a `supervisor` model hands the question to a
//! `researcher` agent (with a lookup tool) and then to a `math` agent (with
//! a calculator), and answers once both reported back. Workers are ordinary
//! ReAct agents running as subgraphs; the stream shows which agent is
//! working, with subgraph namespaces.
//!
//! ```bash
//! cargo run -p sample-17-multi-agent                 # offline, scripted models
//! cargo run -p sample-17-multi-agent -- --ollama     # local model for every agent
//! cargo run -p sample-17-multi-agent -- --model openai:gpt-4.1-mini
//! ```

use std::io::Write;

use futures_util::StreamExt;
use sample_common::{pick_model, BoxError};
use serde_json::{json, Value};
use sinew::graph::{
    prebuilt::{AgentState, ReactAgent, Supervisor},
    GraphEvent, StreamMode, StreamOptions,
};
use sinew::prelude::*;

fn call(id: &str, name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args,
    }
}

fn lookup_tool() -> ToolBox {
    ToolBox::new().with(FnTool::new(
        "lookup_population",
        "Population of a city (2021 census). Args: {\"city\": string}",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
        |a: Value| {
            let city = a["city"].as_str().unwrap_or_default().to_lowercase();
            Ok(match city.as_str() {
                c if c.contains("montr") => json!({"city": "Montreal", "population": 1_762_949}),
                c if c.contains("quebec") || c.contains("québec") => {
                    json!({"city": "Quebec City", "population": 549_459})
                }
                _ => json!({"error": "unknown city"}),
            })
        },
    ))
}

fn calculator() -> ToolBox {
    ToolBox::new().with(FnTool::new(
        "add",
        "Adds two integers. Args: {\"a\": int, \"b\": int}",
        json!({
            "type": "object",
            "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
            "required": ["a", "b"]
        }),
        |a: Value| {
            Ok(json!(
                a["a"].as_i64().unwrap_or(0) + a["b"].as_i64().unwrap_or(0)
            ))
        },
    ))
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // Offline scripts: what each model answers, in order.
    let supervisor_model = pick_model(vec![
        Message::assistant_with_tools("", vec![call("s1", "transfer_to_researcher", json!({}))]),
        Message::assistant_with_tools("", vec![call("s2", "transfer_to_math", json!({}))]),
        Message::assistant(
            "Montreal (1,762,949) and Quebec City (549,459) have 2,312,408 people combined.",
        ),
    ])?;
    let researcher_model = pick_model(vec![
        Message::assistant_with_tools(
            "",
            vec![
                call("r1", "lookup_population", json!({"city": "Montreal"})),
                call("r2", "lookup_population", json!({"city": "Quebec City"})),
            ],
        ),
        Message::assistant("Montreal: 1,762,949. Quebec City: 549,459 (2021 census)."),
    ])?;
    let math_model = pick_model(vec![
        Message::assistant_with_tools(
            "",
            vec![call("m1", "add", json!({"a": 1_762_949, "b": 549_459}))],
        ),
        Message::assistant("1,762,949 + 549,459 = 2,312,408."),
    ])?;

    let researcher = ReactAgent::new(researcher_model, lookup_tool())
        .system_prompt("You look up facts with your tools. Report numbers only, no math.")
        .max_iterations(4)
        .compile()?;
    let math = ReactAgent::new(math_model, calculator())
        .system_prompt("You do arithmetic with the add tool, using numbers from the conversation.")
        .max_iterations(4)
        .compile()?;
    let graph = Supervisor::new(supervisor_model)
        .agent("researcher", "looks up city populations", researcher)
        .agent("math", "adds numbers exactly", math)
        .compile()?
        .with_recursion_limit(12);

    let question = "What is the combined population of Montreal and Quebec City?";
    println!("Q: {question}\n");
    let mut events = graph.stream_with(
        "demo",
        AgentState::new(vec![Message::user(question)]),
        StreamOptions::new([StreamMode::Tasks, StreamMode::Updates]).with_subgraphs(true),
    );
    while let Some(ev) = events.next().await {
        match ev {
            GraphEvent::NodeStart { node, .. } => println!("▶ {node}"),
            GraphEvent::Subgraph { namespace, event } => match *event {
                GraphEvent::NodeEnd { node, state, .. } if node == "tools" => {
                    let last = state["messages"]
                        .as_array()
                        .and_then(|m| m.last())
                        .and_then(|m| m["content"].as_str())
                        .unwrap_or_default()
                        .to_string();
                    println!("    [{}] tool → {last}", namespace.join(" / "));
                }
                GraphEvent::NodeStart { node, .. } => {
                    println!("    [{}] {node}", namespace.join(" / "))
                }
                _ => {}
            },
            GraphEvent::Completed { state } => {
                println!(
                    "\nA: {}",
                    state.last().map(|m| m.content.as_str()).unwrap_or("")
                );
                let handoffs: Vec<&str> = state
                    .messages
                    .iter()
                    .filter(|m| m.content.starts_with("Successfully transferred to"))
                    .map(|m| m.content.as_str())
                    .collect();
                println!("handoffs: {handoffs:?}");
            }
            GraphEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
        std::io::stdout().flush()?;
    }
    Ok(())
}
