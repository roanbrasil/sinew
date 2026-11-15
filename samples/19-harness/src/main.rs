//! Agent harness: an agent works in a temporary workspace. It lists the
//! files, delegates reading them to a subagent (which has its own context
//! window), writes `summary.md` and reports back. Tokens and tool calls are
//! streamed as they happen.
//!
//! ```bash
//! cargo run -p sample-19-harness                    # offline, scripted model
//! cargo run -p sample-19-harness -- --ollama        # local Ollama
//! cargo run -p sample-19-harness -- --model lmstudio:qwen3-8b
//! ```

use std::fs;

use futures_util::StreamExt;
use sample_common::{is_live, pick_model, BoxError};
use serde_json::json;
use sinew::core::{Message, ToolCall};
use sinew_harness::{Agent, AgentEvent};

fn call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args,
    }
}

/// What the offline model says, in order. Parent and subagent share it.
fn script() -> Vec<Message> {
    vec![
        // parent: look around
        Message::assistant_with_tools(
            "Let me see what's in the workspace.",
            vec![call("c1", "list_dir", json!({"path": "notes"}))],
        ),
        // parent: delegate the reading
        Message::assistant_with_tools(
            "",
            vec![call(
                "c2",
                "task",
                json!({
                    "description": "read the notes",
                    "prompt": "Read notes/meeting.md and notes/risks.md and report the launch date, the budget and the top risk.",
                    "subagent_type": "general-purpose"
                }),
            )],
        ),
        // subagent: reads both files in one step
        Message::assistant_with_tools(
            "",
            vec![
                call("s1", "read_file", json!({"path": "notes/meeting.md"})),
                call("s2", "read_file", json!({"path": "notes/risks.md"})),
            ],
        ),
        // subagent: final report (all the parent will see)
        Message::assistant(
            "Launch: Friday, March 6. Budget: 42k CAD (approved). Top risk: the payment vendor may slip its API release by a week.",
        ),
        // parent: write the summary
        Message::assistant_with_tools(
            "",
            vec![call(
                "c3",
                "write_file",
                json!({
                    "path": "summary.md",
                    "content": "# Launch summary\n\n- Date: Friday, March 6\n- Budget: 42k CAD (approved)\n- Top risk: payment vendor API may slip a week\n"
                }),
            )],
        ),
        Message::assistant(
            "I wrote summary.md: the launch is on Friday, March 6 with an approved 42k CAD budget; the main risk is a possible one-week slip of the payment vendor's API.",
        ),
    ]
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let ws = tempfile::tempdir()?;
    fs::create_dir_all(ws.path().join("notes"))?;
    fs::write(
        ws.path().join("notes/meeting.md"),
        "# Weekly sync\n- Launch moved to Friday, March 6.\n- Budget approved: 42k CAD.\n- Ana owns the launch checklist.\n",
    )?;
    fs::write(
        ws.path().join("notes/risks.md"),
        "# Risks\n1. Payment vendor may slip its API release by a week (high).\n2. Docs translation not started (medium).\n",
    )?;

    let model = pick_model(script())?;
    let agent = Agent::builder()
        .model_arc(model)
        .instructions("You are a project assistant. Delegate reading files to a subagent with the task tool, then write a short summary.md in the workspace.")
        .workspace(ws.path())
        .filesystem()
        .general_subagent()
        .max_steps(12)
        .temperature(0.0)
        .build()?;
    println!(
        "tools: {}  (model: {}, live: {})\n",
        agent.tool_names().join(", "),
        agent.model_name(),
        is_live()
    );

    let mut events = agent
        .stream(
            "demo",
            "Summarize the notes folder into summary.md (date, budget, top risk). Use a subagent to read the notes.",
        )
        .await?;
    while let Some(ev) = events.next().await {
        match ev {
            AgentEvent::Token { text } => print!("{text}"),
            AgentEvent::ToolStart {
                name, arguments, ..
            } => {
                println!("\n→ {name} {arguments}");
            }
            AgentEvent::ToolEnd { output, .. } => {
                let first = output.lines().next().unwrap_or("");
                println!("  ← {first}");
            }
            AgentEvent::Done { result } => {
                println!(
                    "\n\n[{} steps, {} tokens{}]",
                    result.steps,
                    result.usage.total_tokens(),
                    result
                        .stop
                        .map(|s| format!(", stopped: {s}"))
                        .unwrap_or_default()
                );
            }
            AgentEvent::Error { message } => return Err(message.into()),
            _ => {}
        }
    }

    match fs::read_to_string(ws.path().join("summary.md")) {
        Ok(s) => println!("\nsummary.md:\n{s}"),
        Err(_) => println!("\n(the model did not write summary.md)"),
    }
    Ok(())
}
