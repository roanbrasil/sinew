//! Live tests against a local Ollama (127.0.0.1:11434). Ignored by default:
//!
//! ```bash
//! ollama pull qwen3:4b-instruct-2507-q4_K_M
//! cargo test -p sinew-harness --test live_ollama -- --ignored --test-threads=1
//! ```
//!
//! `SINEW_MODEL` overrides the model tag.

use std::fs;

use futures_util::StreamExt;
use sinew_harness::{Agent, AgentEvent, PermissionPolicy};

fn model_spec() -> String {
    format!(
        "ollama:{}",
        std::env::var("SINEW_MODEL").unwrap_or_else(|_| "qwen3:4b-instruct-2507-q4_K_M".into())
    )
}

fn workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("meeting.md"),
        "# Weekly sync\n- The launch moved to Friday, March 6.\n- Budget approved: 42k CAD.\n- Owner of the launch checklist: Ana.\n",
    )
    .unwrap();
    dir
}

#[tokio::test]
#[ignore = "needs a local Ollama"]
async fn live_reads_a_file_and_answers() {
    let ws = workspace();
    let agent = Agent::builder()
        .model_spec(model_spec())
        .workspace(ws.path())
        .filesystem_read_only()
        .temperature(0.0)
        .max_steps(6)
        .build()
        .unwrap();
    let mut events = agent
        .stream(
            "live-1",
            "Read meeting.md and tell me who owns the launch checklist. Answer in one sentence.",
        )
        .await
        .unwrap();
    let mut tokens = 0;
    let mut tools = Vec::new();
    let mut result = None;
    while let Some(ev) = events.next().await {
        match ev {
            AgentEvent::Token { .. } => tokens += 1,
            AgentEvent::ToolStart { name, .. } => tools.push(name),
            AgentEvent::Done { result: r } => result = Some(r),
            AgentEvent::Error { message } => panic!("{message}"),
            _ => {}
        }
    }
    let r = result.expect("done");
    println!("tools={tools:?} tokens={tokens} answer={}", r.text);
    assert!(tools.iter().any(|t| t == "read_file"), "{tools:?}");
    assert!(tokens > 1, "answer should be streamed");
    assert!(r.text.contains("Ana"), "{}", r.text);
}

#[tokio::test]
#[ignore = "needs a local Ollama"]
async fn live_writes_a_summary_after_approval() {
    let ws = workspace();
    let agent = Agent::builder()
        .model_spec(model_spec())
        .workspace(ws.path())
        .filesystem()
        .permissions(PermissionPolicy::interactive())
        .temperature(0.0)
        .max_steps(8)
        .build()
        .unwrap();
    let mut outcome = agent
        .run(
            "live-2",
            "Read meeting.md, then use write_file to save a two-line summary to summary.md. Then say done.",
        )
        .await
        .unwrap();
    let mut approvals = 0;
    while let sinew_harness::AgentOutcome::ApprovalRequired { requests, .. } = outcome {
        approvals += requests.len();
        assert!(requests
            .iter()
            .all(|r| r.tool == "write_file" || r.tool == "edit_file"));
        outcome = agent
            .resume(
                "live-2",
                sinew_harness::ApprovalDecision::approve_all(&requests),
            )
            .await
            .unwrap();
        assert!(approvals < 5, "too many approvals");
    }
    let r = outcome.completed().unwrap();
    let summary = fs::read_to_string(ws.path().join("summary.md")).expect("summary.md written");
    println!(
        "approvals={approvals} summary={summary:?} answer={}",
        r.text
    );
    assert!(approvals >= 1);
    assert!(summary.to_lowercase().contains("friday") || summary.contains("42"));
}

#[tokio::test]
#[ignore = "needs a local Ollama"]
async fn live_delegates_to_a_subagent() {
    let ws = workspace();
    let agent = Agent::builder()
        .model_spec(model_spec())
        .workspace(ws.path())
        .filesystem_read_only()
        .general_subagent()
        .temperature(0.0)
        .max_steps(6)
        .build()
        .unwrap();
    let r = agent
        .ask("Use the task tool to have a subagent read meeting.md and report the approved budget. Then tell me the budget.")
        .await
        .unwrap();
    let state = agent.state(&r.thread_id).await.unwrap().unwrap();
    let used_task = state
        .messages
        .iter()
        .flat_map(|m| &m.tool_calls)
        .any(|c| c.name == "task");
    println!("used_task={used_task} answer={}", r.text);
    assert!(used_task);
    assert!(r.text.contains("42"), "{}", r.text);
}
