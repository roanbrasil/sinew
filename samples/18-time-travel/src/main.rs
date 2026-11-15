//! Time travel: a trip planner drafts an itinerary, then a node asks a human
//! for approval with `ctx.interrupt(...)`. The run pauses (checkpointed),
//! the sample answers, and the node resumes with that answer. Then we list
//! the thread's history, go back to the checkpoint right after the draft,
//! edit it (a fork) and run the new branch; both branches stay in history.
//!
//! ```bash
//! cargo run -p sample-18-time-travel                 # offline, scripted model
//! cargo run -p sample-18-time-travel -- --ollama
//! ```

use sample_common::{pick_model, BoxError};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sinew::graph::{Command, RunConfig, RunInput, RunOutcome};
use sinew::prelude::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Trip {
    request: String,
    plan: String,
    approved: Option<bool>,
    notes: Option<String>,
    booked: Vec<String>,
}

/// The distinctive tail of a checkpoint id (ids start with a timestamp).
fn short(id: &str) -> &str {
    &id[id.len().saturating_sub(8)..]
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = pick_model(vec![Message::assistant(
        "Day 1: Old Montreal walk and the Notre-Dame Basilica. Day 2: Mount Royal hike, then Jean-Talon Market.",
    )])?;

    let mut g = StateGraph::<Trip>::new();
    g.add_node("plan", move |mut t: Trip| {
        let model = model.clone();
        async move {
            let prompt = format!("Write a two-day itinerary, one line per day: {}", t.request);
            t.plan = model
                .chat(ChatRequest::new(vec![Message::user(prompt)]))
                .await?
                .message
                .content;
            Ok(t)
        }
    });
    g.add_node_with_context("approve", |mut t: Trip, ctx| async move {
        // Pauses the run the first time; on resume returns the human's answer.
        let answer: serde_json::Value = ctx.interrupt(json!({
            "question": "Book this plan?",
            "plan": t.plan,
        }))?;
        t.approved = answer["approved"].as_bool();
        t.notes = answer["notes"].as_str().map(String::from);
        Ok(t)
    });
    g.add_node("book", |mut t: Trip| async move {
        if t.approved == Some(true) {
            t.booked = t.plan.split(". ").map(|s| format!("booked: {s}")).collect();
        }
        Ok(t)
    });
    g.add_edge("plan", "approve");
    g.add_edge("approve", "book");
    g.add_edge("book", END);
    g.set_entry_point("plan");
    let graph = g.compile()?.with_checkpointer(MemoryCheckpointer::new());
    let thread = "trip-42";

    // 1. Run until the approval node interrupts.
    let out = graph
        .run(
            thread,
            Some(Trip {
                request: "a weekend in Montreal".into(),
                ..Default::default()
            }),
        )
        .await?;
    let RunOutcome::Interrupted { interrupts, .. } = &out else {
        unreachable!("the approval node always asks first")
    };
    println!("⏸  interrupted in `{}`:", interrupts[0].node);
    println!("   {}\n", interrupts[0].value["question"]);

    // 2. The human answers; the node re-runs and `interrupt` returns it.
    let done = graph
        .run_with(
            thread,
            Command::resume(json!({ "approved": true, "notes": "window seat" })),
        )
        .await?
        .into_state();
    println!("✅ approved={:?} notes={:?}", done.approved, done.notes);
    for b in &done.booked {
        println!("   {b}");
    }

    // 3. History, newest first. Every checkpoint has an id and a parent.
    println!("\n📜 history:");
    let history = graph.get_state_history(thread).await?;
    for h in &history {
        println!(
            "   {}  step {}  next {:?}  source {}",
            short(&h.checkpoint_id),
            h.step,
            h.next,
            h.metadata
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("?"),
        );
    }

    // 4. Fork: go back to just after `plan`, edit the plan, run from there.
    let after_plan = history
        .iter()
        .find(|h| h.next == vec!["approve".to_string()] && h.interrupts.is_empty())
        .expect("checkpoint before approval");
    let mut edited = after_plan.state.clone();
    edited.plan = "Day 1: Biodome and Olympic Park. Day 2: Lachine Canal by bike.".into();
    let fork_id = graph
        .update_state_with(
            RunConfig::new(thread).at_checkpoint(&after_plan.checkpoint_id),
            edited,
            None,
        )
        .await?;
    println!(
        "\n🔀 forked at {} → {}",
        short(&after_plan.checkpoint_id),
        short(&fork_id)
    );

    // The fork pauses at the approval again; reject it this time.
    let out = graph.run_with(thread, RunInput::Continue).await?;
    assert!(out.is_interrupted());
    let branch = graph
        .resume(thread, json!({ "approved": false, "notes": "too sporty" }))
        .await?
        .into_state();
    println!(
        "   new branch: approved={:?} booked={}",
        branch.approved,
        branch.booked.len()
    );

    // 5. The original branch is still there; replay it from its last
    //    checkpoint before booking.
    let original = graph
        .get_state_at(thread, &history[0].checkpoint_id)
        .await?;
    println!(
        "   original branch: approved={:?} booked={}",
        original.state.approved,
        original.state.booked.len()
    );
    println!(
        "\n{} checkpoints in the thread now",
        graph.get_state_history(thread).await?.len()
    );
    Ok(())
}
