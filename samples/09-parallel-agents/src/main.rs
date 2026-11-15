//! Parallel fan-out and fan-in. Three reviewers with different personas read
//! the same proposal at the same time; a reducer merges their reviews into the
//! shared state; a final `decide` node reads all three.
//!
//! Graphs run in *supersteps*: every node scheduled for a step runs
//! concurrently, then the reducer folds their outputs (in a fixed order, so
//! the result does not depend on which branch finished first).
//!
//! ```bash
//! cargo run -p sample-09-parallel-agents
//! cargo run -p sample-09-parallel-agents -- --ollama
//! ```

use std::{sync::Arc, time::Instant};

use futures_util::StreamExt;
use sample_common::{pick_model, BoxError};
use serde::{Deserialize, Serialize};
use sinew::prelude::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Review {
    proposal: String,
    /// (reviewer, verdict). Each branch returns only its own entry.
    reviews: Vec<(String, String)>,
    decision: String,
}

const PROPOSAL: &str = "Rewrite our Python ingestion service (2k req/s, 6 engineers, no Rust experience) in Rust this quarter.";

fn reviewer(
    model: Arc<dyn ChatModel>,
    name: &'static str,
    persona: &'static str,
) -> impl Fn(
    Review,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Review, sinew::graph::BoxError>> + Send>,
> + Send
       + Sync
       + 'static {
    move |s: Review| {
        let model = model.clone();
        Box::pin(async move {
            let prompt = format!(
                "You are {persona}. In one sentence, review this proposal:\n{}",
                s.proposal
            );
            let req = ChatRequest::new(vec![Message::user(prompt)]).with_max_tokens(100);
            let verdict = model.chat(req).await?.message.content.trim().to_string();
            Ok(Review {
                reviews: vec![(name.to_string(), verdict)],
                ..Default::default()
            })
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // Branches run concurrently, so the scripted answers are generic on purpose.
    let model = pick_model(vec![
        Message::assistant("Plausible, but the team needs time to learn the language."),
        Message::assistant("Plausible, but the team needs time to learn the language."),
        Message::assistant("Plausible, but the team needs time to learn the language."),
        Message::assistant(
            "Not this quarter: start with one hot path in Rust and measure before a full rewrite.",
        ),
    ])?;

    let mut g = StateGraph::<Review>::new();
    g.add_node("intake", |s: Review| async move { Ok(s) });
    g.add_node(
        "optimist",
        reviewer(model.clone(), "optimist", "an optimistic staff engineer"),
    );
    g.add_node(
        "skeptic",
        reviewer(model.clone(), "skeptic", "a skeptical engineering manager"),
    );
    g.add_node(
        "sre",
        reviewer(
            model.clone(),
            "sre",
            "an SRE who carries the pager for this service",
        ),
    );

    let judge = model.clone();
    g.add_node("decide", move |mut s: Review| {
        let model = judge.clone();
        async move {
            let notes: Vec<String> = s
                .reviews
                .iter()
                .map(|(who, v)| format!("{who}: {v}"))
                .collect();
            let prompt = format!(
                "Given these reviews, give a one-sentence go/no-go decision.\n{}",
                notes.join("\n")
            );
            s.decision = model
                .chat(ChatRequest::new(vec![Message::user(prompt)]))
                .await?
                .message
                .content
                .trim()
                .to_string();
            Ok(s)
        }
    });

    g.add_fan_out("intake", ["optimist", "skeptic", "sre"]); // all three in one superstep
    for r in ["optimist", "skeptic", "sre"] {
        g.add_edge(r, "decide"); // they join here; `decide` runs once
    }
    g.add_edge("decide", END);
    g.set_entry_point("intake");
    g.with_reducer(|mut acc: Review, branch: Review| {
        acc.reviews.extend(branch.reviews);
        acc
    });

    let graph = g.compile()?;
    let t0 = Instant::now();
    let mut events = graph.stream(
        "review-1",
        Some(Review {
            proposal: PROPOSAL.into(),
            ..Default::default()
        }),
    );
    while let Some(event) = events.next().await {
        let ms = t0.elapsed().as_millis();
        match event {
            GraphEvent::NodeStart { node, step, .. } => {
                println!("{ms:>6} ms  start {node:<9} (step {step})")
            }
            GraphEvent::NodeEnd { node, .. } => println!("{ms:>6} ms  end   {node}"),
            GraphEvent::Completed { state } => {
                println!("\nreviews:");
                for (who, v) in &state.reviews {
                    println!("  {who:<9} {v}");
                }
                println!("\ndecision: {}", state.decision);
            }
            GraphEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }
    Ok(())
}
