//! A custom graph with typed state, a cycle and a router: a writer drafts, a
//! reviewer scores, and the graph loops until the score passes or three drafts
//! have been written. `compile()` validates the wiring before anything runs.
//!
//! ```bash
//! cargo run -p sample-04-custom-graph
//! cargo run -p sample-04-custom-graph -- --ollama
//! ```

use std::sync::Arc;

use sample_common::{pick_model, BoxError};
use serde::{Deserialize, Serialize};
use sinew::prelude::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Draft {
    topic: String,
    text: String,
    feedback: String,
    score: u8,
    revisions: u8,
}

const MAX_REVISIONS: u8 = 3;
const PASSING_SCORE: u8 = 8;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = pick_model(vec![
        Message::assistant("Rust makes agents fast."),
        Message::assistant("SCORE: 5\nToo vague: say why it is fast and what else Rust brings."),
        Message::assistant("Rust agents start in milliseconds, use little memory and catch state bugs at compile time."),
        Message::assistant("SCORE: 9\nConcrete and clear."),
    ])?;

    let mut g = StateGraph::<Draft>::new();

    let writer = model.clone();
    g.add_node("write", move |mut s: Draft| {
        let model = writer.clone();
        async move {
            let prompt = if s.text.is_empty() {
                format!("Write one sentence pitching: {}", s.topic)
            } else {
                format!(
                    "Rewrite this sentence.\nSentence: {}\nFeedback: {}",
                    s.text, s.feedback
                )
            };
            s.text = model
                .chat(ChatRequest::new(vec![Message::user(prompt)]))
                .await?
                .message
                .content
                .trim()
                .to_string();
            s.revisions += 1;
            Ok(s)
        }
    });

    let reviewer: Arc<dyn ChatModel> = model.clone();
    g.add_node("review", move |mut s: Draft| {
        let model = reviewer.clone();
        async move {
            let prompt = format!(
                "Score this pitch from 0 to 10. Reply as 'SCORE: <n>' on the first line, then one line of feedback.\n\n{}",
                s.text
            );
            let reply = model.chat(ChatRequest::new(vec![Message::user(prompt)])).await?.message.content;
            s.score = reply
                .lines()
                .find_map(|l| l.trim().strip_prefix("SCORE:"))
                .and_then(|n| n.trim().parse().ok())
                .unwrap_or(0);
            s.feedback = reply.lines().skip(1).collect::<Vec<_>>().join(" ");
            println!("draft {}: score {:>2}  \"{}\"", s.revisions, s.score, s.text);
            Ok(s)
        }
    });

    g.add_edge("write", "review");
    g.add_conditional_edges(
        "review",
        |s: &Draft| {
            if s.score >= PASSING_SCORE || s.revisions >= MAX_REVISIONS {
                END.into()
            } else {
                "write".into()
            }
        },
        ["write", END],
    );
    g.set_entry_point("write");

    let graph = g.compile()?.with_recursion_limit(10);
    let out = graph
        .invoke(Draft {
            topic: "writing AI agents in Rust".into(),
            ..Default::default()
        })
        .await?;
    println!(
        "\nfinal ({} revisions, score {}): {}",
        out.revisions, out.score, out.text
    );
    Ok(())
}
