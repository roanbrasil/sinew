//! Map-reduce over documents: one parallel `summarize` branch per document,
//! merged by a reducer, then a single `combine` step.
//!
//! Offline (scripted model):   cargo run --example fan_out
//! Local Ollama:               SINEW_MODEL=qwen3:4b-instruct-2507-q4_K_M \
//!                               cargo run --example fan_out -- --ollama

use std::{sync::Arc, time::Instant};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sinew::core::mock::ScriptedModel;
use sinew::graph::{GraphEvent, NodeContext, SendTo};
use sinew::prelude::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Job {
    documents: Vec<String>,
    /// Set only on a branch's own input: the document it summarizes.
    #[serde(default)]
    document: Option<(usize, String)>,
    /// (document index, summary), filled by the branches.
    summaries: Vec<(usize, String)>,
    report: String,
}

const DOCS: [&str; 4] = [
    "Rust's ownership model guarantees memory safety without a garbage collector by tracking which variable owns each value.",
    "Tokio is an asynchronous runtime for Rust that provides an event loop, timers and async I/O for building network services.",
    "Server-Sent Events let a server push a stream of text events to a client over a single long-lived HTTP response.",
    "Map-reduce splits a large job into independent map tasks that run in parallel and a reduce step that merges their results.",
];

async fn ask(model: &dyn ChatModel, prompt: String) -> Result<String, BoxError> {
    let req = ChatRequest::new(vec![Message::user(prompt)]).with_max_tokens(120);
    Ok(model.chat(req).await?.message.content.trim().to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model: Arc<dyn ChatModel> = if std::env::args().any(|a| a == "--ollama") {
        Arc::new(OpenAiCompat::ollama(
            std::env::var("SINEW_MODEL").unwrap_or_else(|_| "qwen3:4b-instruct-2507-q4_K_M".into()),
        ))
    } else {
        // Branches call the scripted model concurrently, so the order in which
        // they pop answers is not fixed; the answers are generic on purpose.
        let mut answers: Vec<Message> = (0..DOCS.len())
            .map(|_| Message::assistant("A short summary of one document."))
            .collect();
        answers.push(Message::assistant(
            "Rust gives memory safety and async I/O; SSE and map-reduce structure streaming and parallel work.",
        ));
        Arc::new(ScriptedModel::new(answers))
    };

    let mut g = StateGraph::<Job>::new();
    g.add_node("plan", |s: Job| async move { Ok(s) });

    let m = model.clone();
    g.add_node_with_context("summarize", move |s: Job, ctx: NodeContext| {
        let m = m.clone();
        async move {
            let (i, doc) = s.document.clone().ok_or("summarize needs a document")?;
            let summary = ask(
                m.as_ref(),
                format!("Summarize in one short sentence:\n\n{doc}"),
            )
            .await?;
            ctx.emit(serde_json::json!({ "summarized": i }));
            Ok(Job {
                summaries: vec![(i, summary)],
                ..Default::default()
            })
        }
    });

    let m = model.clone();
    g.add_node("combine", move |mut s: Job| {
        let m = m.clone();
        async move {
            let bullets: Vec<String> = s.summaries.iter().map(|(_, t)| format!("- {t}")).collect();
            s.report = ask(
                m.as_ref(),
                format!(
                    "Combine these notes into a two-sentence overview:\n{}",
                    bullets.join("\n")
                ),
            )
            .await?;
            Ok(s)
        }
    });

    // One branch per document, each with its own input.
    g.add_send_edges(
        "plan",
        |s: &Job| {
            s.documents
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, d)| {
                    SendTo::new(
                        "summarize",
                        Job {
                            document: Some((i, d)),
                            ..Default::default()
                        },
                    )
                })
                .collect()
        },
        ["summarize"],
    );
    g.add_edge("summarize", "combine"); // all branches join here, once
    g.add_edge("combine", END);
    g.set_entry_point("plan");
    // Fold branch outputs into the shared state, in document order.
    g.with_reducer(|mut acc: Job, branch: Job| {
        acc.summaries.extend(branch.summaries);
        acc
    });

    let graph = g.compile()?;
    let input = Job {
        documents: DOCS.iter().map(|d| d.to_string()).collect(),
        ..Default::default()
    };

    let t0 = Instant::now();
    let mut events = graph.stream("docs", Some(input));
    while let Some(ev) = events.next().await {
        let ms = t0.elapsed().as_millis();
        match ev {
            GraphEvent::NodeStart { node, step, task } => {
                println!("{ms:>6}ms  start  {node}[{task}] step {step}")
            }
            GraphEvent::NodeEnd {
                node, step, task, ..
            } => {
                println!("{ms:>6}ms  end    {node}[{task}] step {step}")
            }
            GraphEvent::Custom { value, .. } => println!("{ms:>6}ms  event  {value}"),
            GraphEvent::Completed { state } => {
                println!("\nSummaries:");
                for (i, s) in &state.summaries {
                    println!("  [{i}] {s}");
                }
                println!("\nReport:\n  {}", state.report);
            }
            GraphEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }
    Ok(())
}
