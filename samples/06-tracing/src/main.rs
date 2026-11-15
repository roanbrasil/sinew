//! Tracing: attach a `Tracer` to a graph and every run (graph → node →
//! llm/tool) is recorded with timings, inputs, outputs and token usage.
//! Here traces go to three sinks at once: a JSONL file, memory, and a custom
//! sink that prints slow runs.
//!
//! ```bash
//! cargo run -p sample-06-tracing
//! cargo run -p sinew-scope -- traces.jsonl      # view the tree
//! ```

use std::sync::Arc;

use sample_common::{pick_model, BoxError};
use serde_json::{json, Value};
use sinew::prelude::*;
use sinew::trace::{Run, RunKind, TraceSink};

/// Custom sink: anything slower than a threshold gets reported.
struct SlowRunAlert {
    threshold_ms: u64,
}

impl TraceSink for SlowRunAlert {
    fn record(&self, run: &Run) {
        if run.duration_ms().unwrap_or(0) >= self.threshold_ms {
            eprintln!(
                "⚠ slow {:?} `{}`: {} ms",
                run.kind,
                run.name,
                run.duration_ms().unwrap_or(0)
            );
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = pick_model(vec![
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "1".into(),
                name: "word_count".into(),
                arguments: json!({"text": "tracing makes agents debuggable"}),
            }],
        ),
        Message::assistant("The text has 4 words."),
    ])?;
    let tools = ToolBox::new().with(FnTool::new(
        "word_count",
        "Counts words in a text",
        json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
        |a: Value| {
            Ok(json!(a["text"]
                .as_str()
                .unwrap_or_default()
                .split_whitespace()
                .count()))
        },
    ));

    let memory = Arc::new(MemorySink::new());
    let tracer = Tracer::new()
        .with_sink(JsonlSink::open("traces.jsonl")?)
        .with_shared_sink(memory.clone())
        .with_sink(SlowRunAlert { threshold_ms: 500 });

    let agent = react_agent(model, tools)?.with_tracer(tracer.clone());
    agent
        .invoke(AgentState::new(vec![Message::user(
            "How many words in 'tracing makes agents debuggable'?",
        )]))
        .await?;

    // You can also trace your own code around the graph.
    let mut job = tracer.start(
        "nightly-report",
        RunKind::Chain,
        None,
        json!({ "date": "2026-11-15" }),
    );
    job.set_metadata("owner", json!("data-team"));
    job.finish(json!({ "rows": 1280 }));

    let runs = memory.runs();
    let tokens: u32 = runs.iter().filter_map(|r| r.output_tokens).sum();
    println!("{} runs recorded, {} output tokens", runs.len(), tokens);
    for kind in [
        RunKind::Graph,
        RunKind::Node,
        RunKind::Llm,
        RunKind::Tool,
        RunKind::Chain,
    ] {
        println!(
            "  {:<6} {}",
            format!("{kind:?}"),
            runs.iter().filter(|r| r.kind == kind).count()
        );
    }
    println!(
        "\nappended to traces.jsonl — view it with: cargo run -p sinew-scope -- traces.jsonl --io"
    );
    Ok(())
}
