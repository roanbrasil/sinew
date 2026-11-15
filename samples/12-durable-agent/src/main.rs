//! Durable execution: a deployment workflow pauses for approval, the process
//! "restarts" (everything is dropped), and a fresh graph instance reopens the
//! SQLite file and resumes the same thread. Run it in two real processes with
//! `--plan` and then `--approve` / `--reject`.
//!
//! ```bash
//! cargo run -p sample-12-durable-agent                 # both phases in one run
//! cargo run -p sample-12-durable-agent -- --plan       # phase 1, then exit
//! cargo run -p sample-12-durable-agent -- --approve    # phase 2, new process
//! ```

use sample_common::BoxError;
use serde::{Deserialize, Serialize};
use sinew::checkpoint::{CheckpointStore, SqliteCheckpointer};
use sinew::graph::CompiledGraph;
use sinew::prelude::*;

const THREAD: &str = "deploy-v2.3.0";
const DB: &str = "sqlite://durable-sample.db";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Deploy {
    version: String,
    plan: Vec<String>,
    approved: Option<bool>,
    log: Vec<String>,
}

async fn graph() -> Result<(CompiledGraph<Deploy>, SqliteCheckpointer), BoxError> {
    let store = SqliteCheckpointer::connect(DB).await?.with_retention(20);
    let mut g = StateGraph::<Deploy>::new();
    g.add_node("plan", |mut d: Deploy| async move {
        d.plan = vec![
            format!("build {}", d.version),
            "run migrations".into(),
            "canary 5%".into(),
            "roll out 100%".into(),
        ];
        d.log.push("planned".into());
        Ok(d)
    });
    g.add_node("deploy", |mut d: Deploy| async move {
        if d.approved == Some(true) {
            for step in &d.plan {
                d.log.push(format!("done: {step}"));
            }
        } else {
            d.log.push("cancelled by reviewer".into());
        }
        Ok(d)
    });
    g.add_edge("plan", "deploy");
    g.add_edge("deploy", END);
    g.set_entry_point("plan");
    Ok((
        g.compile()?
            .with_checkpointer(store.clone())
            .interrupt_before(["deploy"]),
        store,
    ))
}

async fn phase_plan() -> Result<(), BoxError> {
    let (graph, store) = graph().await?;
    store.delete_thread(THREAD).await?; // start clean each time
    let out = graph
        .run(
            THREAD,
            Some(Deploy {
                version: "v2.3.0".into(),
                ..Default::default()
            }),
        )
        .await?;
    println!("phase 1 → paused: {}", out.is_interrupted());
    for s in &out.state().plan {
        println!("   plan: {s}");
    }
    Ok(())
}

async fn phase_decide(approve: bool) -> Result<(), BoxError> {
    // A brand-new graph and connection, as after a restart.
    let (graph, store) = graph().await?;
    let (mut state, next) = graph.get_state(THREAD).await?;
    println!("phase 2 → found thread `{THREAD}` waiting before `{next}`");
    state.approved = Some(approve);
    graph.update_state(THREAD, state).await?;
    let done = graph.run(THREAD, None).await?.into_state();
    for line in &done.log {
        println!("   {line}");
    }
    println!("threads in the database: {:?}", store.list_threads().await?);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |f: &str| args.iter().any(|a| a == f);
    if has("--plan") {
        phase_plan().await
    } else if has("--approve") || has("--reject") {
        phase_decide(has("--approve")).await
    } else {
        phase_plan().await?;
        println!("   (everything dropped here: the state lives only in durable-sample.db)\n");
        phase_decide(true).await
    }
}
