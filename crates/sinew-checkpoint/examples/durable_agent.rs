//! A human-in-the-loop agent whose state survives a process restart.
//!
//! ```bash
//! cargo run -p sinew-checkpoint --example durable_agent -- --step1   # plans, then pauses
//! cargo run -p sinew-checkpoint --example durable_agent -- --step2   # a human approves, it resumes
//! cargo run -p sinew-checkpoint --example durable_agent -- --step2 --reject
//! ```
//!
//! The two runs are separate processes: the only thing they share is the
//! SQLite file (`durable_agent.db`, or the path in `SINEW_DB`).

use serde::{Deserialize, Serialize};
use sinew_checkpoint::{CheckpointStore, Checkpointer, SqliteCheckpointer};
use sinew_graph::{CompiledGraph, StateGraph, END};

const THREAD: &str = "refund-4711";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Ticket {
    request: String,
    plan: Option<String>,
    approved: Option<bool>,
    log: Vec<String>,
}

fn build(store: SqliteCheckpointer) -> CompiledGraph<Ticket> {
    let mut g = StateGraph::<Ticket>::new();
    // Stand-in for an LLM call that proposes an action.
    g.add_node("plan", |mut t: Ticket| async move {
        t.plan = Some("refund $120 to the original card".into());
        t.log.push("plan: proposed a refund".into());
        Ok(t)
    });
    // Side effect that must not happen without a human's OK.
    g.add_node("execute", |mut t: Ticket| async move {
        let entry = match t.approved {
            Some(true) => format!("execute: done ({})", t.plan.as_deref().unwrap_or("-")),
            _ => "execute: skipped, not approved".to_string(),
        };
        t.log.push(entry);
        Ok(t)
    });
    g.add_edge("plan", "execute");
    g.add_edge("execute", END);
    g.set_entry_point("plan");
    g.compile()
        .expect("valid graph")
        .with_checkpointer(store)
        .interrupt_before(["execute"])
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = std::env::var("SINEW_DB").unwrap_or_else(|_| "durable_agent.db".into());
    let store = SqliteCheckpointer::connect(&format!("sqlite://{path}")).await?;
    let graph = build(store.clone());

    if args.iter().any(|a| a == "--step1") {
        store.delete_thread(THREAD).await?; // start fresh on every --step1
        let ticket = Ticket {
            request: "Customer asks for a refund of order 4711".into(),
            plan: None,
            approved: None,
            log: vec![],
        };
        let out = graph.run(THREAD, Some(ticket)).await?;
        println!("paused: {}", out.is_interrupted());
        println!("plan:   {:?}", out.state().plan);
        println!("state saved in {path}; now run with --step2 (optionally --reject)");
    } else if args.iter().any(|a| a == "--step2") {
        let (mut ticket, next) = graph.get_state(THREAD).await?;
        println!(
            "resuming `{THREAD}` before `{next}`, plan: {:?}",
            ticket.plan
        );
        let approve = !args.iter().any(|a| a == "--reject");
        ticket.approved = Some(approve);
        ticket.log.push(format!(
            "human: {}",
            if approve { "approved" } else { "rejected" }
        ));
        graph.update_state(THREAD, ticket).await?;

        let done = graph.run(THREAD, None).await?.into_state();
        for line in &done.log {
            println!("  {line}");
        }
        println!("{} checkpoints stored:", store.history(THREAD).await?.len());
        for cp in store.history(THREAD).await? {
            println!("  step {} -> next `{}`", cp.step, cp.next);
        }
    } else {
        eprintln!("usage: durable_agent --step1 | --step2 [--reject]");
        std::process::exit(2);
    }
    Ok(())
}
