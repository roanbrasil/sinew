//! Durable tool calls: crash mid-node, resume, no repeated side effects.

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{
    mock::ScriptedModel, tool::idempotency_key, Message, Role, Tool, ToolBox, ToolCall, ToolSpec,
};
use sinew_graph::{
    durable::{unknown_outcome_reply, UnknownOutcome},
    prebuilt::{AgentState, ReactAgent},
    MemoryCheckpointer, RunOutcome,
};

/// Counts calls, records idempotency keys, and hangs forever on its first
/// call when `crash_once` is set; the test then drops the run, which is what
/// a killed process looks like to the checkpointer.
struct Probe {
    name: &'static str,
    side_effects: bool,
    calls: Arc<AtomicUsize>,
    crash_once: Arc<AtomicBool>,
    keys: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Tool for Probe {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.into(),
            description: "probe".into(),
            parameters: json!({"type": "object"}),
        }
    }
    fn side_effects(&self) -> bool {
        self.side_effects
    }
    async fn call(&self, _: Value) -> sinew_core::Result<Value> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.keys
            .lock()
            .unwrap()
            .push(idempotency_key().unwrap_or_default());
        if self.crash_once.swap(false, Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        Ok(json!(format!("{} ok", self.name)))
    }
}

struct Fixture {
    lookup: Arc<AtomicUsize>,
    charge: Arc<AtomicUsize>,
    charge_keys: Arc<Mutex<Vec<String>>>,
    tools: ToolBox,
}

fn fixture(crash_lookup: bool, crash_charge: bool) -> Fixture {
    let lookup = Arc::new(AtomicUsize::new(0));
    let charge = Arc::new(AtomicUsize::new(0));
    let charge_keys = Arc::new(Mutex::new(Vec::new()));
    let tools = ToolBox::new()
        .with(Probe {
            name: "lookup",
            side_effects: false,
            calls: lookup.clone(),
            crash_once: Arc::new(AtomicBool::new(crash_lookup)),
            keys: Arc::default(),
        })
        .with(Probe {
            name: "charge",
            side_effects: true,
            calls: charge.clone(),
            crash_once: Arc::new(AtomicBool::new(crash_charge)),
            keys: charge_keys.clone(),
        });
    Fixture {
        lookup,
        charge,
        charge_keys,
        tools,
    }
}

fn model() -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "",
            vec![
                ToolCall {
                    id: "c1".into(),
                    name: "charge".into(),
                    arguments: json!({"usd": 10}),
                },
                ToolCall {
                    id: "c2".into(),
                    name: "lookup".into(),
                    arguments: json!({"q": "x"}),
                },
            ],
        ),
        Message::assistant("Charged and looked up."),
    ]))
}

/// Runs until the hanging tool, then "kills the process".
async fn crash<F: std::future::Future>(run: F) {
    let r = tokio::time::timeout(std::time::Duration::from_millis(200), run).await;
    assert!(r.is_err(), "the run should hang in the crashing tool");
}

fn input() -> Option<AgentState> {
    Some(AgentState::new(vec![Message::user("charge and look up")]))
}

fn tool_results(s: &AgentState) -> Vec<String> {
    s.messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .map(|m| m.content.clone())
        .collect()
}

#[tokio::test]
async fn finished_calls_are_not_repeated_after_a_crash() {
    // charge runs and finishes; lookup crashes the node.
    let f = fixture(true, false);
    let agent = ReactAgent::new(model(), f.tools.clone())
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());
    crash(agent.run("t", input())).await;
    assert_eq!(f.charge.load(Ordering::SeqCst), 1);

    // Resume: charge comes from the ledger, lookup runs again.
    let out = agent.run("t", None).await.unwrap();
    assert_eq!(
        f.charge.load(Ordering::SeqCst),
        1,
        "the card is charged once"
    );
    assert_eq!(f.lookup.load(Ordering::SeqCst), 2);
    assert_eq!(tool_results(out.state()), ["charge ok", "lookup ok"]);
    assert_eq!(
        out.state().last().unwrap().content,
        "Charged and looked up."
    );
}

#[tokio::test]
async fn unknown_outcome_of_a_side_effect_fails_by_default() {
    // charge crashes while running: it may or may not have charged.
    let f = fixture(false, true);
    let agent = ReactAgent::new(model(), f.tools.clone())
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());
    crash(agent.run("t", input())).await;
    let out = agent.run("t", None).await.unwrap();
    assert_eq!(f.charge.load(Ordering::SeqCst), 1, "not repeated");
    assert_eq!(
        tool_results(out.state())[0],
        unknown_outcome_reply("charge")
    );
    assert_eq!(f.lookup.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn retry_policy_reruns_with_the_same_idempotency_key() {
    let f = fixture(false, true);
    let agent = ReactAgent::new(model(), f.tools.clone())
        .unknown_tool_outcome(UnknownOutcome::Retry)
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());
    crash(agent.run("t", input())).await;
    let out = agent.run("t", None).await.unwrap();
    assert_eq!(f.charge.load(Ordering::SeqCst), 2);
    let keys = f.charge_keys.lock().unwrap().clone();
    assert_eq!(keys.len(), 2);
    assert!(!keys[0].is_empty());
    assert_eq!(keys[0], keys[1], "an external API can drop the duplicate");
    assert_eq!(tool_results(out.state())[0], "charge ok");
}

#[tokio::test]
async fn interrupt_policy_asks_a_human() {
    let f = fixture(false, true);
    let agent = ReactAgent::new(model(), f.tools.clone())
        .unknown_tool_outcome(UnknownOutcome::Interrupt)
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());
    crash(agent.run("t", input())).await;
    let out = agent.run("t", None).await.unwrap();
    let RunOutcome::Interrupted { interrupts, .. } = &out else {
        panic!("expected an interrupt, got {out:?}")
    };
    assert_eq!(
        interrupts[0].value["unknown_tool_outcome"]["name"],
        "charge"
    );
    // The human checked the payment provider: it did not go through.
    let out = agent.resume("t", true).await.unwrap();
    assert_eq!(f.charge.load(Ordering::SeqCst), 2);
    assert_eq!(tool_results(out.state())[0], "charge ok");
}

#[tokio::test]
async fn without_a_ledger_tools_run_as_before() {
    let f = fixture(false, false);
    let agent = ReactAgent::new(model(), f.tools.clone()).compile().unwrap();
    let out = agent.invoke(input().unwrap()).await.unwrap();
    assert_eq!(tool_results(&out), ["charge ok", "lookup ok"]);
    assert_eq!(f.charge.load(Ordering::SeqCst), 1);
    assert!(
        f.charge_keys.lock().unwrap()[0].is_empty(),
        "no key outside durable runs"
    );
}
