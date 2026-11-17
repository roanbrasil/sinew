//! `ReactAgent::loop_guard`: repeated tool calls and stuck runs.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, FnTool, Message, Role, ToolBox, ToolCall};
use sinew_graph::{
    prebuilt::{AgentState, LoopAction, LoopGuard, ReactAgent, LOOP_STOP_REPLY},
    MemoryCheckpointer, RunOutcome,
};

fn search(id: &str, q: &str) -> Message {
    Message::assistant_with_tools(
        "",
        vec![ToolCall {
            id: id.into(),
            name: "search".into(),
            arguments: json!({ "q": q }),
        }],
    )
}

fn counting_tool(calls: Arc<AtomicUsize>, fail: bool) -> ToolBox {
    ToolBox::new().with(FnTool::new(
        "search",
        "search the web",
        json!({"type": "object"}),
        move |_: Value| {
            calls.fetch_add(1, Ordering::SeqCst);
            if fail {
                Err(sinew_core::Error::Tool {
                    name: "search".into(),
                    message: "backend down".into(),
                })
            } else {
                Ok(json!("no results"))
            }
        },
    ))
}

fn looping_model(final_answer: &str) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new([
        search("1", "rust"),
        search("2", "rust"),
        search("3", "rust"),
        Message::assistant(final_answer),
    ]))
}

fn tool_results(s: &AgentState) -> Vec<String> {
    s.messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .map(|m| m.content.clone())
        .collect()
}

#[tokio::test]
async fn without_a_guard_every_call_runs() {
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = ReactAgent::new(looping_model("ok"), counting_tool(calls.clone(), false))
        .compile()
        .unwrap();
    agent
        .invoke(AgentState::new(vec![Message::user("find rust")]))
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn nudge_answers_the_repeated_call_instead_of_running_it() {
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = ReactAgent::new(
        looping_model("Nothing found."),
        counting_tool(calls.clone(), false),
    )
    .loop_guard(LoopGuard::default())
    .compile()
    .unwrap();
    let out = agent
        .invoke(AgentState::new(vec![Message::user("find rust")]))
        .await
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "third identical call is not executed"
    );
    let results = tool_results(&out);
    assert!(
        results[2].starts_with("error: loop detected"),
        "{}",
        results[2]
    );
    assert!(results[2].contains("`search`") && results[2].contains("2 times"));
    assert_eq!(out.last().unwrap().content, "Nothing found.");
}

#[tokio::test]
async fn stop_ends_the_run_with_a_reply() {
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = ReactAgent::new(
        looping_model("unreachable"),
        counting_tool(calls.clone(), false),
    )
    .loop_guard(LoopGuard::default().on_loop(LoopAction::Stop))
    .compile()
    .unwrap();
    let out = agent
        .invoke(AgentState::new(vec![Message::user("find rust")]))
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let last = out.last().unwrap();
    assert_eq!(last.content, LOOP_STOP_REPLY);
    // Every tool call still has a result, as providers require.
    assert!(tool_results(&out)[2].starts_with("skipped"));
}

#[tokio::test]
async fn consecutive_failures_stop_the_run_even_in_nudge_mode() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(ScriptedModel::new([
        search("1", "a"),
        search("2", "b"),
        search("3", "c"),
        search("4", "d"),
        Message::assistant("unreachable"),
    ]));
    let agent = ReactAgent::new(model, counting_tool(calls.clone(), true))
        .loop_guard(LoopGuard::default().max_consecutive_errors(3))
        .compile()
        .unwrap();
    let out = agent
        .invoke(AgentState::new(vec![Message::user("find")]))
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(out.last().unwrap().content, LOOP_STOP_REPLY);
}

#[tokio::test]
async fn interrupt_lets_a_human_decide() {
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = ReactAgent::new(looping_model("Done."), counting_tool(calls.clone(), false))
        .loop_guard(LoopGuard::default().on_loop(LoopAction::Interrupt))
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());

    let out = agent
        .run("t", Some(AgentState::new(vec![Message::user("find rust")])))
        .await
        .unwrap();
    let RunOutcome::Interrupted { interrupts, .. } = &out else {
        panic!("expected an interrupt, got {out:?}");
    };
    assert_eq!(
        interrupts[0].value["loop_detected"]["repeated"][0]["name"],
        "search"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    // The human says "run it anyway".
    let out = agent.resume("t", true).await.unwrap();
    assert!(!out.is_interrupted());
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(out.state().last().unwrap().content, "Done.");
}

#[tokio::test]
async fn detections_are_emitted_as_custom_events() {
    use futures_util::StreamExt;
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = ReactAgent::new(looping_model("ok"), counting_tool(calls, false))
        .loop_guard(LoopGuard::default())
        .compile()
        .unwrap();
    let mut events = agent.stream("t", Some(AgentState::new(vec![Message::user("find rust")])));
    let mut seen = false;
    while let Some(ev) = events.next().await {
        if let sinew_graph::GraphEvent::Custom { value, .. } = ev {
            seen |= value.get("loop_detected").is_some();
        }
    }
    assert!(seen);
}
