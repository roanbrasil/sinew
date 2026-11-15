//! Commands, dynamic interrupts and interrupt_after.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_graph::*;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct S {
    log: Vec<String>,
    n: i64,
    #[serde(default)]
    answer: Option<String>,
}

fn push(mut s: S, x: &str) -> S {
    s.log.push(x.into());
    s
}

#[tokio::test]
async fn command_goto_routes_without_edges() {
    let mut g = StateGraph::<S>::new();
    g.add_command_node("route", ["even", "odd"], |s: S, _ctx| async move {
        let to = if s.n % 2 == 0 { "even" } else { "odd" };
        Ok(Command::goto(to).with_update(push(s, "route")))
    });
    g.add_node("even", |s: S| async move { Ok(push(s, "even")) });
    g.add_node("odd", |s: S| async move { Ok(push(s, "odd")) });
    g.add_edge("even", END);
    g.add_edge("odd", END);
    g.set_entry_point("route");
    let graph = g.compile().unwrap();
    let out = graph
        .invoke(S {
            n: 3,
            ..S::default()
        })
        .await
        .unwrap();
    assert_eq!(out.log, vec!["route", "odd"]);
}

#[tokio::test]
async fn command_goto_many_end_and_json_update() {
    let mut g = StateGraph::<S>::new();
    g.add_command_node("fan", ["a", "b"], |_s: S, _ctx| async move {
        // No `update`: the node writes nothing, the branches do.
        Ok(Command::goto_many(["a", "b"]))
    });
    g.add_command_node("a", [END], |_s: S, _ctx| async move {
        Ok(Command::end().with_json_update(json!({ "log": ["a"] })))
    });
    g.add_command_node("b", [END], |_s: S, _ctx| async move {
        Ok(Command::end().with_json_update(json!({ "log": ["b"], "n": 7 })))
    });
    g.with_reducer(|mut acc: S, b: S| {
        acc.log.extend(b.log);
        acc.n += b.n;
        acc
    });
    g.set_entry_point("fan");
    let out = g.compile().unwrap().invoke(S::default()).await.unwrap();
    assert_eq!(out.log, vec!["a", "b"]);
    assert_eq!(out.n, 7);
}

#[tokio::test]
async fn command_goto_unknown_node_is_an_error() {
    let mut g = StateGraph::<S>::new();
    g.add_command_node("x", Vec::<String>::new(), |_s: S, _ctx| async move {
        Ok(Command::goto("nope"))
    });
    g.set_entry_point("x");
    let err = g.compile().unwrap().invoke(S::default()).await.unwrap_err();
    assert!(matches!(err, GraphError::UnknownRoute { to, .. } if to == "nope"));
}

#[tokio::test]
async fn parent_command_in_root_graph_is_an_error() {
    let mut g = StateGraph::<S>::new();
    g.add_command_node("x", [END], |_s: S, _ctx| async move {
        Ok(Command::goto(END).to_parent())
    });
    g.set_entry_point("x");
    let err = g.compile().unwrap().invoke(S::default()).await.unwrap_err();
    assert!(matches!(err, GraphError::InvalidCommand(_)), "{err}");
}

fn approval_graph(runs: Arc<AtomicUsize>) -> StateGraph<S> {
    let mut g = StateGraph::<S>::new();
    g.add_node("draft", |s: S| async move { Ok(push(s, "draft")) });
    g.add_node_with_context("review", move |mut s: S, ctx| {
        let runs = runs.clone();
        async move {
            runs.fetch_add(1, Ordering::SeqCst);
            let name: String = ctx.interrupt(json!({ "question": "name?" }))?;
            let ok: bool = ctx.interrupt(json!({ "question": format!("approve for {name}?") }))?;
            s.answer = Some(format!("{name}:{ok}"));
            Ok(push(s, "review"))
        }
    });
    g.add_node("publish", |s: S| async move { Ok(push(s, "publish")) });
    g.add_edge("draft", "review");
    g.add_edge("review", "publish");
    g.add_edge("publish", END);
    g.set_entry_point("draft");
    g
}

#[tokio::test]
async fn dynamic_interrupt_pauses_and_resumes_with_values() {
    let runs = Arc::new(AtomicUsize::new(0));
    let graph = approval_graph(runs.clone())
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());

    let out = graph.run("t", Some(S::default())).await.unwrap();
    assert!(out.is_interrupted());
    let RunOutcome::Interrupted {
        next, interrupts, ..
    } = &out
    else {
        unreachable!()
    };
    assert_eq!(next, "review");
    assert_eq!(interrupts.len(), 1);
    assert_eq!(interrupts[0].value["question"], "name?");
    assert_eq!(interrupts[0].node, "review");
    assert_eq!(out.state().log, vec!["draft"]);

    // The snapshot exposes the pending interrupt too.
    let snap = graph.get_state_snapshot("t").await.unwrap();
    assert_eq!(snap.interrupts, *interrupts);

    // First resume: the first interrupt returns "Ana", the second pauses.
    let out = graph.resume("t", "Ana").await.unwrap();
    assert_eq!(out.interrupts()[0].value["question"], "approve for Ana?");

    // Second resume: the node re-runs; earlier interrupt replays "Ana".
    let out = graph.resume("t", true).await.unwrap();
    assert!(!out.is_interrupted());
    assert_eq!(out.state().answer.as_deref(), Some("Ana:true"));
    assert_eq!(out.state().log, vec!["draft", "review", "publish"]);
    assert_eq!(runs.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn resume_by_interrupt_id() {
    let graph = approval_graph(Arc::default())
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());
    let out = graph.run("t", Some(S::default())).await.unwrap();
    let id = out.interrupts()[0].id.clone();
    let out = graph
        .run_with("t", Command::resume_map([(id, json!("Bo"))]))
        .await
        .unwrap();
    assert_eq!(out.interrupts()[0].value["question"], "approve for Bo?");
}

#[tokio::test]
async fn interrupt_without_checkpointer_fails_clearly() {
    let graph = approval_graph(Arc::default()).compile().unwrap();
    let err = graph.invoke(S::default()).await.unwrap_err();
    assert!(
        matches!(&err, GraphError::Interrupted(n) if n == "review"),
        "{err}"
    );
    let err = graph.run("t", Some(S::default())).await.unwrap_err();
    assert!(matches!(err, GraphError::Interrupted(_)));
}

#[tokio::test]
async fn interrupted_superstep_keeps_finished_branches() {
    let fast_runs = Arc::new(AtomicUsize::new(0));
    let fr = fast_runs.clone();
    let mut g = StateGraph::<S>::new();
    g.add_node("start", |s: S| async move { Ok(s) });
    g.add_node("fast", move |_s: S| {
        let fr = fr.clone();
        async move {
            fr.fetch_add(1, Ordering::SeqCst);
            Ok(S {
                log: vec!["fast".into()],
                ..S::default()
            })
        }
    });
    g.add_node_with_context("ask", |_s: S, ctx| async move {
        let a: String = ctx.interrupt("why?")?;
        Ok(S {
            log: vec![format!("ask:{a}")],
            ..S::default()
        })
    });
    g.add_node("join", |s: S| async move { Ok(push(s, "join")) });
    g.add_fan_out("start", ["fast", "ask"]);
    g.add_edge("fast", "join");
    g.add_edge("ask", "join");
    g.add_edge("join", END);
    g.with_reducer(|mut acc: S, b: S| {
        acc.log.extend(b.log);
        acc
    });
    g.set_entry_point("start");
    let graph = g
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());

    let out = graph.run("t", Some(S::default())).await.unwrap();
    assert_eq!(out.interrupts().len(), 1);
    let out = graph.resume("t", "because").await.unwrap();
    assert_eq!(out.state().log, vec!["fast", "ask:because", "join"]);
    // `fast` finished before the pause and was not run again.
    assert_eq!(fast_runs.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn interrupt_after_pauses_once_the_node_ran() {
    let mut g = StateGraph::<S>::new();
    g.add_node("a", |s: S| async move { Ok(push(s, "a")) });
    g.add_node("b", |s: S| async move { Ok(push(s, "b")) });
    g.add_edge("a", "b");
    g.add_edge("b", END);
    g.set_entry_point("a");
    let graph = g
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new())
        .interrupt_after(["a"])
        .interrupt_before(["b"]);
    let out = graph.run("t", Some(S::default())).await.unwrap();
    assert!(out.is_interrupted());
    assert_eq!(out.state().log, vec!["a"]);
    // Paused *after* `a`, so `interrupt_before(b)` still applies on resume.
    let out = graph.run("t", None).await.unwrap();
    assert!(out.is_interrupted());
    assert_eq!(out.state().log, vec!["a"]);
    let out = graph.run("t", None).await.unwrap();
    assert_eq!(out.into_state().log, vec!["a", "b"]);
}

#[tokio::test]
async fn command_input_updates_and_jumps() {
    let graph = approval_graph(Arc::default())
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());
    graph.run("t", Some(S::default())).await.unwrap();
    // Skip the review entirely: edit the state and jump to `publish`.
    let out = graph
        .run_with(
            "t",
            Command::goto("publish").with_json_update(json!({ "answer": "skipped" })),
        )
        .await
        .unwrap();
    assert_eq!(out.state().answer.as_deref(), Some("skipped"));
    assert_eq!(out.state().log, vec!["draft", "publish"]);
}

#[tokio::test]
async fn conditional_entry_point() {
    let mut g = StateGraph::<S>::new();
    g.add_node("small", |s: S| async move { Ok(push(s, "small")) });
    g.add_node("big", |s: S| async move { Ok(push(s, "big")) });
    g.add_edge("small", END);
    g.add_edge("big", END);
    g.add_conditional_edges(
        START,
        |s: &S| {
            if s.n > 10 {
                "big".into()
            } else {
                "small".into()
            }
        },
        ["small", "big"],
    );
    let graph = g.compile().unwrap();
    let big = graph
        .invoke(S {
            n: 11,
            ..S::default()
        })
        .await
        .unwrap();
    assert_eq!(big.log, vec!["big"]);
    assert_eq!(graph.invoke(S::default()).await.unwrap().log, vec!["small"]);
}

#[tokio::test]
async fn interrupt_streams_its_payload() {
    use futures_util::StreamExt;
    let graph = approval_graph(Arc::default())
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());
    let events: Vec<_> = graph.stream("t", Some(S::default())).collect().await;
    let Some(GraphEvent::Interrupted {
        interrupts, next, ..
    }) = events.last()
    else {
        panic!("{:?}", events.last())
    };
    assert_eq!(next, &vec!["review".to_string()]);
    assert_eq!(interrupts[0].value, json!({ "question": "name?" }));
    let _: Value = interrupts[0].value.clone();
}
