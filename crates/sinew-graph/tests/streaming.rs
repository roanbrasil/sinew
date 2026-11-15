//! `CompiledGraph::stream`: event order, node emissions, terminal events.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sinew_graph::{
    BoxError, GraphError, GraphEvent, MemoryCheckpointer, NodeContext, StateGraph, END,
};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct S {
    log: Vec<String>,
}

fn push(name: &'static str) -> impl Fn(S) -> std::future::Ready<Result<S, BoxError>> + Send + Sync {
    move |mut s: S| {
        s.log.push(name.into());
        std::future::ready(Ok(s))
    }
}

fn label(e: &GraphEvent<S>) -> String {
    match e {
        GraphEvent::NodeStart { node, step, .. } => format!("start {node}@{step}"),
        GraphEvent::NodeEnd { node, step, .. } => format!("end {node}@{step}"),
        GraphEvent::Token { node, delta, .. } => format!("token {node} {delta}"),
        GraphEvent::Custom { node, value, .. } => format!("custom {node} {value}"),
        GraphEvent::Interrupted { next, .. } => format!("interrupted {next:?}"),
        GraphEvent::Completed { .. } => "completed".into(),
        GraphEvent::Error(e) => format!("error {e}"),
        other => format!("other {other:?}"),
    }
}

fn chain() -> StateGraph<S> {
    let mut g = StateGraph::new();
    g.add_node_with_context("talk", |mut s: S, ctx: NodeContext| async move {
        for t in ["Hi", " there"] {
            ctx.emit_token(t);
        }
        ctx.emit(json!({"progress": 1.0}));
        s.log.push("talk".into());
        Ok(s)
    });
    g.add_node("done", push("done"));
    g.add_edge("talk", "done");
    g.add_edge("done", END);
    g.set_entry_point("talk");
    g
}

#[tokio::test]
async fn sequential_events_arrive_in_order() {
    let graph = chain().compile().unwrap();
    let events: Vec<_> = graph.stream("t", Some(S::default())).collect().await;
    let labels: Vec<String> = events.iter().map(label).collect();
    assert_eq!(
        labels,
        vec![
            "start talk@0",
            "token talk Hi",
            "token talk  there",
            "custom talk {\"progress\":1.0}",
            "end talk@0",
            "start done@1",
            "end done@1",
            "completed",
        ]
    );
    let GraphEvent::NodeEnd { state, .. } = &events[4] else {
        panic!()
    };
    assert_eq!(state.log, vec!["talk"]);
    let Some(GraphEvent::Completed { state }) = events.last() else {
        panic!()
    };
    assert_eq!(state.log, vec!["talk", "done"]);
}

#[tokio::test]
async fn invoke_ignores_emissions() {
    let out = chain()
        .compile()
        .unwrap()
        .invoke(S::default())
        .await
        .unwrap();
    assert_eq!(out.log, vec!["talk", "done"]);
}

#[tokio::test]
async fn context_reports_whether_someone_is_listening() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut g = StateGraph::<S>::new();
    let s2 = seen.clone();
    g.add_node_with_context("n", move |s: S, ctx: NodeContext| {
        s2.lock().unwrap().push(ctx.is_streaming());
        async move { Ok(s) }
    });
    g.add_edge("n", END);
    g.set_entry_point("n");
    let graph = g.compile().unwrap();
    graph.invoke(S::default()).await.unwrap();
    let _ = graph
        .stream("t", Some(S::default()))
        .collect::<Vec<_>>()
        .await;
    assert_eq!(*seen.lock().unwrap(), vec![false, true]);
    // A detached context is usable directly in unit tests.
    let ctx = NodeContext::new("n", 0, "t");
    ctx.emit_token("dropped");
    assert!(!ctx.is_streaming());
}

#[tokio::test]
async fn fan_out_events_carry_task_indices() {
    let mut g = StateGraph::<S>::new();
    g.add_node("start", push("start"));
    for (name, delay) in [("slow", 40u64), ("fast", 0)] {
        g.add_node_with_context(name, move |mut s: S, ctx: NodeContext| async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            ctx.emit_token(name);
            s.log.push(name.into());
            Ok(s)
        });
        g.add_edge(name, END);
    }
    g.add_fan_out("start", ["slow", "fast"]);
    g.set_entry_point("start");
    g.with_reducer_from_base(|base: &S, mut acc: S, b: S| {
        acc.log.extend(b.log.into_iter().skip(base.log.len()));
        acc
    });
    let events: Vec<_> = g
        .compile()
        .unwrap()
        .stream("t", Some(S::default()))
        .collect()
        .await;
    let labels: Vec<String> = events.iter().map(label).collect();
    // Both branches start before either ends; `fast` finishes first.
    assert_eq!(&labels[2..4], &["start slow@1", "start fast@1"]);
    assert_eq!(&labels[4..6], &["token fast fast", "end fast@1"]);
    let tasks: Vec<(String, usize)> = events
        .iter()
        .filter_map(|e| match e {
            GraphEvent::Token { node, task, .. } => Some((node.clone(), *task)),
            _ => None,
        })
        .collect();
    assert_eq!(tasks, vec![("fast".into(), 1), ("slow".into(), 0)]);
    // ...but the merged state follows task order.
    let Some(GraphEvent::Completed { state }) = events.last() else {
        panic!()
    };
    assert_eq!(state.log, vec!["start", "slow", "fast"]);
}

#[tokio::test]
async fn interrupt_then_resume_by_streaming_again() {
    let graph = chain()
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new())
        .interrupt_before(["done"]);
    let first: Vec<String> = graph
        .stream("t", Some(S::default()))
        .map(|e| label(&e))
        .collect()
        .await;
    assert_eq!(first.last().unwrap(), "interrupted [\"done\"]");
    let second: Vec<String> = graph.stream("t", None).map(|e| label(&e)).collect().await;
    assert_eq!(second, vec!["start done@1", "end done@1", "completed"]);
}

#[tokio::test]
async fn errors_end_the_stream() {
    let mut g = StateGraph::<S>::new();
    g.add_node(
        "boom",
        |_: S| async move { Err::<S, BoxError>("kaput".into()) },
    );
    g.add_edge("boom", END);
    g.set_entry_point("boom");
    let events: Vec<_> = g
        .compile()
        .unwrap()
        .stream("t", Some(S::default()))
        .collect()
        .await;
    assert_eq!(events.len(), 2);
    assert!(
        matches!(&events[1], GraphEvent::Error(GraphError::Node { node, .. }) if node == "boom")
    );
    assert!(events[1].is_terminal());

    // Resuming a thread that has no checkpointer is an error event too.
    let events: Vec<_> = chain().compile().unwrap().stream("x", None).collect().await;
    assert!(matches!(
        &events[..],
        [GraphEvent::Error(GraphError::NoCheckpointer)]
    ));
}

#[tokio::test]
async fn dropping_the_stream_cancels_the_run() {
    let reached = Arc::new(Mutex::new(false));
    let mut g = StateGraph::<S>::new();
    g.add_node_with_context("first", |s: S, ctx: NodeContext| async move {
        ctx.emit(json!("hello"));
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(s)
    });
    let r = reached.clone();
    g.add_node("second", move |s: S| {
        *r.lock().unwrap() = true;
        async move { Ok(s) }
    });
    g.add_edge("first", "second");
    g.add_edge("second", END);
    g.set_entry_point("first");
    let mut stream = g.compile().unwrap().stream("t", Some(S::default()));
    assert!(matches!(
        stream.next().await,
        Some(GraphEvent::NodeStart { .. })
    ));
    assert!(matches!(
        stream.next().await,
        Some(GraphEvent::Custom { .. })
    ));
    drop(stream);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!*reached.lock().unwrap());
}
