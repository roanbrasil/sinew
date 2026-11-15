//! Parallel fan-out / fan-in: supersteps, reducers, checkpoints, tracing.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::json;
use sinew_graph::{
    BoxError, Checkpointer, GraphError, MemoryCheckpointer, SendTo, StateGraph, END,
};
use sinew_trace::{MemorySink, RunKind, Tracer};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Doc {
    docs: Vec<String>,
    /// The single document a `summarize` branch works on.
    #[serde(default)]
    doc: Option<String>,
    summaries: Vec<String>,
    log: Vec<String>,
}

type NodeFuture =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Doc, BoxError>> + Send>>;
type Spans = Arc<Mutex<Vec<(&'static str, Instant, Instant)>>>;

fn append_reducer(mut acc: Doc, branch: Doc) -> Doc {
    acc.summaries.extend(branch.summaries);
    acc.log.extend(branch.log);
    acc
}

/// Branches get a clone of the shared state, so they return the old log too;
/// keep only the entries each branch added on top of `base`.
fn log_tail(base: &Doc, mut acc: Doc, branch: Doc) -> Doc {
    acc.log.extend(branch.log.into_iter().skip(base.log.len()));
    acc
}

fn logging(name: &'static str, delay_ms: u64) -> impl Fn(Doc) -> NodeFuture + Send + Sync {
    move |mut s: Doc| {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            s.log.push(name.to_string());
            Ok(s)
        })
    }
}

/// start -> {a, b} -> join -> END
fn diamond(delay_ms: u64) -> StateGraph<Doc> {
    let mut g = StateGraph::new();
    g.add_node("start", logging("start", 0));
    g.add_node("a", logging("a", delay_ms));
    g.add_node("b", logging("b", delay_ms));
    g.add_node("join", logging("join", 0));
    g.add_fan_out("start", ["a", "b"]);
    g.add_edge("a", "join");
    g.add_edge("b", "join");
    g.add_edge("join", END);
    g.set_entry_point("start");
    g.with_reducer_from_base(log_tail);
    g
}

#[tokio::test]
async fn branches_really_run_concurrently() {
    let spans: Spans = Arc::default();
    let mut g = StateGraph::<Doc>::new();
    g.add_node("start", |s: Doc| async move { Ok(s) });
    for name in ["a", "b", "c"] {
        let spans = spans.clone();
        g.add_node(name, move |s: Doc| {
            let spans = spans.clone();
            async move {
                let t0 = Instant::now();
                tokio::time::sleep(Duration::from_millis(200)).await;
                spans.lock().unwrap().push((name, t0, Instant::now()));
                Ok(s)
            }
        });
        g.add_edge(name, END);
    }
    g.add_fan_out("start", ["a", "b", "c"]);
    g.set_entry_point("start");
    g.with_reducer(|acc, _| acc);

    let t0 = Instant::now();
    g.compile().unwrap().invoke(Doc::default()).await.unwrap();
    let elapsed = t0.elapsed();

    assert!(
        elapsed < Duration::from_millis(450),
        "took {elapsed:?}; branches ran sequentially?"
    );
    let spans = spans.lock().unwrap();
    assert_eq!(spans.len(), 3);
    let latest_start = spans.iter().map(|s| s.1).max().unwrap();
    let earliest_end = spans.iter().map(|s| s.2).min().unwrap();
    assert!(
        latest_start < earliest_end,
        "branch intervals do not overlap"
    );
}

#[tokio::test]
async fn branches_join_at_the_next_node_once() {
    let out = diamond(10)
        .compile()
        .unwrap()
        .invoke(Doc::default())
        .await
        .unwrap();
    assert_eq!(out.log, vec!["start", "a", "b", "join"]);
}

#[tokio::test]
async fn reducer_folds_in_task_order_not_completion_order() {
    let mut g = StateGraph::<Doc>::new();
    g.add_node("plan", |s: Doc| async move { Ok(s) });
    g.add_node("summarize", |s: Doc| async move {
        let doc = s.doc.clone().unwrap();
        // Later documents finish first.
        let delay = 60u64.saturating_sub(10 * doc.len() as u64);
        tokio::time::sleep(Duration::from_millis(delay)).await;
        Ok(Doc {
            summaries: vec![format!("<{doc}>")],
            ..Default::default()
        })
    });
    g.add_node("reduce", |mut s: Doc| async move {
        s.log.push(format!("reduced {}", s.summaries.len()));
        Ok(s)
    });
    g.add_send_edges(
        "plan",
        |s: &Doc| {
            s.docs
                .iter()
                .map(|d| {
                    SendTo::new(
                        "summarize",
                        Doc {
                            doc: Some(d.clone()),
                            ..Default::default()
                        },
                    )
                })
                .collect()
        },
        ["summarize"],
    );
    g.add_edge("summarize", "reduce");
    g.add_edge("reduce", END);
    g.set_entry_point("plan");
    g.with_reducer(append_reducer);

    let input = Doc {
        docs: vec!["a".into(), "bb".into(), "ccc".into(), "dddd".into()],
        summaries: vec!["existing".into()],
        ..Default::default()
    };
    let out = g.compile().unwrap().invoke(input).await.unwrap();
    assert_eq!(
        out.summaries,
        vec!["existing", "<a>", "<bb>", "<ccc>", "<dddd>"]
    );
    // The shared state outside what the reducer touches is preserved.
    assert_eq!(out.docs.len(), 4);
    // `reduce` ran once, after all branches.
    assert_eq!(out.log, vec!["reduced 5"]);
}

#[tokio::test]
async fn send_with_a_single_item_still_uses_the_reducer() {
    let mut g = StateGraph::<Doc>::new();
    g.add_node("plan", |s: Doc| async move { Ok(s) });
    g.add_node("work", |_s: Doc| async move {
        Ok(Doc {
            summaries: vec!["w".into()],
            ..Default::default()
        })
    });
    g.add_send_edges(
        "plan",
        |_: &Doc| vec![SendTo::new("work", Doc::default())],
        ["work"],
    );
    g.add_edge("work", END);
    g.set_entry_point("plan");
    g.with_reducer(append_reducer);
    let out = g
        .compile()
        .unwrap()
        .invoke(Doc {
            docs: vec!["kept".into()],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(out.docs, vec!["kept"]);
    assert_eq!(out.summaries, vec!["w"]);
}

#[tokio::test]
async fn multi_router_picks_a_subset_and_validates_targets() {
    let mut g = StateGraph::<Doc>::new();
    g.add_node("route", logging("route", 0));
    g.add_node("x", logging("x", 0));
    g.add_node("y", logging("y", 0));
    g.add_node("z", logging("z", 0));
    g.add_conditional_edges_multi(
        "route",
        |s: &Doc| s.docs.clone(),
        ["x", "y", "z", "bogus-but-declared"],
    );
    for n in ["x", "y", "z"] {
        g.add_edge(n, END);
    }
    g.set_entry_point("route");
    g.with_reducer_from_base(log_tail);
    // `bogus-but-declared` is not a node, so compile rejects it.
    assert!(
        matches!(g.compile(), Err(GraphError::Build(e)) if e.iter().any(|m| m.contains("bogus")))
    );

    let mut g = StateGraph::<Doc>::new();
    g.add_node("route", logging("route", 0));
    g.add_node("x", logging("x", 0));
    g.add_node("z", logging("z", 0));
    g.add_conditional_edges_multi("route", |s: &Doc| s.docs.clone(), ["x", "z", END]);
    g.add_edge("x", END);
    g.add_edge("z", END);
    g.set_entry_point("route");
    g.with_reducer_from_base(log_tail);
    let graph = g.compile().unwrap();

    let docs = |d: &[&str]| Doc {
        docs: d.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    let out = graph.invoke(docs(&["z", "x"])).await.unwrap();
    assert_eq!(out.log, vec!["route", "z", "x"]);
    // A single pick behaves like a plain conditional edge (no reducer).
    let out = graph.invoke(docs(&["x"])).await.unwrap();
    assert_eq!(out.log, vec!["route", "x"]);
    // An empty pick ends the run.
    let out = graph.invoke(docs(&[])).await.unwrap();
    assert_eq!(out.log, vec!["route"]);
    let err = graph.invoke(docs(&["x", "nope"])).await.unwrap_err();
    assert!(matches!(err, GraphError::UnknownRoute { to, .. } if to == "nope"));
}

#[test]
fn fan_out_without_reducer_is_a_build_error() {
    let mut g = StateGraph::<Doc>::new();
    g.add_node("start", logging("start", 0));
    g.add_node("a", logging("a", 0));
    g.add_node("b", logging("b", 0));
    g.add_fan_out("start", ["a", "b"]);
    g.add_edge("a", END);
    g.add_edge("b", END);
    g.set_entry_point("start");
    let Err(GraphError::Build(errs)) = g.compile() else {
        panic!("expected a build error")
    };
    assert!(errs.iter().any(|e| e.contains("no reducer")), "{errs:?}");
}

#[tokio::test]
async fn recursion_limit_counts_supersteps_with_fan_out() {
    let count = Arc::new(Mutex::new(0usize));
    let mut g = StateGraph::<Doc>::new();
    for name in ["split", "a", "b"] {
        let count = count.clone();
        g.add_node(name, move |s: Doc| {
            *count.lock().unwrap() += 1;
            async move { Ok(s) }
        });
    }
    g.add_fan_out("split", ["a", "b"]);
    g.add_edge("a", "split");
    g.add_edge("b", "split");
    g.set_entry_point("split");
    g.with_reducer(|acc, _| acc);
    let err = g
        .compile()
        .unwrap()
        .with_recursion_limit(5)
        .invoke(Doc::default())
        .await
        .unwrap_err();
    assert!(matches!(err, GraphError::RecursionLimit(5)));
    // Supersteps: split, {a,b}, split, {a,b}, split  => 7 node executions.
    assert_eq!(*count.lock().unwrap(), 7);
}

#[tokio::test]
async fn interrupt_and_resume_across_a_fan_out() {
    let cp = Arc::new(MemoryCheckpointer::new());
    let graph = diamond(5)
        .compile()
        .unwrap()
        .with_shared_checkpointer(cp.clone())
        .interrupt_before(["b"]);

    let out = graph.run("t", Some(Doc::default())).await.unwrap();
    let sinew_graph::RunOutcome::Interrupted {
        next,
        next_nodes,
        state,
    } = out
    else {
        panic!("expected an interrupt")
    };
    assert_eq!(next, "a");
    assert_eq!(next_nodes, vec!["a", "b"]);
    // Neither branch ran: the whole superstep pauses together.
    assert_eq!(state.log, vec!["start"]);

    let snap = graph.get_state_snapshot("t").await.unwrap();
    assert_eq!(snap.next, vec!["a", "b"]);
    assert_eq!(snap.step, 1);
    let mut edited = snap.state;
    edited.log[0] = "START".into();
    graph.update_state("t", edited).await.unwrap();

    let done = graph.run("t", None).await.unwrap();
    assert!(!done.is_interrupted());
    assert_eq!(done.state().log, vec!["START", "a", "b", "join"]);

    // One checkpoint per superstep (+ the interrupt and the human edit).
    let history = cp.history("t").await.unwrap();
    let steps: Vec<usize> = history.iter().map(|c| c.step).collect();
    assert_eq!(steps, vec![1, 1, 1, 2, 3]);
    assert_eq!(history[3].next_nodes(), vec!["join"]);
    assert!(history[4].next_nodes().is_empty());
}

#[tokio::test]
async fn send_inputs_survive_an_interrupt() {
    let mut g = StateGraph::<Doc>::new();
    g.add_node("plan", |s: Doc| async move { Ok(s) });
    g.add_node("summarize", |s: Doc| async move {
        Ok(Doc {
            summaries: vec![s.doc.unwrap().to_uppercase()],
            ..Default::default()
        })
    });
    g.add_send_edges(
        "plan",
        |s: &Doc| {
            s.docs
                .iter()
                .map(|d| {
                    SendTo::new(
                        "summarize",
                        Doc {
                            doc: Some(d.clone()),
                            ..Default::default()
                        },
                    )
                })
                .collect()
        },
        ["summarize"],
    );
    g.add_edge("summarize", END);
    g.set_entry_point("plan");
    g.with_reducer(append_reducer);
    let graph = g
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new())
        .interrupt_before(["summarize"]);
    let input = Doc {
        docs: vec!["x".into(), "y".into()],
        ..Default::default()
    };
    assert!(graph.run("t", Some(input)).await.unwrap().is_interrupted());
    let done = graph.run("t", None).await.unwrap().into_state();
    assert_eq!(done.summaries, vec!["X", "Y"]);
}

#[tokio::test]
async fn failing_branch_fails_the_run_and_cancels_siblings() {
    let finished = Arc::new(Mutex::new(false));
    let mut g = StateGraph::<Doc>::new();
    g.add_node("start", |s: Doc| async move { Ok(s) });
    g.add_node("boom", |_: Doc| async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        Err::<Doc, BoxError>("kaput".into())
    });
    let f = finished.clone();
    g.add_node("slow", move |s: Doc| {
        let f = f.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            *f.lock().unwrap() = true;
            Ok(s)
        }
    });
    g.add_fan_out("start", ["slow", "boom"]);
    g.add_edge("boom", END);
    g.add_edge("slow", END);
    g.set_entry_point("start");
    g.with_reducer(|acc, _| acc);
    let err = g
        .compile()
        .unwrap()
        .invoke(Doc::default())
        .await
        .unwrap_err();
    assert_eq!(err.to_string(), "node `boom` failed: kaput");
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert!(
        !*finished.lock().unwrap(),
        "sibling kept running after the failure"
    );
}

#[tokio::test]
async fn panicking_branch_is_reported_as_that_node() {
    let mut g = StateGraph::<Doc>::new();
    g.add_node("start", |s: Doc| async move { Ok(s) });
    g.add_node("ok", |s: Doc| async move { Ok(s) });
    g.add_node("bad", |s: Doc| async move {
        if s.docs.is_empty() {
            panic!("bad node");
        }
        Ok(s)
    });
    g.add_fan_out("start", ["ok", "bad"]);
    g.add_edge("ok", END);
    g.add_edge("bad", END);
    g.set_entry_point("start");
    g.with_reducer(|acc, _| acc);
    let err = g
        .compile()
        .unwrap()
        .invoke(Doc::default())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, GraphError::Node { node, .. } if node == "bad"),
        "{err}"
    );
}

#[tokio::test]
async fn parallel_nodes_share_the_graph_run_as_parent() {
    let sink = Arc::new(MemorySink::new());
    let graph = diamond(5)
        .compile()
        .unwrap()
        .with_tracer(Tracer::new().with_shared_sink(sink.clone()));
    graph.invoke(Doc::default()).await.unwrap();

    let runs = sink.runs();
    let root = runs.iter().find(|r| r.kind == RunKind::Graph).unwrap();
    let nodes: Vec<_> = runs.iter().filter(|r| r.kind == RunKind::Node).collect();
    assert_eq!(nodes.len(), 4);
    assert!(nodes
        .iter()
        .all(|n| n.parent_id.as_deref() == Some(root.id.as_str()) && n.trace_id == root.trace_id));
    let a = nodes.iter().find(|n| n.name == "a").unwrap();
    let b = nodes.iter().find(|n| n.name == "b").unwrap();
    assert_eq!(a.metadata["step"], json!(1));
    assert_eq!(b.metadata["step"], json!(1));
    assert_eq!(
        (a.metadata["task"].clone(), b.metadata["task"].clone()),
        (json!(0), json!(1))
    );
    assert!(a.error.is_none() && b.error.is_none());
    // Branch runs overlap in time.
    assert!(a.start_ms <= b.end_ms.unwrap() && b.start_ms <= a.end_ms.unwrap());
    let join = nodes.iter().find(|n| n.name == "join").unwrap();
    assert_eq!(join.metadata["step"], json!(2));
}

#[tokio::test]
async fn uneven_branches_reach_the_join_in_different_supersteps() {
    // a -> join, b -> b2 -> join: like Pregel/LangGraph, `join` runs when the
    // first branch arrives and again when the longer branch arrives.
    let mut g = StateGraph::<Doc>::new();
    g.add_node("start", logging("start", 0));
    g.add_node("a", logging("a", 0));
    g.add_node("b", logging("b", 0));
    g.add_node("b2", logging("b2", 0));
    g.add_node("join", logging("join", 0));
    g.add_fan_out("start", ["a", "b"]);
    g.add_edge("a", "join");
    g.add_edge("b", "b2");
    g.add_edge("b2", "join");
    g.add_edge("join", END);
    g.set_entry_point("start");
    g.with_reducer_from_base(log_tail);
    let out = g.compile().unwrap().invoke(Doc::default()).await.unwrap();
    assert_eq!(out.log, vec!["start", "a", "b", "join", "b2", "join"]);
}
