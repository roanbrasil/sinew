//! Subgraphs: shared and mapped state, interrupts inside, Command::PARENT,
//! namespaced checkpoints and streamed subgraph events.

use std::sync::Arc;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sinew_graph::*;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct S {
    log: Vec<String>,
    #[serde(default)]
    route: Option<String>,
}

fn push(mut s: S, x: &str) -> S {
    s.log.push(x.into());
    s
}

fn child() -> CompiledGraph<S> {
    let mut g = StateGraph::<S>::new();
    g.add_node("c1", |s: S| async move { Ok(push(s, "c1")) });
    g.add_node("c2", |s: S| async move { Ok(push(s, "c2")) });
    g.add_edge("c1", "c2");
    g.add_edge("c2", END);
    g.set_entry_point("c1");
    g.compile().unwrap()
}

fn parent_with(sub: CompiledGraph<S>) -> StateGraph<S> {
    let mut g = StateGraph::<S>::new();
    g.add_node("before", |s: S| async move { Ok(push(s, "before")) });
    g.add_subgraph("sub", sub);
    g.add_node("after", |s: S| async move { Ok(push(s, "after")) });
    g.add_edge("before", "sub");
    g.add_edge("sub", "after");
    g.add_edge("after", END);
    g.set_entry_point("before");
    g
}

#[tokio::test]
async fn shared_state_subgraph_runs_inline() {
    let out = parent_with(child())
        .compile()
        .unwrap()
        .invoke(S::default())
        .await
        .unwrap();
    assert_eq!(out.log, vec!["before", "c1", "c2", "after"]);
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Inner {
    text: String,
    words: usize,
}

#[tokio::test]
async fn mapped_subgraph_translates_state() {
    let mut inner = StateGraph::<Inner>::new();
    inner.add_node("count", |mut s: Inner| async move {
        s.words = s.text.split_whitespace().count();
        Ok(s)
    });
    inner.add_edge("count", END);
    inner.set_entry_point("count");

    let mut g = StateGraph::<S>::new();
    g.add_subgraph_mapped(
        "counter",
        inner.compile().unwrap(),
        |s: &S| Inner {
            text: s.log.join(" "),
            words: 0,
        },
        |mut s: S, c: Inner| {
            s.log.push(format!("{} words", c.words));
            s
        },
    );
    g.add_edge("counter", END);
    g.set_entry_point("counter");
    let out = g
        .compile()
        .unwrap()
        .invoke(S {
            log: vec!["a b".into(), "c".into()],
            route: None,
        })
        .await
        .unwrap();
    assert_eq!(out.log.last().unwrap(), "3 words");
}

fn asking_child() -> CompiledGraph<S> {
    let mut g = StateGraph::<S>::new();
    g.add_node("c1", |s: S| async move { Ok(push(s, "c1")) });
    g.add_node_with_context("ask", |s: S, ctx| async move {
        let a: String = ctx.interrupt(json!({ "q": "color?" }))?;
        Ok(push(s, &format!("ask:{a}")))
    });
    g.add_edge("c1", "ask");
    g.add_edge("ask", END);
    g.set_entry_point("c1");
    g.compile().unwrap()
}

#[tokio::test]
async fn interrupt_inside_subgraph_resumes_inside_it() {
    let cp = Arc::new(MemoryCheckpointer::new());
    let graph = parent_with(asking_child())
        .compile()
        .unwrap()
        .with_shared_checkpointer(cp.clone());

    let out = graph.run("t", Some(S::default())).await.unwrap();
    let intr = &out.interrupts()[0];
    assert_eq!(intr.node, "ask");
    assert_eq!(intr.namespace, vec!["sub:0"]);
    assert_eq!(out.state().log, vec!["before"]);

    // The subgraph checkpointed under its namespace.
    let sub_cp = cp.latest_in("t", "sub:0").await.unwrap().unwrap();
    assert_eq!(sub_cp.next_nodes(), vec!["ask"]);

    let out = graph.resume("t", "blue").await.unwrap();
    assert!(!out.is_interrupted());
    // `c1` ran once: the subgraph resumed from its own checkpoint.
    assert_eq!(out.state().log, vec!["before", "c1", "ask:blue", "after"]);
    // Root history only shows root checkpoints.
    let hist = graph.get_state_history("t").await.unwrap();
    assert!(hist.iter().all(|h| h.state.log.first().map(String::as_str) != Some("c1")));
}

#[tokio::test]
async fn subgraph_interrupt_before_pauses_parent() {
    let mut g = StateGraph::<S>::new();
    g.add_node("c1", |s: S| async move { Ok(push(s, "c1")) });
    g.add_node("c2", |s: S| async move { Ok(push(s, "c2")) });
    g.add_edge("c1", "c2");
    g.add_edge("c2", END);
    g.set_entry_point("c1");
    let sub = g.compile().unwrap().interrupt_before(["c2"]);
    let graph = parent_with(sub)
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());
    let out = graph.run("t", Some(S::default())).await.unwrap();
    assert!(out.is_interrupted());
    assert_eq!(out.interrupts()[0].namespace, vec!["sub:0"]);
    let out = graph.run("t", None).await.unwrap();
    assert_eq!(out.into_state().log, vec!["before", "c1", "c2", "after"]);
}

#[tokio::test]
async fn command_parent_routes_the_parent_graph() {
    let mut inner = StateGraph::<S>::new();
    inner.add_command_node("decide", Vec::<String>::new(), |s: S, _ctx| async move {
        // Leave the subgraph and send the parent to `escalate`.
        Ok(Command::goto("escalate")
            .with_update(push(s, "decide"))
            .with_json_update(json!({ "route": "escalate" }))
            .to_parent())
    });
    inner.add_node("never", |s: S| async move { Ok(push(s, "never")) });
    inner.add_edge("never", END);
    inner.set_entry_point("decide");

    let mut g = StateGraph::<S>::new();
    g.add_subgraph("triage", inner.compile().unwrap());
    g.set_node_destinations("triage", ["escalate"]);
    g.add_node("escalate", |s: S| async move { Ok(push(s, "escalate")) });
    g.add_node("normal", |s: S| async move { Ok(push(s, "normal")) });
    g.add_edge("triage", END);
    g.add_edge("escalate", END);
    g.add_edge("normal", END);
    g.add_conditional_edges(START, |_: &S| "triage".into(), ["triage", "normal"]);
    let out = g.compile().unwrap().invoke(S::default()).await.unwrap();
    assert_eq!(out.log, vec!["decide", "escalate"]);
    assert_eq!(out.route.as_deref(), Some("escalate"));
}

#[tokio::test]
async fn subgraph_events_stream_with_namespace() {
    let graph = parent_with(child()).compile().unwrap();
    let events: Vec<_> = graph
        .stream_with(
            "t",
            S::default(),
            StreamOptions::new([StreamMode::Updates]).with_subgraphs(true),
        )
        .collect()
        .await;
    let sub: Vec<(Vec<String>, String)> = events
        .iter()
        .filter_map(|e| match e {
            GraphEvent::Subgraph { namespace, event } => match event.as_ref() {
                GraphEvent::NodeEnd { node, .. } => Some((namespace.clone(), node.clone())),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(
        sub,
        vec![
            (vec!["sub:0".to_string()], "c1".to_string()),
            (vec!["sub:0".to_string()], "c2".to_string())
        ]
    );
    // Without `subgraphs`, only root events.
    let plain: Vec<_> = graph.stream("t2", Some(S::default())).collect().await;
    assert!(!plain
        .iter()
        .any(|e| matches!(e, GraphEvent::Subgraph { .. })));
}

#[tokio::test]
async fn nested_subgraph_namespaces_flatten() {
    let mut mid = StateGraph::<S>::new();
    mid.add_subgraph("inner", child());
    mid.add_edge("inner", END);
    mid.set_entry_point("inner");
    let mut top = StateGraph::<S>::new();
    top.add_subgraph("mid", mid.compile().unwrap());
    top.add_edge("mid", END);
    top.set_entry_point("mid");
    let events: Vec<_> = top
        .compile()
        .unwrap()
        .stream_with(
            "t",
            S::default(),
            StreamOptions::new([StreamMode::Updates]).with_subgraphs(true),
        )
        .collect()
        .await;
    let deepest = events.iter().find_map(|e| match e {
        GraphEvent::Subgraph { namespace, event } if namespace.len() == 2 => {
            Some((namespace.clone(), event.as_ref()))
        }
        _ => None,
    });
    let (ns, ev) = deepest.expect("an event two levels down");
    assert_eq!(ns, vec!["mid:0", "inner:0"]);
    assert!(matches!(ev, GraphEvent::NodeEnd { node, .. } if node == "c1"));
}
