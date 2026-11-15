//! Dynamic interrupts, subgraphs and time travel on a durable checkpointer,
//! across a simulated restart.
#![cfg(feature = "sqlite")]

use serde::{Deserialize, Serialize};
use serde_json::json;
use sinew_checkpoint::SqliteCheckpointer;
use sinew_graph::{CompiledGraph, RunConfig, RunInput, StateGraph, END};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct S {
    log: Vec<String>,
}

fn push(mut s: S, x: &str) -> S {
    s.log.push(x.into());
    s
}

fn build(store: SqliteCheckpointer) -> CompiledGraph<S> {
    let mut inner = StateGraph::<S>::new();
    inner.add_node("prep", |s: S| async move { Ok(push(s, "prep")) });
    inner.add_node_with_context("ask", |s: S, ctx| async move {
        let who: String = ctx.interrupt(json!({ "q": "who?" }))?;
        let ok: bool = ctx.interrupt(json!({ "q": format!("ok for {who}?") }))?;
        Ok(push(s, &format!("ask:{who}:{ok}")))
    });
    inner.add_edge("prep", "ask");
    inner.add_edge("ask", END);
    inner.set_entry_point("prep");

    let mut g = StateGraph::<S>::new();
    g.add_node("start", |s: S| async move { Ok(push(s, "start")) });
    g.add_subgraph("review", inner.compile().unwrap());
    g.add_node("finish", |s: S| async move { Ok(push(s, "finish")) });
    g.add_edge("start", "review");
    g.add_edge("review", "finish");
    g.add_edge("finish", END);
    g.set_entry_point("start");
    g.compile().unwrap().with_checkpointer(store)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupts_in_subgraphs_survive_restarts() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}", dir.path().join("g.db").display());

    {
        let store = SqliteCheckpointer::connect(&url).await.unwrap();
        let out = build(store.clone())
            .run("t", Some(S::default()))
            .await
            .unwrap();
        assert_eq!(out.interrupts()[0].value["q"], "who?");
        store.pool().close().await;
    }
    {
        let store = SqliteCheckpointer::connect(&url).await.unwrap();
        let out = build(store.clone()).resume("t", "Ana").await.unwrap();
        assert_eq!(out.interrupts()[0].value["q"], "ok for Ana?");
        store.pool().close().await;
    }
    let store = SqliteCheckpointer::connect(&url).await.unwrap();
    let graph = build(store);
    let out = graph.resume("t", true).await.unwrap();
    assert!(!out.is_interrupted());
    assert_eq!(
        out.state().log,
        vec!["start", "prep", "ask:Ana:true", "finish"]
    );

    // Time travel over the durable history: fork before `finish`.
    let hist = graph.get_state_history("t").await.unwrap();
    let before_finish = hist
        .iter()
        .find(|h| h.next == vec!["finish".to_string()])
        .unwrap()
        .clone();
    let mut edited = before_finish.state.clone();
    edited.log.push("edited".into());
    graph
        .update_state_with(
            RunConfig::new("t").at_checkpoint(&before_finish.checkpoint_id),
            edited,
            None,
        )
        .await
        .unwrap();
    let out = graph.run_with("t", RunInput::Continue).await.unwrap();
    assert_eq!(out.state().log.last().unwrap(), "finish");
    assert!(out.state().log.contains(&"edited".to_string()));
}
