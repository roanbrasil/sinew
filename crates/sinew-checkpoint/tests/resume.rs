//! End to end: a graph paused by `interrupt_before` survives a "restart"
//! (everything dropped, the database re-opened by a new graph instance).
#![cfg(feature = "sqlite")]

use serde::{Deserialize, Serialize};
use sinew_checkpoint::{CheckpointStore, Checkpointer, SqliteCheckpointer};
use sinew_graph::{CompiledGraph, StateGraph, END};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Draft {
    text: String,
    revisions: u32,
    published: bool,
}

/// write -> (loop until 2 revisions) -> publish -> END, pausing before publish.
fn build(store: SqliteCheckpointer) -> CompiledGraph<Draft> {
    let mut g = StateGraph::<Draft>::new();
    g.add_node("write", |mut s: Draft| async move {
        s.revisions += 1;
        s.text.push_str(&format!(" r{}", s.revisions));
        Ok(s)
    });
    g.add_node("publish", |mut s: Draft| async move {
        s.published = true;
        Ok(s)
    });
    g.add_conditional_edges(
        "write",
        |s: &Draft| {
            if s.revisions < 2 {
                "write".into()
            } else {
                "publish".into()
            }
        },
        ["write", "publish"],
    );
    g.add_edge("publish", END);
    g.set_entry_point("write");
    g.compile()
        .unwrap()
        .with_checkpointer(store)
        .interrupt_before(["publish"])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_after_restart() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}", dir.path().join("agent.db").display());

    // "Process" 1: run until the interrupt, then drop everything.
    {
        let store = SqliteCheckpointer::connect(&url).await.unwrap();
        let graph = build(store.clone());
        let input = Draft {
            text: "draft".into(),
            revisions: 0,
            published: false,
        };
        let out = graph.run("doc-1", Some(input)).await.unwrap();
        assert!(out.is_interrupted());
        assert_eq!(out.state().text, "draft r1 r2");
        assert!(!out.state().published);
        drop(graph);
        store.pool().close().await;
    }

    // "Process" 2: a brand-new graph on the re-opened database.
    let store = SqliteCheckpointer::connect(&url).await.unwrap();
    let graph = build(store.clone());

    let (mut state, next) = graph.get_state("doc-1").await.unwrap();
    assert_eq!(next, "publish");
    assert_eq!(state.revisions, 2);

    // A human edits the paused state.
    state.text = "edited by a human".into();
    graph.update_state("doc-1", state).await.unwrap();

    let done = graph.run("doc-1", None).await.unwrap();
    assert!(!done.is_interrupted());
    let done = done.into_state();
    assert!(done.published);
    assert_eq!(done.text, "edited by a human");

    // The full history is there, ordered by step.
    let history = store.history("doc-1").await.unwrap();
    assert!(history.windows(2).all(|w| w[0].step <= w[1].step));
    assert_eq!(history.last().unwrap().next, END);
    assert_eq!(store.list_threads().await.unwrap(), vec!["doc-1"]);

    // Resuming a finished thread after another restart returns the final state.
    drop(graph);
    store.pool().close().await;
    let graph = build(SqliteCheckpointer::connect(&url).await.unwrap());
    let again = graph.run("doc-1", None).await.unwrap();
    assert!(!again.is_interrupted());
    assert_eq!(again.into_state(), done);
}
