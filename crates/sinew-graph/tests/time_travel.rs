//! History, replay and forks.

use serde::{Deserialize, Serialize};
use sinew_graph::*;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct S {
    n: i64,
    log: Vec<String>,
}

fn graph() -> CompiledGraph<S> {
    let mut g = StateGraph::<S>::new();
    g.add_node("double", |mut s: S| async move {
        s.n *= 2;
        s.log.push(format!("double={}", s.n));
        Ok(s)
    });
    g.add_node("inc", |mut s: S| async move {
        s.n += 1;
        s.log.push(format!("inc={}", s.n));
        Ok(s)
    });
    g.add_edge("double", "inc");
    g.add_edge("inc", END);
    g.set_entry_point("double");
    g.compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new())
}

#[tokio::test]
async fn history_is_newest_first_and_linked() {
    let g = graph();
    g.run("t", Some(S { n: 3, log: vec![] })).await.unwrap();
    let hist = g.get_state_history("t").await.unwrap();
    let ns: Vec<i64> = hist.iter().map(|h| h.state.n).collect();
    assert_eq!(ns, vec![7, 6, 3]);
    let next: Vec<Vec<String>> = hist.iter().map(|h| h.next.clone()).collect();
    assert_eq!(
        next,
        vec![vec![], vec!["inc".to_string()], vec!["double".to_string()]]
    );
    assert_eq!(hist[0].parent_id.as_deref(), Some(hist[1].checkpoint_id.as_str()));
    assert_eq!(hist[1].parent_id.as_deref(), Some(hist[2].checkpoint_id.as_str()));
    assert_eq!(hist[2].parent_id, None);
    assert_eq!(hist[2].metadata["source"], "input");
    assert_eq!(hist[1].metadata["writes"], serde_json::json!(["double"]));
}

#[tokio::test]
async fn replay_from_a_past_checkpoint_creates_a_branch() {
    let g = graph();
    g.run("t", Some(S { n: 3, log: vec![] })).await.unwrap();
    let hist = g.get_state_history("t").await.unwrap();
    let after_double = hist[1].clone();

    let out = g
        .run_with(
            RunConfig::new("t").at_checkpoint(&after_double.checkpoint_id),
            RunInput::Continue,
        )
        .await
        .unwrap();
    assert_eq!(out.state().n, 7);
    let hist = g.get_state_history("t").await.unwrap();
    assert_eq!(hist.len(), 4);
    // The re-executed step hangs off the replayed checkpoint.
    assert_eq!(
        hist[0].parent_id.as_deref(),
        Some(after_double.checkpoint_id.as_str())
    );
    let at = g
        .get_state_at("t", &after_double.checkpoint_id)
        .await
        .unwrap();
    assert_eq!(at.state.n, 6);
}

#[tokio::test]
async fn fork_edits_a_past_state() {
    let g = graph();
    g.run("t", Some(S { n: 3, log: vec![] })).await.unwrap();
    let hist = g.get_state_history("t").await.unwrap();
    let after_double = &hist[1];
    let mut edited = after_double.state.clone();
    edited.n = 100;
    let fork_id = g
        .update_state_with(
            RunConfig::new("t").at_checkpoint(&after_double.checkpoint_id),
            edited,
            None,
        )
        .await
        .unwrap();
    let snap = g.get_state_snapshot("t").await.unwrap();
    assert_eq!(snap.checkpoint_id, fork_id);
    assert_eq!(snap.next, vec!["inc"]);
    assert_eq!(snap.metadata["source"], "fork");

    let out = g.run("t", None).await.unwrap();
    assert_eq!(out.state().n, 101);
    // The original branch is still there.
    let original = g.get_state_at("t", &hist[0].checkpoint_id).await.unwrap();
    assert_eq!(original.state.n, 7);
}

#[tokio::test]
async fn update_as_node_moves_the_thread() {
    let g = graph();
    g.run("t", Some(S { n: 1, log: vec![] })).await.unwrap();
    // Pretend `double` produced n = 50: next is `inc`.
    g.update_state_with(
        "t",
        S {
            n: 50,
            log: vec![],
        },
        Some("double"),
    )
    .await
    .unwrap();
    let snap = g.get_state_snapshot("t").await.unwrap();
    assert_eq!(snap.next, vec!["inc"]);
    assert_eq!(g.run("t", None).await.unwrap().state().n, 51);
    assert!(g
        .update_state_with("t", S::default(), Some("ghost"))
        .await
        .is_err());
}

#[tokio::test]
async fn unknown_checkpoint_id_is_reported() {
    let g = graph();
    g.run("t", Some(S::default())).await.unwrap();
    let err = g
        .run_with(RunConfig::new("t").at_checkpoint("nope"), RunInput::Continue)
        .await
        .unwrap_err();
    assert!(matches!(err, GraphError::CheckpointNotFound { .. }));
}
