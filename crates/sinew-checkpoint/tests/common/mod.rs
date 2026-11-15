//! Backend-independent test scenarios, run against SQLite and Postgres.
#![allow(dead_code)]

use std::sync::Arc;

use serde_json::json;
use sinew_checkpoint::{Checkpoint, CheckpointStore};
use sinew_graph::new_checkpoint_id;

pub fn cp(thread: &str, step: usize, next: &str, n: i64) -> Checkpoint {
    Checkpoint {
        thread_id: thread.into(),
        step,
        next: next.into(),
        tasks: Vec::new(),
        state: json!({ "n": n }),
        checkpoint_id: new_checkpoint_id(),
        ..Checkpoint::default()
    }
}

fn ns(history: &[Checkpoint]) -> Vec<i64> {
    history
        .iter()
        .map(|c| c.state["n"].as_i64().unwrap())
        .collect()
}

pub async fn round_trip<S: CheckpointStore>(store: &S) {
    assert_eq!(store.latest("t").await.unwrap(), None);
    assert!(store.history("t").await.unwrap().is_empty());

    let rich = Checkpoint {
        thread_id: "t".into(),
        step: 7,
        next: "review".into(),
        tasks: Vec::new(),
        state: json!({
            "text": "olá, \"world\" \u{1F680}\n",
            "nested": { "list": [1, 2.5, -3, null, true], "empty": {} },
            "big": u64::MAX,
            "neg": i64::MIN,
            "float": 0.1,
        }),
        checkpoint_id: new_checkpoint_id(),
        ..Checkpoint::default()
    };
    store.put(rich.clone()).await.unwrap();
    assert_eq!(store.latest("t").await.unwrap(), Some(rich.clone()));
    assert_eq!(store.history("t").await.unwrap(), vec![rich]);
}

pub async fn ordering<S: CheckpointStore>(store: &S) {
    store.put(cp("t", 0, "a", 0)).await.unwrap();
    store.put(cp("t", 1, "b", 1)).await.unwrap();
    store.put(cp("t", 2, "c", 2)).await.unwrap();
    // `update_state` writes a second checkpoint with the same step.
    store.put(cp("t", 2, "c", 3)).await.unwrap();
    // A late write with a lower step sorts by step in the history...
    store.put(cp("t", 1, "b", 4)).await.unwrap();

    let history = store.history("t").await.unwrap();
    assert_eq!(ns(&history), vec![0, 1, 4, 2, 3]);
    let steps: Vec<usize> = history.iter().map(|c| c.step).collect();
    assert_eq!(steps, vec![0, 1, 1, 2, 2]);
    // ...but `latest` is the most recently written one.
    assert_eq!(store.latest("t").await.unwrap().unwrap().state["n"], 4);
}

pub async fn multiple_threads<S: CheckpointStore>(store: &S) {
    store.put(cp("a", 0, "x", 1)).await.unwrap();
    store.put(cp("b", 0, "x", 10)).await.unwrap();
    store.put(cp("a", 1, "y", 2)).await.unwrap();
    store.put(cp("c", 0, "x", 100)).await.unwrap();

    assert_eq!(ns(&store.history("a").await.unwrap()), vec![1, 2]);
    assert_eq!(ns(&store.history("b").await.unwrap()), vec![10]);
    assert_eq!(store.latest("c").await.unwrap().unwrap().state["n"], 100);
    // Most recently written first.
    assert_eq!(store.list_threads().await.unwrap(), vec!["c", "a", "b"]);

    assert_eq!(store.delete_thread("a").await.unwrap(), 2);
    assert_eq!(store.delete_thread("a").await.unwrap(), 0);
    assert_eq!(store.latest("a").await.unwrap(), None);
    assert_eq!(store.list_threads().await.unwrap(), vec!["c", "b"]);
    assert_eq!(ns(&store.history("b").await.unwrap()), vec![10]);
}

pub async fn pruning<S: CheckpointStore>(store: &S) {
    for i in 0..5 {
        store.put(cp("p", i, "n", i as i64)).await.unwrap();
        store.put(cp("q", i, "n", 10 + i as i64)).await.unwrap();
    }
    store.put(cp("r", 0, "n", 99)).await.unwrap();

    assert_eq!(store.prune("p", 2).await.unwrap(), 3);
    assert_eq!(ns(&store.history("p").await.unwrap()), vec![3, 4]);
    assert_eq!(store.prune("p", 2).await.unwrap(), 0);
    assert_eq!(store.history("q").await.unwrap().len(), 5);

    // q: 5 -> 3, p: 2 (untouched), r: 1 (untouched).
    assert_eq!(store.prune_all(3).await.unwrap(), 2);
    assert_eq!(ns(&store.history("q").await.unwrap()), vec![12, 13, 14]);
    assert_eq!(ns(&store.history("p").await.unwrap()), vec![3, 4]);
    assert_eq!(ns(&store.history("r").await.unwrap()), vec![99]);

    assert_eq!(store.prune("q", 0).await.unwrap(), 3);
    assert_eq!(store.latest("q").await.unwrap(), None);
}

/// `store` must have been built with `with_retention(3)`.
pub async fn retention_on_put<S: CheckpointStore>(store: &S) {
    for i in 0..10 {
        store.put(cp("keep3", i, "n", i as i64)).await.unwrap();
    }
    store.put(cp("other", 0, "n", 42)).await.unwrap();
    assert_eq!(ns(&store.history("keep3").await.unwrap()), vec![7, 8, 9]);
    assert_eq!(store.latest("keep3").await.unwrap().unwrap().step, 9);
    assert_eq!(ns(&store.history("other").await.unwrap()), vec![42]);
}

/// Many tasks writing at once, to one shared thread and to their own threads.
pub async fn concurrent_puts<S: CheckpointStore + 'static>(store: Arc<S>) {
    const TASKS: usize = 8;
    const PUTS: usize = 25;
    let mut handles = Vec::new();
    for t in 0..TASKS {
        let store = store.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..PUTS {
                let n = (t * 1000 + i) as i64;
                store.put(cp("shared", i, "n", n)).await.unwrap();
                store.put(cp(&format!("own-{t}"), i, "n", n)).await.unwrap();
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let shared = store.history("shared").await.unwrap();
    assert_eq!(shared.len(), TASKS * PUTS);
    let mut seen = ns(&shared);
    seen.sort_unstable();
    let mut expected: Vec<i64> = (0..TASKS)
        .flat_map(|t| (0..PUTS).map(move |i| (t * 1000 + i) as i64))
        .collect();
    expected.sort_unstable();
    assert_eq!(seen, expected);
    // History is sorted by step.
    assert!(shared.windows(2).all(|w| w[0].step <= w[1].step));

    for t in 0..TASKS {
        let own = store.history(&format!("own-{t}")).await.unwrap();
        let expected: Vec<i64> = (0..PUTS).map(|i| (t * 1000 + i) as i64).collect();
        assert_eq!(ns(&own), expected);
    }
    assert_eq!(store.list_threads().await.unwrap().len(), TASKS + 1);
}

/// Concurrent writers with retention enabled never exceed the limit.
/// `store` must have been built with `with_retention(5)`.
pub async fn concurrent_puts_with_retention<S: CheckpointStore + 'static>(store: Arc<S>) {
    let mut handles = Vec::new();
    for t in 0..6 {
        let store = store.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..20 {
                store
                    .put(cp("hot", i, "n", t * 100 + i as i64))
                    .await
                    .unwrap();
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    assert_eq!(store.history("hot").await.unwrap().len(), 5);
}

/// Namespaces, lookups by id and write-order history (time travel).
pub async fn namespaces_and_ids<S: CheckpointStore>(store: &S) {
    let root = cp("t", 0, "a", 1);
    let sub = cp("t", 0, "x", 2).with_namespace("sub:0");
    let root2 = cp("t", 1, "b", 3).with_parent(Some(root.checkpoint_id.clone()));
    // A fork written later with a lower step.
    let fork = cp("t", 0, "a", 4).with_parent(Some(root.checkpoint_id.clone()));
    for c in [&root, &sub, &root2, &fork] {
        store.put(c.clone()).await.unwrap();
    }
    assert_eq!(store.latest("t").await.unwrap(), Some(fork.clone()));
    assert_eq!(
        store.latest_in("t", "sub:0").await.unwrap(),
        Some(sub.clone())
    );
    assert_eq!(store.latest_in("t", "nope").await.unwrap(), None);
    assert_eq!(
        store.get("t", &root2.checkpoint_id).await.unwrap(),
        Some(root2.clone())
    );
    assert_eq!(
        store.get("other", &root2.checkpoint_id).await.unwrap(),
        None
    );
    assert_eq!(
        ns(&store.history_in("t", "").await.unwrap()),
        vec![1, 3, 4],
        "write order, root namespace only"
    );
    assert_eq!(store.history("t").await.unwrap().len(), 4);
}
