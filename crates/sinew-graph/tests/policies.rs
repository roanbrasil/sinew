//! Retry, cache and timeout policies, stream modes, and the store from nodes.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sinew_graph::{store::SearchQuery, *};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct S {
    n: i64,
    log: Vec<String>,
}

fn single(name: &str, g: &mut StateGraph<S>) {
    g.add_edge(name, END);
    g.set_entry_point(name);
}

#[tokio::test]
async fn retry_until_success() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let mut g = StateGraph::<S>::new();
    g.add_node("flaky", move |mut s: S| {
        let c = c.clone();
        async move {
            if c.fetch_add(1, Ordering::SeqCst) < 2 {
                return Err("transient".into());
            }
            s.n = 1;
            Ok(s)
        }
    });
    single("flaky", &mut g);
    g.set_node_retry(
        "flaky",
        RetryPolicy::new(3).with_initial_interval(Duration::from_millis(1)),
    );
    let out = g.compile().unwrap().invoke(S::default()).await.unwrap();
    assert_eq!(out.n, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn retry_gives_up_and_respects_predicate() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let mut g = StateGraph::<S>::new();
    g.add_node("bad", move |_s: S| {
        let c = c.clone();
        async move {
            c.fetch_add(1, Ordering::SeqCst);
            Err::<S, BoxError>("fatal".into())
        }
    });
    single("bad", &mut g);
    g.set_node_retry(
        "bad",
        RetryPolicy::new(5)
            .with_initial_interval(Duration::from_millis(1))
            .retry_on(|e| e.to_string() != "fatal"),
    );
    let err = g.compile().unwrap().invoke(S::default()).await.unwrap_err();
    assert_eq!(err.to_string(), "node `bad` failed: fatal");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn timeout_fails_and_is_retried() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let mut g = StateGraph::<S>::new();
    g.add_node("slow", move |s: S| {
        let c = c.clone();
        async move {
            if c.fetch_add(1, Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Ok(s)
        }
    });
    single("slow", &mut g);
    g.set_node_timeout("slow", Duration::from_millis(20));
    let graph = g.compile().unwrap();
    let err = graph.invoke(S::default()).await.unwrap_err();
    let GraphError::Node { source, .. } = &err else {
        panic!("{err}")
    };
    assert!(source.downcast_ref::<NodeTimeout>().is_some());
    assert!(err.to_string().contains("timed out"));

    // With a retry policy the second (fast) attempt succeeds.
    let c = Arc::new(AtomicUsize::new(0));
    let c2 = c.clone();
    let mut g = StateGraph::<S>::new();
    g.add_node("slow", move |s: S| {
        let c = c2.clone();
        async move {
            if c.fetch_add(1, Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Ok(s)
        }
    });
    single("slow", &mut g);
    g.set_node_timeout("slow", Duration::from_millis(20));
    g.set_node_retry(
        "slow",
        RetryPolicy::new(2).with_initial_interval(Duration::from_millis(1)),
    );
    g.compile().unwrap().invoke(S::default()).await.unwrap();
    assert_eq!(c.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn cache_skips_repeated_inputs() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let mut g = StateGraph::<S>::new();
    g.add_node("expensive", move |mut s: S| {
        let c = c.clone();
        async move {
            c.fetch_add(1, Ordering::SeqCst);
            s.log.push(format!("sq={}", s.n * s.n));
            Ok(s)
        }
    });
    single("expensive", &mut g);
    g.set_node_cache("expensive", CachePolicy::new());
    let graph = g.compile().unwrap();
    let a = graph.invoke(S { n: 4, log: vec![] }).await.unwrap();
    let b = graph.clone().invoke(S { n: 4, log: vec![] }).await.unwrap();
    graph.invoke(S { n: 5, log: vec![] }).await.unwrap();
    assert_eq!(a, b);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    graph.clear_cache();
    graph.invoke(S { n: 4, log: vec![] }).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn recursion_limit_per_run_config() {
    let mut g = StateGraph::<S>::new();
    g.add_node("spin", |mut s: S| async move {
        s.n += 1;
        Ok(s)
    });
    g.add_conditional_edges(
        "spin",
        |s: &S| if s.n < 10 { "spin".into() } else { END.into() },
        ["spin", END],
    );
    g.set_entry_point("spin");
    let graph = g.compile().unwrap();
    assert!(graph.invoke(S::default()).await.is_ok());
    let err = graph
        .run_with(RunConfig::new("t").with_recursion_limit(3), S::default())
        .await
        .unwrap_err();
    assert!(matches!(err, GraphError::RecursionLimit(3)));
}

fn two_steps() -> StateGraph<S> {
    let mut g = StateGraph::<S>::new();
    g.add_node_with_context("a", |mut s: S, ctx| async move {
        ctx.emit_token("tok");
        ctx.emit(json!({"progress": 1}));
        s.n += 1;
        Ok(s)
    });
    g.add_node("b", |mut s: S| async move {
        s.n += 10;
        Ok(s)
    });
    g.add_edge("a", "b");
    g.add_edge("b", END);
    g.set_entry_point("a");
    g
}

fn kinds(events: &[GraphEvent<S>]) -> Vec<&'static str> {
    events
        .iter()
        .map(|e| match e {
            GraphEvent::NodeStart { .. } => "start",
            GraphEvent::NodeEnd { .. } => "end",
            GraphEvent::Token { .. } => "token",
            GraphEvent::Custom { .. } => "custom",
            GraphEvent::Values { .. } => "values",
            GraphEvent::Checkpoint { .. } => "checkpoint",
            GraphEvent::Subgraph { .. } => "subgraph",
            GraphEvent::Interrupted { .. } => "interrupted",
            GraphEvent::Completed { .. } => "completed",
            GraphEvent::Error(_) => "error",
        })
        .collect()
}

#[tokio::test]
async fn stream_modes_select_events() {
    let graph = two_steps().compile().unwrap();
    let values: Vec<_> = graph
        .stream_with("t", S::default(), StreamMode::Values)
        .collect()
        .await;
    assert_eq!(
        kinds(&values),
        vec!["values", "values", "values", "completed"]
    );
    let ns: Vec<i64> = values
        .iter()
        .filter_map(|e| match e {
            GraphEvent::Values { state, .. } => Some(state.n),
            _ => None,
        })
        .collect();
    assert_eq!(ns, vec![0, 1, 11]);

    let msgs: Vec<_> = graph
        .stream_with(
            "t",
            S::default(),
            [StreamMode::Messages, StreamMode::Custom],
        )
        .collect()
        .await;
    assert_eq!(kinds(&msgs), vec!["token", "custom", "completed"]);

    let updates: Vec<_> = graph
        .stream_with("t", S::default(), StreamMode::Updates)
        .collect()
        .await;
    assert_eq!(kinds(&updates), vec!["end", "end", "completed"]);

    let debug: Vec<_> = graph
        .clone()
        .with_checkpointer(MemoryCheckpointer::new())
        .stream_with("t", S::default(), StreamMode::Debug)
        .collect()
        .await;
    let k = kinds(&debug);
    assert_eq!(k.iter().filter(|k| **k == "checkpoint").count(), 3);
    assert!(k.contains(&"start") && k.contains(&"end") && k.contains(&"token"));

    // The default stream is unchanged: tasks, updates, messages, custom.
    let default: Vec<_> = graph.stream("t", Some(S::default())).collect().await;
    assert_eq!(
        kinds(&default),
        vec![
            "start",
            "token",
            "custom",
            "end",
            "start",
            "end",
            "completed"
        ]
    );
}

#[tokio::test]
async fn nodes_use_the_store() {
    let mut g = StateGraph::<S>::new();
    g.add_node_with_context("remember", |mut s: S, ctx| async move {
        let store = ctx.store().expect("store attached");
        let ns = ["memories", "ana"];
        let seen = store
            .get(&ns, "visits")
            .await?
            .map_or(0, |i| i.value.as_i64().unwrap_or(0));
        store.put(&ns, "visits", json!(seen + 1)).await?;
        s.n = seen + 1;
        Ok(s)
    });
    single("remember", &mut g);
    let graph = g.compile().unwrap().with_store(InMemoryStore::new());
    graph.invoke(S::default()).await.unwrap();
    let out = graph.invoke(S::default()).await.unwrap();
    assert_eq!(out.n, 2);
    let hits = graph
        .store()
        .unwrap()
        .search(&["memories"], SearchQuery::new())
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
}
