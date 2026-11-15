//! Superstep overhead of the graph engine.
//!
//! ```bash
//! cargo bench -p sinew-graph --bench superstep
//! ```
//!
//! Every node is trivial, so the numbers are the engine's own cost per run
//! (and, divided by the node count, per superstep).

use async_trait::async_trait;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde::{Deserialize, Serialize};
use sinew_graph::{Checkpoint, Checkpointer, CompiledGraph, StateGraph, END};

/// Serializes like a real checkpointer would (the graph builds every
/// checkpoint) but stores nothing, so memory stays flat across iterations.
struct NullCheckpointer;

#[async_trait]
impl Checkpointer for NullCheckpointer {
    async fn put(&self, checkpoint: Checkpoint) -> Result<(), String> {
        std::hint::black_box(checkpoint);
        Ok(())
    }
    async fn latest(&self, _: &str) -> Result<Option<Checkpoint>, String> {
        Ok(None)
    }
    async fn history(&self, _: &str) -> Result<Vec<Checkpoint>, String> {
        Ok(Vec::new())
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct S {
    n: u64,
    history: Vec<String>,
}

const CHAIN: usize = 10;

fn chain() -> CompiledGraph<S> {
    let mut g = StateGraph::<S>::new();
    for i in 0..CHAIN {
        g.add_node(format!("n{i}"), |mut s: S| async move {
            s.n += 1;
            Ok(s)
        });
        let next = if i + 1 == CHAIN {
            END.to_string()
        } else {
            format!("n{}", i + 1)
        };
        g.add_edge(format!("n{i}"), next);
    }
    g.set_entry_point("n0");
    g.compile().unwrap().with_recursion_limit(CHAIN + 1)
}

fn fan_out(width: usize) -> CompiledGraph<S> {
    let mut g = StateGraph::<S>::new();
    let branches: Vec<String> = (0..width).map(|i| format!("b{i}")).collect();
    g.add_node("start", |s: S| async move { Ok(s) });
    for b in &branches {
        g.add_node(b.clone(), |mut s: S| async move {
            s.n = 1;
            Ok(s)
        });
        g.add_edge(b.clone(), "join");
    }
    g.add_node("join", |s: S| async move { Ok(s) });
    g.add_fan_out("start", branches);
    g.add_edge("join", END);
    g.with_reducer(|mut acc: S, b: S| {
        acc.n += b.n;
        acc
    });
    g.set_entry_point("start");
    g.compile().unwrap()
}

fn state(history: usize) -> S {
    S {
        n: 0,
        history: (0..history)
            .map(|i| format!("message number {i}"))
            .collect(),
    }
}

fn bench(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("chain10");
    group.throughput(Throughput::Elements(CHAIN as u64));
    for history in [0usize, 1000] {
        let g = chain();
        let input = state(history);
        group.bench_with_input(
            BenchmarkId::new("no_checkpointer", history),
            &input,
            |b, input| b.iter(|| rt.block_on(g.invoke(input.clone())).unwrap()),
        );
        let g = chain().with_checkpointer(NullCheckpointer);
        group.bench_with_input(
            BenchmarkId::new("checkpointing", history),
            &input,
            |b, input| b.iter(|| rt.block_on(g.invoke(input.clone())).unwrap()),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("fan_out");
    for width in [2usize, 8] {
        let g = fan_out(width);
        let input = state(0);
        group.bench_with_input(BenchmarkId::from_parameter(width), &input, |b, input| {
            b.iter(|| rt.block_on(g.invoke(input.clone())).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
