//! In-memory vector search throughput.
//!
//! ```bash
//! cargo bench -p sinew-rag --bench search
//! ```

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sinew_rag::{Document, FakeEmbeddings, Filter, InMemoryVectorStore, VectorStore};

/// Deterministic pseudo-random vectors (xorshift), no extra dependency.
fn vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s % 2000) as f32 / 1000.0 - 1.0
                })
                .collect()
        })
        .collect()
}

fn build(n: usize, dim: usize, threads: usize) -> InMemoryVectorStore {
    let store = InMemoryVectorStore::new(FakeEmbeddings::new(dim)).with_threads(threads);
    let docs: Vec<Document> = (0..n)
        .map(|i| Document::new(format!("doc {i}")).with_metadata("bucket", (i % 10) as u64))
        .collect();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(store.add_vectors(docs, vectors(n, dim, 42)))
        .unwrap();
    store
}

fn search(c: &mut Criterion) {
    let dim = 384;
    let query = vectors(1, dim, 7).pop().unwrap();
    let mut g = c.benchmark_group("in_memory_top10");
    for &n in &[1_000usize, 10_000, 100_000] {
        g.throughput(Throughput::Elements(n as u64));
        let serial = build(n, dim, 1);
        g.bench_with_input(BenchmarkId::new("1_thread", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    serial
                        .search_vector_sync(black_box(&query), 10, None)
                        .unwrap(),
                )
            })
        });
        let parallel = build(n, dim, 8).with_parallel_threshold(0);
        g.bench_with_input(BenchmarkId::new("8_threads", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    parallel
                        .search_vector_sync(black_box(&query), 10, None)
                        .unwrap(),
                )
            })
        });
        let filter = Filter::eq("bucket", 3);
        g.bench_with_input(BenchmarkId::new("filtered", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    serial
                        .search_vector_sync(black_box(&query), 10, Some(&filter))
                        .unwrap(),
                )
            })
        });
    }
    g.finish();
}

criterion_group!(benches, search);
criterion_main!(benches);
