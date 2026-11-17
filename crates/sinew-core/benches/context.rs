//! Context strategies on long transcripts.
//!
//! ```bash
//! cargo bench -p sinew-core --bench context
//! ```
//!
//! Each transcript has a system prompt and user turns with several rounds of
//! parallel tool calls and large results. Before measuring, every policy is
//! checked once: the output fits the budget and tool calls stay paired.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde_json::json;
use sinew_core::context::{
    check_tool_pairing, ClearToolResults, ContextPolicy, KeepLast, KeepRecent, Summarize,
};
use sinew_core::{mock::ScriptedModel, Message, ToolCall};

fn transcript(n: usize) -> Vec<Message> {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move |m: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % m
    };
    let mut out = vec![Message::system("You are a careful research agent.")];
    let mut id = 0;
    while out.len() < n {
        out.push(Message::user(format!("look up topic {}", next(1000))));
        for _ in 0..next(4) {
            let calls: Vec<ToolCall> = (0..1 + next(3))
                .map(|_| {
                    id += 1;
                    ToolCall {
                        id: format!("c{id}"),
                        name: "search".into(),
                        arguments: json!({ "q": id }),
                    }
                })
                .collect();
            let results: Vec<Message> = calls
                .iter()
                .map(|c| {
                    Message::tool_result(
                        c.id.clone(),
                        "result data ".repeat(1 + next(250) as usize),
                    )
                })
                .collect();
            out.push(Message::assistant_with_tools("checking", calls));
            out.extend(results);
        }
        out.push(Message::assistant("answer ".repeat(1 + next(40) as usize)));
    }
    out
}

fn summarizer() -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new(
        (0..100_000).map(|_| Message::assistant("summary of the earlier work")),
    ))
}

fn policies(budget: usize) -> Vec<(&'static str, ContextPolicy)> {
    vec![
        (
            "keep_last",
            ContextPolicy::new(budget).then(KeepLast::fit_budget()),
        ),
        ("trimming", ContextPolicy::trimming(budget)),
        (
            "summarizing",
            ContextPolicy::new(budget)
                .then(ClearToolResults::new())
                .then(Summarize::new(summarizer()).keep_recent(KeepRecent::Tokens(budget / 2)))
                .then(KeepLast::fit_budget()),
        ),
    ]
}

fn bench(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("context_policy");
    for n in [100usize, 500, 2_000] {
        let msgs = transcript(n);
        group.throughput(Throughput::Elements(msgs.len() as u64));
        for (name, policy) in policies(8_000) {
            let report = rt
                .block_on(policy.apply_with_report(msgs.clone(), 0))
                .unwrap();
            assert!(
                report.fits(),
                "{name}/{n}: {} > {}",
                report.tokens_after,
                report.budget
            );
            check_tool_pairing(&report.messages).unwrap();
            group.bench_with_input(BenchmarkId::new(name, n), &msgs, |b, msgs| {
                b.iter(|| rt.block_on(policy.apply(msgs.clone())).unwrap())
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
