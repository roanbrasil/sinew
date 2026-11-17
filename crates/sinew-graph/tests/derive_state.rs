//! `#[derive(State)]`: field-wise reducers.
#![cfg(feature = "macros")]

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use sinew_graph::{Reduce, SendTo, State, StateGraph, END};

fn longest(_base: &String, acc: String, branch: String) -> String {
    if branch.len() > acc.len() {
        branch
    } else {
        acc
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, State)]
struct S {
    #[state(append)]
    log: Vec<String>,
    #[state(add)]
    hits: u32,
    #[state(add)]
    cost: f64,
    status: String,
    #[state(overwrite)]
    last: u8,
    #[state(keep)]
    owner: String,
    #[state(merge)]
    scores: HashMap<String, i32>,
    #[state(merge)]
    tags: BTreeMap<String, bool>,
    #[state(reducer = "longest")]
    title: String,
}

#[test]
fn strategies_against_a_shared_base() {
    let base = S {
        log: vec!["start".into()],
        hits: 1,
        cost: 0.5,
        status: "new".into(),
        owner: "ana".into(),
        scores: HashMap::from([("a".into(), 1)]),
        title: "t".into(),
        ..Default::default()
    };
    // Two branches started from a clone of base.
    let mut b1 = base.clone();
    b1.log.push("b1".into());
    b1.hits += 2;
    b1.cost += 1.0;
    b1.last = 1;
    b1.owner = "bo".into();
    b1.scores.insert("b".into(), 2);
    b1.tags.insert("x".into(), true);
    b1.title = "longer title".into();
    let mut b2 = base.clone();
    b2.log.push("b2".into());
    b2.hits += 3;
    b2.status = "done".into();
    b2.last = 2;
    b2.scores.insert("a".into(), 9);
    b2.title = "mid".into();

    let acc = S::reduce(&base, base.clone(), b1);
    let out = S::reduce(&base, acc, b2);
    assert_eq!(out.log, ["start", "b1", "b2"]);
    assert_eq!(out.hits, 6);
    assert!((out.cost - 1.5).abs() < 1e-9);
    assert_eq!(out.status, "done", "only b2 changed it");
    assert_eq!(out.last, 2, "overwrite: last branch wins");
    assert_eq!(out.owner, "ana", "keep ignores branches");
    assert_eq!(
        out.scores,
        HashMap::from([("a".into(), 9), ("b".into(), 2)])
    );
    assert_eq!(out.tags.get("x"), Some(&true));
    assert_eq!(out.title, "longer title");
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, State)]
struct Docs {
    #[state(keep)]
    docs: Vec<String>,
    doc: Option<String>,
    #[state(append)]
    summaries: Vec<String>,
}

#[tokio::test]
async fn map_reduce_with_minimal_send_inputs() {
    let mut g = StateGraph::<Docs>::new();
    g.add_node("plan", |s: Docs| async move { Ok(s) });
    g.add_node("summarize", |s: Docs| async move {
        Ok(Docs {
            summaries: vec![format!("sum of {}", s.doc.unwrap_or_default())],
            ..Default::default()
        })
    });
    g.set_entry_point("plan");
    g.add_send_edges(
        "plan",
        |s: &Docs| {
            s.docs
                .iter()
                .map(|d| {
                    SendTo::new(
                        "summarize",
                        Docs {
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
    g.with_derived_reducer();
    let out = g
        .compile()
        .unwrap()
        .invoke(Docs {
            docs: vec!["a".into(), "b".into(), "c".into()],
            summaries: vec!["earlier".into()],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        out.summaries,
        ["earlier", "sum of a", "sum of b", "sum of c"]
    );
    assert_eq!(out.docs.len(), 3);
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, State)]
struct Pair(#[state(add)] i64, #[state(append)] Vec<u8>);

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, State)]
struct Wrapper<T: PartialEq> {
    value: T,
}

#[test]
fn tuple_and_generic_structs() {
    let out = Pair::reduce(&Pair(1, vec![1]), Pair(3, vec![1, 2]), Pair(5, vec![1, 3]));
    assert_eq!(out, Pair(7, vec![1, 2, 3]));
    let w = Wrapper::reduce(
        &Wrapper { value: 1 },
        Wrapper { value: 1 },
        Wrapper { value: 2 },
    );
    assert_eq!(w.value, 2);
}

#[test]
fn compile_errors() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
