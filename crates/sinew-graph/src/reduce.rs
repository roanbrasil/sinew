//! Field-wise reducers for parallel branches, usually derived with
//! `#[derive(State)]` (feature `macros`).

/// How the outputs of parallel branches merge into one state.
///
/// `reduce(base, acc, branch)`: `base` is the state before the superstep,
/// `acc` the state merged so far (it starts as `base`), and `branch` one
/// branch's output. Derive it with `#[derive(State)]` and pick a strategy per
/// field with `#[state(...)]`:
///
/// | Strategy | Field types | Result |
/// |---|---|---|
/// | `replace` (default) | any `PartialEq` | the branch value if the branch changed it from `base`, else `acc` |
/// | `overwrite` | any | always the branch value (last branch in task order wins) |
/// | `keep` | any | ignore branch values |
/// | `append` | `Vec<T: PartialEq>` | `acc` plus what the branch added (items after `base` if the branch started from it, else all its items) |
/// | `add` | numbers | `acc + (branch - base)`: counters summed across branches |
/// | `merge` | `HashMap`, `BTreeMap` | `acc` plus the entries the branch added or changed |
/// | `reducer = "path"` | any | `path(&base.f, acc.f, branch.f)` |
///
/// ```
/// # #[cfg(feature = "macros")] {
/// use serde::{Deserialize, Serialize};
/// use sinew_graph::{StateGraph, State, END};
///
/// #[derive(Clone, Default, Serialize, Deserialize, State)]
/// struct Research {
///     #[state(append)]
///     notes: Vec<String>,
///     #[state(add)]
///     sources: u32,
///     answer: Option<String>,          // replace
/// }
///
/// let mut g = StateGraph::<Research>::new();
/// g.add_node("plan", |s: Research| async move { Ok(s) });
/// g.add_node("web", |mut s: Research| async move { s.notes.push("web".into()); s.sources += 2; Ok(s) });
/// g.add_node("docs", |mut s: Research| async move { s.notes.push("docs".into()); s.sources += 1; Ok(s) });
/// g.set_entry_point("plan");
/// g.add_fan_out("plan", ["web", "docs"]);
/// g.add_edge("web", END);
/// g.add_edge("docs", END);
/// g.with_derived_reducer();
/// # let out = tokio_test_block_on(g.compile().unwrap().invoke(Research::default())).unwrap();
/// # assert_eq!(out.notes, ["web", "docs"]);
/// # assert_eq!(out.sources, 3);
/// # fn tokio_test_block_on<F: std::future::Future>(f: F) -> F::Output {
/// #     tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(f)
/// # }
/// # }
/// ```
pub trait Reduce: Sized {
    fn reduce(base: &Self, acc: Self, branch: Self) -> Self;
}

/// Helpers called by `#[derive(State)]` expansions. Not public API.
#[doc(hidden)]
pub mod __private {
    pub mod reduce {
        use std::{
            collections::{BTreeMap, HashMap},
            hash::Hash,
            ops::{Add, Sub},
        };

        pub fn replace<T: PartialEq>(base: &T, acc: T, branch: T) -> T {
            if &branch != base {
                branch
            } else {
                acc
            }
        }

        pub fn append<T: PartialEq>(base: &[T], mut acc: Vec<T>, branch: Vec<T>) -> Vec<T> {
            let skip = if branch.len() >= base.len() && branch[..base.len()] == *base {
                base.len()
            } else {
                0
            };
            acc.extend(branch.into_iter().skip(skip));
            acc
        }

        pub fn add<T>(base: &T, acc: T, branch: T) -> T
        where
            T: Copy + Add<Output = T> + Sub<Output = T>,
        {
            acc + (branch - *base)
        }

        /// Maps whose entries can be merged by `#[state(merge)]`.
        pub trait MergeMap: Sized {
            fn merge(base: &Self, acc: Self, branch: Self) -> Self;
        }

        impl<K: Eq + Hash, V: PartialEq, S: std::hash::BuildHasher> MergeMap for HashMap<K, V, S> {
            fn merge(base: &Self, mut acc: Self, branch: Self) -> Self {
                for (k, v) in branch {
                    if base.get(&k) != Some(&v) {
                        acc.insert(k, v);
                    }
                }
                acc
            }
        }

        impl<K: Ord, V: PartialEq> MergeMap for BTreeMap<K, V> {
            fn merge(base: &Self, mut acc: Self, branch: Self) -> Self {
                for (k, v) in branch {
                    if base.get(&k) != Some(&v) {
                        acc.insert(k, v);
                    }
                }
                acc
            }
        }

        pub fn merge<M: MergeMap>(base: &M, acc: M, branch: M) -> M {
            M::merge(base, acc, branch)
        }
    }
}
