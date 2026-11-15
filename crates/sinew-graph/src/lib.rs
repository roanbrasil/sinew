//! A typed state graph for agents.
//!
//! Nodes are async functions `S -> S`. Edges are either fixed or decided by a
//! router over the current state. Execution supports cycles (bounded by a
//! recursion limit), checkpoints after every step, and interrupts before
//! chosen nodes so a human can inspect or edit the state and resume.
//!
//! ```no_run
//! # use sinew_graph::*;
//! # #[derive(Clone, serde::Serialize, serde::Deserialize)] struct S { n: u32 }
//! # async fn demo() -> Result<(), GraphError> {
//! let mut g = StateGraph::<S>::new();
//! g.add_node("inc", |mut s: S| async move { s.n += 1; Ok(s) });
//! g.add_conditional_edges("inc", |s: &S| if s.n < 3 { "inc".into() } else { END.into() }, ["inc", END]);
//! g.set_entry_point("inc");
//! let out = g.compile()?.invoke(S { n: 0 }).await?;
//! assert_eq!(out.n, 3);
//! # Ok(()) }
//! ```

mod checkpoint;
mod graph;
pub mod prebuilt;

pub use checkpoint::{Checkpoint, Checkpointer, MemoryCheckpointer};
pub use graph::{
    BoxError, CompiledGraph, GraphError, NodeContext, NodeResult, RunOutcome, State, StateGraph,
    END,
};
