//! A typed state graph for agents.
//!
//! Nodes are async functions `S -> S`. Edges are either fixed or decided by a
//! router over the current state. Execution supports cycles (bounded by a
//! recursion limit), checkpoints after every step, and interrupts before
//! chosen nodes so a human can inspect or edit the state and resume.
//!
//! Routers may fan out to several nodes (or `SendTo` one branch per item);
//! branches run concurrently in a superstep and are merged with a reducer.
//! [`CompiledGraph::stream`] yields per-node events and token deltas.
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
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

mod checkpoint;
mod events;
mod graph;
pub mod prebuilt;

pub use checkpoint::{Checkpoint, Checkpointer, MemoryCheckpointer, PendingTask};
pub use events::{GraphEvent, GraphStream};
pub use graph::{
    BoxError, CompiledGraph, GraphError, NodeContext, NodeResult, RunOutcome, SendTo, State,
    StateGraph, StateSnapshot, END,
};
