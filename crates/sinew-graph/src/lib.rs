//! A typed state graph for agents.
//!
//! Nodes are async functions `S -> S`. Edges are either fixed or decided by a
//! router over the current state. Execution supports cycles (bounded by a
//! recursion limit), checkpoints after every step, and interrupts before
//! chosen nodes so a human can inspect or edit the state and resume.
//!
//! Routers may fan out to several nodes (or `SendTo` one branch per item);
//! branches run concurrently in a superstep and are merged with a reducer.
//! [`CompiledGraph::stream`] yields per-node events and token deltas;
//! [`CompiledGraph::stream_with`] selects [`StreamMode`]s and subgraph events.
//!
//! Nodes can also return a [`Command`] (update + `goto`, including into the
//! parent graph from a subgraph), pause with
//! [`NodeContext::interrupt`] and be resumed with [`Command::resume`].
//! Compiled graphs nest as nodes ([`StateGraph::add_subgraph`]); threads keep
//! a checkpoint tree for time travel ([`CompiledGraph::get_state_history`],
//! [`RunConfig::at_checkpoint`], [`CompiledGraph::update_state_with`]);
//! nodes get retry, cache and timeout policies; and a long-term [`Store`]
//! is reachable from every node.
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
mod command;
pub mod durable;
mod events;
mod graph;
mod policy;
pub mod prebuilt;
mod pregel;
mod reduce;
pub mod store;
mod viz;

pub use checkpoint::{
    new_checkpoint_id, Checkpoint, Checkpointer, MemoryCheckpointer, PendingTask, ToolEntry,
};
pub use command::{
    Command, CommandGraph, Goto, GraphInterrupt, Interrupt, NodeTimeout, ParentCommand, Resume,
    RunConfig, RunInput,
};
pub use events::{GraphEvent, GraphStream, StreamMode, StreamModes, StreamOptions};
pub use graph::{
    BoxError, CompiledGraph, GraphError, NodeContext, NodeResult, RunOutcome, SendTo, State,
    StateGraph, StateSnapshot, END, START,
};
pub use policy::{CachePolicy, RetryPolicy};
#[doc(hidden)]
pub use reduce::__private;
pub use reduce::Reduce;
/// Derives [`Reduce`] with a strategy per field (feature `macros`).
#[cfg(feature = "macros")]
pub use sinew_macros::State;
pub use store::{InMemoryStore, Store};
pub use viz::{EdgeKind, GraphEdge, GraphNode, GraphView, NodeKind};
