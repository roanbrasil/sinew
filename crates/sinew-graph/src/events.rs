//! Graph streaming: [`GraphEvent`]s and the [`GraphStream`] that yields them.

use std::{
    fmt,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures_core::Stream;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::GraphError;

/// Something that happened while a graph ran. Produced by
/// [`CompiledGraph::stream`](crate::CompiledGraph::stream).
///
/// Every stream ends with exactly one terminal event: `Completed`,
/// `Interrupted` or `Error`. `task` is the index of the node execution within
/// its superstep (always 0 without fan-out), so events of parallel branches,
/// including several `SendTo` runs of the same node, can be told apart.
#[derive(Debug)]
pub enum GraphEvent<S> {
    /// A node started executing in superstep `step`.
    NodeStart {
        node: String,
        step: usize,
        task: usize,
    },
    /// A node finished. `state` is that node's own output, before branch
    /// outputs are merged by the reducer.
    NodeEnd {
        node: String,
        step: usize,
        task: usize,
        state: S,
    },
    /// A token (text delta) emitted by a node, e.g. a streaming model reply
    /// in [`react_agent`](crate::prebuilt::react_agent).
    Token {
        node: String,
        step: usize,
        task: usize,
        delta: String,
    },
    /// A custom event emitted by a node with [`NodeContext::emit`](crate::NodeContext::emit).
    Custom {
        node: String,
        step: usize,
        task: usize,
        value: Value,
    },
    /// The run paused before `next` (interrupt). Resume with
    /// `stream(thread_id, None)` or `run(thread_id, None)`.
    Interrupted { next: Vec<String>, state: S },
    /// The run reached `END`.
    Completed { state: S },
    /// The run failed.
    Error(GraphError),
}

impl<S> GraphEvent<S> {
    /// Whether this is the last event of a stream.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            GraphEvent::Completed { .. } | GraphEvent::Interrupted { .. } | GraphEvent::Error(_)
        )
    }
}

/// What a node can send through its [`NodeContext`](crate::NodeContext).
pub(crate) enum Emission {
    Token(String),
    Custom(Value),
}

/// Type-erased sink for node emissions (the context is not generic over `S`).
#[derive(Clone)]
pub(crate) struct Emitter(Arc<dyn Fn(Emission) + Send + Sync>);

impl Emitter {
    pub(crate) fn new(f: impl Fn(Emission) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
    pub(crate) fn send(&self, e: Emission) {
        (self.0)(e)
    }
}

impl fmt::Debug for Emitter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Emitter")
    }
}

pub(crate) type EventTx<S> = mpsc::UnboundedSender<GraphEvent<S>>;

/// Stream of [`GraphEvent`]s for one run.
///
/// The run is driven by polling this stream: nothing executes in the
/// background, and dropping the stream cancels the run (including any
/// parallel branches still in flight). Checkpoints already written stay, so a
/// cancelled thread can be resumed from its last completed superstep.
pub struct GraphStream<S> {
    run: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
    rx: mpsc::UnboundedReceiver<GraphEvent<S>>,
}

// The stream never pins `S` values: events are moved out of the channel.
impl<S> Unpin for GraphStream<S> {}

impl<S> GraphStream<S> {
    pub(crate) fn new(
        run: Pin<Box<dyn Future<Output = ()> + Send>>,
        rx: mpsc::UnboundedReceiver<GraphEvent<S>>,
    ) -> Self {
        Self { run: Some(run), rx }
    }
}

impl<S> Stream for GraphStream<S> {
    type Item = GraphEvent<S>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Poll::Ready(Some(ev)) = this.rx.poll_recv(cx) {
                return Poll::Ready(Some(ev));
            }
            match this.run.as_mut() {
                Some(run) => match run.as_mut().poll(cx) {
                    // The run finished; drain what it sent on the next turn.
                    Poll::Ready(()) => this.run = None,
                    Poll::Pending => return Poll::Pending,
                },
                None => return Poll::Ready(this.rx.try_recv().ok()),
            }
        }
    }
}
