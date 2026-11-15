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

use crate::{GraphError, Interrupt};

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
    /// The full state after a superstep (and the state a run starts from).
    /// Only with [`StreamMode::Values`].
    Values { step: usize, state: S },
    /// A checkpoint was written. Only with [`StreamMode::Debug`].
    Checkpoint {
        checkpoint_id: String,
        step: usize,
        next: Vec<String>,
    },
    /// An event from a subgraph (only when [`StreamOptions::subgraphs`] is
    /// on). `namespace` is the path of subgraph nodes from the root graph
    /// (`"node:task"` segments); the inner event's state is serialized.
    Subgraph {
        namespace: Vec<String>,
        event: Box<GraphEvent<Value>>,
    },
    /// The run paused before `next`: a static interrupt
    /// (`interrupt_before`/`interrupt_after`, `interrupts` empty) or dynamic
    /// [`interrupt`](crate::NodeContext::interrupt)s. Resume with
    /// `run(thread_id, None)` or `resume(thread_id, value)`.
    Interrupted {
        next: Vec<String>,
        state: S,
        interrupts: Vec<Interrupt>,
    },
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

impl<S: serde::Serialize> GraphEvent<S> {
    /// The same event with its state serialized (used to forward subgraph
    /// events). Terminal events have no JSON counterpart and give `None`.
    pub fn to_json(&self) -> Option<GraphEvent<Value>> {
        let j = |s: &S| serde_json::to_value(s).unwrap_or(Value::Null);
        Some(match self {
            GraphEvent::NodeStart { node, step, task } => GraphEvent::NodeStart {
                node: node.clone(),
                step: *step,
                task: *task,
            },
            GraphEvent::NodeEnd {
                node,
                step,
                task,
                state,
            } => GraphEvent::NodeEnd {
                node: node.clone(),
                step: *step,
                task: *task,
                state: j(state),
            },
            GraphEvent::Token {
                node,
                step,
                task,
                delta,
            } => GraphEvent::Token {
                node: node.clone(),
                step: *step,
                task: *task,
                delta: delta.clone(),
            },
            GraphEvent::Custom {
                node,
                step,
                task,
                value,
            } => GraphEvent::Custom {
                node: node.clone(),
                step: *step,
                task: *task,
                value: value.clone(),
            },
            GraphEvent::Values { step, state } => GraphEvent::Values {
                step: *step,
                state: j(state),
            },
            GraphEvent::Checkpoint {
                checkpoint_id,
                step,
                next,
            } => GraphEvent::Checkpoint {
                checkpoint_id: checkpoint_id.clone(),
                step: *step,
                next: next.clone(),
            },
            GraphEvent::Subgraph { namespace, event } => GraphEvent::Subgraph {
                namespace: namespace.clone(),
                event: Box::new(event.to_json()?),
            },
            GraphEvent::Interrupted { .. }
            | GraphEvent::Completed { .. }
            | GraphEvent::Error(_) => return None,
        })
    }
}

/// A kind of streamed output (LangGraph's `stream_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamMode {
    /// [`GraphEvent::Values`]: the full state after every superstep.
    Values,
    /// [`GraphEvent::NodeEnd`]: each node's output as it finishes.
    Updates,
    /// [`GraphEvent::Token`]: LLM tokens, tagged with node, step and task.
    Messages,
    /// [`GraphEvent::Custom`]: values nodes emit with `NodeContext::emit`.
    Custom,
    /// [`GraphEvent::NodeStart`]: a node starts.
    Tasks,
    /// Everything above plus [`GraphEvent::Checkpoint`].
    Debug,
}

/// A set of [`StreamMode`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamModes(u8);

impl StreamModes {
    /// What [`CompiledGraph::stream`](crate::CompiledGraph::stream) emits:
    /// tasks, updates, messages and custom events.
    pub const DEFAULT: StreamModes = StreamModes(0b01_1110);
    pub const NONE: StreamModes = StreamModes(0);

    fn bit(m: StreamMode) -> u8 {
        match m {
            StreamMode::Values => 1,
            StreamMode::Updates => 2,
            StreamMode::Messages => 4,
            StreamMode::Custom => 8,
            StreamMode::Tasks => 16,
            StreamMode::Debug => 32,
        }
    }

    pub fn with(self, m: StreamMode) -> Self {
        StreamModes(self.0 | Self::bit(m))
    }

    /// Whether events of mode `m` are wanted (`Debug` implies all but
    /// `Values`).
    pub fn contains(&self, m: StreamMode) -> bool {
        let debug = self.0 & 32 != 0;
        self.0 & Self::bit(m) != 0 || (debug && m != StreamMode::Values)
    }
}

impl Default for StreamModes {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl From<StreamMode> for StreamModes {
    fn from(m: StreamMode) -> Self {
        StreamModes::NONE.with(m)
    }
}

impl From<&[StreamMode]> for StreamModes {
    fn from(ms: &[StreamMode]) -> Self {
        ms.iter().fold(StreamModes::NONE, |acc, m| acc.with(*m))
    }
}

impl<const N: usize> From<[StreamMode; N]> for StreamModes {
    fn from(ms: [StreamMode; N]) -> Self {
        StreamModes::from(&ms[..])
    }
}

/// What a stream yields: the modes, and whether subgraph events are
/// forwarded (wrapped in [`GraphEvent::Subgraph`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamOptions {
    pub modes: StreamModes,
    pub subgraphs: bool,
}

impl StreamOptions {
    pub fn new(modes: impl Into<StreamModes>) -> Self {
        Self {
            modes: modes.into(),
            subgraphs: false,
        }
    }

    pub fn with_subgraphs(mut self, on: bool) -> Self {
        self.subgraphs = on;
        self
    }
}

impl From<StreamMode> for StreamOptions {
    fn from(m: StreamMode) -> Self {
        Self::new(m)
    }
}

impl<const N: usize> From<[StreamMode; N]> for StreamOptions {
    fn from(ms: [StreamMode; N]) -> Self {
        Self::new(ms)
    }
}

impl From<StreamModes> for StreamOptions {
    fn from(m: StreamModes) -> Self {
        Self::new(m)
    }
}

/// What a node can send through its [`NodeContext`](crate::NodeContext).
pub(crate) enum Emission {
    Token(String),
    Custom(Value),
    /// A (serialized) event of a subgraph run by this node.
    Subgraph {
        namespace: Vec<String>,
        event: GraphEvent<Value>,
    },
}

/// Where a run's events go, and which ones are wanted.
pub(crate) struct EventSink<S> {
    f: Arc<dyn Fn(GraphEvent<S>) + Send + Sync>,
    pub(crate) opts: StreamOptions,
}

impl<S> Clone for EventSink<S> {
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            opts: self.opts,
        }
    }
}

impl<S> EventSink<S> {
    pub(crate) fn new(
        opts: StreamOptions,
        f: impl Fn(GraphEvent<S>) + Send + Sync + 'static,
    ) -> Self {
        Self {
            f: Arc::new(f),
            opts,
        }
    }

    pub(crate) fn wants(&self, m: StreamMode) -> bool {
        self.opts.modes.contains(m)
    }

    pub(crate) fn send(&self, ev: GraphEvent<S>) {
        (self.f)(ev)
    }
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
