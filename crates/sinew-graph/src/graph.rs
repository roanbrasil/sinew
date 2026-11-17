use std::{
    collections::{HashMap, HashSet, VecDeque},
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    time::Duration,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{Map, Value};
use sinew_trace::{RunRef, Tracer};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::{
    command::{overlay, ParentCommand},
    events::{Emission, Emitter, EventSink, GraphEvent, GraphStream, StreamOptions},
    policy::{CachePolicy, NodeCache, RetryPolicy},
    pregel::{Run, SubOutcome, TaskResult},
    store::Store,
    Checkpoint, Checkpointer, Command, Goto, GraphInterrupt, Interrupt, RunConfig, RunInput,
};

/// Terminal pseudo-node.
pub const END: &str = "__end__";
/// Entry pseudo-node: `add_edge(START, "a")` is `set_entry_point("a")`, and
/// `add_conditional_edges(START, ...)` picks the first node from the input.
pub const START: &str = "__start__";

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type NodeResult<S> = Result<S, BoxError>;
pub(crate) type BoxFut<T> = Pin<Box<dyn Future<Output = Result<T, BoxError>> + Send>>;
pub(crate) type NodeFn<S> = Arc<dyn Fn(S, NodeContext) -> BoxFut<Command<S>> + Send + Sync>;
type RouterFn<S> = Arc<dyn Fn(&S) -> Vec<String> + Send + Sync>;
type SendFn<S> = Arc<dyn Fn(&S) -> Vec<SendTo<S>> + Send + Sync>;
pub(crate) type ResumeMap = Mutex<HashMap<String, Value>>;

/// Bound satisfied by any graph state.
pub trait State: Clone + Send + Sync + Serialize + DeserializeOwned + 'static {}
impl<T: Clone + Send + Sync + Serialize + DeserializeOwned + 'static> State for T {}

/// Resume values and the interrupt counter of one node execution.
#[derive(Debug, Default)]
pub(crate) struct TaskInterrupts {
    pub(crate) resume: Vec<Value>,
    pub(crate) counter: usize,
}

/// Passed to nodes registered with [`StateGraph::add_node_with_context`]
/// (and to command nodes).
#[derive(Clone)]
pub struct NodeContext {
    pub node: String,
    /// Superstep this execution belongs to.
    pub step: usize,
    pub thread_id: String,
    pub tracer: Option<Tracer>,
    /// The node's own run, for attaching model/tool child runs.
    pub run: Option<RunRef>,
    /// Index of this execution within its superstep (0 without fan-out).
    pub task: usize,
    pub(crate) emitter: Option<Emitter>,
    pub(crate) stream: StreamOptions,
    pub(crate) namespace: Vec<String>,
    pub(crate) store: Option<Arc<dyn Store>>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    pub(crate) resume_map: Option<Arc<ResumeMap>>,
    /// The task had pending interrupts (a subgraph should resume its own
    /// checkpoint instead of starting over).
    pub(crate) resuming: bool,
    /// Resume values and the call counter; `None` when the task has no
    /// resume values (its first `interrupt` call always pauses), which
    /// saves an allocation per node execution.
    pub(crate) interrupts: Option<Arc<Mutex<TaskInterrupts>>>,
}

impl NodeContext {
    /// A context outside any graph run (no tracer, emissions are dropped).
    /// Handy for calling node functions directly in tests.
    pub fn new(node: impl Into<String>, step: usize, thread_id: impl Into<String>) -> Self {
        Self {
            node: node.into(),
            step,
            thread_id: thread_id.into(),
            tracer: None,
            run: None,
            task: 0,
            emitter: None,
            stream: StreamOptions::default(),
            namespace: Vec::new(),
            store: None,
            checkpointer: None,
            resume_map: None,
            resuming: false,
            interrupts: None,
        }
    }

    /// Attaches a store (for tests that call nodes directly).
    pub fn with_store(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self
    }

    /// Pre-loads resume values, as if the node had been interrupted and
    /// resumed that many times (for tests that call nodes directly).
    pub fn with_resume_values(mut self, values: Vec<Value>) -> Self {
        self.interrupts = Some(Arc::new(Mutex::new(TaskInterrupts {
            resume: values,
            counter: 0,
        })));
        self
    }

    /// Whether the graph is being consumed with [`CompiledGraph::stream`]
    /// (and LLM tokens are wanted). Nodes can use it to pick a streaming
    /// model call only when someone listens.
    pub fn is_streaming(&self) -> bool {
        self.emitter.is_some() && self.stream.modes.contains(crate::StreamMode::Messages)
    }

    /// Emits a [`GraphEvent::Custom`] (no-op unless streaming).
    pub fn emit(&self, value: Value) {
        if let Some(e) = &self.emitter {
            e.send(Emission::Custom(value));
        }
    }

    /// Emits a [`GraphEvent::Token`] (no-op unless streaming).
    pub fn emit_token(&self, delta: impl Into<String>) {
        if let Some(e) = &self.emitter {
            e.send(Emission::Token(delta.into()));
        }
    }

    /// Subgraph path of this node (empty in the root graph).
    pub fn namespace(&self) -> &[String] {
        &self.namespace
    }

    /// The long-term store attached to the graph, if any.
    pub fn store(&self) -> Option<&Arc<dyn Store>> {
        self.store.as_ref()
    }

    /// Pauses the run and surfaces `value` (e.g. a question) to the caller;
    /// propagate the error with `?`:
    ///
    /// ```ignore
    /// let approved: bool = ctx.interrupt(json!({"question": "Send it?"}))?;
    /// ```
    ///
    /// The run stops with [`RunOutcome::Interrupted`] listing the
    /// [`Interrupt`]. When the caller resumes with
    /// [`Command::resume`] (or [`CompiledGraph::resume`]), the node runs
    /// again **from the start**, and this call returns the resume value
    /// instead of pausing. A node may call `interrupt` several times: on each
    /// re-execution the calls already answered return their stored values in
    /// order, and the first unanswered one pauses again. Code before an
    /// `interrupt` therefore runs once per resume; keep it idempotent.
    ///
    /// Needs a checkpointer to resume; without one the run fails with
    /// [`GraphError::Interrupted`].
    pub fn interrupt<T: DeserializeOwned>(&self, value: impl Serialize) -> Result<T, BoxError> {
        let idx = match &self.interrupts {
            Some(st) => {
                let mut st = st.lock().unwrap();
                let idx = st.counter;
                st.counter += 1;
                if let Some(v) = st.resume.get(idx) {
                    return Ok(serde_json::from_value(v.clone())?);
                }
                idx
            }
            None => 0,
        };
        let value = serde_json::to_value(value)?;
        Err(Box::new(GraphInterrupt {
            interrupts: vec![Interrupt {
                id: self.interrupt_id(idx),
                value,
                node: self.node.clone(),
                namespace: self.namespace.clone(),
            }],
        }))
    }

    fn interrupt_id(&self, idx: usize) -> String {
        let mut id = self.namespace.join("|");
        if !id.is_empty() {
            id.push('|');
        }
        id.push_str(&format!(
            "{}:{}:{}:{}",
            self.node, self.step, self.task, idx
        ));
        id
    }
}

/// A branch created by a [`StateGraph::add_send_edges`] router: run `node`
/// in the next superstep on `input` instead of the shared state. This is how
/// map-reduce is expressed: one `SendTo` per item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "S: Serialize", deserialize = "S: Deserialize<'de>"))]
pub struct SendTo<S> {
    pub node: String,
    pub input: S,
}

impl<S> SendTo<S> {
    pub fn new(node: impl Into<String>, input: S) -> Self {
        Self {
            node: node.into(),
            input,
        }
    }
}

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("invalid graph:\n  - {}", .0.join("\n  - "))]
    Build(Vec<String>),
    #[error("node `{node}` failed: {source}")]
    Node { node: String, source: BoxError },
    #[error("recursion limit of {0} supersteps reached before the graph hit END; raise it with `with_recursion_limit` or `RunConfig::with_recursion_limit` if the loop is intended")]
    RecursionLimit(usize),
    #[error("router of `{from}` returned unknown target `{to}`")]
    UnknownRoute { from: String, to: String },
    #[error("graph was interrupted at `{0}`; use `run` with a checkpointer to resume")]
    Interrupted(String),
    #[error("no checkpoint for thread `{0}`")]
    NoCheckpoint(String),
    #[error("no checkpoint `{checkpoint_id}` in thread `{thread_id}`")]
    CheckpointNotFound {
        thread_id: String,
        checkpoint_id: String,
    },
    #[error("this operation needs a checkpointer")]
    NoCheckpointer,
    #[error("checkpointer error: {0}")]
    Checkpoint(String),
    #[error("state serialization failed: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("superstep runs several branches ({}) but no reducer is set; call `StateGraph::with_reducer`", .0.join(", "))]
    NoReducer(Vec<String>),
    #[error("invalid command: {0}")]
    InvalidCommand(String),
    /// A node of a subgraph returned a [`Command`] for the parent graph.
    /// Handled by the subgraph node; only seen when calling a subgraph's
    /// internals directly.
    #[error("a subgraph node sent a command to its parent graph")]
    ParentCommand(Box<ParentCommand>),
}

#[derive(Debug, Clone)]
pub enum RunOutcome<S> {
    Completed(S),
    /// Paused; resume with `run(thread_id, None)` or, for dynamic
    /// interrupts, [`CompiledGraph::resume`].
    Interrupted {
        state: S,
        /// First node of the paused superstep.
        next: String,
        /// Every node of the paused superstep (more than one after a fan-out).
        next_nodes: Vec<String>,
        /// Dynamic interrupts raised by nodes (empty for
        /// `interrupt_before`/`interrupt_after` pauses).
        interrupts: Vec<Interrupt>,
    },
}

impl<S> RunOutcome<S> {
    pub fn state(&self) -> &S {
        match self {
            RunOutcome::Completed(s) | RunOutcome::Interrupted { state: s, .. } => s,
        }
    }
    pub fn into_state(self) -> S {
        match self {
            RunOutcome::Completed(s) | RunOutcome::Interrupted { state: s, .. } => s,
        }
    }
    pub fn is_interrupted(&self) -> bool {
        matches!(self, RunOutcome::Interrupted { .. })
    }
    /// Pending dynamic interrupts (empty when completed or statically paused).
    pub fn interrupts(&self) -> &[Interrupt] {
        match self {
            RunOutcome::Interrupted { interrupts, .. } => interrupts,
            RunOutcome::Completed(_) => &[],
        }
    }
}

/// A decoded checkpoint of a thread.
#[derive(Debug, Clone)]
pub struct StateSnapshot<S> {
    pub state: S,
    /// Nodes of the next superstep; empty when the run finished.
    pub next: Vec<String>,
    /// Supersteps completed so far.
    pub step: usize,
    /// Id to replay or fork from (see [`RunConfig::at_checkpoint`]).
    pub checkpoint_id: String,
    pub parent_id: Option<String>,
    /// Pending dynamic interrupts.
    pub interrupts: Vec<Interrupt>,
    pub metadata: Map<String, Value>,
    pub created_at: Option<u64>,
}

impl<S: State> StateSnapshot<S> {
    pub(crate) fn from_checkpoint(cp: Checkpoint) -> Result<Self, GraphError> {
        Ok(Self {
            next: cp.next_nodes(),
            interrupts: cp.interrupts(),
            step: cp.step,
            state: serde_json::from_value(cp.state)?,
            checkpoint_id: cp.checkpoint_id,
            parent_id: cp.parent_id,
            metadata: cp.metadata,
            created_at: cp.created_at,
        })
    }
}

pub(crate) enum Edge<S> {
    /// Fixed targets; more than one is a static fan-out.
    Static(Vec<String>),
    Router {
        router: RouterFn<S>,
        targets: Vec<String>,
        multi: bool,
    },
    Send {
        router: SendFn<S>,
        targets: Vec<String>,
    },
}

impl<S> Edge<S> {
    pub(crate) fn targets(&self) -> Vec<&str> {
        match self {
            Edge::Static(t) | Edge::Router { targets: t, .. } | Edge::Send { targets: t, .. } => {
                t.iter().map(String::as_str).collect()
            }
        }
    }

    fn can_fan_out(&self) -> bool {
        match self {
            Edge::Static(t) => t.len() > 1,
            Edge::Router { multi, .. } => *multi,
            Edge::Send { .. } => true,
        }
    }
}

/// A registered node and its execution policies.
pub(crate) struct NodeSpec<S> {
    pub(crate) f: NodeFn<S>,
    /// May return a command without an update: keep a copy of its input.
    pub(crate) command: bool,
    pub(crate) destinations: Vec<String>,
    pub(crate) retry: Option<RetryPolicy>,
    pub(crate) cache: Option<CachePolicy>,
    pub(crate) timeout: Option<Duration>,
    /// Shape of the compiled graph behind a subgraph node, for diagrams.
    pub(crate) subgraph: Option<crate::viz::GraphView>,
}

type PlainReducer<S> = Arc<dyn Fn(S, S) -> S + Send + Sync>;
type BaseReducer<S> = Arc<dyn Fn(&S, S, S) -> S + Send + Sync>;

pub(crate) enum Reducer<S> {
    Plain(PlainReducer<S>),
    WithBase(BaseReducer<S>),
}

impl<S> Clone for Reducer<S> {
    fn clone(&self) -> Self {
        match self {
            Reducer::Plain(f) => Reducer::Plain(f.clone()),
            Reducer::WithBase(f) => Reducer::WithBase(f.clone()),
        }
    }
}

pub struct StateGraph<S> {
    nodes: HashMap<String, NodeSpec<S>>,
    order: Vec<String>,
    edges: HashMap<String, Edge<S>>,
    entry: Option<Edge<S>>,
    reducer: Option<Reducer<S>>,
    retry: HashMap<String, RetryPolicy>,
    cache: HashMap<String, CachePolicy>,
    timeouts: HashMap<String, Duration>,
    errors: Vec<String>,
}

impl<S: State> Default for StateGraph<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: State> StateGraph<S> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            order: Vec::new(),
            edges: HashMap::new(),
            entry: None,
            reducer: None,
            retry: HashMap::new(),
            cache: HashMap::new(),
            timeouts: HashMap::new(),
            errors: Vec::new(),
        }
    }

    pub fn add_node<F, Fut>(&mut self, name: impl Into<String>, f: F) -> &mut Self
    where
        F: Fn(S) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = NodeResult<S>> + Send + 'static,
    {
        let f: NodeFn<S> = Arc::new(move |s, _ctx| {
            let fut = f(s);
            Box::pin(async move { fut.await.map(Command::update) })
        });
        self.insert(name.into(), f, false, Vec::new())
    }

    pub fn add_node_with_context<F, Fut>(&mut self, name: impl Into<String>, f: F) -> &mut Self
    where
        F: Fn(S, NodeContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = NodeResult<S>> + Send + 'static,
    {
        let f: NodeFn<S> = Arc::new(move |s, ctx| {
            let fut = f(s, ctx);
            Box::pin(async move { fut.await.map(Command::update) })
        });
        self.insert(name.into(), f, false, Vec::new())
    }

    /// A node that returns a [`Command`]: a state update plus navigation
    /// (`goto` one or more nodes, `END`, or the parent graph's nodes with
    /// [`Command::to_parent`]).
    ///
    /// `destinations` lists the nodes the command may go to; it is used to
    /// check reachability at compile time (pass an empty list to skip that
    /// check for the whole graph). A command node does not need an outgoing
    /// edge; if it has one, the edge's targets run in addition to `goto`.
    pub fn add_command_node<F, Fut, I, T>(
        &mut self,
        name: impl Into<String>,
        destinations: I,
        f: F,
    ) -> &mut Self
    where
        F: Fn(S, NodeContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Command<S>, BoxError>> + Send + 'static,
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let f: NodeFn<S> = Arc::new(move |s, ctx| Box::pin(f(s, ctx)));
        let dest = destinations.into_iter().map(Into::into).collect();
        self.insert(name.into(), f, true, dest)
    }

    /// Adds a compiled graph with the same state type as a node. The
    /// subgraph runs on the current state and its final state replaces it.
    ///
    /// Inside a subgraph, dynamic interrupts pause the whole run (resuming
    /// continues inside the subgraph), the subgraph checkpoints under its own
    /// namespace with the parent's checkpointer (unless it has its own), and
    /// its nodes can route the parent with [`Command::to_parent`].
    pub fn add_subgraph(&mut self, name: impl Into<String>, graph: CompiledGraph<S>) -> &mut Self {
        self.add_subgraph_mapped(name, graph, S::clone, |_parent, child| child)
    }

    /// Adds a compiled graph with a different state type as a node:
    /// `input` builds the subgraph's input from the current state and
    /// `output` folds the subgraph's final state back into it.
    pub fn add_subgraph_mapped<C, In, Out>(
        &mut self,
        name: impl Into<String>,
        graph: CompiledGraph<C>,
        input: In,
        output: Out,
    ) -> &mut Self
    where
        C: State,
        In: Fn(&S) -> C + Send + Sync + 'static,
        Out: Fn(S, C) -> S + Send + Sync + 'static,
    {
        let name = name.into();
        let view = graph.view();
        let input = Arc::new(input);
        let output = Arc::new(output);
        let f: NodeFn<S> = Arc::new(move |s: S, ctx: NodeContext| {
            let (graph, input, output) = (graph.clone(), input.clone(), output.clone());
            Box::pin(async move {
                let child_in = input(&s);
                match graph.run_as_subgraph(child_in, &ctx).await {
                    Ok(SubOutcome::Done(c)) => Ok(Command::update(output(s, c))),
                    Ok(SubOutcome::Interrupted(interrupts)) => {
                        Err(Box::new(GraphInterrupt { interrupts }) as BoxError)
                    }
                    Err(GraphError::ParentCommand(pc)) => Ok(parent_command_into(s, *pc)?),
                    Err(e) => Err(Box::new(e) as BoxError),
                }
            })
        });
        self.insert(name.clone(), f, false, Vec::new());
        if let Some(n) = self.nodes.get_mut(&name) {
            n.subgraph = Some(view);
        }
        self
    }

    fn insert(
        &mut self,
        name: String,
        f: NodeFn<S>,
        command: bool,
        dest: Vec<String>,
    ) -> &mut Self {
        if name.is_empty() || name == END || name == START {
            self.errors
                .push(format!("`{name}` is not a valid node name"));
        } else if self.nodes.contains_key(&name) {
            self.errors.push(format!("node `{name}` added twice"));
        } else {
            self.nodes.insert(
                name.clone(),
                NodeSpec {
                    f,
                    command,
                    destinations: dest,
                    retry: None,
                    cache: None,
                    timeout: None,
                    subgraph: None,
                },
            );
            self.order.push(name);
        }
        self
    }

    /// Declares nodes this node may route to at run time (through a
    /// [`Command`] `goto`, e.g. one a subgraph sends with
    /// [`Command::to_parent`]); used for the reachability check.
    pub fn set_node_destinations<I, T>(
        &mut self,
        node: impl Into<String>,
        destinations: I,
    ) -> &mut Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let node = node.into();
        let dest: Vec<String> = destinations.into_iter().map(Into::into).collect();
        match self.nodes.get_mut(&node) {
            Some(spec) => spec.destinations.extend(dest),
            None => self
                .errors
                .push(format!("destinations set for unknown node `{node}`")),
        }
        self
    }

    /// Retries the node on failure (see [`RetryPolicy`]).
    pub fn set_node_retry(&mut self, node: impl Into<String>, policy: RetryPolicy) -> &mut Self {
        self.retry.insert(node.into(), policy);
        self
    }

    /// Caches the node's result by input (see [`CachePolicy`]).
    pub fn set_node_cache(&mut self, node: impl Into<String>, policy: CachePolicy) -> &mut Self {
        self.cache.insert(node.into(), policy);
        self
    }

    /// Fails an attempt of the node that runs longer than `timeout` with a
    /// [`NodeTimeout`](crate::NodeTimeout) error (retried if the node has a
    /// retry policy).
    pub fn set_node_timeout(&mut self, node: impl Into<String>, timeout: Duration) -> &mut Self {
        self.timeouts.insert(node.into(), timeout);
        self
    }

    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) -> &mut Self {
        self.set_edge(from.into(), Edge::Static(vec![to.into()]))
    }

    /// Static fan-out: after `from`, all `targets` run in parallel in the next
    /// superstep. Needs a reducer ([`with_reducer`](Self::with_reducer)).
    pub fn add_fan_out<I, T>(&mut self, from: impl Into<String>, targets: I) -> &mut Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let targets = targets.into_iter().map(Into::into).collect();
        self.set_edge(from.into(), Edge::Static(targets))
    }

    /// `targets` lists every value the router may return, for validation.
    pub fn add_conditional_edges<R, I, T>(
        &mut self,
        from: impl Into<String>,
        router: R,
        targets: I,
    ) -> &mut Self
    where
        R: Fn(&S) -> String + Send + Sync + 'static,
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let targets = targets.into_iter().map(Into::into).collect();
        self.set_edge(
            from.into(),
            Edge::Router {
                router: Arc::new(move |s| vec![router(s)]),
                targets,
                multi: false,
            },
        )
    }

    /// Like [`add_conditional_edges`](Self::add_conditional_edges), but the
    /// router picks any number of targets; they all run in parallel in the
    /// next superstep. An empty list ends this branch. Needs a reducer.
    pub fn add_conditional_edges_multi<R, I, T>(
        &mut self,
        from: impl Into<String>,
        router: R,
        targets: I,
    ) -> &mut Self
    where
        R: Fn(&S) -> Vec<String> + Send + Sync + 'static,
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let targets = targets.into_iter().map(Into::into).collect();
        self.set_edge(
            from.into(),
            Edge::Router {
                router: Arc::new(router),
                targets,
                multi: true,
            },
        )
    }

    /// Map-style fan-out: the router returns one [`SendTo`] per branch, each
    /// with its own input state. Branches run in parallel in the next
    /// superstep and their outputs are folded into the shared state with the
    /// reducer (always, even for a single branch). `targets` lists every node
    /// the router may send to.
    pub fn add_send_edges<R, I, T>(
        &mut self,
        from: impl Into<String>,
        router: R,
        targets: I,
    ) -> &mut Self
    where
        R: Fn(&S) -> Vec<SendTo<S>> + Send + Sync + 'static,
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let targets = targets.into_iter().map(Into::into).collect();
        self.set_edge(
            from.into(),
            Edge::Send {
                router: Arc::new(router),
                targets,
            },
        )
    }

    /// Merges parallel branch outputs into the shared state.
    ///
    /// After a superstep with several node executions (or any `SendTo`
    /// branch), the new state is
    /// `reducer(...reducer(reducer(before, out_0), out_1)..., out_n)`:
    /// a left fold that starts from the state *before* the superstep and
    /// takes branch outputs in task order (deterministic, independent of
    /// which branch finished first). Each output is a whole state, so write
    /// the reducer to copy over only what a branch owns, or use
    /// [`with_reducer_from_base`](Self::with_reducer_from_base) when you need
    /// the pre-superstep state to tell what a branch changed. Command nodes
    /// that return no update are left out of the fold. See
    /// `docs/fan-out.md`.
    pub fn with_reducer<F>(&mut self, reducer: F) -> &mut Self
    where
        F: Fn(S, S) -> S + Send + Sync + 'static,
    {
        self.reducer = Some(Reducer::Plain(Arc::new(reducer)));
        self
    }

    /// Like [`with_reducer`](Self::with_reducer), but the reducer also gets
    /// the state from before the superstep (`base`, the fold's starting
    /// value). Use it when branches receive a clone of the shared state and
    /// you need to keep only what each branch added, e.g.
    /// `acc.log.extend(branch.log.into_iter().skip(base.log.len()))`.
    pub fn with_reducer_from_base<F>(&mut self, reducer: F) -> &mut Self
    where
        F: Fn(&S, S, S) -> S + Send + Sync + 'static,
    {
        self.reducer = Some(Reducer::WithBase(Arc::new(reducer)));
        self
    }

    fn set_edge(&mut self, from: String, edge: Edge<S>) -> &mut Self {
        if from == START {
            self.entry = Some(edge);
        } else if self.edges.insert(from.clone(), edge).is_some() {
            self.errors.push(format!(
                "node `{from}` has more than one outgoing edge definition"
            ));
        }
        self
    }

    pub fn set_entry_point(&mut self, name: impl Into<String>) -> &mut Self {
        self.entry = Some(Edge::Static(vec![name.into()]));
        self
    }

    /// Picks the first node(s) from the input state (like
    /// `add_conditional_edges(START, ...)`).
    pub fn set_conditional_entry_point<R, I, T>(&mut self, router: R, targets: I) -> &mut Self
    where
        R: Fn(&S) -> String + Send + Sync + 'static,
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.add_conditional_edges(START, router, targets)
    }

    pub fn compile(self) -> Result<CompiledGraph<S>, GraphError> {
        let mut errors = self.errors;
        let mut nodes = self.nodes;
        let known = |n: &str| n == END || nodes.contains_key(n);
        match &self.entry {
            None => errors.push("no entry point set".into()),
            Some(e) => {
                for t in e.targets() {
                    if t == END || !nodes.contains_key(t) {
                        errors.push(format!("entry point `{t}` is not a node"));
                    }
                }
                if e.can_fan_out() && self.reducer.is_none() {
                    errors.push("the entry point fans out but no reducer is set".into());
                }
            }
        }
        for (from, edge) in &self.edges {
            if !nodes.contains_key(from) {
                errors.push(format!("edge from unknown node `{from}`"));
            }
            let targets = edge.targets();
            if targets.is_empty() {
                errors.push(format!("edge from `{from}` lists no targets"));
            }
            for t in targets {
                if !known(t) {
                    errors.push(format!("edge `{from}` -> `{t}` points to an unknown node"));
                }
            }
            if edge.can_fan_out() && self.reducer.is_none() {
                errors.push(format!(
                    "node `{from}` fans out but no reducer is set (call `with_reducer`)"
                ));
            }
        }
        let mut open_commands = false;
        for n in &self.order {
            let spec = &nodes[n];
            for d in &spec.destinations {
                if !known(d) {
                    errors.push(format!("node `{n}` lists unknown destination `{d}`"));
                }
            }
            if spec.command {
                open_commands |= spec.destinations.is_empty();
            } else if !self.edges.contains_key(n) {
                errors.push(format!(
                    "node `{n}` has no outgoing edge (add one to END to finish)"
                ));
            }
        }
        if !open_commands {
            if let Some(entry) = &self.entry {
                let mut seen = HashSet::new();
                let mut queue = VecDeque::new();
                for t in entry.targets() {
                    if nodes.contains_key(t) && seen.insert(t) {
                        queue.push_back(t);
                    }
                }
                while let Some(n) = queue.pop_front() {
                    let mut next = self.edges.get(n).map(|e| e.targets()).unwrap_or_default();
                    next.extend(nodes[n].destinations.iter().map(String::as_str));
                    for t in next {
                        if t != END && nodes.contains_key(t) && seen.insert(t) {
                            queue.push_back(t);
                        }
                    }
                }
                let entry_name = entry.targets().join("`, `");
                for n in &self.order {
                    if !seen.contains(n.as_str()) {
                        errors.push(format!("node `{n}` is unreachable from `{entry_name}`"));
                    }
                }
            }
        }
        for (what, names) in [
            ("retry policy", self.retry.keys().collect::<Vec<_>>()),
            ("cache policy", self.cache.keys().collect()),
            ("timeout", self.timeouts.keys().collect()),
        ] {
            for n in names {
                if !nodes.contains_key(n) {
                    errors.push(format!("{what} set for unknown node `{n}`"));
                }
            }
        }
        if !errors.is_empty() {
            errors.sort();
            return Err(GraphError::Build(errors));
        }
        for (n, p) in self.retry {
            nodes.get_mut(&n).unwrap().retry = Some(p);
        }
        for (n, p) in self.cache {
            nodes.get_mut(&n).unwrap().cache = Some(p);
        }
        for (n, t) in self.timeouts {
            nodes.get_mut(&n).unwrap().timeout = Some(t);
        }
        Ok(CompiledGraph {
            nodes: Arc::new(nodes.into_iter().map(|(k, v)| (k, Arc::new(v))).collect()),
            edges: Arc::new(self.edges),
            entry: Arc::new(self.entry.unwrap()),
            order: Arc::new(self.order),
            reducer: self.reducer,
            recursion_limit: 25,
            checkpointer: None,
            interrupt_before: Arc::new(HashSet::new()),
            interrupt_after: Arc::new(HashSet::new()),
            tracer: None,
            store: None,
            cache: Arc::new(NodeCache::new()),
        })
    }
}

/// Converts a subgraph's [`ParentCommand`] into a command of the parent
/// graph, applied on top of the parent's state `s`.
fn parent_command_into<S: State>(mut s: S, pc: ParentCommand) -> Result<Command<S>, BoxError> {
    if let Some(patch) = pc.update {
        let mut v = serde_json::to_value(&s)?;
        overlay(&mut v, patch);
        s = serde_json::from_value(v)?;
    }
    let goto = pc
        .goto
        .into_iter()
        .map(|g| match g {
            Goto::Node(n) => Ok(Goto::Node(n)),
            Goto::Send(send) => Ok(Goto::Send(SendTo::new(
                send.node,
                serde_json::from_value(send.input)?,
            ))),
        })
        .collect::<Result<Vec<_>, serde_json::Error>>()?;
    Ok(Command {
        update: Some(s),
        goto,
        ..Command::default()
    })
}

/// An executable graph. Cheap to clone and share across tasks.
///
/// Execution proceeds in *supersteps* (as in Pregel / LangGraph): every node
/// scheduled for a step runs concurrently on a snapshot of the state, their
/// outputs are merged (a single plain node simply replaces the state; several
/// branches go through the reducer), then every executed node's outgoing edge
/// (plus any `goto` it returned) is evaluated on the merged state to schedule
/// the next step. Identical target nodes are deduplicated, which is how
/// parallel branches join.
pub struct CompiledGraph<S> {
    pub(crate) nodes: Arc<HashMap<String, Arc<NodeSpec<S>>>>,
    pub(crate) edges: Arc<HashMap<String, Edge<S>>>,
    pub(crate) entry: Arc<Edge<S>>,
    /// Node names in the order they were added (for diagrams).
    pub(crate) order: Arc<Vec<String>>,
    pub(crate) reducer: Option<Reducer<S>>,
    pub(crate) recursion_limit: usize,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    pub(crate) interrupt_before: Arc<HashSet<String>>,
    pub(crate) interrupt_after: Arc<HashSet<String>>,
    pub(crate) tracer: Option<Tracer>,
    pub(crate) store: Option<Arc<dyn Store>>,
    pub(crate) cache: Arc<NodeCache<TaskResult<S>>>,
}

impl<S> Clone for CompiledGraph<S> {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            entry: self.entry.clone(),
            order: self.order.clone(),
            reducer: self.reducer.clone(),
            recursion_limit: self.recursion_limit,
            checkpointer: self.checkpointer.clone(),
            interrupt_before: self.interrupt_before.clone(),
            interrupt_after: self.interrupt_after.clone(),
            tracer: self.tracer.clone(),
            store: self.store.clone(),
            cache: self.cache.clone(),
        }
    }
}

impl<S: State> CompiledGraph<S> {
    /// Maximum supersteps per run (default 25). Without fan-out a superstep
    /// is exactly one node execution. Each `run`/`resume` call gets the full
    /// budget.
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }

    pub fn with_checkpointer(mut self, cp: impl Checkpointer + 'static) -> Self {
        self.checkpointer = Some(Arc::new(cp));
        self
    }

    pub fn with_shared_checkpointer(mut self, cp: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = Some(cp);
        self
    }

    /// Attaches a long-term [`Store`], reachable from nodes through
    /// [`NodeContext::store`] (subgraphs inherit it).
    pub fn with_store(mut self, store: impl Store + 'static) -> Self {
        self.store = Some(Arc::new(store));
        self
    }

    pub fn with_shared_store(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self
    }

    /// The attached store, if any.
    pub fn store(&self) -> Option<&Arc<dyn Store>> {
        self.store.as_ref()
    }

    /// Pause before these nodes (human-in-the-loop). Needs a checkpointer to
    /// resume. If any node of a superstep is listed, the whole superstep
    /// pauses and resumes together.
    pub fn interrupt_before<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_before = Arc::new(nodes.into_iter().map(Into::into).collect());
        self
    }

    /// Pause after a superstep in which any of these nodes ran (when there
    /// is something left to run). Needs a checkpointer to resume.
    pub fn interrupt_after<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_after = Arc::new(nodes.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_tracer(mut self, tracer: Tracer) -> Self {
        self.tracer = Some(tracer);
        self
    }

    /// Drops every cached node result.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Runs to completion on a throwaway thread.
    pub async fn invoke(&self, state: S) -> Result<S, GraphError> {
        let thread = uuid::Uuid::new_v4().to_string();
        match self
            .drive(RunConfig::new(thread), RunInput::Start(state), None)
            .await?
        {
            RunOutcome::Completed(s) => Ok(s),
            RunOutcome::Interrupted {
                next, interrupts, ..
            } => Err(GraphError::Interrupted(
                interrupts.first().map_or(next, |i| i.node.clone()),
            )),
        }
    }

    /// Starts a thread with `Some(input)`, or resumes its latest checkpoint with `None`.
    pub async fn run(
        &self,
        thread_id: &str,
        input: Option<S>,
    ) -> Result<RunOutcome<S>, GraphError> {
        self.drive(RunConfig::new(thread_id), input.into(), None)
            .await
    }

    /// The general entry point: `config` picks the thread (and optionally
    /// the checkpoint to continue from, for replay and forks); `input` is a
    /// state (start over), `None`/[`RunInput::Continue`] (continue), or a
    /// [`Command`] (resume interrupts, edit the state, or jump).
    pub async fn run_with(
        &self,
        config: impl Into<RunConfig>,
        input: impl Into<RunInput<S>>,
    ) -> Result<RunOutcome<S>, GraphError> {
        self.drive(config.into(), input.into(), None).await
    }

    /// Resumes a thread paused by [`NodeContext::interrupt`]: the first
    /// pending interrupt returns `value`. Shorthand for
    /// `run_with(thread_id, Command::resume(value))`.
    pub async fn resume(
        &self,
        thread_id: &str,
        value: impl Serialize,
    ) -> Result<RunOutcome<S>, GraphError> {
        self.run_with(thread_id, Command::resume(value)).await
    }

    /// Like [`run`](Self::run), but yields [`GraphEvent`]s as the graph
    /// executes: node start/end, tokens and custom events emitted by nodes,
    /// and a final `Completed`, `Interrupted` or `Error`.
    ///
    /// The run is driven by polling the stream; dropping it cancels the run.
    pub fn stream(&self, thread_id: impl Into<String>, input: Option<S>) -> GraphStream<S> {
        self.stream_with(
            RunConfig::new(thread_id),
            RunInput::from(input),
            StreamOptions::default(),
        )
    }

    /// [`stream`](Self::stream) with a config, any [`RunInput`] and a choice
    /// of [`StreamMode`](crate::StreamMode)s (and subgraph events).
    pub fn stream_with(
        &self,
        config: impl Into<RunConfig>,
        input: impl Into<RunInput<S>>,
        options: impl Into<StreamOptions>,
    ) -> GraphStream<S> {
        let (tx, rx) = mpsc::unbounded_channel();
        let graph = self.clone();
        let (config, input, options) = (config.into(), input.into(), options.into());
        let sink_tx = tx.clone();
        let sink = EventSink::new(options, move |ev| {
            let _ = sink_tx.send(ev);
        });
        let run = async move {
            let terminal = match graph.drive(config, input, Some(sink)).await {
                Ok(RunOutcome::Completed(state)) => GraphEvent::Completed { state },
                Ok(RunOutcome::Interrupted {
                    state,
                    next_nodes,
                    interrupts,
                    ..
                }) => GraphEvent::Interrupted {
                    next: next_nodes,
                    state,
                    interrupts,
                },
                Err(e) => GraphEvent::Error(e),
            };
            let _ = tx.send(terminal);
        };
        GraphStream::new(Box::pin(run), rx)
    }

    async fn drive(
        &self,
        config: RunConfig,
        input: RunInput<S>,
        sink: Option<EventSink<S>>,
    ) -> Result<RunOutcome<S>, GraphError> {
        let run = Run {
            thread_id: config.thread_id.clone(),
            ns: Vec::new(),
            ns_str: String::new(),
            checkpointer: self.checkpointer.clone(),
            store: self.store.clone(),
            tracer: self.tracer.clone(),
            parent_run: None,
            sink,
            resume: None,
            limit: config.recursion_limit.unwrap_or(self.recursion_limit),
        };
        match input {
            RunInput::Start(s) => self.start(run, s, config.checkpoint_id).await,
            RunInput::Continue => self.resume_from(run, config.checkpoint_id, None).await,
            RunInput::Command(c) => self.resume_from(run, config.checkpoint_id, Some(c)).await,
        }
    }

    /// Latest state of a thread and the (first) node that runs next.
    pub async fn get_state(&self, thread_id: &str) -> Result<(S, String), GraphError> {
        let cp = self.latest(thread_id).await?;
        Ok((serde_json::from_value(cp.state)?, cp.next))
    }

    /// Latest checkpoint of a thread, decoded.
    pub async fn get_state_snapshot(
        &self,
        thread_id: &str,
    ) -> Result<StateSnapshot<S>, GraphError> {
        StateSnapshot::from_checkpoint(self.latest(thread_id).await?)
    }

    /// A specific checkpoint of a thread, decoded.
    pub async fn get_state_at(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<StateSnapshot<S>, GraphError> {
        StateSnapshot::from_checkpoint(self.checkpoint_by_id(thread_id, checkpoint_id).await?)
    }

    /// Every checkpoint of a thread's root graph, **newest first** (all
    /// branches, in write order). Replay from any of them with
    /// [`RunConfig::at_checkpoint`], or fork with
    /// [`update_state_with`](Self::update_state_with).
    pub async fn get_state_history(
        &self,
        thread_id: &str,
    ) -> Result<Vec<StateSnapshot<S>>, GraphError> {
        let cp = self
            .checkpointer
            .as_ref()
            .ok_or(GraphError::NoCheckpointer)?;
        cp.history_in(thread_id, "")
            .await
            .map_err(GraphError::Checkpoint)?
            .into_iter()
            .rev()
            .map(StateSnapshot::from_checkpoint)
            .collect()
    }

    /// Replaces a paused thread's state (e.g. a human edit) without moving it.
    /// Pending tasks (including `SendTo` inputs and interrupts) are kept.
    pub async fn update_state(&self, thread_id: &str, state: S) -> Result<(), GraphError> {
        self.update_state_with(thread_id, state, None)
            .await
            .map(|_| ())
    }

    /// Writes a new checkpoint with `state` and returns its id.
    ///
    /// - With `config.checkpoint_id`, the new checkpoint follows that
    ///   (older) checkpoint: a **fork**. It becomes the thread's latest, so
    ///   `run(thread, None)` continues the new branch.
    /// - With `as_node`, the update counts as that node's output: the next
    ///   tasks are that node's outgoing edge evaluated on `state` (like
    ///   LangGraph's `as_node`). Otherwise the pending tasks are kept.
    pub async fn update_state_with(
        &self,
        config: impl Into<RunConfig>,
        state: S,
        as_node: Option<&str>,
    ) -> Result<String, GraphError> {
        let config = config.into();
        let (base, source) = match &config.checkpoint_id {
            Some(id) => (self.checkpoint_by_id(&config.thread_id, id).await?, "fork"),
            None => (self.latest(&config.thread_id).await?, "update"),
        };
        let mut cp = base.clone();
        cp.state = serde_json::to_value(&state)?;
        if let Some(node) = as_node {
            if !self.nodes.contains_key(node) {
                return Err(GraphError::InvalidCommand(format!(
                    "`as_node` names unknown node `{node}`"
                )));
            }
            let next = self.next_from_node(node, &state)?;
            cp.tasks = self.pending_of(&next)?;
            cp.next = cp
                .tasks
                .first()
                .map_or_else(|| END.to_string(), |t| t.node.clone());
            cp.step += 1;
        }
        let mut new = Checkpoint::new(&config.thread_id, cp.step, cp.tasks, cp.state)
            .with_parent((!base.checkpoint_id.is_empty()).then(|| base.checkpoint_id.clone()))
            .with_metadata("source", Value::from(source));
        if let Some(node) = as_node {
            new = new.with_metadata("writes", Value::from(vec![node.to_string()]));
        }
        let id = new.checkpoint_id.clone();
        self.put(new).await?;
        Ok(id)
    }

    pub(crate) async fn latest(&self, thread_id: &str) -> Result<Checkpoint, GraphError> {
        let cp = self
            .checkpointer
            .as_ref()
            .ok_or(GraphError::NoCheckpointer)?;
        cp.latest_in(thread_id, "")
            .await
            .map_err(GraphError::Checkpoint)?
            .ok_or_else(|| GraphError::NoCheckpoint(thread_id.into()))
    }

    pub(crate) async fn checkpoint_by_id(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<Checkpoint, GraphError> {
        let cp = self
            .checkpointer
            .as_ref()
            .ok_or(GraphError::NoCheckpointer)?;
        cp.get(thread_id, checkpoint_id)
            .await
            .map_err(GraphError::Checkpoint)?
            .ok_or_else(|| GraphError::CheckpointNotFound {
                thread_id: thread_id.into(),
                checkpoint_id: checkpoint_id.into(),
            })
    }

    async fn put(&self, checkpoint: Checkpoint) -> Result<(), GraphError> {
        match &self.checkpointer {
            Some(cp) => cp.put(checkpoint).await.map_err(GraphError::Checkpoint),
            None => Err(GraphError::NoCheckpointer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryCheckpointer;
    use serde::Deserialize;
    use sinew_trace::{MemorySink, RunKind};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Counter {
        n: u32,
        log: Vec<String>,
    }

    fn counter(n: u32) -> Counter {
        Counter { n, log: vec![] }
    }

    fn looping() -> StateGraph<Counter> {
        let mut g = StateGraph::new();
        g.add_node("inc", |mut s: Counter| async move {
            s.n += 1;
            s.log.push("inc".into());
            Ok(s)
        });
        g.add_node("report", |mut s: Counter| async move {
            s.log.push(format!("done at {}", s.n));
            Ok(s)
        });
        g.add_conditional_edges(
            "inc",
            |s: &Counter| {
                if s.n < 3 {
                    "inc".into()
                } else {
                    "report".into()
                }
            },
            ["inc", "report"],
        );
        g.add_edge("report", END);
        g.set_entry_point("inc");
        g
    }

    #[tokio::test]
    async fn runs_cycles_until_router_exits() {
        let out = looping()
            .compile()
            .unwrap()
            .invoke(counter(0))
            .await
            .unwrap();
        assert_eq!(out.n, 3);
        assert_eq!(out.log.last().unwrap(), "done at 3");
    }

    #[tokio::test]
    async fn enforces_recursion_limit() {
        let mut g = StateGraph::new();
        g.add_node("spin", |s: Counter| async move { Ok(s) });
        g.add_edge("spin", "spin");
        g.set_entry_point("spin");
        let err = g
            .compile()
            .unwrap()
            .with_recursion_limit(5)
            .invoke(counter(0))
            .await
            .unwrap_err();
        assert!(matches!(err, GraphError::RecursionLimit(5)));
        assert!(err.to_string().contains("with_recursion_limit"));
    }

    #[test]
    fn compile_reports_every_problem() {
        let mut g: StateGraph<Counter> = StateGraph::new();
        g.add_node("a", |s: Counter| async move { Ok(s) });
        g.add_node("orphan", |s: Counter| async move { Ok(s) });
        g.add_node("a", |s: Counter| async move { Ok(s) });
        g.add_edge("a", "ghost");
        g.set_node_retry("nobody", RetryPolicy::default());
        g.set_entry_point("a");
        let Err(GraphError::Build(errs)) = g.compile() else {
            panic!("expected build error")
        };
        let all = errs.join("|");
        assert!(all.contains("added twice"), "{all}");
        assert!(
            all.contains("unknown node `ghost`") || all.contains("`a` -> `ghost`"),
            "{all}"
        );
        assert!(all.contains("`orphan` has no outgoing edge"), "{all}");
        assert!(all.contains("`orphan` is unreachable"), "{all}");
        assert!(
            all.contains("retry policy set for unknown node `nobody`"),
            "{all}"
        );
    }

    #[tokio::test]
    async fn unknown_route_is_an_error() {
        let mut g = StateGraph::new();
        g.add_node("a", |s: Counter| async move { Ok(s) });
        g.add_conditional_edges("a", |_: &Counter| "nowhere".into(), [END]);
        g.set_entry_point("a");
        let err = g.compile().unwrap().invoke(counter(0)).await.unwrap_err();
        assert!(matches!(err, GraphError::UnknownRoute { .. }));
    }

    #[tokio::test]
    async fn node_errors_carry_node_name() {
        let mut g = StateGraph::new();
        g.add_node("boom", |_: Counter| async move {
            Err::<Counter, BoxError>("kaput".into())
        });
        g.add_edge("boom", END);
        g.set_entry_point("boom");
        let err = g.compile().unwrap().invoke(counter(0)).await.unwrap_err();
        assert_eq!(err.to_string(), "node `boom` failed: kaput");
    }

    #[tokio::test]
    async fn interrupt_edit_and_resume() {
        let graph = looping()
            .compile()
            .unwrap()
            .with_checkpointer(MemoryCheckpointer::new())
            .interrupt_before(["report"]);

        let out = graph.run("t1", Some(counter(0))).await.unwrap();
        assert!(out.is_interrupted());
        assert_eq!(out.state().n, 3);

        let (mut state, next) = graph.get_state("t1").await.unwrap();
        assert_eq!(next, "report");
        state.n = 42; // human edits the state while paused
        graph.update_state("t1", state).await.unwrap();

        let done = graph.run("t1", None).await.unwrap();
        assert!(!done.is_interrupted());
        assert_eq!(done.state().log.last().unwrap(), "done at 42");

        // Resuming a finished thread returns its final state.
        let again = graph.run("t1", None).await.unwrap().into_state();
        assert_eq!(again.n, 42);
    }

    #[tokio::test]
    async fn invoke_refuses_to_silently_skip_interrupts() {
        let graph = looping().compile().unwrap().interrupt_before(["report"]);
        assert!(
            matches!(graph.invoke(counter(0)).await, Err(GraphError::Interrupted(n)) if n == "report")
        );
    }

    #[tokio::test]
    async fn traces_graph_and_nodes() {
        let sink = Arc::new(MemorySink::new());
        let graph = looping()
            .compile()
            .unwrap()
            .with_tracer(Tracer::new().with_shared_sink(sink.clone()));
        graph.invoke(counter(0)).await.unwrap();
        let runs = sink.runs();
        let root = runs.iter().find(|r| r.kind == RunKind::Graph).unwrap();
        let nodes: Vec<_> = runs.iter().filter(|r| r.kind == RunKind::Node).collect();
        assert_eq!(nodes.len(), 4); // inc x3 + report
        assert!(nodes
            .iter()
            .all(|n| n.parent_id.as_deref() == Some(root.id.as_str())));
    }
}
