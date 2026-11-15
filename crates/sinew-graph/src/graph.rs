use std::{
    collections::{HashMap, HashSet, VecDeque},
    future::Future,
    pin::Pin,
    sync::Arc,
};

use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};
use sinew_trace::{RunHandle, RunKind, RunRef, Tracer};
use thiserror::Error;
use tokio::{sync::mpsc, task::JoinSet};

use crate::{
    events::{Emission, Emitter, EventTx, GraphEvent, GraphStream},
    Checkpoint, Checkpointer, PendingTask,
};

/// Terminal pseudo-node.
pub const END: &str = "__end__";

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type NodeResult<S> = Result<S, BoxError>;
type BoxFut<S> = Pin<Box<dyn Future<Output = NodeResult<S>> + Send>>;
type NodeFn<S> = Arc<dyn Fn(S, NodeContext) -> BoxFut<S> + Send + Sync>;
type RouterFn<S> = Arc<dyn Fn(&S) -> Vec<String> + Send + Sync>;
type SendFn<S> = Arc<dyn Fn(&S) -> Vec<SendTo<S>> + Send + Sync>;
type ReducerFn<S> = Arc<dyn Fn(&S, S, S) -> S + Send + Sync>;

/// Bound satisfied by any graph state.
pub trait State: Clone + Send + Sync + Serialize + DeserializeOwned + 'static {}
impl<T: Clone + Send + Sync + Serialize + DeserializeOwned + 'static> State for T {}

/// Passed to nodes registered with [`StateGraph::add_node_with_context`].
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
    emitter: Option<Emitter>,
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
        }
    }

    /// Whether the graph is being consumed with [`CompiledGraph::stream`].
    /// Nodes can use it to pick a streaming model call only when someone
    /// listens.
    pub fn is_streaming(&self) -> bool {
        self.emitter.is_some()
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
}

/// A branch created by a [`StateGraph::add_send_edges`] router: run `node`
/// in the next superstep on `input` instead of the shared state. This is how
/// map-reduce is expressed: one `SendTo` per item.
#[derive(Debug, Clone)]
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
    #[error("recursion limit of {0} steps reached")]
    RecursionLimit(usize),
    #[error("router of `{from}` returned unknown target `{to}`")]
    UnknownRoute { from: String, to: String },
    #[error("graph was interrupted before `{0}`; use `run` with a checkpointer to resume")]
    Interrupted(String),
    #[error("no checkpoint for thread `{0}`")]
    NoCheckpoint(String),
    #[error("this operation needs a checkpointer")]
    NoCheckpointer,
    #[error("checkpointer error: {0}")]
    Checkpoint(String),
    #[error("state serialization failed: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("superstep runs several branches ({}) but no reducer is set; call `StateGraph::with_reducer`", .0.join(", "))]
    NoReducer(Vec<String>),
}

pub enum RunOutcome<S> {
    Completed(S),
    /// Paused before the next superstep; resume with `run(thread_id, None)`.
    Interrupted {
        state: S,
        /// First node of the paused superstep.
        next: String,
        /// Every node of the paused superstep (more than one after a fan-out).
        next_nodes: Vec<String>,
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
}

/// A thread's latest checkpoint, decoded.
#[derive(Debug, Clone)]
pub struct StateSnapshot<S> {
    pub state: S,
    /// Nodes of the next superstep; empty when the run finished.
    pub next: Vec<String>,
    /// Supersteps completed so far.
    pub step: usize,
}

enum Edge<S> {
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
    fn targets(&self) -> Vec<&str> {
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

struct Task<S> {
    node: String,
    input: Option<S>,
}

pub struct StateGraph<S> {
    nodes: HashMap<String, NodeFn<S>>,
    order: Vec<String>,
    edges: HashMap<String, Edge<S>>,
    entry: Option<String>,
    reducer: Option<ReducerFn<S>>,
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
            errors: Vec::new(),
        }
    }

    pub fn add_node<F, Fut>(&mut self, name: impl Into<String>, f: F) -> &mut Self
    where
        F: Fn(S) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = NodeResult<S>> + Send + 'static,
    {
        self.insert(name.into(), Arc::new(move |s, _ctx| Box::pin(f(s))))
    }

    pub fn add_node_with_context<F, Fut>(&mut self, name: impl Into<String>, f: F) -> &mut Self
    where
        F: Fn(S, NodeContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = NodeResult<S>> + Send + 'static,
    {
        self.insert(name.into(), Arc::new(move |s, ctx| Box::pin(f(s, ctx))))
    }

    fn insert(&mut self, name: String, f: NodeFn<S>) -> &mut Self {
        if name.is_empty() || name == END {
            self.errors
                .push(format!("`{name}` is not a valid node name"));
        } else if self.nodes.insert(name.clone(), f).is_some() {
            self.errors.push(format!("node `{name}` added twice"));
        } else {
            self.order.push(name);
        }
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
    /// the pre-superstep state to tell what a branch changed. See
    /// `docs/fan-out.md`.
    pub fn with_reducer<F>(&mut self, reducer: F) -> &mut Self
    where
        F: Fn(S, S) -> S + Send + Sync + 'static,
    {
        self.reducer = Some(Arc::new(move |_base: &S, acc, branch| reducer(acc, branch)));
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
        self.reducer = Some(Arc::new(reducer));
        self
    }

    fn set_edge(&mut self, from: String, edge: Edge<S>) -> &mut Self {
        if self.edges.insert(from.clone(), edge).is_some() {
            self.errors.push(format!(
                "node `{from}` has more than one outgoing edge definition"
            ));
        }
        self
    }

    pub fn set_entry_point(&mut self, name: impl Into<String>) -> &mut Self {
        self.entry = Some(name.into());
        self
    }

    pub fn compile(self) -> Result<CompiledGraph<S>, GraphError> {
        let mut errors = self.errors;
        let known = |n: &str| n == END || self.nodes.contains_key(n);
        match &self.entry {
            None => errors.push("no entry point set".into()),
            Some(e) if !self.nodes.contains_key(e) => {
                errors.push(format!("entry point `{e}` is not a node"))
            }
            _ => {}
        }
        for (from, edge) in &self.edges {
            if !self.nodes.contains_key(from) {
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
        for n in &self.order {
            if !self.edges.contains_key(n) {
                errors.push(format!(
                    "node `{n}` has no outgoing edge (add one to END to finish)"
                ));
            }
        }
        if let Some(entry) = self.entry.as_ref().filter(|e| self.nodes.contains_key(*e)) {
            let mut seen = HashSet::from([entry.as_str()]);
            let mut queue = VecDeque::from([entry.as_str()]);
            while let Some(n) = queue.pop_front() {
                for t in self.edges.get(n).map(|e| e.targets()).unwrap_or_default() {
                    if t != END && self.nodes.contains_key(t) && seen.insert(t) {
                        queue.push_back(t);
                    }
                }
            }
            for n in &self.order {
                if !seen.contains(n.as_str()) {
                    errors.push(format!("node `{n}` is unreachable from `{entry}`"));
                }
            }
        }
        if !errors.is_empty() {
            errors.sort();
            return Err(GraphError::Build(errors));
        }
        Ok(CompiledGraph {
            nodes: Arc::new(self.nodes),
            edges: Arc::new(self.edges),
            entry: self.entry.unwrap(),
            reducer: self.reducer,
            recursion_limit: 25,
            checkpointer: None,
            interrupt_before: Arc::new(HashSet::new()),
            tracer: None,
        })
    }
}

/// An executable graph. Cheap to clone and share across tasks.
///
/// Execution proceeds in *supersteps* (as in Pregel / LangGraph): every node
/// scheduled for a step runs concurrently on a snapshot of the state, their
/// outputs are merged (a single plain node simply replaces the state; several
/// branches go through the reducer), then every executed node's outgoing edge
/// is evaluated on the merged state to schedule the next step. Identical
/// target nodes are deduplicated, which is how parallel branches join.
pub struct CompiledGraph<S> {
    nodes: Arc<HashMap<String, NodeFn<S>>>,
    edges: Arc<HashMap<String, Edge<S>>>,
    entry: String,
    reducer: Option<ReducerFn<S>>,
    recursion_limit: usize,
    checkpointer: Option<Arc<dyn Checkpointer>>,
    interrupt_before: Arc<HashSet<String>>,
    tracer: Option<Tracer>,
}

impl<S> Clone for CompiledGraph<S> {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            entry: self.entry.clone(),
            reducer: self.reducer.clone(),
            recursion_limit: self.recursion_limit,
            checkpointer: self.checkpointer.clone(),
            interrupt_before: self.interrupt_before.clone(),
            tracer: self.tracer.clone(),
        }
    }
}

impl<S: State> CompiledGraph<S> {
    /// Maximum supersteps per thread (default 25). Without fan-out a
    /// superstep is exactly one node execution.
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

    pub fn with_tracer(mut self, tracer: Tracer) -> Self {
        self.tracer = Some(tracer);
        self
    }

    /// Runs to completion on a throwaway thread.
    pub async fn invoke(&self, state: S) -> Result<S, GraphError> {
        let thread = uuid::Uuid::new_v4().to_string();
        match self
            .execute(&thread, state, self.entry_tasks(), 0, false, None)
            .await?
        {
            RunOutcome::Completed(s) => Ok(s),
            RunOutcome::Interrupted { next, .. } => Err(GraphError::Interrupted(next)),
        }
    }

    /// Starts a thread with `Some(input)`, or resumes its latest checkpoint with `None`.
    pub async fn run(
        &self,
        thread_id: &str,
        input: Option<S>,
    ) -> Result<RunOutcome<S>, GraphError> {
        self.run_with(thread_id, input, None).await
    }

    /// Like [`run`](Self::run), but yields [`GraphEvent`]s as the graph
    /// executes: node start/end, tokens and custom events emitted by nodes,
    /// and a final `Completed`, `Interrupted` or `Error`.
    ///
    /// The run is driven by polling the stream; dropping it cancels the run.
    pub fn stream(&self, thread_id: impl Into<String>, input: Option<S>) -> GraphStream<S> {
        let (tx, rx) = mpsc::unbounded_channel();
        let graph = self.clone();
        let thread_id = thread_id.into();
        let run = async move {
            let terminal = match graph.run_with(&thread_id, input, Some(tx.clone())).await {
                Ok(RunOutcome::Completed(state)) => GraphEvent::Completed { state },
                Ok(RunOutcome::Interrupted {
                    state, next_nodes, ..
                }) => GraphEvent::Interrupted {
                    next: next_nodes,
                    state,
                },
                Err(e) => GraphEvent::Error(e),
            };
            let _ = tx.send(terminal);
        };
        GraphStream::new(Box::pin(run), rx)
    }

    async fn run_with(
        &self,
        thread_id: &str,
        input: Option<S>,
        events: Option<EventTx<S>>,
    ) -> Result<RunOutcome<S>, GraphError> {
        match input {
            Some(s) => {
                self.execute(thread_id, s, self.entry_tasks(), 0, false, events)
                    .await
            }
            None => {
                let cp = self.latest(thread_id).await?;
                let state: S = serde_json::from_value(cp.state.clone())?;
                let tasks = Self::tasks_from(&cp)?;
                if tasks.is_empty() {
                    return Ok(RunOutcome::Completed(state));
                }
                self.execute(thread_id, state, tasks, cp.step, true, events)
                    .await
            }
        }
    }

    /// Latest state of a thread and the (first) node that runs next.
    pub async fn get_state(&self, thread_id: &str) -> Result<(S, String), GraphError> {
        let cp = self.latest(thread_id).await?;
        Ok((serde_json::from_value(cp.state)?, cp.next))
    }

    /// Latest state of a thread with every node of the next superstep.
    pub async fn get_state_snapshot(
        &self,
        thread_id: &str,
    ) -> Result<StateSnapshot<S>, GraphError> {
        let cp = self.latest(thread_id).await?;
        Ok(StateSnapshot {
            next: cp.next_nodes(),
            step: cp.step,
            state: serde_json::from_value(cp.state)?,
        })
    }

    /// Replaces a paused thread's state (e.g. a human edit) without moving it.
    /// Pending `SendTo` inputs are kept as they were.
    pub async fn update_state(&self, thread_id: &str, state: S) -> Result<(), GraphError> {
        let mut cp = self.latest(thread_id).await?;
        cp.state = serde_json::to_value(&state)?;
        self.put(cp).await
    }

    fn entry_tasks(&self) -> Vec<Task<S>> {
        vec![Task {
            node: self.entry.clone(),
            input: None,
        }]
    }

    fn tasks_from(cp: &Checkpoint) -> Result<Vec<Task<S>>, GraphError> {
        if cp.tasks.is_empty() {
            return Ok(cp
                .next_nodes()
                .into_iter()
                .map(|node| Task { node, input: None })
                .collect());
        }
        cp.tasks
            .iter()
            .map(|t| {
                Ok(Task {
                    node: t.node.clone(),
                    input: t.input.clone().map(serde_json::from_value).transpose()?,
                })
            })
            .collect()
    }

    async fn latest(&self, thread_id: &str) -> Result<Checkpoint, GraphError> {
        let cp = self
            .checkpointer
            .as_ref()
            .ok_or(GraphError::NoCheckpointer)?;
        cp.latest(thread_id)
            .await
            .map_err(GraphError::Checkpoint)?
            .ok_or_else(|| GraphError::NoCheckpoint(thread_id.into()))
    }

    async fn put(&self, checkpoint: Checkpoint) -> Result<(), GraphError> {
        match &self.checkpointer {
            Some(cp) => cp.put(checkpoint).await.map_err(GraphError::Checkpoint),
            None => Ok(()),
        }
    }

    async fn save(
        &self,
        thread_id: &str,
        step: usize,
        tasks: &[Task<S>],
        state: &S,
    ) -> Result<(), GraphError> {
        if self.checkpointer.is_none() {
            return Ok(());
        }
        let pending = tasks
            .iter()
            .map(|t| {
                Ok(PendingTask {
                    node: t.node.clone(),
                    input: t.input.as_ref().map(serde_json::to_value).transpose()?,
                })
            })
            .collect::<Result<Vec<_>, serde_json::Error>>()?;
        self.put(Checkpoint::new(
            thread_id,
            step,
            pending,
            serde_json::to_value(state)?,
        ))
        .await
    }

    /// Evaluates the outgoing edge of every node executed in a superstep
    /// (once per distinct node) on the merged state.
    fn next_tasks(&self, executed: &[Task<S>], state: &S) -> Result<Vec<Task<S>>, GraphError> {
        let mut out = Vec::new();
        let mut seen_targets = HashSet::new();
        let mut seen_sources = HashSet::new();
        let mut push_plain = |out: &mut Vec<Task<S>>, to: &str| {
            if to != END && seen_targets.insert(to.to_string()) {
                out.push(Task {
                    node: to.to_string(),
                    input: None,
                });
            }
        };
        for t in executed {
            if !seen_sources.insert(t.node.as_str()) {
                continue;
            }
            let from = t.node.as_str();
            let unknown = |to: String| GraphError::UnknownRoute {
                from: from.into(),
                to,
            };
            match &self.edges[from] {
                Edge::Static(targets) => {
                    for to in targets {
                        push_plain(&mut out, to);
                    }
                }
                Edge::Router {
                    router, targets, ..
                } => {
                    for to in router(state) {
                        if !targets.contains(&to) {
                            return Err(unknown(to));
                        }
                        push_plain(&mut out, &to);
                    }
                }
                Edge::Send { router, targets } => {
                    for send in router(state) {
                        if !targets.contains(&send.node) {
                            return Err(unknown(send.node));
                        }
                        if send.node != END {
                            out.push(Task {
                                node: send.node,
                                input: Some(send.input),
                            });
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    async fn execute(
        &self,
        thread_id: &str,
        state: S,
        tasks: Vec<Task<S>>,
        step: usize,
        resuming: bool,
        events: Option<EventTx<S>>,
    ) -> Result<RunOutcome<S>, GraphError> {
        let start: Vec<&str> = tasks.iter().map(|t| t.node.as_str()).collect();
        let graph_run = self.tracer.as_ref().map(|t| {
            t.start(
                "graph",
                RunKind::Graph,
                None,
                json!({ "thread_id": thread_id, "start": start, "resumed": resuming }),
            )
        });
        let parent = graph_run.as_ref().map(|r| r.reference());
        let result = self
            .step_loop(
                thread_id,
                state,
                tasks,
                step,
                resuming,
                parent.as_ref(),
                events.as_ref(),
            )
            .await;
        if let Some(run) = graph_run {
            match &result {
                Ok(RunOutcome::Completed(s)) => {
                    run.finish(json!({ "status": "completed", "state": s }))
                }
                Ok(RunOutcome::Interrupted { next_nodes, .. }) => {
                    run.finish(json!({ "status": "interrupted", "next": next_nodes }))
                }
                Err(e) => run.fail(e),
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    async fn step_loop(
        &self,
        thread_id: &str,
        mut state: S,
        mut tasks: Vec<Task<S>>,
        mut step: usize,
        mut skip_interrupt: bool,
        parent: Option<&RunRef>,
        events: Option<&EventTx<S>>,
    ) -> Result<RunOutcome<S>, GraphError> {
        loop {
            if tasks.is_empty() {
                return Ok(RunOutcome::Completed(state));
            }
            if !skip_interrupt
                && tasks
                    .iter()
                    .any(|t| self.interrupt_before.contains(&t.node))
            {
                self.save(thread_id, step, &tasks, &state).await?;
                let next_nodes: Vec<String> = tasks.iter().map(|t| t.node.clone()).collect();
                return Ok(RunOutcome::Interrupted {
                    state,
                    next: next_nodes[0].clone(),
                    next_nodes,
                });
            }
            skip_interrupt = false;
            if step >= self.recursion_limit {
                return Err(GraphError::RecursionLimit(self.recursion_limit));
            }
            let mut outputs = self
                .superstep(thread_id, step, &state, &tasks, parent, events)
                .await?;
            state = if tasks.len() == 1 && tasks[0].input.is_none() {
                outputs.pop().expect("one output per task")
            } else {
                let reducer = self.reducer.as_ref().ok_or_else(|| {
                    GraphError::NoReducer(tasks.iter().map(|t| t.node.clone()).collect())
                })?;
                let base = state.clone();
                outputs
                    .into_iter()
                    .fold(state, |acc, out| reducer(&base, acc, out))
            };
            step += 1;
            tasks = self.next_tasks(&tasks, &state)?;
            self.save(thread_id, step, &tasks, &state).await?;
        }
    }

    /// Runs every task of one superstep and returns their outputs in task
    /// order. A single task runs inline; several run as concurrent tokio
    /// tasks, and the first failure cancels the rest.
    async fn superstep(
        &self,
        thread_id: &str,
        step: usize,
        state: &S,
        tasks: &[Task<S>],
        parent: Option<&RunRef>,
        events: Option<&EventTx<S>>,
    ) -> Result<Vec<S>, GraphError> {
        let parallel = tasks.len() > 1;
        let mut jobs = Vec::with_capacity(tasks.len());
        for (index, t) in tasks.iter().enumerate() {
            let input = t.input.clone().unwrap_or_else(|| state.clone());
            let run = self.tracer.as_ref().map(|tr| {
                let mut r = tr.start(
                    t.node.clone(),
                    RunKind::Node,
                    parent,
                    serde_json::to_value(&input).unwrap_or_default(),
                );
                r.set_metadata("step", json!(step));
                if parallel {
                    r.set_metadata("task", json!(index));
                }
                r
            });
            let emitter = events.map(|tx| {
                let (tx, node) = (tx.clone(), t.node.clone());
                Emitter::new(move |e| {
                    let ev = match e {
                        Emission::Token(delta) => GraphEvent::Token {
                            node: node.clone(),
                            step,
                            task: index,
                            delta,
                        },
                        Emission::Custom(value) => GraphEvent::Custom {
                            node: node.clone(),
                            step,
                            task: index,
                            value,
                        },
                    };
                    let _ = tx.send(ev);
                })
            });
            let ctx = NodeContext {
                node: t.node.clone(),
                step,
                thread_id: thread_id.into(),
                tracer: self.tracer.clone(),
                run: run.as_ref().map(|r| r.reference()),
                task: index,
                emitter,
            };
            jobs.push(Job {
                node: t.node.clone(),
                f: self.nodes[&t.node].clone(),
                input,
                ctx,
                run,
                events: events.cloned(),
            });
        }

        if !parallel {
            let job = jobs.pop().expect("superstep with no tasks");
            return Ok(vec![job.run().await?]);
        }

        let names: Vec<String> = jobs.iter().map(|j| j.node.clone()).collect();
        let mut set = JoinSet::new();
        let mut ids = HashMap::new();
        for (index, job) in jobs.into_iter().enumerate() {
            let handle = set.spawn(async move { (index, job.run().await) });
            ids.insert(handle.id(), index);
        }
        let mut outputs: Vec<Option<S>> = (0..names.len()).map(|_| None).collect();
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok((index, Ok(s))) => outputs[index] = Some(s),
                Ok((_, Err(e))) => return Err(e),
                Err(join_err) => {
                    let index = ids.get(&join_err.id()).copied().unwrap_or(0);
                    return Err(GraphError::Node {
                        node: names[index].clone(),
                        source: format!("branch task failed: {join_err}").into(),
                    });
                }
            }
        }
        Ok(outputs
            .into_iter()
            .map(|o| o.expect("every branch finished"))
            .collect())
    }
}

/// One node execution, ready to run (possibly on another tokio task).
struct Job<S> {
    node: String,
    f: NodeFn<S>,
    input: S,
    ctx: NodeContext,
    run: Option<RunHandle>,
    events: Option<EventTx<S>>,
}

impl<S: State> Job<S> {
    async fn run(self) -> Result<S, GraphError> {
        let (step, task) = (self.ctx.step, self.ctx.task);
        if let Some(tx) = &self.events {
            let _ = tx.send(GraphEvent::NodeStart {
                node: self.node.clone(),
                step,
                task,
            });
        }
        match (self.f)(self.input, self.ctx).await {
            Ok(s) => {
                if let Some(r) = self.run {
                    r.finish(serde_json::to_value(&s).unwrap_or_default());
                }
                if let Some(tx) = &self.events {
                    let _ = tx.send(GraphEvent::NodeEnd {
                        node: self.node,
                        step,
                        task,
                        state: s.clone(),
                    });
                }
                Ok(s)
            }
            Err(source) => {
                if let Some(r) = self.run {
                    r.fail(&source);
                }
                Err(GraphError::Node {
                    node: self.node,
                    source,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryCheckpointer;
    use serde::Deserialize;
    use sinew_trace::MemorySink;

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
    }

    #[test]
    fn compile_reports_every_problem() {
        let mut g: StateGraph<Counter> = StateGraph::new();
        g.add_node("a", |s: Counter| async move { Ok(s) });
        g.add_node("orphan", |s: Counter| async move { Ok(s) });
        g.add_node("a", |s: Counter| async move { Ok(s) });
        g.add_edge("a", "ghost");
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
