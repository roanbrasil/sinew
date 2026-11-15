use std::{
    collections::{HashMap, HashSet, VecDeque},
    future::Future,
    pin::Pin,
    sync::Arc,
};

use serde::{de::DeserializeOwned, Serialize};
use serde_json::json;
use sinew_trace::{RunKind, RunRef, Tracer};
use thiserror::Error;

use crate::{Checkpoint, Checkpointer};

/// Terminal pseudo-node.
pub const END: &str = "__end__";

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type NodeResult<S> = Result<S, BoxError>;
type BoxFut<S> = Pin<Box<dyn Future<Output = NodeResult<S>> + Send>>;
type NodeFn<S> = Arc<dyn Fn(S, NodeContext) -> BoxFut<S> + Send + Sync>;
type RouterFn<S> = Arc<dyn Fn(&S) -> String + Send + Sync>;

/// Bound satisfied by any graph state.
pub trait State: Clone + Send + Sync + Serialize + DeserializeOwned + 'static {}
impl<T: Clone + Send + Sync + Serialize + DeserializeOwned + 'static> State for T {}

/// Passed to nodes registered with [`StateGraph::add_node_with_context`].
#[derive(Clone)]
pub struct NodeContext {
    pub node: String,
    pub step: usize,
    pub thread_id: String,
    pub tracer: Option<Tracer>,
    /// The node's own run, for attaching model/tool child runs.
    pub run: Option<RunRef>,
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
}

pub enum RunOutcome<S> {
    Completed(S),
    /// Paused before `next`; resume with `run(thread_id, None)`.
    Interrupted {
        state: S,
        next: String,
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

enum Edge<S> {
    Direct(String),
    Conditional {
        router: RouterFn<S>,
        targets: Vec<String>,
    },
}

impl<S> Clone for Edge<S> {
    fn clone(&self) -> Self {
        match self {
            Edge::Direct(t) => Edge::Direct(t.clone()),
            Edge::Conditional { router, targets } => Edge::Conditional {
                router: router.clone(),
                targets: targets.clone(),
            },
        }
    }
}

impl<S> Edge<S> {
    fn targets(&self) -> Vec<&str> {
        match self {
            Edge::Direct(t) => vec![t.as_str()],
            Edge::Conditional { targets, .. } => targets.iter().map(String::as_str).collect(),
        }
    }
}

pub struct StateGraph<S> {
    nodes: HashMap<String, NodeFn<S>>,
    order: Vec<String>,
    edges: HashMap<String, Edge<S>>,
    entry: Option<String>,
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
        self.set_edge(from.into(), Edge::Direct(to.into()))
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
            Edge::Conditional {
                router: Arc::new(router),
                targets,
            },
        )
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
                errors.push(format!("conditional edge from `{from}` lists no targets"));
            }
            for t in targets {
                if !known(t) {
                    errors.push(format!("edge `{from}` -> `{t}` points to an unknown node"));
                }
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
            recursion_limit: 25,
            checkpointer: None,
            interrupt_before: Arc::new(HashSet::new()),
            tracer: None,
        })
    }
}

/// An executable graph. Cheap to clone and share across tasks.
pub struct CompiledGraph<S> {
    nodes: Arc<HashMap<String, NodeFn<S>>>,
    edges: Arc<HashMap<String, Edge<S>>>,
    entry: String,
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
            recursion_limit: self.recursion_limit,
            checkpointer: self.checkpointer.clone(),
            interrupt_before: self.interrupt_before.clone(),
            tracer: self.tracer.clone(),
        }
    }
}

impl<S: State> CompiledGraph<S> {
    /// Maximum node executions per run (default 25).
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

    /// Pause before these nodes (human-in-the-loop). Needs a checkpointer to resume.
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
            .execute(&thread, state, self.entry.clone(), 0, false)
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
        match input {
            Some(s) => {
                self.execute(thread_id, s, self.entry.clone(), 0, false)
                    .await
            }
            None => {
                let cp = self.latest(thread_id).await?;
                let state: S = serde_json::from_value(cp.state)?;
                if cp.next == END {
                    return Ok(RunOutcome::Completed(state));
                }
                self.execute(thread_id, state, cp.next, cp.step, true).await
            }
        }
    }

    /// Latest state of a thread and the node that runs next.
    pub async fn get_state(&self, thread_id: &str) -> Result<(S, String), GraphError> {
        let cp = self.latest(thread_id).await?;
        Ok((serde_json::from_value(cp.state)?, cp.next))
    }

    /// Replaces a paused thread's state (e.g. a human edit) without moving it.
    pub async fn update_state(&self, thread_id: &str, state: S) -> Result<(), GraphError> {
        let cp = self.latest(thread_id).await?;
        self.save(thread_id, cp.step, &cp.next, &state).await
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

    async fn save(
        &self,
        thread_id: &str,
        step: usize,
        next: &str,
        state: &S,
    ) -> Result<(), GraphError> {
        if let Some(cp) = &self.checkpointer {
            let checkpoint = Checkpoint {
                thread_id: thread_id.into(),
                step,
                next: next.into(),
                state: serde_json::to_value(state)?,
            };
            cp.put(checkpoint).await.map_err(GraphError::Checkpoint)?;
        }
        Ok(())
    }

    fn route(&self, from: &str, state: &S) -> Result<String, GraphError> {
        match &self.edges[from] {
            Edge::Direct(t) => Ok(t.clone()),
            Edge::Conditional { router, targets } => {
                let to = router(state);
                if targets.contains(&to) {
                    Ok(to)
                } else {
                    Err(GraphError::UnknownRoute {
                        from: from.into(),
                        to,
                    })
                }
            }
        }
    }

    async fn execute(
        &self,
        thread_id: &str,
        state: S,
        start: String,
        step: usize,
        resuming: bool,
    ) -> Result<RunOutcome<S>, GraphError> {
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
            .step_loop(thread_id, state, start, step, resuming, parent.as_ref())
            .await;
        if let Some(run) = graph_run {
            match &result {
                Ok(RunOutcome::Completed(s)) => {
                    run.finish(json!({ "status": "completed", "state": s }))
                }
                Ok(RunOutcome::Interrupted { next, .. }) => {
                    run.finish(json!({ "status": "interrupted", "next": next }))
                }
                Err(e) => run.fail(e),
            }
        }
        result
    }

    async fn step_loop(
        &self,
        thread_id: &str,
        mut state: S,
        mut current: String,
        mut step: usize,
        mut skip_interrupt: bool,
        parent: Option<&RunRef>,
    ) -> Result<RunOutcome<S>, GraphError> {
        loop {
            if current == END {
                return Ok(RunOutcome::Completed(state));
            }
            if !skip_interrupt && self.interrupt_before.contains(&current) {
                self.save(thread_id, step, &current, &state).await?;
                return Ok(RunOutcome::Interrupted {
                    state,
                    next: current,
                });
            }
            skip_interrupt = false;
            if step >= self.recursion_limit {
                return Err(GraphError::RecursionLimit(self.recursion_limit));
            }
            let node = self.nodes[&current].clone();
            let run = self.tracer.as_ref().map(|t| {
                let mut r = t.start(
                    current.clone(),
                    RunKind::Node,
                    parent,
                    serde_json::to_value(&state).unwrap_or_default(),
                );
                r.set_metadata("step", json!(step));
                r
            });
            let ctx = NodeContext {
                node: current.clone(),
                step,
                thread_id: thread_id.into(),
                tracer: self.tracer.clone(),
                run: run.as_ref().map(|r| r.reference()),
            };
            state = match node(state, ctx).await {
                Ok(s) => {
                    if let Some(r) = run {
                        r.finish(serde_json::to_value(&s).unwrap_or_default());
                    }
                    s
                }
                Err(source) => {
                    if let Some(r) = run {
                        r.fail(&source);
                    }
                    return Err(GraphError::Node {
                        node: current,
                        source,
                    });
                }
            };
            step += 1;
            let next = self.route(&current, &state)?;
            self.save(thread_id, step, &next, &state).await?;
            current = next;
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
