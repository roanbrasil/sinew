//! The superstep engine behind [`CompiledGraph`].

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_trace::{RunHandle, RunKind, RunRef, Tracer};
use tokio::task::JoinSet;

use crate::{
    command::{overlay, ParentCommand},
    events::{Emission, Emitter, EventSink, GraphEvent, StreamMode},
    graph::{Edge, NodeSpec, Reducer, ResumeMap, TaskInterrupts},
    policy::NodeCache,
    store::Store,
    Checkpoint, Checkpointer, Command, CommandGraph, CompiledGraph, Goto, GraphError,
    GraphInterrupt, Interrupt, NodeContext, NodeTimeout, PendingTask, Resume, RunOutcome, SendTo,
    State, END,
};

/// Everything one (sub)graph run needs besides the graph itself.
pub(crate) struct Run<S> {
    pub(crate) thread_id: String,
    pub(crate) ns: Vec<String>,
    pub(crate) ns_str: String,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    pub(crate) store: Option<Arc<dyn Store>>,
    pub(crate) tracer: Option<Tracer>,
    /// Tracing parent of this run's graph run (the subgraph node's run).
    pub(crate) parent_run: Option<RunRef>,
    pub(crate) sink: Option<EventSink<S>>,
    pub(crate) resume: Option<Arc<ResumeMap>>,
    pub(crate) limit: usize,
}

/// A finished task's writes, as stored in checkpoints when a superstep is
/// interrupted (and in the node cache).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "S: Serialize", deserialize = "S: Deserialize<'de>"))]
pub(crate) struct TaskResult<S> {
    /// The task's resolved output state; `None` when it wrote nothing.
    pub(crate) output: Option<S>,
    #[serde(default)]
    pub(crate) goto: Vec<Goto<S>>,
}

pub(crate) enum SubOutcome<S> {
    Done(S),
    Interrupted(Vec<Interrupt>),
}

enum TaskOutcome<S> {
    Done(TaskResult<S>),
    Interrupted(Vec<Interrupt>),
    Parent(ParentCommand),
}

pub(crate) struct Task<S> {
    pub(crate) node: String,
    pub(crate) input: Option<S>,
    resume: Vec<Value>,
    interrupts: Vec<Interrupt>,
    result: Option<TaskResult<S>>,
}

impl<S> Task<S> {
    fn plain(node: impl Into<String>) -> Self {
        Self {
            node: node.into(),
            input: None,
            resume: Vec::new(),
            interrupts: Vec::new(),
            result: None,
        }
    }

    fn send(node: String, input: S) -> Self {
        Self {
            input: Some(input),
            ..Self::plain(node)
        }
    }
}

struct LoopState<S> {
    state: S,
    tasks: Vec<Task<S>>,
    step: usize,
    parent: Option<String>,
    skip_before: bool,
    /// Resumed from an `interrupt_after` pause: a pause before the next
    /// superstep must be checkpointed again, or resuming would stop there
    /// forever.
    resave_on_pause: bool,
}

fn to_value<T: Serialize>(v: &T) -> Result<Value, GraphError> {
    Ok(serde_json::to_value(v)?)
}

impl<S: State> CompiledGraph<S> {
    /// Starts over from the entry point (writing an `input` checkpoint).
    pub(crate) async fn start(
        &self,
        run: Run<S>,
        state: S,
        parent: Option<String>,
    ) -> Result<RunOutcome<S>, GraphError> {
        let tasks = self.entry_tasks(&state)?;
        let parent = self
            .save(&run, 0, &tasks, &state, parent, "input", None, false)
            .await?;
        self.execute(
            &run,
            LoopState {
                state,
                tasks,
                step: 0,
                parent,
                skip_before: false,
                resave_on_pause: false,
            },
        )
        .await
    }

    /// Continues a thread from its latest (or a given) checkpoint.
    pub(crate) async fn resume_from(
        &self,
        run: Run<S>,
        checkpoint_id: Option<String>,
        cmd: Option<Command<S>>,
    ) -> Result<RunOutcome<S>, GraphError> {
        let cp_store = run.checkpointer.clone().ok_or(GraphError::NoCheckpointer)?;
        let cp = match &checkpoint_id {
            Some(id) => Some(
                cp_store
                    .get(&run.thread_id, id)
                    .await
                    .map_err(GraphError::Checkpoint)?
                    .ok_or_else(|| GraphError::CheckpointNotFound {
                        thread_id: run.thread_id.clone(),
                        checkpoint_id: id.clone(),
                    })?,
            ),
            None => cp_store
                .latest_in(&run.thread_id, &run.ns_str)
                .await
                .map_err(GraphError::Checkpoint)?,
        };
        match cp {
            Some(cp) => self.resume_loaded(run, cp, cmd).await,
            // A command with a state update can start a fresh thread.
            None => match cmd.and_then(|c| c.update) {
                Some(state) => self.start(run, state, None).await,
                None => Err(GraphError::NoCheckpoint(run.thread_id.clone())),
            },
        }
    }

    pub(crate) async fn resume_loaded(
        &self,
        mut run: Run<S>,
        cp: Checkpoint,
        cmd: Option<Command<S>>,
    ) -> Result<RunOutcome<S>, GraphError> {
        let mut state: S = serde_json::from_value(cp.state.clone())?;
        let mut tasks = self.tasks_from(&cp)?;
        let mut parent = (!cp.checkpoint_id.is_empty()).then(|| cp.checkpoint_id.clone());
        let paused_after = cp.metadata.get("paused").and_then(Value::as_str) == Some("after");
        if let Some(c) = cmd {
            if c.graph == CommandGraph::Parent {
                return Err(GraphError::InvalidCommand(
                    "a run input cannot target the parent graph".into(),
                ));
            }
            let mut changed = false;
            if c.update.is_some() || c.update_json.is_some() {
                state = apply_update(c.update.unwrap_or(state), c.update_json)?;
                changed = true;
            }
            if !c.goto.is_empty() {
                let mut seen = HashSet::new();
                tasks = Vec::new();
                for g in c.goto {
                    self.push_goto(START_LABEL, g, &mut tasks, &mut seen)?;
                }
                changed = true;
            }
            if changed {
                parent = self
                    .save(&run, cp.step, &tasks, &state, parent, "update", None, false)
                    .await?;
            }
            let map = match c.resume {
                Some(Resume::Value(v)) => tasks
                    .iter()
                    .flat_map(|t| t.interrupts.first())
                    .next()
                    .map(|i| HashMap::from([(i.id.clone(), v)]))
                    .unwrap_or_default(),
                Some(Resume::ById(m)) => m,
                None => HashMap::new(),
            };
            run.resume = Some(Arc::new(Mutex::new(map)));
        }
        if tasks.is_empty() {
            return Ok(RunOutcome::Completed(state));
        }
        if let Some(map) = &run.resume {
            let mut map = map.lock().unwrap();
            for t in &mut tasks {
                for i in &t.interrupts {
                    if i.namespace == run.ns {
                        if let Some(v) = map.remove(&i.id) {
                            t.resume.push(v);
                        }
                    }
                }
            }
        }
        self.execute(
            &run,
            LoopState {
                state,
                tasks,
                step: cp.step,
                parent,
                skip_before: !paused_after,
                resave_on_pause: paused_after,
            },
        )
        .await
    }

    /// Runs this graph as a node of a parent graph.
    pub(crate) async fn run_as_subgraph(
        &self,
        input: S,
        ctx: &NodeContext,
    ) -> Result<SubOutcome<S>, GraphError> {
        let seg = format!("{}:{}", ctx.node, ctx.task);
        let mut ns = ctx.namespace.clone();
        ns.push(seg.clone());
        let sink = match (&ctx.emitter, ctx.stream.subgraphs) {
            (Some(emitter), true) => {
                let emitter = emitter.clone();
                Some(EventSink::new(ctx.stream, move |ev: GraphEvent<S>| {
                    if let Some(j) = ev.to_json() {
                        let (namespace, event) = match j {
                            GraphEvent::Subgraph { namespace, event } => {
                                let mut full = vec![seg.clone()];
                                full.extend(namespace);
                                (full, *event)
                            }
                            other => (vec![seg.clone()], other),
                        };
                        emitter.send(Emission::Subgraph { namespace, event });
                    }
                }))
            }
            _ => None,
        };
        let run = Run {
            thread_id: ctx.thread_id.clone(),
            ns_str: ns.join("|"),
            ns,
            checkpointer: self
                .checkpointer
                .clone()
                .or_else(|| ctx.checkpointer.clone()),
            store: self.store.clone().or_else(|| ctx.store.clone()),
            tracer: self.tracer.clone().or_else(|| ctx.tracer.clone()),
            parent_run: ctx.run.clone(),
            sink,
            resume: ctx.resume_map.clone(),
            limit: self.recursion_limit,
        };
        let saved = match (&run.checkpointer, ctx.resuming) {
            (Some(cp), true) => cp
                .latest_in(&run.thread_id, &run.ns_str)
                .await
                .map_err(GraphError::Checkpoint)?
                .filter(|c| !c.tasks.is_empty()),
            _ => None,
        };
        let ns_str = run.ns_str.clone();
        let ns = run.ns.clone();
        let outcome = match saved {
            Some(cp) => self.resume_loaded(run, cp, None).await?,
            None => self.start(run, input, None).await?,
        };
        Ok(match outcome {
            RunOutcome::Completed(s) => SubOutcome::Done(s),
            RunOutcome::Interrupted {
                interrupts,
                next_nodes,
                ..
            } if interrupts.is_empty() => SubOutcome::Interrupted(vec![Interrupt {
                id: format!("{ns_str}|paused:{}", next_nodes.join(",")),
                value: json!({ "paused_before": next_nodes }),
                node: next_nodes.first().cloned().unwrap_or_default(),
                namespace: ns,
            }]),
            RunOutcome::Interrupted { interrupts, .. } => SubOutcome::Interrupted(interrupts),
        })
    }

    async fn execute(&self, run: &Run<S>, ls: LoopState<S>) -> Result<RunOutcome<S>, GraphError> {
        let graph_run = run.tracer.as_ref().map(|t| {
            let start: Vec<&str> = ls.tasks.iter().map(|t| t.node.as_str()).collect();
            let name = if run.ns.is_empty() {
                "graph".to_string()
            } else {
                run.ns_str.clone()
            };
            t.start(
                name,
                RunKind::Graph,
                run.parent_run.as_ref(),
                json!({ "thread_id": run.thread_id, "start": start, "resumed": ls.skip_before }),
            )
        });
        let parent = graph_run.as_ref().map(|r| r.reference());
        let result = self.step_loop(run, ls, parent.as_ref()).await;
        if let Some(gr) = graph_run {
            match &result {
                Ok(RunOutcome::Completed(s)) => {
                    gr.finish(json!({ "status": "completed", "state": s }))
                }
                Ok(RunOutcome::Interrupted { next_nodes, .. }) => {
                    gr.finish(json!({ "status": "interrupted", "next": next_nodes }))
                }
                Err(GraphError::ParentCommand(_)) => {
                    gr.finish(json!({ "status": "parent_command" }))
                }
                Err(e) => gr.fail(e),
            }
        }
        result
    }

    async fn step_loop(
        &self,
        run: &Run<S>,
        ls: LoopState<S>,
        trace_parent: Option<&RunRef>,
    ) -> Result<RunOutcome<S>, GraphError> {
        let LoopState {
            mut state,
            mut tasks,
            mut step,
            mut parent,
            mut skip_before,
            resave_on_pause,
        } = ls;
        if let Some(sink) = &run.sink {
            if sink.wants(StreamMode::Values) {
                sink.send(GraphEvent::Values {
                    step,
                    state: state.clone(),
                });
            }
        }
        let mut steps_run = 0;
        loop {
            if tasks.is_empty() {
                return Ok(RunOutcome::Completed(state));
            }
            if !skip_before
                && tasks
                    .iter()
                    .any(|t| self.interrupt_before.contains(&t.node))
            {
                // The current state and tasks were checkpointed already.
                if resave_on_pause && steps_run == 0 {
                    self.save(run, step, &tasks, &state, parent, "interrupt", None, false)
                        .await?;
                }
                return Ok(interrupted(state, &tasks, Vec::new()));
            }
            skip_before = false;
            if steps_run >= run.limit {
                return Err(GraphError::RecursionLimit(run.limit));
            }
            steps_run += 1;

            // A lone plain task without a checkpointer gets the state moved
            // in: nothing needs the pre-step state afterwards.
            let single_move = tasks.len() == 1
                && tasks[0].input.is_none()
                && tasks[0].result.is_none()
                && run.checkpointer.is_none()
                && run.ns.is_empty()
                && !self.nodes[&tasks[0].node].command;
            let (inputs, kept) = if single_move {
                (vec![Some(state)], None)
            } else {
                let inputs = tasks
                    .iter()
                    .map(|t| {
                        t.result
                            .is_none()
                            .then(|| t.input.clone().unwrap_or_else(|| state.clone()))
                    })
                    .collect();
                (inputs, Some(state))
            };
            let outcomes = self
                .superstep(run, step, &tasks, inputs, trace_parent)
                .await?;

            let mut results = Vec::with_capacity(tasks.len());
            let mut any_interrupt = false;
            let mut parent_cmd = None;
            for (i, o) in outcomes.into_iter().enumerate() {
                match o {
                    TaskOutcome::Done(r) => results.push(Some(r)),
                    TaskOutcome::Interrupted(v) => {
                        tasks[i].interrupts = v;
                        any_interrupt = true;
                        results.push(None);
                    }
                    TaskOutcome::Parent(pc) => {
                        parent_cmd.get_or_insert(pc);
                        results.push(None);
                    }
                }
            }
            if let Some(pc) = parent_cmd {
                if run.ns.is_empty() {
                    return Err(GraphError::InvalidCommand(
                        "a node returned a command for the parent graph, but it runs in the root graph".into(),
                    ));
                }
                return Err(GraphError::ParentCommand(Box::new(pc)));
            }
            if any_interrupt {
                let state = match (kept, &run.checkpointer) {
                    (Some(s), Some(_)) => s,
                    _ => {
                        let node = tasks
                            .iter()
                            .find(|t| !t.interrupts.is_empty())
                            .map(|t| t.node.clone())
                            .unwrap_or_default();
                        return Err(GraphError::Interrupted(node));
                    }
                };
                for (t, r) in tasks.iter_mut().zip(results) {
                    if r.is_some() {
                        t.result = r;
                        t.interrupts.clear();
                    }
                }
                self.save(run, step, &tasks, &state, parent, "interrupt", None, false)
                    .await?;
                let interrupts = tasks.iter().flat_map(|t| t.interrupts.clone()).collect();
                return Ok(interrupted(state, &tasks, interrupts));
            }
            let mut results: Vec<TaskResult<S>> = results.into_iter().map(Option::unwrap).collect();
            state = self.merge(kept, &tasks, &mut results)?;
            step += 1;
            let next = self.next_tasks(&tasks, &results, &state)?;
            let pause_after = !next.is_empty()
                && !self.interrupt_after.is_empty()
                && tasks.iter().any(|t| self.interrupt_after.contains(&t.node));
            let writes = run
                .checkpointer
                .is_some()
                .then(|| tasks.iter().map(|t| t.node.clone()).collect());
            parent = self
                .save(
                    run,
                    step,
                    &next,
                    &state,
                    parent,
                    "loop",
                    writes,
                    pause_after,
                )
                .await?;
            if let Some(sink) = &run.sink {
                if sink.wants(StreamMode::Values) {
                    sink.send(GraphEvent::Values {
                        step,
                        state: state.clone(),
                    });
                }
            }
            tasks = next;
            if pause_after {
                return Ok(interrupted(state, &tasks, Vec::new()));
            }
        }
    }

    /// Folds a superstep's outputs into the new state.
    fn merge(
        &self,
        kept: Option<S>,
        tasks: &[Task<S>],
        results: &mut [TaskResult<S>],
    ) -> Result<S, GraphError> {
        if tasks.len() == 1 && tasks[0].input.is_none() {
            return match results[0].output.take() {
                Some(o) => Ok(o),
                None => Ok(kept.expect("nodes without an update keep their input")),
            };
        }
        let base = kept.expect("parallel supersteps keep the pre-step state");
        let writes: Vec<S> = results.iter_mut().filter_map(|r| r.output.take()).collect();
        if writes.is_empty() {
            return Ok(base);
        }
        let reducer = self
            .reducer
            .as_ref()
            .ok_or_else(|| GraphError::NoReducer(tasks.iter().map(|t| t.node.clone()).collect()))?;
        Ok(match reducer {
            Reducer::Plain(f) => writes.into_iter().fold(base, |acc, o| f(acc, o)),
            Reducer::WithBase(f) => {
                let b = base.clone();
                writes.into_iter().fold(base, |acc, o| f(&b, acc, o))
            }
        })
    }

    pub(crate) fn entry_tasks(&self, state: &S) -> Result<Vec<Task<S>>, GraphError> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        self.eval_edge(START_LABEL, &self.entry, state, &mut out, &mut seen)?;
        Ok(out)
    }

    /// The tasks that follow `node` if it had just produced `state`.
    pub(crate) fn next_from_node(&self, node: &str, state: &S) -> Result<Vec<Task<S>>, GraphError> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        if let Some(edge) = self.edges.get(node) {
            self.eval_edge(node, edge, state, &mut out, &mut seen)?;
        }
        Ok(out)
    }

    fn eval_edge(
        &self,
        from: &str,
        edge: &Edge<S>,
        state: &S,
        out: &mut Vec<Task<S>>,
        seen: &mut HashSet<String>,
    ) -> Result<(), GraphError> {
        let unknown = |to: String| GraphError::UnknownRoute {
            from: from.into(),
            to,
        };
        match edge {
            Edge::Static(targets) => {
                for to in targets {
                    push_plain(out, seen, to);
                }
            }
            Edge::Router {
                router, targets, ..
            } => {
                for to in router(state) {
                    if !targets.contains(&to) {
                        return Err(unknown(to));
                    }
                    push_plain(out, seen, &to);
                }
            }
            Edge::Send { router, targets } => {
                for send in router(state) {
                    if !targets.contains(&send.node) {
                        return Err(unknown(send.node));
                    }
                    if send.node != END {
                        out.push(Task::send(send.node, send.input));
                    }
                }
            }
        }
        Ok(())
    }

    fn push_goto(
        &self,
        from: &str,
        g: Goto<S>,
        out: &mut Vec<Task<S>>,
        seen: &mut HashSet<String>,
    ) -> Result<(), GraphError> {
        let check = |n: &str| {
            if n == END || self.nodes.contains_key(n) {
                Ok(())
            } else {
                Err(GraphError::UnknownRoute {
                    from: from.into(),
                    to: n.into(),
                })
            }
        };
        match g {
            Goto::Node(n) => {
                check(&n)?;
                push_plain(out, seen, &n);
            }
            Goto::Send(SendTo { node, input }) => {
                check(&node)?;
                if node != END {
                    out.push(Task::send(node, input));
                }
            }
        }
        Ok(())
    }

    /// Evaluates the outgoing edge of every node executed in a superstep
    /// (once per distinct node) on the merged state, plus every `goto`.
    fn next_tasks(
        &self,
        executed: &[Task<S>],
        results: &[TaskResult<S>],
        state: &S,
    ) -> Result<Vec<Task<S>>, GraphError> {
        let mut out = Vec::new();
        let mut seen_targets = HashSet::new();
        let mut seen_sources = HashSet::new();
        for (t, r) in executed.iter().zip(results) {
            let from = t.node.as_str();
            if seen_sources.insert(from) {
                if let Some(edge) = self.edges.get(from) {
                    self.eval_edge(from, edge, state, &mut out, &mut seen_targets)?;
                }
            }
            for g in &r.goto {
                self.push_goto(from, g.clone(), &mut out, &mut seen_targets)?;
            }
        }
        Ok(out)
    }

    fn tasks_from(&self, cp: &Checkpoint) -> Result<Vec<Task<S>>, GraphError> {
        if cp.tasks.is_empty() {
            return Ok(cp.next_nodes().into_iter().map(Task::plain).collect());
        }
        cp.tasks
            .iter()
            .map(|t| {
                Ok(Task {
                    node: t.node.clone(),
                    input: t.input.clone().map(serde_json::from_value).transpose()?,
                    resume: t.resume.clone(),
                    interrupts: t.interrupts.clone(),
                    result: t.result.clone().map(serde_json::from_value).transpose()?,
                })
            })
            .collect()
    }

    pub(crate) fn pending_of(&self, tasks: &[Task<S>]) -> Result<Vec<PendingTask>, GraphError> {
        tasks
            .iter()
            .map(|t| {
                Ok(PendingTask {
                    node: t.node.clone(),
                    input: t.input.as_ref().map(to_value).transpose()?,
                    resume: t.resume.clone(),
                    interrupts: t.interrupts.clone(),
                    result: t.result.as_ref().map(to_value).transpose()?,
                })
            })
            .collect()
    }

    /// Writes a checkpoint (no-op without a checkpointer) and returns the id
    /// the next checkpoint should point to.
    #[allow(clippy::too_many_arguments)]
    async fn save(
        &self,
        run: &Run<S>,
        step: usize,
        tasks: &[Task<S>],
        state: &S,
        parent: Option<String>,
        source: &str,
        writes: Option<Vec<String>>,
        paused_after: bool,
    ) -> Result<Option<String>, GraphError> {
        let Some(cp) = &run.checkpointer else {
            return Ok(parent);
        };
        let mut c = Checkpoint::new(
            &run.thread_id,
            step,
            self.pending_of(tasks)?,
            to_value(state)?,
        )
        .with_parent(parent)
        .with_namespace(run.ns_str.clone())
        .with_metadata("source", Value::from(source));
        if let Some(w) = writes {
            c = c.with_metadata("writes", Value::from(w));
        }
        if paused_after {
            c = c.with_metadata("paused", Value::from("after"));
        }
        let id = c.checkpoint_id.clone();
        let debug = run
            .sink
            .as_ref()
            .filter(|s| s.wants(StreamMode::Debug))
            .map(|s| (s.clone(), c.next_nodes()));
        cp.put(c).await.map_err(GraphError::Checkpoint)?;
        if let Some((sink, next)) = debug {
            sink.send(GraphEvent::Checkpoint {
                checkpoint_id: id.clone(),
                step,
                next,
            });
        }
        Ok(Some(id))
    }

    /// Runs every task of one superstep and returns their outcomes in task
    /// order. A single task runs inline; several run as concurrent tokio
    /// tasks. A failure cancels the rest; interrupts let the others finish.
    async fn superstep(
        &self,
        run: &Run<S>,
        step: usize,
        tasks: &[Task<S>],
        inputs: Vec<Option<S>>,
        trace_parent: Option<&RunRef>,
    ) -> Result<Vec<TaskOutcome<S>>, GraphError> {
        let parallel = tasks.len() > 1;
        let mut outcomes: Vec<Option<TaskOutcome<S>>> = (0..tasks.len()).map(|_| None).collect();
        let mut jobs = Vec::with_capacity(tasks.len());
        for (index, (t, input)) in tasks.iter().zip(inputs).enumerate() {
            let Some(input) = input else {
                // Completed before an interrupt: reuse its stored result.
                outcomes[index] = t.result.clone().map(TaskOutcome::Done);
                continue;
            };
            jobs.push((
                index,
                self.job(run, step, index, parallel, t, input, trace_parent),
            ));
        }

        if jobs.len() == 1 && !parallel {
            let (index, job) = jobs.pop().expect("one job");
            outcomes[index] = Some(job.run().await?);
        } else if !jobs.is_empty() {
            let mut names = HashMap::new();
            let mut set = JoinSet::new();
            for (index, job) in jobs {
                let node = job.node.clone();
                let handle = set.spawn(async move { (index, job.run().await) });
                names.insert(handle.id(), node);
            }
            while let Some(joined) = set.join_next().await {
                match joined {
                    Ok((index, Ok(o))) => outcomes[index] = Some(o),
                    Ok((_, Err(e))) => return Err(e),
                    Err(join_err) => {
                        return Err(GraphError::Node {
                            node: names.get(&join_err.id()).cloned().unwrap_or_default(),
                            source: format!("branch task failed: {join_err}").into(),
                        });
                    }
                }
            }
        }
        Ok(outcomes
            .into_iter()
            .map(|o| o.expect("every task has an outcome"))
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn job(
        &self,
        run: &Run<S>,
        step: usize,
        index: usize,
        parallel: bool,
        t: &Task<S>,
        input: S,
        trace_parent: Option<&RunRef>,
    ) -> Job<S> {
        let spec = self.nodes[&t.node].clone();
        let trace = run.tracer.as_ref().map(|tr| {
            let mut r = tr.start(
                t.node.clone(),
                RunKind::Node,
                trace_parent,
                serde_json::to_value(&input).unwrap_or_default(),
            );
            r.set_metadata("step", json!(step));
            if parallel {
                r.set_metadata("task", json!(index));
            }
            if !run.ns.is_empty() {
                r.set_metadata("namespace", json!(run.ns));
            }
            r
        });
        let emitter = run.sink.as_ref().map(|sink| {
            let (sink, node) = (sink.clone(), t.node.clone());
            Emitter::new(move |e| match e {
                Emission::Token(delta) => {
                    if sink.wants(StreamMode::Messages) {
                        sink.send(GraphEvent::Token {
                            node: node.clone(),
                            step,
                            task: index,
                            delta,
                        })
                    }
                }
                Emission::Custom(value) => {
                    if sink.wants(StreamMode::Custom) {
                        sink.send(GraphEvent::Custom {
                            node: node.clone(),
                            step,
                            task: index,
                            value,
                        })
                    }
                }
                Emission::Subgraph { namespace, event } => {
                    if sink.opts.subgraphs {
                        sink.send(GraphEvent::Subgraph {
                            namespace,
                            event: Box::new(event),
                        })
                    }
                }
            })
        });
        let ctx = NodeContext {
            node: t.node.clone(),
            step,
            thread_id: run.thread_id.clone(),
            tracer: run.tracer.clone(),
            run: trace.as_ref().map(|r| r.reference()),
            task: index,
            stream: run.sink.as_ref().map(|s| s.opts).unwrap_or_default(),
            emitter,
            namespace: run.ns.clone(),
            store: run.store.clone(),
            checkpointer: run.checkpointer.clone(),
            resume_map: run.resume.clone(),
            resuming: !t.interrupts.is_empty(),
            interrupts: Arc::new(Mutex::new(TaskInterrupts {
                resume: t.resume.clone(),
                counter: 0,
            })),
        };
        Job {
            node: t.node.clone(),
            spec,
            input,
            ctx,
            trace,
            sink: run.sink.clone(),
            cache: self.cache.clone(),
        }
    }
}

const START_LABEL: &str = crate::START;

fn push_plain<S>(out: &mut Vec<Task<S>>, seen: &mut HashSet<String>, to: &str) {
    if to != END && seen.insert(to.to_string()) {
        out.push(Task::plain(to));
    }
}

fn interrupted<S>(state: S, tasks: &[Task<S>], interrupts: Vec<Interrupt>) -> RunOutcome<S> {
    let next_nodes: Vec<String> = tasks.iter().map(|t| t.node.clone()).collect();
    RunOutcome::Interrupted {
        state,
        next: next_nodes.first().cloned().unwrap_or_default(),
        next_nodes,
        interrupts,
    }
}

/// `update` with the JSON `patch` overlaid on its top-level fields.
fn apply_update<S: State>(update: S, patch: Option<Value>) -> Result<S, GraphError> {
    match patch {
        None => Ok(update),
        Some(p) => {
            let mut v = to_value(&update)?;
            overlay(&mut v, p);
            Ok(serde_json::from_value(v)?)
        }
    }
}

/// One node execution, ready to run (possibly on another tokio task).
struct Job<S> {
    node: String,
    spec: Arc<NodeSpec<S>>,
    input: S,
    ctx: NodeContext,
    trace: Option<RunHandle>,
    sink: Option<EventSink<S>>,
    cache: Arc<NodeCache<TaskResult<S>>>,
}

impl<S: State> Job<S> {
    async fn run(self) -> Result<TaskOutcome<S>, GraphError> {
        let Job {
            node,
            spec,
            input,
            ctx,
            trace,
            sink,
            cache,
        } = self;
        let (step, task) = (ctx.step, ctx.task);
        if let Some(s) = sink.as_ref().filter(|s| s.wants(StreamMode::Tasks)) {
            s.send(GraphEvent::NodeStart {
                node: node.clone(),
                step,
                task,
            });
        }
        let emit_end = |state: &S| {
            if let Some(s) = sink.as_ref().filter(|s| s.wants(StreamMode::Updates)) {
                s.send(GraphEvent::NodeEnd {
                    node: node.clone(),
                    step,
                    task,
                    state: state.clone(),
                });
            }
        };

        let cache_key = spec.cache.as_ref().and_then(|p| {
            serde_json::to_value(&input)
                .ok()
                .map(|v| (p.key(&node, &v), p.ttl))
        });
        if let Some((key, _)) = cache_key {
            if let Some(hit) = cache.get(key) {
                if let Some(o) = &hit.output {
                    emit_end(o);
                }
                if let Some(mut r) = trace {
                    r.set_metadata("cached", json!(true));
                    r.finish(serde_json::to_value(&hit.output).unwrap_or_default());
                }
                return Ok(TaskOutcome::Done(hit));
            }
        }

        let keep = spec.command.then(|| input.clone());
        let mut input = Some(input);
        let mut attempt = 0;
        let res = loop {
            attempt += 1;
            let more = spec
                .retry
                .as_ref()
                .is_some_and(|p| attempt < p.max_attempts);
            let inp = if more {
                input.clone().expect("input kept for retries")
            } else {
                input.take().expect("input available for the last attempt")
            };
            ctx.interrupts.lock().unwrap().counter = 0;
            let fut = (spec.f)(inp, ctx.clone());
            let r = match spec.timeout {
                Some(after) => match tokio::time::timeout(after, fut).await {
                    Ok(r) => r,
                    Err(_) => Err(Box::new(NodeTimeout { after }) as crate::BoxError),
                },
                None => fut.await,
            };
            match r {
                Ok(c) => break Ok(c),
                Err(e) => match &spec.retry {
                    Some(p) if p.should_retry(attempt, &e) => {
                        tokio::time::sleep(p.backoff(attempt)).await;
                    }
                    _ => break Err(e),
                },
            }
        };

        let cmd = match res {
            Ok(c) => c,
            Err(e) => {
                if let Some(gi) = e.downcast_ref::<GraphInterrupt>() {
                    let interrupts = gi.interrupts.clone();
                    if let Some(r) = trace {
                        r.finish(json!({ "interrupted": interrupts }));
                    }
                    return Ok(TaskOutcome::Interrupted(interrupts));
                }
                if let Some(r) = trace {
                    r.fail(&e);
                }
                return Err(GraphError::Node { node, source: e });
            }
        };

        if cmd.graph == CommandGraph::Parent {
            let update = match (&cmd.update, cmd.update_json) {
                (None, None) => None,
                (u, j) => {
                    let mut v = match u {
                        Some(u) => to_value(u)?,
                        None => Value::Object(Default::default()),
                    };
                    if let Some(j) = j {
                        overlay(&mut v, j);
                    }
                    Some(v)
                }
            };
            let goto = cmd
                .goto
                .into_iter()
                .map(|g| {
                    Ok(match g {
                        Goto::Node(n) => Goto::Node(n),
                        Goto::Send(s) => Goto::Send(SendTo::new(s.node, to_value(&s.input)?)),
                    })
                })
                .collect::<Result<Vec<_>, GraphError>>()?;
            if let Some(r) = trace {
                r.finish(json!({ "parent_command": { "update": update, "goto": goto.len() } }));
            }
            return Ok(TaskOutcome::Parent(ParentCommand { update, goto }));
        }

        let output = if cmd.has_write() {
            let base = match cmd.update {
                Some(u) => u,
                None => keep.clone().expect("command nodes keep their input"),
            };
            Some(apply_update(base, cmd.update_json)?)
        } else {
            None
        };
        if let Some(o) = output.as_ref().or(keep.as_ref()) {
            emit_end(o);
        }
        if let Some(r) = trace {
            r.finish(serde_json::to_value(&output).unwrap_or_default());
        }
        let result = TaskResult {
            output,
            goto: cmd.goto,
        };
        if let Some((key, ttl)) = cache_key {
            cache.put(key, ttl, result.clone());
        }
        Ok(TaskOutcome::Done(result))
    }
}
