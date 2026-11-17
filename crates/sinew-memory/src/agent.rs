//! [`MemoryAgent`]: a prebuilt ReAct agent that recalls before a turn and
//! remembers after it.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use sinew_core::Message;
use sinew_graph::{
    prebuilt::{AgentState, ReactAgent},
    CompiledGraph, GraphError, NodeContext, RunConfig, RunInput, RunOutcome,
};
use tokio::task::JoinHandle;

use crate::{
    manager::{last_turn, last_user_text},
    memory::{MemoryError, RememberReport, Result},
    MemoryManager,
};

type Pending = Mutex<Vec<JoinHandle<Result<RememberReport>>>>;

/// Which messages of a finished run are passed to
/// [`MemoryManager::remember`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RememberScope {
    /// From the last user message to the end (default): each turn is
    /// extracted once even when the state carries the whole thread.
    #[default]
    LastTurn,
    /// Every message of the final state.
    AllMessages,
}

/// Options shared by [`MemoryAgent`] and the harness `MemoryHook`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryOptions {
    /// Memories recalled per turn (default 5).
    pub recall_limit: usize,
    /// Save memories in a spawned task so the answer is returned without
    /// waiting for the extraction calls (default `true`). Wait for them with
    /// `flush`.
    pub background: bool,
    pub scope: RememberScope,
}

impl Default for MemoryOptions {
    fn default() -> Self {
        Self {
            recall_limit: 5,
            background: true,
            scope: RememberScope::LastTurn,
        }
    }
}

impl MemoryOptions {
    pub(crate) fn select<'a>(&self, messages: &'a [Message]) -> &'a [Message] {
        match self.scope {
            RememberScope::LastTurn => last_turn(messages),
            RememberScope::AllMessages => messages,
        }
    }
}

struct Shared {
    memory: MemoryManager,
    recall_limit: AtomicUsize,
    /// Run thread id -> user id, for the `system_context` callback.
    users: Mutex<HashMap<String, String>>,
    /// Run thread id -> (query, recalled block): one recall per turn.
    recalled: Mutex<HashMap<String, (String, Option<String>)>>,
}

impl Shared {
    async fn context(&self, state: &AgentState, thread_id: &str) -> Result<Option<String>> {
        let Some(user) = self.users.lock().unwrap().get(thread_id).cloned() else {
            return Ok(None);
        };
        let query = last_user_text(&state.messages).unwrap_or_default();
        if let Some((q, block)) = self.recalled.lock().unwrap().get(thread_id) {
            if *q == query {
                return Ok(block.clone());
            }
        }
        let block = self
            .memory
            .context(&user, &query, self.recall_limit.load(Ordering::Relaxed))
            .await?;
        self.recalled
            .lock()
            .unwrap()
            .insert(thread_id.to_string(), (query, block.clone()));
        Ok(block)
    }
}

/// A [`ReactAgent`] with long-term memory per user.
///
/// Before every model call of a run, the memories most relevant to the last
/// user message are recalled (once per turn) and appended to the system
/// prompt through [`ReactAgent::system_context`]; they are not stored in the
/// state. When a run completes, the last turn is passed to
/// [`MemoryManager::remember`], by default in a background task so the
/// answer is not delayed (see [`MemoryOptions`]). Interrupted runs save
/// nothing until they complete.
///
/// The agent should be the top-level graph: the user of a run is looked up
/// by the run's thread id.
///
/// ```
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// use std::sync::Arc;
/// use serde_json::json;
/// use sinew_core::{mock::ScriptedModel, Message, ToolBox, ToolCall};
/// use sinew_graph::{prebuilt::{AgentState, ReactAgent}, InMemoryStore};
/// use sinew_memory::{MemoryAgent, MemoryManager, MemoryOptions};
/// use sinew_rag::FakeEmbeddings;
///
/// let save = |args| Message::assistant_with_tools("", vec![ToolCall {
///     id: "m".into(), name: "save_memories".into(), arguments: args,
/// }]);
/// let model = Arc::new(ScriptedModel::new([
///     Message::assistant("Nice to meet you, Ana."),                       // turn 1
///     save(json!({"memories": [{"content": "The user is called Ana.", "kind": "fact"}]})),
///     Message::assistant("Your name is Ana."),                            // turn 2
///     save(json!({"memories": []})),
/// ]));
/// let memory = MemoryManager::new(Arc::new(InMemoryStore::new()), model.clone())
///     .with_embeddings(FakeEmbeddings::new(64));
/// let agent = MemoryAgent::new(ReactAgent::new(model.clone(), ToolBox::new()), memory.clone())?
///     .with_options(MemoryOptions { background: false, ..Default::default() });
///
/// agent.invoke("ana", AgentState::new(vec![Message::user("Hi, I'm Ana")])).await?;
/// agent.invoke("ana", AgentState::new(vec![Message::user("What is my name?")])).await?;
/// // The second turn's model call saw the memory in its system prompt.
/// assert!(model.requests()[2].messages[0].content.contains("The user is called Ana."));
/// # Ok::<_, sinew_memory::MemoryError>(()) }).unwrap();
/// ```
pub struct MemoryAgent {
    graph: CompiledGraph<AgentState>,
    shared: Arc<Shared>,
    options: MemoryOptions,
    pending: Pending,
}

impl std::fmt::Debug for MemoryAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryAgent")
            .field("memory", &self.shared.memory)
            .field("options", &self.options)
            .finish()
    }
}

impl MemoryAgent {
    /// Compiles `agent` with memory recall wired in. `agent` must not set
    /// its own [`ReactAgent::system_context`] (this replaces it).
    pub fn new(agent: ReactAgent, memory: MemoryManager) -> std::result::Result<Self, GraphError> {
        let options = MemoryOptions::default();
        let shared = Arc::new(Shared {
            memory,
            recall_limit: AtomicUsize::new(options.recall_limit),
            users: Mutex::new(HashMap::new()),
            recalled: Mutex::new(HashMap::new()),
        });
        let s = shared.clone();
        let graph = agent
            .system_context(move |state: AgentState, ctx: NodeContext| {
                let s = s.clone();
                async move { Ok(s.context(&state, &ctx.thread_id).await?) }
            })
            .compile()?;
        Ok(Self {
            graph,
            shared,
            options,
            pending: Mutex::new(Vec::new()),
        })
    }

    /// Sets the recall limit, background saving and remember scope.
    ///
    /// With background saving, a turn that starts before the previous
    /// turn's save has finished does not see the memories that save adds.
    pub fn with_options(mut self, options: MemoryOptions) -> Self {
        self.shared
            .recall_limit
            .store(options.recall_limit, Ordering::Relaxed);
        self.options = options;
        self
    }

    /// Transforms the compiled graph, e.g. to attach a checkpointer, a
    /// store or a tracer: `agent.map_graph(|g| g.with_checkpointer(cp))`.
    pub fn map_graph(
        mut self,
        f: impl FnOnce(CompiledGraph<AgentState>) -> CompiledGraph<AgentState>,
    ) -> Self {
        self.graph = f(self.graph);
        self
    }

    pub fn graph(&self) -> &CompiledGraph<AgentState> {
        &self.graph
    }

    pub fn memory(&self) -> &MemoryManager {
        &self.shared.memory
    }

    /// Runs one turn for `user_id` on a fresh thread and returns the final
    /// state, like [`CompiledGraph::invoke`].
    pub async fn invoke(&self, user_id: &str, input: AgentState) -> Result<AgentState> {
        let thread = uuid::Uuid::new_v4().to_string();
        match self
            .run_with(user_id, RunConfig::new(thread), input)
            .await?
        {
            RunOutcome::Completed(s) => Ok(s),
            RunOutcome::Interrupted {
                next, interrupts, ..
            } => Err(
                GraphError::Interrupted(interrupts.first().map_or(next, |i| i.node.clone())).into(),
            ),
        }
    }

    /// Like [`CompiledGraph::run`], for `user_id`.
    pub async fn run(
        &self,
        user_id: &str,
        thread_id: &str,
        input: Option<AgentState>,
    ) -> Result<RunOutcome<AgentState>> {
        self.run_with(user_id, RunConfig::new(thread_id), input)
            .await
    }

    /// Like [`CompiledGraph::run_with`], for `user_id`.
    pub async fn run_with(
        &self,
        user_id: &str,
        config: impl Into<RunConfig>,
        input: impl Into<RunInput<AgentState>>,
    ) -> Result<RunOutcome<AgentState>> {
        let config = config.into();
        let thread = config.thread_id.clone();
        self.shared
            .users
            .lock()
            .unwrap()
            .insert(thread.clone(), user_id.to_string());
        let outcome = self.graph.run_with(config, input).await;
        self.shared.users.lock().unwrap().remove(&thread);
        self.shared.recalled.lock().unwrap().remove(&thread);
        let outcome = outcome?;
        if let RunOutcome::Completed(state) = &outcome {
            self.save(user_id, &thread, &state.messages).await?;
        }
        Ok(outcome)
    }

    async fn save(&self, user_id: &str, thread: &str, messages: &[Message]) -> Result<()> {
        let turn = self.options.select(messages).to_vec();
        let memory = self.shared.memory.clone();
        let (user, thread) = (user_id.to_string(), thread.to_string());
        if self.options.background {
            let handle =
                tokio::spawn(async move { memory.remember(&user, &turn, Some(&thread)).await });
            self.pending.lock().unwrap().push(handle);
        } else {
            memory.remember(&user, &turn, Some(&thread)).await?;
        }
        Ok(())
    }

    /// Waits for every background save started so far and returns their
    /// results, oldest first. Handles of finished saves are kept until this
    /// is called, so long-running services should call it now and then (or
    /// at shutdown) to collect errors.
    pub async fn flush(&self) -> Vec<Result<RememberReport>> {
        flush(&self.pending).await
    }
}

pub(crate) async fn flush(pending: &Pending) -> Vec<Result<RememberReport>> {
    let handles: Vec<_> = std::mem::take(&mut *pending.lock().unwrap());
    let mut out = Vec::with_capacity(handles.len());
    for h in handles {
        out.push(match h.await {
            Ok(r) => r,
            Err(e) => Err(MemoryError::Task(e.to_string())),
        });
    }
    out
}
