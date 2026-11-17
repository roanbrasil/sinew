//! [`MemoryHook`]: long-term memory for the `sinew-harness` agent.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use sinew_core::{ChatRequest, ChatResponse, Message, Role};
use sinew_harness::{HarnessError, Hook, HookContext};
use tokio::task::JoinHandle;

use crate::{
    agent::{flush, MemoryOptions},
    manager::last_user_text,
    memory::{RememberReport, Result},
    MemoryManager,
};

type UserResolver = Arc<dyn Fn(&HookContext) -> String + Send + Sync>;
/// Recalled block and the conversation of the last request.
type Turn = (Option<String>, Vec<Message>);

struct Inner {
    memory: MemoryManager,
    user: UserResolver,
    /// Thread id -> (recalled block, messages of the last request).
    turns: Mutex<HashMap<String, Turn>>,
    pending: Mutex<Vec<JoinHandle<Result<RememberReport>>>>,
}

/// A harness [`Hook`] that gives the agent long-term memory.
///
/// - `before_model`, on the first model call of a turn, recalls the memories
///   most relevant to the last user message and appends them to the system
///   message; later calls of the same turn reuse that block.
/// - `after_model`, when the reply has no tool calls (the turn is ending),
///   passes the turn to [`MemoryManager::remember`], in a background task by
///   default ([`MemoryOptions::background`]).
///
/// Subagents (`depth > 0`) are left alone. The hook is cheap to clone; keep
/// a clone to call [`flush`](Self::flush).
///
/// ```no_run
/// # async fn demo(memory: sinew_memory::MemoryManager) -> sinew_harness::Result<()> {
/// use sinew_harness::Agent;
/// use sinew_memory::MemoryHook;
///
/// let hook = MemoryHook::new(memory, "ana");
/// let agent = Agent::builder().model_spec("ollama:qwen3").hook(hook.clone()).build()?;
/// agent.ask("I'm vegetarian, suggest a dinner").await?;
/// hook.flush().await;
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct MemoryHook {
    inner: Arc<Inner>,
    options: MemoryOptions,
}

impl std::fmt::Debug for MemoryHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryHook")
            .field("memory", &self.inner.memory)
            .field("options", &self.options)
            .finish()
    }
}

impl MemoryHook {
    /// Memory for one user (an agent per user, or a single-user CLI).
    pub fn new(memory: MemoryManager, user_id: impl Into<String>) -> Self {
        let user = user_id.into();
        Self::with_user_resolver(memory, move |_| user.clone())
    }

    /// Picks the user per call, e.g. from the thread id
    /// (`"<user>/<conversation>"`).
    pub fn with_user_resolver(
        memory: MemoryManager,
        resolve: impl Fn(&HookContext) -> String + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                memory,
                user: Arc::new(resolve),
                turns: Mutex::new(HashMap::new()),
                pending: Mutex::new(Vec::new()),
            }),
            options: MemoryOptions::default(),
        }
    }

    /// Sets the recall limit, background saving and remember scope. Clones
    /// made earlier keep their options.
    pub fn with_options(mut self, options: MemoryOptions) -> Self {
        self.options = options;
        self
    }

    pub fn memory(&self) -> &MemoryManager {
        &self.inner.memory
    }

    /// Waits for every background save started so far and returns their
    /// results, oldest first.
    pub async fn flush(&self) -> Vec<Result<RememberReport>> {
        flush(&self.inner.pending).await
    }
}

fn hook_err(e: crate::MemoryError) -> HarnessError {
    HarnessError::Hook(format!("memory: {e}"))
}

#[async_trait]
impl Hook for MemoryHook {
    async fn before_model(
        &self,
        ctx: &HookContext,
        request: &mut ChatRequest,
    ) -> sinew_harness::Result<()> {
        if ctx.depth > 0 {
            return Ok(());
        }
        let inner = &self.inner;
        let cached = if ctx.step > 0 {
            inner
                .turns
                .lock()
                .unwrap()
                .get(&ctx.thread_id)
                .map(|(block, _)| block.clone())
        } else {
            None
        };
        let block = match cached {
            Some(block) => block,
            None => {
                let user = (inner.user)(ctx);
                let query = last_user_text(&request.messages).unwrap_or_default();
                inner
                    .memory
                    .context(&user, &query, self.options.recall_limit)
                    .await
                    .map_err(hook_err)?
            }
        };
        let conversation: Vec<Message> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .cloned()
            .collect();
        if let Some(text) = &block {
            match request.messages.first_mut() {
                Some(m) if m.role == Role::System => {
                    if !m.content.is_empty() {
                        m.content.push_str("\n\n");
                    }
                    m.content.push_str(text);
                }
                _ => request.messages.insert(0, Message::system(text.clone())),
            }
        }
        inner
            .turns
            .lock()
            .unwrap()
            .insert(ctx.thread_id.clone(), (block, conversation));
        Ok(())
    }

    async fn after_model(
        &self,
        ctx: &HookContext,
        response: &mut ChatResponse,
    ) -> sinew_harness::Result<()> {
        if ctx.depth > 0 || !response.message.tool_calls.is_empty() {
            return Ok(());
        }
        let inner = &self.inner;
        let Some((_, mut messages)) = inner.turns.lock().unwrap().remove(&ctx.thread_id) else {
            return Ok(());
        };
        messages.push(response.message.clone());
        let turn = self.options.select(&messages).to_vec();
        let user = (inner.user)(ctx);
        let memory = inner.memory.clone();
        let thread = ctx.thread_id.clone();
        if self.options.background {
            let handle =
                tokio::spawn(async move { memory.remember(&user, &turn, Some(&thread)).await });
            inner.pending.lock().unwrap().push(handle);
        } else {
            memory
                .remember(&user, &turn, Some(&thread))
                .await
                .map_err(hook_err)?;
        }
        Ok(())
    }
}
