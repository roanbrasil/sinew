//! The agent side of the server: [`AgentExecutor`] runs one request and
//! reports progress through an [`EventQueue`].

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};

use crate::types::{
    Artifact, Message, MessageSendConfiguration, Metadata, Part, Role, Task, TaskState, TaskStatus,
};

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// What the server hands an executor for one `message/send` or
/// `message/stream` call.
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub task_id: String,
    pub context_id: String,
    /// The incoming user message (its `taskId`/`contextId` are filled in).
    pub message: Message,
    /// The task as it was before this message, when the message continues
    /// an existing task (e.g. one paused in `input-required`). Its history
    /// does not include `message` yet.
    pub task: Option<Task>,
    pub configuration: Option<MessageSendConfiguration>,
    pub metadata: Option<Metadata>,
    /// The caller, when the server has an identity verifier
    /// ([`A2aServer::with_identity_verifier`](crate::A2aServer::with_identity_verifier))
    /// and the caller's proof verified.
    pub caller: Option<sinew_core::identity::VerifiedCaller>,
}

impl RequestContext {
    /// Text of the incoming message.
    pub fn text(&self) -> String {
        self.message.text()
    }

    /// Whether this message answers a task that was waiting for input.
    pub fn is_resume(&self) -> bool {
        self.task
            .as_ref()
            .is_some_and(|t| t.status.state.is_interrupted())
    }
}

/// Something an executor reports.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// New task status. `completed`, `failed`, `canceled`, `rejected`,
    /// `input-required` and `auth-required` end the turn.
    Status(TaskStatus),
    /// A new artifact, or a chunk of one (`append`).
    Artifact {
        artifact: Artifact,
        append: bool,
        last_chunk: bool,
    },
    /// A direct reply. As the first event of a new task it is returned
    /// instead of a task; it always ends the turn.
    Message(Message),
}

// Events are the hot path (one per token); boxing them would add an
// allocation each, so the rare `Cancel` stays small instead.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Control {
    Event(AgentEvent),
    Cancel(oneshot::Sender<Task>),
}

/// Sends an executor's events to the server, which applies them to the task
/// and streams them to clients immediately. Sending never blocks and never
/// fails: if the task was canceled the events are dropped.
#[derive(Clone)]
pub struct EventQueue {
    tx: mpsc::UnboundedSender<Control>,
    task_id: String,
    context_id: String,
}

impl EventQueue {
    pub(crate) fn new(
        tx: mpsc::UnboundedSender<Control>,
        task_id: String,
        context_id: String,
    ) -> Self {
        Self {
            tx,
            task_id,
            context_id,
        }
    }

    /// A queue whose events go to `tx`, for driving an executor without the
    /// server (tests, custom transports).
    pub fn channel(
        task_id: impl Into<String>,
        context_id: impl Into<String>,
    ) -> (Self, EventReceiver) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self::new(tx, task_id.into(), context_id.into()),
            EventReceiver { rx },
        )
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn context_id(&self) -> &str {
        &self.context_id
    }

    /// Whether the task is gone (canceled or finished); further events are
    /// dropped. Long-running executors can poll it to stop early.
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }

    pub fn send(&self, event: AgentEvent) {
        let _ = self.tx.send(Control::Event(event));
    }

    /// An agent message bound to this task and context.
    pub fn agent_message(&self, parts: Vec<Part>) -> Message {
        let mut m = Message::new(Role::Agent, parts);
        m.task_id = Some(self.task_id.clone());
        m.context_id = Some(self.context_id.clone());
        m
    }

    pub fn status(&self, state: TaskState, message: Option<Message>) {
        self.send(AgentEvent::Status(TaskStatus::new(state, message)));
    }

    /// `working`, no message.
    pub fn working(&self) {
        self.status(TaskState::Working, None);
    }

    /// `working` with a progress note.
    pub fn progress(&self, text: impl Into<String>) {
        let m = self.agent_message(vec![Part::text(text)]);
        self.status(TaskState::Working, Some(m));
    }

    pub fn complete(&self, message: Option<Message>) {
        self.status(TaskState::Completed, message);
    }

    pub fn fail(&self, text: impl Into<String>) {
        let m = self.agent_message(vec![Part::text(text)]);
        self.status(TaskState::Failed, Some(m));
    }

    /// Pause and ask the client for more input; the client answers with a
    /// message carrying this task's id.
    pub fn input_required(&self, message: Message) {
        self.status(TaskState::InputRequired, Some(message));
    }

    /// Reply with a message instead of task updates (ends the turn).
    pub fn reply(&self, parts: Vec<Part>) {
        let m = self.agent_message(parts);
        self.send(AgentEvent::Message(m));
    }

    pub fn artifact(&self, artifact: Artifact, append: bool, last_chunk: bool) {
        self.send(AgentEvent::Artifact {
            artifact,
            append,
            last_chunk,
        });
    }

    /// A writer that streams text into one artifact chunk by chunk.
    pub fn text_artifact(&self, name: impl Into<String>) -> TextArtifactWriter {
        TextArtifactWriter {
            queue: self.clone(),
            artifact_id: crate::types::new_id(),
            name: Some(name.into()),
            started: false,
            closed: false,
        }
    }
}

/// Receiving end of [`EventQueue::channel`].
pub struct EventReceiver {
    rx: mpsc::UnboundedReceiver<Control>,
}

impl EventReceiver {
    pub async fn recv(&mut self) -> Option<AgentEvent> {
        loop {
            match self.rx.recv().await? {
                Control::Event(e) => return Some(e),
                Control::Cancel(_) => continue,
            }
        }
    }
}

/// Streams text into one artifact: the first chunk creates it (carrying the
/// name), later chunks append, [`finish`](Self::finish) marks the last chunk.
/// Each `push` is sent at once, so tokens reach the client as they are made.
pub struct TextArtifactWriter {
    queue: EventQueue,
    artifact_id: String,
    name: Option<String>,
    started: bool,
    closed: bool,
}

impl TextArtifactWriter {
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    /// Whether any chunk was sent.
    pub fn has_content(&self) -> bool {
        self.started
    }

    pub fn push(&mut self, text: impl Into<String>) {
        let text = text.into();
        if text.is_empty() || self.closed {
            return;
        }
        self.send(vec![Part::text(text)], false);
    }

    /// Sends the closing `lastChunk: true` update (with no new parts) if
    /// anything was written. Idempotent.
    pub fn finish(&mut self) {
        if self.started && !self.closed {
            self.send(Vec::new(), true);
        }
        self.closed = true;
    }

    fn send(&mut self, parts: Vec<Part>, last_chunk: bool) {
        let mut artifact = Artifact::new(self.artifact_id.clone(), parts);
        if !self.started {
            artifact.name = self.name.take();
        }
        let append = self.started;
        self.started = true;
        self.queue.artifact(artifact, append, last_chunk);
    }
}

/// Runs the agent for one request. Report progress, artifacts and the final
/// state through `events`; every event is streamed to the client right away.
///
/// Returning `Ok` without a final status completes the task; returning `Err`
/// fails it with the error text. `tasks/cancel` aborts the `execute` future
/// (it is dropped at its next `.await`) and marks the task `canceled`.
#[async_trait]
pub trait AgentExecutor: Send + Sync + 'static {
    async fn execute(&self, ctx: RequestContext, events: EventQueue) -> Result<(), BoxError>;
}

#[async_trait]
impl<T: AgentExecutor + ?Sized> AgentExecutor for std::sync::Arc<T> {
    async fn execute(&self, ctx: RequestContext, events: EventQueue) -> Result<(), BoxError> {
        (**self).execute(ctx, events).await
    }
}
