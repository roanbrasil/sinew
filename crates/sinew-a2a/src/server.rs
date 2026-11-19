//! A2A server over HTTP (axum): the Agent Card at
//! `/.well-known/agent-card.json` and the JSON-RPC methods at the RPC path
//! (`/` by default).
//!
//! # How a request runs
//!
//! `message/send` and `message/stream` create (or continue) a task and spawn
//! the [`AgentExecutor`] on its own tokio task, so it keeps running if the
//! client disconnects and `tasks/resubscribe` can pick it up again. A per-task
//! driver applies the executor's events to the task and fans each one out to
//! every subscribed stream as soon as it arrives: the event is serialized
//! once and written to the SSE body without buffering. Snapshots go to the
//! [`TaskStore`] when the task is announced, on every status change and when
//! the turn ends, never per token.

use std::{
    collections::HashMap,
    future::Future,
    net::SocketAddr,
    sync::{Arc, Mutex, OnceLock, RwLock},
};

use axum::{
    body::Bytes,
    extract::State,
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use futures_util::stream;
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;
use tokio::{
    net::{TcpListener, ToSocketAddrs},
    sync::{mpsc, oneshot},
    task::{AbortHandle, JoinHandle},
};

use sinew_core::identity::{
    with_identity_scope, AgentCredentials, AgentKey, IdentityVerifier, UserCredential,
    VerifiedCaller, Via, META_CREDENTIAL, META_IDENTITY, META_MESSAGE,
};

use crate::{
    executor::{AgentEvent, AgentExecutor, Control, EventQueue, RequestContext},
    jsonrpc::{self, method, RpcError},
    store::{InMemoryTaskStore, TaskStore},
    types::{
        new_id, task_text, AgentCard, Message, MessageSendParams, SendMessageResult,
        StreamResponse, Task, TaskArtifactUpdateEvent, TaskIdParams, TaskQueryParams, TaskState,
        TaskStatus, TaskStatusUpdateEvent, AGENT_CARD_PATH, LEGACY_AGENT_CARD_PATH,
    },
};

/// One streamed event, serialized once and shared by every subscriber.
struct Frame {
    event: StreamResponse,
    json: String,
    terminal: bool,
}

impl Frame {
    fn new(event: StreamResponse, terminal: bool) -> Arc<Self> {
        let json = serde_json::to_string(&event).unwrap_or_else(|_| "null".into());
        Arc::new(Self {
            event,
            json,
            terminal,
        })
    }
}

type FrameTx = mpsc::UnboundedSender<Arc<Frame>>;
type FrameRx = mpsc::UnboundedReceiver<Arc<Frame>>;

struct LiveState {
    task: Task,
    subscribers: Vec<FrameTx>,
    /// The task snapshot was streamed (lazily, before the first status or
    /// artifact event, so a direct message reply creates no task).
    announced: bool,
    /// The task existed before this turn.
    existing: bool,
    /// The turn ended; later events are ignored.
    done: bool,
}

impl LiveState {
    fn broadcast(&mut self, frame: &Arc<Frame>) {
        self.subscribers.retain(|s| s.send(frame.clone()).is_ok());
    }
}

/// A task whose executor is running.
struct LiveTask {
    state: Mutex<LiveState>,
    ctl: mpsc::UnboundedSender<Control>,
    abort: OnceLock<AbortHandle>,
}

struct Inner {
    card: RwLock<AgentCard>,
    executor: Arc<dyn AgentExecutor>,
    store: Arc<dyn TaskStore>,
    live: Mutex<HashMap<String, Arc<LiveTask>>>,
    rpc_path: String,
    verifier: Option<IdentityVerifier>,
    card_key: Option<AgentKey>,
    signer: Option<AgentCredentials>,
}

/// Who called, as established from the request's metadata.
#[derive(Default)]
struct CallerScope {
    caller: Option<VerifiedCaller>,
    credential: Option<UserCredential>,
    /// The agent the caller's proof names, verified or not: the addressee
    /// of a signed reply.
    claimed: Option<String>,
}

impl CallerScope {
    /// Who a signed reply is addressed to: the verified caller, else the
    /// agent its proof claims, else anyone (`*`).
    fn reply_recipient(&self) -> String {
        self.caller
            .as_ref()
            .map(|c| c.agent().to_string())
            .or_else(|| self.claimed.clone())
            .unwrap_or_else(|| "*".into())
    }
}

/// Serves an [`AgentExecutor`] over A2A JSON-RPC.
///
/// ```no_run
/// # async fn demo(executor: impl sinew_a2a::AgentExecutor) -> std::io::Result<()> {
/// use sinew_a2a::{A2aServer, AgentCard};
/// let card = AgentCard::new("echo", "Repeats what you say", "http://127.0.0.1:9999/");
/// let listener = tokio::net::TcpListener::bind("127.0.0.1:9999").await?;
/// A2aServer::new(card, executor).serve(listener).await
/// # }
/// ```
#[derive(Clone)]
pub struct A2aServer {
    inner: Arc<Inner>,
}

impl A2aServer {
    pub fn new(card: AgentCard, executor: impl AgentExecutor) -> Self {
        Self::from_arc(card, Arc::new(executor))
    }

    pub fn from_arc(card: AgentCard, executor: Arc<dyn AgentExecutor>) -> Self {
        Self {
            inner: Arc::new(Inner {
                card: RwLock::new(card),
                executor,
                store: Arc::new(InMemoryTaskStore::new()),
                live: Mutex::new(HashMap::new()),
                rpc_path: "/".into(),
                verifier: None,
                card_key: None,
                signer: None,
            }),
        }
    }

    fn inner_mut(&mut self) -> &mut Inner {
        Arc::get_mut(&mut self.inner).expect("configure the server before cloning it")
    }

    /// Replaces the default [`InMemoryTaskStore`].
    pub fn with_store(mut self, store: Arc<dyn TaskStore>) -> Self {
        self.inner_mut().store = store;
        self
    }

    /// Path of the JSON-RPC endpoint (default `/`).
    pub fn with_rpc_path(mut self, path: impl Into<String>) -> Self {
        let mut path = path.into();
        if !path.starts_with('/') {
            path.insert(0, '/');
        }
        self.inner_mut().rpc_path = path;
        self
    }

    /// Verifies the caller's identity proof (see
    /// [`A2aClient::with_credentials`](crate::A2aClient::with_credentials))
    /// on every request; rejected requests get the JSON-RPC error
    /// [`IDENTITY_REJECTED`](crate::jsonrpc::error_code::IDENTITY_REJECTED).
    /// The audience is the verifier's, else the card's
    /// [`agent_id`](AgentCard::agent_id). The executor runs inside the
    /// caller's [`CallContext`](sinew_core::identity::CallContext) with this
    /// agent appended to the delegation chain
    /// ([`current_context`](sinew_core::identity::current_context)), sees
    /// the caller in [`RequestContext::caller`], and gets the caller's
    /// on-behalf-of credential as
    /// [`user_credential`](sinew_core::identity::user_credential) and as the
    /// context's subject token. Without a verifier, identity metadata is
    /// ignored.
    pub fn with_identity_verifier(mut self, verifier: IdentityVerifier) -> Self {
        self.inner_mut().verifier = Some(verifier);
        self
    }

    /// Signs the card with the agent's key (see [`AgentCard::sign`]), now
    /// and again after `serve` / `spawn` fill in an empty card `url`, so the
    /// served card always verifies. Add the identity extension first
    /// ([`AgentCard::with_identity`]).
    pub fn with_card_signing_key(mut self, key: AgentKey) -> Self {
        let inner = self.inner_mut();
        let card = inner.card.get_mut().unwrap();
        if let Ok(signed) = card.clone().sign(&key) {
            *card = signed;
        }
        inner.card_key = Some(key);
        self
    }

    /// Signs every `message/send` reply as a message from this agent
    /// (OWASP ASI07): a [`SignedMessage`](sinew_core::identity::SignedMessage)
    /// over the reply text (the message text, or the task's
    /// [`task_text`]) addressed to the caller (the verified caller, else the
    /// agent its identity proof names) with the A2A `contextId` as thread,
    /// in the reply's `metadata` under
    /// [`META_MESSAGE`]. Clients check it
    /// with [`A2aClient::with_message_verifier`](crate::A2aClient::with_message_verifier).
    pub fn with_message_signing(mut self, credentials: AgentCredentials) -> Self {
        self.inner_mut().signer = Some(credentials);
        self
    }

    pub fn card(&self) -> AgentCard {
        self.inner.card.read().unwrap().clone()
    }

    pub fn store(&self) -> Arc<dyn TaskStore> {
        self.inner.store.clone()
    }

    /// The axum router; nest or merge it into a larger app.
    pub fn router(&self) -> Router {
        Router::new()
            .route(AGENT_CARD_PATH, get(card_handler))
            .route(LEGACY_AGENT_CARD_PATH, get(card_handler))
            .route(&self.inner.rpc_path, post(rpc_handler))
            .with_state(self.inner.clone())
    }

    /// Serves on `listener` until the process ends. Sets `TCP_NODELAY` on
    /// every connection so small SSE events leave immediately. An empty card
    /// `url` is set to `http://<listener address><rpc path>`.
    pub async fn serve(self, listener: TcpListener) -> std::io::Result<()> {
        self.serve_with_shutdown(listener, std::future::pending())
            .await
    }

    pub async fn serve_with_shutdown(
        self,
        listener: TcpListener,
        signal: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        use axum::serve::ListenerExt;
        self.fill_card_url(listener.local_addr()?);
        let listener = listener.tap_io(|tcp| {
            let _ = tcp.set_nodelay(true);
        });
        axum::serve(listener, self.router())
            .with_graceful_shutdown(signal)
            .await
    }

    /// Sets the card's `url` to this server's RPC endpoint if it is empty;
    /// returns the endpoint URL.
    fn fill_card_url(&self, addr: SocketAddr) -> String {
        let url = format!("http://{addr}{}", self.inner.rpc_path);
        let mut card = self.inner.card.write().unwrap();
        if card.url.is_empty() {
            card.url = url.clone();
            if let Some(key) = &self.inner.card_key {
                if let Ok(signed) = card.clone().sign(key) {
                    *card = signed;
                }
            }
        }
        url
    }

    /// Binds `addr` (use port 0 for a free port) and serves in the
    /// background. Like [`serve`](Self::serve), it fills in an empty card
    /// `url`.
    pub async fn spawn(self, addr: impl ToSocketAddrs) -> std::io::Result<ServerHandle> {
        let listener = TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        let url = self.fill_card_url(local);
        let (tx, rx) = oneshot::channel::<()>();
        let task = tokio::spawn(self.serve_with_shutdown(listener, async {
            let _ = rx.await;
        }));
        Ok(ServerHandle {
            addr: local,
            url,
            shutdown: Some(tx),
            task,
        })
    }
}

/// A server started with [`A2aServer::spawn`]. Dropping it stops the server.
pub struct ServerHandle {
    addr: SocketAddr,
    url: String,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<std::io::Result<()>>,
}

impl ServerHandle {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Base URL, e.g. `http://127.0.0.1:41234` (where the card is served).
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// JSON-RPC endpoint URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Graceful shutdown: stops accepting and waits for open requests.
    pub async fn shutdown(mut self) -> std::io::Result<()> {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        (&mut self.task)
            .await
            .unwrap_or_else(|e| Err(std::io::Error::other(e)))
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if self.shutdown.is_some() {
            self.task.abort();
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

async fn card_handler(State(inner): State<Arc<Inner>>) -> Response {
    let body = serde_json::to_string(&*inner.card.read().unwrap()).unwrap_or_default();
    json_response(body)
}

fn json_response(body: String) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}

fn rpc_error(id: &Value, e: RpcError) -> Response {
    json_response(jsonrpc::error_response(id, &e))
}

#[derive(Deserialize)]
struct RawRequest {
    #[serde(default)]
    jsonrpc: Option<String>,
    #[serde(default)]
    id: Option<Value>,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    params: Option<Value>,
}

fn params<T: DeserializeOwned>(p: Option<Value>) -> Result<T, RpcError> {
    serde_json::from_value(p.unwrap_or(Value::Null)).map_err(RpcError::invalid_params)
}

async fn rpc_handler(State(inner): State<Arc<Inner>>, body: Bytes) -> Response {
    let value: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return rpc_error(&Value::Null, RpcError::parse_error(e)),
    };
    if value.is_array() {
        return rpc_error(
            &Value::Null,
            RpcError::invalid_request("batch requests are not supported"),
        );
    }
    let req: RawRequest = match serde_json::from_value(value) {
        Ok(r) => r,
        Err(e) => {
            return rpc_error(
                &Value::Null,
                RpcError::invalid_request(format!("Invalid JSON-RPC Request: {e}")),
            )
        }
    };
    let id = req.id.unwrap_or(Value::Null);
    if req.jsonrpc.as_deref() != Some("2.0") {
        return rpc_error(&id, RpcError::invalid_request("`jsonrpc` must be \"2.0\""));
    }
    if !(id.is_string() || id.is_number()) {
        return rpc_error(
            &id,
            RpcError::invalid_request("`id` must be a string or a number"),
        );
    }
    let Some(m) = req.method else {
        return rpc_error(&id, RpcError::invalid_request("missing `method`"));
    };
    let mut req_params = req.params;
    let scope = match inner.identify(req_params.as_mut()) {
        Ok(s) => s,
        Err(e) => return rpc_error(&id, e),
    };
    let result = match m.as_str() {
        method::MESSAGE_SEND => match params(req_params) {
            Ok(p) => inner
                .message_send(p, scope)
                .await
                .map(|r| jsonrpc::response(&id, &r)),
            Err(e) => Err(e),
        },
        method::MESSAGE_STREAM => {
            return match params(req_params) {
                Ok(p) => match inner.message_stream(p, scope).await {
                    Ok(rx) => sse_response(&id, rx),
                    Err(e) => rpc_error(&id, e),
                },
                Err(e) => rpc_error(&id, e),
            };
        }
        method::TASKS_RESUBSCRIBE => {
            return match params::<TaskIdParams>(req_params) {
                Ok(p) => match inner.resubscribe(&p.id).await {
                    Ok(rx) => sse_response(&id, rx),
                    Err(e) => rpc_error(&id, e),
                },
                Err(e) => rpc_error(&id, e),
            };
        }
        method::TASKS_GET => match params::<TaskQueryParams>(req_params) {
            Ok(p) => inner
                .get_task(&p.id, p.history_length)
                .await
                .map(|t| jsonrpc::response(&id, &t)),
            Err(e) => Err(e),
        },
        method::TASKS_CANCEL => match params::<TaskIdParams>(req_params) {
            Ok(p) => inner
                .cancel(&p.id)
                .await
                .map(|t| jsonrpc::response(&id, &t)),
            Err(e) => Err(e),
        },
        method::PUSH_CONFIG_SET
        | method::PUSH_CONFIG_GET
        | method::PUSH_CONFIG_LIST
        | method::PUSH_CONFIG_DELETE => Err(RpcError::push_not_supported()),
        method::GET_EXTENDED_CARD => Err(RpcError::extended_card_not_configured()),
        other => Err(RpcError::method_not_found(other)),
    };
    match result {
        Ok(body) => json_response(body),
        Err(e) => rpc_error(&id, e),
    }
}

/// Streams frames as SSE until the terminal one. Each event is written as
/// soon as it is received.
fn sse_response(id: &Value, rx: FrameRx) -> Response {
    let id_json = id.to_string();
    let events = stream::unfold((rx, false), move |(mut rx, done)| {
        let id_json = id_json.clone();
        async move {
            if done {
                return None;
            }
            let frame = rx.recv().await?;
            let data = jsonrpc::response_raw(&id_json, &frame.json);
            let terminal = frame.terminal;
            Some((
                Ok::<_, std::convert::Infallible>(Event::default().data(data)),
                (rx, terminal),
            ))
        }
    });
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ---------------------------------------------------------------------------
// Task management
// ---------------------------------------------------------------------------

/// Moves the current status message (if any) into the history before the
/// status is replaced, so questions asked in `input-required` stay in the
/// conversation.
fn set_status(task: &mut Task, status: TaskStatus) {
    let old = std::mem::replace(&mut task.status, status);
    if let Some(m) = old.message {
        if !task.history.iter().any(|h| h.message_id == m.message_id) {
            task.history.push(m);
        }
    }
}

impl Inner {
    /// Takes the identity proof and the credential out of the params'
    /// metadata (the credential never reaches the executor's metadata or a
    /// store) and verifies the proof when a verifier is set.
    fn identify(&self, params: Option<&mut Value>) -> Result<CallerScope, RpcError> {
        let meta = params
            .and_then(|p| p.get_mut("metadata"))
            .and_then(Value::as_object_mut);
        let (proof, credential) = match meta {
            Some(m) => (m.get(META_IDENTITY).cloned(), m.remove(META_CREDENTIAL)),
            None => (None, None),
        };
        let claimed = proof
            .as_ref()
            .and_then(|p| p.pointer("/assertion/payload/context/agent"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let Some(verifier) = &self.verifier else {
            return Ok(CallerScope {
                claimed,
                ..CallerScope::default()
            });
        };
        let audience = self.card.read().unwrap().agent_id();
        let caller = verifier
            .verify_value(proof.as_ref(), Some(&audience))
            .map_err(|e| RpcError::identity_rejected(e.to_string()))?;
        let credential = match (credential, &caller) {
            (Some(c), Some(_)) => Some(
                serde_json::from_value::<UserCredential>(c)
                    .map_err(|e| RpcError::invalid_params(format!("credential: {e}")))?,
            ),
            _ => None,
        };
        Ok(CallerScope {
            caller,
            credential,
            claimed,
        })
    }

    /// Adds this agent's signature to a `message/send` reply.
    fn sign_reply(&self, result: &mut SendMessageResult, recipient: &str) -> Result<(), RpcError> {
        let Some(signer) = &self.signer else {
            return Ok(());
        };
        let (text, thread, metadata) = match result {
            SendMessageResult::Message(m) => (m.text(), m.context_id.clone(), &mut m.metadata),
            SendMessageResult::Task(t) => {
                (task_text(t), Some(t.context_id.clone()), &mut t.metadata)
            }
        };
        let signed = signer
            .sign_message(recipient, thread.as_deref(), &text)
            .map_err(RpcError::internal)?;
        metadata
            .get_or_insert_with(Default::default)
            .insert(META_MESSAGE.into(), signed.to_value());
        Ok(())
    }

    fn check_streaming(&self) -> Result<(), RpcError> {
        if self.card.read().unwrap().capabilities.streaming == Some(false) {
            return Err(RpcError::unsupported(
                "streaming is not supported by this agent",
            ));
        }
        Ok(())
    }

    /// Creates or continues a task, spawns the executor and its driver, and
    /// returns a subscription to the task's frames.
    async fn start(
        self: &Arc<Self>,
        p: MessageSendParams,
        scope: CallerScope,
    ) -> Result<FrameRx, RpcError> {
        let MessageSendParams {
            mut message,
            configuration,
            metadata,
        } = p;
        let existing = match message.task_id.clone() {
            Some(id) => {
                if self.live.lock().unwrap().contains_key(&id) {
                    return Err(RpcError::unsupported(format!(
                        "task {id} is still running; cancel it or wait for it to finish"
                    )));
                }
                let task = self
                    .store
                    .get(&id)
                    .await
                    .map_err(RpcError::internal)?
                    .ok_or_else(|| RpcError::task_not_found(&id))?;
                if task.status.state.is_terminal() {
                    return Err(RpcError::invalid_params(format!(
                        "task {id} is in terminal state `{}` and can't be continued",
                        serde_json::to_value(task.status.state)
                            .ok()
                            .and_then(|v| v.as_str().map(str::to_owned))
                            .unwrap_or_default()
                    )));
                }
                if let Some(ctx) = &message.context_id {
                    if ctx != &task.context_id {
                        return Err(RpcError::invalid_params(format!(
                            "contextId {ctx} does not match task {id}"
                        )));
                    }
                }
                Some(task)
            }
            None => None,
        };
        let (task_id, context_id) = match &existing {
            Some(t) => (t.id.clone(), t.context_id.clone()),
            None => (new_id(), message.context_id.clone().unwrap_or_else(new_id)),
        };
        message.task_id = Some(task_id.clone());
        message.context_id = Some(context_id.clone());

        let mut task = existing.clone().unwrap_or_else(|| Task {
            id: task_id.clone(),
            context_id: context_id.clone(),
            status: TaskStatus::new(TaskState::Submitted, None),
            history: Vec::new(),
            artifacts: Vec::new(),
            metadata: None,
        });
        if existing.is_some() {
            set_status(&mut task, TaskStatus::new(TaskState::Submitted, None));
        }
        task.history.push(message.clone());

        let (ctl_tx, ctl_rx) = mpsc::unbounded_channel();
        let (frame_tx, frame_rx) = mpsc::unbounded_channel();
        let live = Arc::new(LiveTask {
            state: Mutex::new(LiveState {
                task,
                subscribers: vec![frame_tx],
                announced: false,
                existing: existing.is_some(),
                done: false,
            }),
            ctl: ctl_tx.clone(),
            abort: OnceLock::new(),
        });
        {
            // Re-check under the lock: two messages may continue one task.
            let mut map = self.live.lock().unwrap();
            if map.contains_key(&task_id) {
                return Err(RpcError::unsupported(format!(
                    "task {task_id} is still running; cancel it or wait for it to finish"
                )));
            }
            map.insert(task_id.clone(), live.clone());
        }

        let ctx = RequestContext {
            task_id: task_id.clone(),
            context_id: context_id.clone(),
            message,
            task: existing,
            configuration,
            metadata,
            caller: scope.caller.clone(),
        };
        let call_ctx = scope.caller.as_ref().map(|c| {
            let mut next = c.context_for(&self.card.read().unwrap().agent_id(), Via::A2a);
            next.subject_token = scope.credential.as_ref().map(|cr| cr.token.clone());
            next
        });
        let queue = EventQueue::new(ctl_tx, task_id, context_id);
        let executor = self.executor.clone();
        let exec = tokio::spawn(with_identity_scope(
            call_ctx,
            scope.credential,
            async move {
                executor
                    .execute(ctx, queue)
                    .await
                    .map_err(|e| e.to_string())
            },
        ));
        let _ = live.abort.set(exec.abort_handle());
        tokio::spawn(drive(self.clone(), live, ctl_rx, exec));
        Ok(frame_rx)
    }

    async fn message_stream(
        self: &Arc<Self>,
        p: MessageSendParams,
        scope: CallerScope,
    ) -> Result<FrameRx, RpcError> {
        self.check_streaming()?;
        self.start(p, scope).await
    }

    async fn message_send(
        self: &Arc<Self>,
        p: MessageSendParams,
        scope: CallerScope,
    ) -> Result<SendMessageResult, RpcError> {
        let blocking = p
            .configuration
            .as_ref()
            .and_then(|c| c.blocking)
            .unwrap_or(true);
        let history_length = p.configuration.as_ref().and_then(|c| c.history_length);
        let recipient = scope.reply_recipient();
        let mut result = self
            .message_result(p, scope, blocking, history_length)
            .await?;
        self.sign_reply(&mut result, &recipient)?;
        Ok(result)
    }

    async fn message_result(
        self: &Arc<Self>,
        p: MessageSendParams,
        scope: CallerScope,
        blocking: bool,
        history_length: Option<usize>,
    ) -> Result<SendMessageResult, RpcError> {
        let mut rx = self.start(p, scope).await?;
        let mut task_id = None;
        while let Some(frame) = rx.recv().await {
            match &frame.event {
                StreamResponse::Message(m) if frame.terminal => {
                    return Ok(SendMessageResult::Message(m.clone()))
                }
                StreamResponse::Task(t) => task_id = Some(t.id.clone()),
                StreamResponse::StatusUpdate(s) => task_id = Some(s.task_id.clone()),
                StreamResponse::ArtifactUpdate(a) => task_id = Some(a.task_id.clone()),
                StreamResponse::Message(_) => {}
            }
            if frame.terminal || !blocking {
                break;
            }
        }
        let id = task_id.ok_or_else(|| RpcError::internal("agent produced no events"))?;
        self.get_task(&id, history_length)
            .await
            .map(SendMessageResult::Task)
    }

    async fn get_task(&self, id: &str, history_length: Option<usize>) -> Result<Task, RpcError> {
        let live = self.live.lock().unwrap().get(id).cloned();
        let mut task = match live {
            Some(l) => l.state.lock().unwrap().task.clone(),
            None => self
                .store
                .get(id)
                .await
                .map_err(RpcError::internal)?
                .ok_or_else(|| RpcError::task_not_found(id))?,
        };
        if let Some(n) = history_length {
            task.truncate_history(n);
        }
        Ok(task)
    }

    async fn cancel(&self, id: &str) -> Result<Task, RpcError> {
        let live = self.live.lock().unwrap().get(id).cloned();
        if let Some(l) = live {
            let (tx, rx) = oneshot::channel();
            if l.ctl.send(Control::Cancel(tx)).is_ok() {
                // Stop the agent right away; the driver records the state.
                if let Some(a) = l.abort.get() {
                    a.abort();
                }
                if let Ok(task) = rx.await {
                    if task.status.state == TaskState::Canceled {
                        return Ok(task);
                    }
                    return Err(RpcError::task_not_cancelable(id));
                }
            }
        }
        let mut task = self
            .store
            .get(id)
            .await
            .map_err(RpcError::internal)?
            .ok_or_else(|| RpcError::task_not_found(id))?;
        if task.status.state.is_terminal() {
            return Err(RpcError::task_not_cancelable(id));
        }
        set_status(&mut task, TaskStatus::new(TaskState::Canceled, None));
        self.store.save(&task).await.map_err(RpcError::internal)?;
        Ok(task)
    }

    async fn resubscribe(&self, id: &str) -> Result<FrameRx, RpcError> {
        self.check_streaming()?;
        let (tx, rx) = mpsc::unbounded_channel();
        let live = self.live.lock().unwrap().get(id).cloned();
        if let Some(l) = live {
            let mut st = l.state.lock().unwrap();
            if !st.done {
                // Snapshot and subscription under one lock: every later event
                // reaches this subscriber, every earlier one is in the snapshot.
                let _ = tx.send(Frame::new(StreamResponse::Task(st.task.clone()), false));
                st.subscribers.push(tx);
                return Ok(rx);
            }
        }
        let task = self
            .store
            .get(id)
            .await
            .map_err(RpcError::internal)?
            .ok_or_else(|| RpcError::task_not_found(id))?;
        let _ = tx.send(Frame::new(StreamResponse::Task(task), true));
        Ok(rx)
    }

    /// Applies one event; returns whether the turn ended.
    async fn apply(&self, live: &LiveTask, event: AgentEvent) -> bool {
        let (persist, ended) = {
            let mut st = live.state.lock().unwrap();
            if st.done {
                return true;
            }
            let mut persist = None;
            let ended = match event {
                AgentEvent::Message(mut m) => {
                    m.task_id.get_or_insert_with(|| st.task.id.clone());
                    m.context_id
                        .get_or_insert_with(|| st.task.context_id.clone());
                    if st.announced || st.existing {
                        st.task.history.push(m.clone());
                        persist = Some(st.task.clone());
                    }
                    st.broadcast(&Frame::new(StreamResponse::Message(m), true));
                    true
                }
                AgentEvent::Status(status) => {
                    announce(&mut st);
                    let ended = status.state.ends_turn();
                    set_status(&mut st.task, status.clone());
                    let ev = TaskStatusUpdateEvent {
                        task_id: st.task.id.clone(),
                        context_id: st.task.context_id.clone(),
                        status,
                        is_final: ended,
                        metadata: None,
                    };
                    st.broadcast(&Frame::new(StreamResponse::StatusUpdate(ev), ended));
                    persist = Some(st.task.clone());
                    ended
                }
                AgentEvent::Artifact {
                    artifact,
                    append,
                    last_chunk,
                } => {
                    if announce(&mut st) {
                        persist = Some(st.task.clone());
                    }
                    let ev = TaskArtifactUpdateEvent {
                        task_id: st.task.id.clone(),
                        context_id: st.task.context_id.clone(),
                        artifact,
                        append: Some(append),
                        last_chunk: Some(last_chunk),
                        metadata: None,
                    };
                    let frame = Frame::new(StreamResponse::ArtifactUpdate(ev), false);
                    st.broadcast(&frame);
                    if let StreamResponse::ArtifactUpdate(ev) = &frame.event {
                        merge_artifact(&mut st.task, ev, append);
                    }
                    false
                }
            };
            st.done = ended;
            (persist, ended)
        };
        if let Some(t) = persist {
            // Store errors must not break a running stream.
            let _ = self.store.save(&t).await;
        }
        ended
    }
}

/// Streams the task snapshot once, before its first update.
fn announce(st: &mut LiveState) -> bool {
    if st.announced {
        return false;
    }
    st.announced = true;
    let frame = Frame::new(StreamResponse::Task(st.task.clone()), false);
    st.broadcast(&frame);
    true
}

fn merge_artifact(task: &mut Task, ev: &TaskArtifactUpdateEvent, append: bool) {
    let incoming = &ev.artifact;
    match task
        .artifacts
        .iter_mut()
        .find(|a| a.artifact_id == incoming.artifact_id)
    {
        Some(a) if append => a.append_parts(incoming.parts.clone()),
        Some(a) => *a = incoming.clone(),
        None => task.artifacts.push(incoming.clone()),
    }
}

/// Owns one execution: applies events in order, finishes the task when the
/// executor returns, handles cancellation, then unregisters the task.
async fn drive(
    inner: Arc<Inner>,
    live: Arc<LiveTask>,
    mut rx: mpsc::UnboundedReceiver<Control>,
    mut exec: JoinHandle<Result<(), String>>,
) {
    let mut exec_done = false;
    loop {
        let ended = tokio::select! {
            biased;
            Some(c) = rx.recv() => handle(&inner, &live, &mut exec, c).await,
            res = &mut exec, if !exec_done => {
                exec_done = true;
                // Everything the executor sent happened before it returned.
                let mut ended = false;
                while let Ok(c) = rx.try_recv() {
                    if handle(&inner, &live, &mut exec, c).await {
                        ended = true;
                        break;
                    }
                }
                if !ended {
                    let status = match res {
                        Ok(Ok(())) => TaskStatus::new(TaskState::Completed, None),
                        Ok(Err(e)) => failed(&live, e),
                        Err(e) if e.is_cancelled() => TaskStatus::new(TaskState::Canceled, None),
                        Err(_) => failed(&live, "agent panicked".into()),
                    };
                    inner.apply(&live, AgentEvent::Status(status)).await;
                }
                true
            }
            else => true,
        };
        if ended {
            break;
        }
    }
    let id = live.state.lock().unwrap().task.id.clone();
    inner.live.lock().unwrap().remove(&id);
}

fn failed(live: &LiveTask, error: String) -> TaskStatus {
    let st = live.state.lock().unwrap();
    let mut m = Message::agent_text(error);
    m.task_id = Some(st.task.id.clone());
    m.context_id = Some(st.task.context_id.clone());
    TaskStatus::new(TaskState::Failed, Some(m))
}

async fn handle(
    inner: &Inner,
    live: &LiveTask,
    exec: &mut JoinHandle<Result<(), String>>,
    c: Control,
) -> bool {
    match c {
        Control::Event(ev) => inner.apply(live, ev).await,
        Control::Cancel(reply) => {
            exec.abort();
            inner
                .apply(
                    live,
                    AgentEvent::Status(TaskStatus::new(TaskState::Canceled, None)),
                )
                .await;
            let task = live.state.lock().unwrap().task.clone();
            let _ = reply.send(task);
            true
        }
    }
}
