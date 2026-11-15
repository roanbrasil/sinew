//! MCP client: connect to a server, list its tools and expose them as Sinew
//! [`Tool`]s.

use std::{
    collections::HashMap,
    ffi::OsStr,
    process::Stdio,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, OnceLock, Weak,
    },
    time::Duration,
};

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{Tool, ToolBox, ToolSpec};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader},
    process::{Child, Command},
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

use crate::{
    error::{McpError, Result},
    protocol::{
        self, classify, error_code, CallToolResult, Implementation, Incoming, InitializeResult,
        McpToolInfo, RpcError, LATEST_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
    },
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Sends JSON-RPC messages to the server. Replies arrive on the incoming channel.
#[async_trait]
pub(crate) trait Outbound: Send + Sync {
    async fn send(&self, msg: Value) -> Result<()>;
    /// Called once the protocol version is negotiated.
    fn set_protocol_version(&self, _version: &str) {}
    async fn close(&self) {}
}

struct StreamOut {
    // `None` once closed. Dropping the writer is what closes a child's stdin.
    writer: tokio::sync::Mutex<Option<Box<dyn AsyncWrite + Send + Unpin>>>,
}

#[async_trait]
impl Outbound for StreamOut {
    async fn send(&self, msg: Value) -> Result<()> {
        let mut line = serde_json::to_vec(&msg)?;
        line.push(b'\n');
        let mut guard = self.writer.lock().await;
        let w = guard.as_mut().ok_or(McpError::Closed)?;
        w.write_all(&line).await?;
        w.flush().await?;
        Ok(())
    }

    async fn close(&self) {
        if let Some(mut w) = self.writer.lock().await.take() {
            let _ = w.shutdown().await;
        }
    }
}

type Pending = HashMap<u64, oneshot::Sender<std::result::Result<Value, RpcError>>>;

struct Inner {
    out: Box<dyn Outbound>,
    pending: Mutex<Pending>,
    closed: AtomicBool,
    next_id: AtomicU64,
    timeout_ms: AtomicU64,
    init: OnceLock<InitializeResult>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
    child: Mutex<Option<Child>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        for t in self.tasks.get_mut().unwrap().drain(..) {
            t.abort();
        }
        // `kill_on_drop` on the child takes care of the server process.
    }
}

/// A connection to one MCP server. Cheap to clone; the connection (and a
/// spawned server process) lives until the last clone and the last
/// [`McpTool`] built from it are dropped.
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<Inner>,
}

impl McpClient {
    /// Spawns `program args...` and talks MCP over its stdin/stdout.
    /// The server's stderr is inherited.
    pub async fn connect_stdio<I, S>(program: impl AsRef<OsStr>, args: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut cmd = Command::new(program);
        cmd.args(args);
        Self::connect_command(cmd).await
    }

    /// Like [`McpClient::connect_stdio`] with a fully configured command
    /// (environment, working directory, stderr handling...).
    pub async fn connect_command(mut cmd: Command) -> Result<Self> {
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take().ok_or(McpError::Closed)?;
        let stdout = child.stdout.take().ok_or(McpError::Closed)?;
        Self::start_streams(stdout, stdin, Some(child)).await
    }

    /// Talks MCP over any pair of byte streams (newline-delimited JSON, the
    /// stdio transport framing). Useful for in-process servers and tests.
    pub async fn connect<R, W>(reader: R, writer: W) -> Result<Self>
    where
        R: AsyncRead + Send + Unpin + 'static,
        W: AsyncWrite + Send + Unpin + 'static,
    {
        Self::start_streams(reader, writer, None).await
    }

    /// Connects to a Streamable HTTP endpoint such as `http://localhost:8000/mcp`.
    #[cfg(feature = "http")]
    pub async fn connect_http(url: impl Into<String>) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let out = crate::http::HttpOut::new(url.into(), tx);
        let client = Self::new(Box::new(out), rx, None);
        client.initialize().await?;
        Ok(client)
    }

    async fn start_streams<R, W>(reader: R, writer: W, child: Option<Child>) -> Result<Self>
    where
        R: AsyncRead + Send + Unpin + 'static,
        W: AsyncWrite + Send + Unpin + 'static,
    {
        let (tx, rx) = mpsc::unbounded_channel();
        let reader_task = tokio::spawn(async move {
            let mut lines = BufReader::new(reader).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                // Servers must only write JSON-RPC to stdout; skip anything else.
                if let Ok(v) = serde_json::from_str::<Value>(line) {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
            }
        });
        let out = StreamOut {
            writer: tokio::sync::Mutex::new(Some(Box::new(writer))),
        };
        let client = Self::new(Box::new(out), rx, child);
        client.inner.tasks.lock().unwrap().push(reader_task);
        client.initialize().await?;
        Ok(client)
    }

    fn new(
        out: Box<dyn Outbound>,
        rx: mpsc::UnboundedReceiver<Value>,
        child: Option<Child>,
    ) -> Self {
        let inner = Arc::new(Inner {
            out,
            pending: Mutex::new(HashMap::new()),
            closed: AtomicBool::new(false),
            next_id: AtomicU64::new(1),
            timeout_ms: AtomicU64::new(DEFAULT_TIMEOUT.as_millis() as u64),
            init: OnceLock::new(),
            tasks: Mutex::new(Vec::new()),
            child: Mutex::new(child),
        });
        let dispatcher = tokio::spawn(dispatch(Arc::downgrade(&inner), rx));
        inner.tasks.lock().unwrap().push(dispatcher);
        Self { inner }
    }

    async fn initialize(&self) -> Result<()> {
        let params = json!({
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "sinew-mcp", "version": env!("CARGO_PKG_VERSION") },
        });
        let result: InitializeResult =
            serde_json::from_value(self.request("initialize", Some(params)).await?)?;
        if !SUPPORTED_PROTOCOL_VERSIONS.contains(&result.protocol_version.as_str()) {
            self.close().await;
            return Err(McpError::Protocol(format!(
                "server wants protocol version {}, supported: {}",
                result.protocol_version,
                SUPPORTED_PROTOCOL_VERSIONS.join(", ")
            )));
        }
        self.inner
            .out
            .set_protocol_version(&result.protocol_version);
        let _ = self.inner.init.set(result);
        self.notify("notifications/initialized", None).await
    }

    /// What the server said about itself during the handshake.
    pub fn server_info(&self) -> &InitializeResult {
        self.inner
            .init
            .get()
            .expect("client is initialized by every constructor")
    }

    /// Timeout for each request (default 60 s). A request that times out is
    /// cancelled on the server with `notifications/cancelled`.
    pub fn set_request_timeout(&self, timeout: Duration) {
        self.inner
            .timeout_ms
            .store(timeout.as_millis() as u64, Ordering::Relaxed);
    }

    /// Sends a raw JSON-RPC request and waits for its result.
    pub async fn request(&self, method: &str, params: Option<Value>) -> Result<Value> {
        let inner = &self.inner;
        let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        inner.pending.lock().unwrap().insert(id, tx);
        if inner.closed.load(Ordering::SeqCst) {
            inner.pending.lock().unwrap().remove(&id);
            return Err(McpError::Closed);
        }
        if let Err(e) = inner.out.send(protocol::request(id, method, params)).await {
            inner.pending.lock().unwrap().remove(&id);
            return Err(e);
        }
        let timeout = Duration::from_millis(inner.timeout_ms.load(Ordering::Relaxed));
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(Ok(v))) => Ok(v),
            Ok(Ok(Err(e))) => Err(McpError::Rpc(e)),
            Ok(Err(_)) => Err(McpError::Closed),
            Err(_) => {
                inner.pending.lock().unwrap().remove(&id);
                let _ = self
                    .notify(
                        "notifications/cancelled",
                        Some(json!({ "requestId": id, "reason": "timeout" })),
                    )
                    .await;
                Err(McpError::Timeout(method.into()))
            }
        }
    }

    /// Sends a JSON-RPC notification.
    pub async fn notify(&self, method: &str, params: Option<Value>) -> Result<()> {
        self.inner
            .out
            .send(protocol::notification(method, params))
            .await
    }

    pub async fn ping(&self) -> Result<()> {
        self.request("ping", None).await.map(|_| ())
    }

    /// All tools the server offers (follows `nextCursor` pagination).
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = cursor.as_ref().map(|c| json!({ "cursor": c }));
            let page = self.request("tools/list", params).await?;
            let items = page
                .get("tools")
                .cloned()
                .ok_or_else(|| McpError::Protocol("tools/list result has no `tools`".into()))?;
            tools.extend(serde_json::from_value::<Vec<McpToolInfo>>(items)?);
            match page.get("nextCursor").and_then(Value::as_str) {
                Some(c) if !c.is_empty() => cursor = Some(c.to_string()),
                _ => return Ok(tools),
            }
        }
    }

    /// Calls a tool. A tool-level failure comes back as `Ok` with
    /// `is_error == true`; protocol failures (unknown tool, bad params) are `Err`.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<CallToolResult> {
        let arguments = if arguments.is_null() {
            json!({})
        } else {
            arguments
        };
        let result = self
            .request(
                "tools/call",
                Some(json!({ "name": name, "arguments": arguments })),
            )
            .await?;
        Ok(CallToolResult::from_json(&result))
    }

    /// Every server tool wrapped as a Sinew [`Tool`], ready for `react_agent`.
    pub async fn tools(&self) -> Result<Vec<McpTool>> {
        Ok(self
            .list_tools()
            .await?
            .into_iter()
            .map(|info| McpTool {
                client: self.clone(),
                spec: info.to_spec(),
                info,
            })
            .collect())
    }

    /// A [`ToolBox`] holding every server tool.
    pub async fn toolbox(&self) -> Result<ToolBox> {
        let mut tb = ToolBox::new();
        for t in self.tools().await? {
            tb.register(t);
        }
        Ok(tb)
    }

    /// Closes the connection and stops a spawned server. Pending and later
    /// requests fail with [`McpError::Closed`].
    pub async fn close(&self) {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.inner.pending.lock().unwrap().clear();
        self.inner.out.close().await;
        let child = self.inner.child.lock().unwrap().take();
        if let Some(mut child) = child {
            // Closing stdin asks the server to exit; give it a moment, then kill.
            if tokio::time::timeout(Duration::from_secs(2), child.wait())
                .await
                .is_err()
            {
                let _ = child.kill().await;
            }
        }
    }
}

/// Routes responses to waiting requests and answers server-initiated requests.
async fn dispatch(inner: Weak<Inner>, mut rx: mpsc::UnboundedReceiver<Value>) {
    while let Some(msg) = rx.recv().await {
        let Some(inner) = inner.upgrade() else { return };
        match classify(msg) {
            Ok(Incoming::Response { id, result }) => {
                if let Some(id) = id.as_u64() {
                    if let Some(tx) = inner.pending.lock().unwrap().remove(&id) {
                        let _ = tx.send(result);
                    }
                }
            }
            Ok(Incoming::Request { id, method, .. }) => {
                let reply = if method == "ping" {
                    protocol::response(id, json!({}))
                } else {
                    protocol::error_response(
                        id,
                        RpcError::new(
                            error_code::METHOD_NOT_FOUND,
                            format!("client does not support `{method}`"),
                        ),
                    )
                };
                let _ = inner.out.send(reply).await;
            }
            // Logging, progress and list-changed notifications are ignored.
            Ok(Incoming::Notification { .. }) | Err(_) => {}
        }
    }
    if let Some(inner) = inner.upgrade() {
        inner.closed.store(true, Ordering::SeqCst);
        // Dropping the senders wakes every waiter with `Closed`.
        inner.pending.lock().unwrap().clear();
    }
}

/// One MCP server tool as a Sinew [`Tool`].
///
/// `isError` results and protocol errors both become
/// `sinew_core::Error::Tool`, so `ToolBox::run` turns them into an
/// `error: ...` message the model can react to.
#[derive(Clone)]
pub struct McpTool {
    client: McpClient,
    spec: ToolSpec,
    info: McpToolInfo,
}

impl McpTool {
    /// The tool exactly as the server described it.
    pub fn info(&self) -> &McpToolInfo {
        &self.info
    }

    pub fn client(&self) -> &McpClient {
        &self.client
    }
}

#[async_trait]
impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value) -> sinew_core::Result<Value> {
        let tool_err = |message: String| sinew_core::Error::Tool {
            name: self.spec.name.clone(),
            message,
        };
        let result = self
            .client
            .call_tool(&self.spec.name, arguments)
            .await
            .map_err(|e| tool_err(e.to_string()))?;
        result.into_tool_value().map_err(tool_err)
    }
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let info: Option<&Implementation> = self.inner.init.get().map(|i| &i.server_info);
        f.debug_struct("McpClient").field("server", &info).finish()
    }
}
