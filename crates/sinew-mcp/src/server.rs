//! MCP server: serve a Sinew [`ToolBox`] (and optionally an agent graph as a
//! single tool), resources and prompts over stdio or any pair of byte streams.

use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
};

use serde::Serialize;
use serde_json::{json, Value};
use sinew_core::identity::{
    with_identity_scope, CallContext, IdentityVerifier, SignedAgentIdentity, UserCredential, Via,
    META_CREDENTIAL, META_IDENTITY,
};
use sinew_core::{PromptTemplate, Tool, ToolBox, ToolCall};
use sinew_graph::{prebuilt::AgentState, CompiledGraph};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader},
    sync::mpsc,
    task::{AbortHandle, JoinSet},
};

use crate::{
    agent::AgentTool,
    error::Result,
    hub::Hub,
    protocol::{
        self, classify, error_code, CallToolResult, GetPromptResult, Implementation, Incoming,
        McpPrompt, McpResource, McpResourceTemplate, McpToolInfo, ResourceContents, RpcError,
        UriTemplate, LATEST_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
    },
};

/// What a resource handler receives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceRequest {
    /// The URI the client asked for.
    pub uri: String,
    /// Values of the URI template variables (empty for static resources).
    pub params: HashMap<String, String>,
}

/// What a prompt handler receives. Required arguments are checked before the
/// handler runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptRequest {
    pub name: String,
    pub arguments: HashMap<String, String>,
}

/// Result type of resource and prompt handlers. Return
/// [`RpcError::resource_not_found`], [`RpcError::invalid_params`] or
/// [`RpcError::internal`] to report a failure to the client.
pub type HandlerResult<T> = std::result::Result<T, RpcError>;

type BoxFuture<T> = Pin<Box<dyn Future<Output = HandlerResult<T>> + Send>>;
type ResourceHandler =
    Arc<dyn Fn(ResourceRequest) -> BoxFuture<Vec<ResourceContents>> + Send + Sync>;
type PromptHandler = Arc<dyn Fn(PromptRequest) -> BoxFuture<GetPromptResult> + Send + Sync>;

/// Resources, prompts and connected peers, shared by every clone of a server.
#[derive(Default)]
struct Catalog {
    resources: RwLock<Vec<(McpResource, ResourceHandler)>>,
    templates: RwLock<Vec<(McpResourceTemplate, UriTemplate, ResourceHandler)>>,
    prompts: RwLock<Vec<(McpPrompt, PromptHandler)>>,
    hub: Hub,
    next_peer: AtomicU64,
}

/// Requests that may take a while: served concurrently and cancellable.
const CONCURRENT_METHODS: &[&str] = &["tools/call", "resources/read", "prompts/get"];

/// An MCP server exposing Sinew tools, resources and prompts.
///
/// ```no_run
/// # use sinew_mcp::McpServer;
/// # use sinew_core::ToolBox;
/// # async fn demo(tools: ToolBox) -> sinew_mcp::Result<()> {
/// McpServer::new("calculator", "0.1.0").with_tools(tools).serve_stdio().await
/// # }
/// ```
///
/// Clones share their resources, prompts and connected clients (tools are
/// copied). Keep a clone to change resources and send notifications while
/// `serve_*` runs:
///
/// ```no_run
/// # use sinew_mcp::{McpServer, protocol::McpResource};
/// # async fn demo() {
/// let server = McpServer::new("notes", "0.1.0")
///     .with_text_resource(McpResource::new("notes://today", "today"), "ship it");
/// let handle = server.clone();
/// tokio::spawn(server.serve_stdio());
/// handle.notify_resource_updated("notes://today");
/// # }
/// ```
#[derive(Clone)]
pub struct McpServer {
    info: Implementation,
    instructions: Option<String>,
    tools: ToolBox,
    catalog: Arc<Catalog>,
    page_size: Option<usize>,
    verifier: Option<IdentityVerifier>,
    identity: Option<SignedAgentIdentity>,
}

impl McpServer {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            info: Implementation {
                name: name.into(),
                version: version.into(),
                title: None,
            },
            instructions: None,
            tools: ToolBox::new(),
            catalog: Arc::default(),
            page_size: None,
            verifier: None,
            identity: None,
        }
    }

    /// Verifies the caller's identity proof on every `tools/call` (see
    /// [`McpClient::with_credentials`](crate::McpClient::with_credentials)).
    /// The audience is the verifier's, else this server's identity id
    /// ([`with_identity`](Self::with_identity)), else its name. A call is
    /// rejected with
    /// [`IDENTITY_REJECTED`](crate::protocol::error_code::IDENTITY_REJECTED)
    /// when the proof is missing (and required) or does not verify, or when
    /// the caller's identity does not list the tool in `allowed_tools`. The
    /// tool runs inside the caller's
    /// [`CallContext`] with this server
    /// appended to the delegation chain.
    ///
    /// An on-behalf-of credential in the request's `_meta` is handed to the
    /// tool as [`user_credential`](sinew_core::identity::user_credential)
    /// with or without a verifier: it is a bearer token the downstream
    /// service checks itself.
    pub fn with_identity_verifier(mut self, verifier: IdentityVerifier) -> Self {
        self.verifier = Some(verifier);
        self
    }

    /// Presents this server's signed identity in its `initialize` answer
    /// (`_meta`), for clients to check with
    /// [`McpClient::verify_server`](crate::McpClient::verify_server).
    pub fn with_identity(mut self, identity: SignedAgentIdentity) -> Self {
        self.identity = Some(identity);
        self
    }

    /// This server's id as an audience and a hop in the delegation chain.
    fn agent_id(&self) -> &str {
        self.identity
            .as_ref()
            .map_or(self.info.name.as_str(), |i| i.payload.id.as_str())
    }

    /// The caller (when verified) and the credential from a request's
    /// `_meta`.
    fn identify(
        &self,
        meta: Option<&Value>,
        tool: &str,
    ) -> HandlerResult<(Option<CallContext>, Option<UserCredential>)> {
        let credential = match meta.and_then(|m| m.get(META_CREDENTIAL)) {
            Some(c) => Some(
                serde_json::from_value::<UserCredential>(c.clone())
                    .map_err(|e| RpcError::invalid_params(format!("credential: {e}")))?,
            ),
            None => None,
        };
        let rejected = |e: String| {
            RpcError::new(
                error_code::IDENTITY_REJECTED,
                format!("identity rejected: {e}"),
            )
        };
        let caller = match &self.verifier {
            Some(v) => v
                .verify_value(
                    meta.and_then(|m| m.get(META_IDENTITY)),
                    Some(self.agent_id()),
                )
                .map_err(|e| rejected(e.to_string()))?,
            None => None,
        };
        let ctx = match caller {
            Some(c) if !c.identity.identity.tool_allowed(tool) => {
                return Err(rejected(format!("`{}` may not call `{tool}`", c.agent())))
            }
            Some(c) => {
                let mut ctx = c.context_for(self.agent_id(), Via::Mcp);
                ctx.subject_token = credential.as_ref().map(|cr| cr.token.clone());
                Some(ctx)
            }
            None => None,
        };
        Ok((ctx, credential))
    }

    /// Replaces the served tools with `tools`.
    pub fn with_tools(mut self, tools: ToolBox) -> Self {
        self.tools = tools;
        self
    }

    /// Adds one tool.
    pub fn with_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.register(tool);
        self
    }

    /// Adds an agent graph (e.g. from `react_agent`) as one tool that takes
    /// `{"input": "..."}` and returns the agent's final answer.
    pub fn with_agent(
        self,
        name: &str,
        description: &str,
        graph: CompiledGraph<AgentState>,
    ) -> Self {
        self.with_tool(AgentTool::new(name, description, graph))
    }

    /// Instructions sent to the client in the `initialize` result.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Splits `resources/list`, `resources/templates/list` and `prompts/list`
    /// into pages of `n` items linked by `nextCursor`. By default every item
    /// is returned in one page.
    pub fn with_page_size(mut self, n: usize) -> Self {
        self.page_size = Some(n.max(1));
        self
    }

    /// Adds a resource whose contents `handler` produces on every
    /// `resources/read`. See [`add_resource`](Self::add_resource).
    pub fn with_resource<F, Fut>(self, resource: McpResource, handler: F) -> Self
    where
        F: Fn(ResourceRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = HandlerResult<Vec<ResourceContents>>> + Send + 'static,
    {
        self.add_resource(resource, handler);
        self
    }

    /// Adds a resource with fixed text contents (tagged with the resource's
    /// MIME type).
    pub fn with_text_resource(self, resource: McpResource, text: impl Into<String>) -> Self {
        let text = text.into();
        let mime = resource.mime_type.clone();
        self.with_resource(resource, move |req: ResourceRequest| {
            let mut c = ResourceContents::text(req.uri, text.clone());
            c.mime_type = mime.clone();
            async move { Ok(vec![c]) }
        })
    }

    /// Adds a resource template: `resources/read` for any URI matching
    /// `template.uri_template` calls `handler` with the template variables in
    /// [`ResourceRequest::params`]. Static resources are matched first, then
    /// templates in registration order. See [`UriTemplate`] for the
    /// supported syntax.
    ///
    /// # Panics
    ///
    /// If `template.uri_template` is not a valid URI template (an unclosed
    /// `{` or an empty `{}`). Use
    /// [`add_resource_template`](Self::add_resource_template) to get an error
    /// instead.
    pub fn with_resource_template<F, Fut>(self, template: McpResourceTemplate, handler: F) -> Self
    where
        F: Fn(ResourceRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = HandlerResult<Vec<ResourceContents>>> + Send + 'static,
    {
        if let Err(e) = self.add_resource_template(template, handler) {
            panic!("{}", e.message);
        }
        self
    }

    /// Adds a prompt. `handler` runs on `prompts/get` once the server has
    /// checked that every required argument is present.
    pub fn with_prompt<F, Fut>(self, prompt: McpPrompt, handler: F) -> Self
    where
        F: Fn(PromptRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = HandlerResult<GetPromptResult>> + Send + 'static,
    {
        self.add_prompt(prompt, handler);
        self
    }

    /// Adds a prompt rendered from a Sinew [`PromptTemplate`]: every template
    /// variable becomes a required argument, and `prompts/get` returns the
    /// rendered text as one user message.
    pub fn with_prompt_template(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        template: PromptTemplate,
    ) -> Self {
        let mut prompt = McpPrompt::new(name).with_description(description);
        for v in template.variables() {
            prompt.arguments.push(protocol::PromptArgument {
                name: v,
                title: None,
                description: None,
                required: true,
            });
        }
        let template = Arc::new(template);
        self.with_prompt(prompt, move |req: PromptRequest| {
            let rendered = template
                .render_vars(&req.arguments)
                .map(GetPromptResult::user)
                .map_err(|e| RpcError::invalid_params(e.to_string()));
            async move { rendered }
        })
    }

    /// Adds (or replaces, by URI) a resource at runtime and sends
    /// `notifications/resources/list_changed` to connected clients.
    pub fn add_resource<F, Fut>(&self, resource: McpResource, handler: F)
    where
        F: Fn(ResourceRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = HandlerResult<Vec<ResourceContents>>> + Send + 'static,
    {
        let handler: ResourceHandler = Arc::new(move |req| Box::pin(handler(req)));
        {
            let mut list = self.catalog.resources.write().unwrap();
            list.retain(|(r, _)| r.uri != resource.uri);
            list.push((resource, handler));
        }
        self.notify_resource_list_changed();
    }

    /// Adds (or replaces, by template string) a resource template at runtime
    /// and notifies connected clients. Fails with `-32602` on an invalid URI
    /// template.
    pub fn add_resource_template<F, Fut>(
        &self,
        template: McpResourceTemplate,
        handler: F,
    ) -> HandlerResult<()>
    where
        F: Fn(ResourceRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = HandlerResult<Vec<ResourceContents>>> + Send + 'static,
    {
        let matcher =
            UriTemplate::new(template.uri_template.clone()).map_err(RpcError::invalid_params)?;
        let handler: ResourceHandler = Arc::new(move |req| Box::pin(handler(req)));
        {
            let mut list = self.catalog.templates.write().unwrap();
            list.retain(|(t, _, _)| t.uri_template != template.uri_template);
            list.push((template, matcher, handler));
        }
        self.notify_resource_list_changed();
        Ok(())
    }

    /// Adds (or replaces, by name) a prompt at runtime and sends
    /// `notifications/prompts/list_changed` to connected clients.
    pub fn add_prompt<F, Fut>(&self, prompt: McpPrompt, handler: F)
    where
        F: Fn(PromptRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = HandlerResult<GetPromptResult>> + Send + 'static,
    {
        let handler: PromptHandler = Arc::new(move |req| Box::pin(handler(req)));
        {
            let mut list = self.catalog.prompts.write().unwrap();
            list.retain(|(p, _)| p.name != prompt.name);
            list.push((prompt, handler));
        }
        self.notify_prompt_list_changed();
    }

    /// Removes a resource by URI and notifies clients if it existed.
    pub fn remove_resource(&self, uri: &str) -> bool {
        let removed = retain_changed(&self.catalog.resources, |(r, _)| r.uri != uri);
        if removed {
            self.notify_resource_list_changed();
        }
        removed
    }

    /// Removes a resource template by its template string and notifies
    /// clients if it existed.
    pub fn remove_resource_template(&self, uri_template: &str) -> bool {
        let removed = retain_changed(&self.catalog.templates, |(t, _, _)| {
            t.uri_template != uri_template
        });
        if removed {
            self.notify_resource_list_changed();
        }
        removed
    }

    /// Removes a prompt by name and notifies clients if it existed.
    pub fn remove_prompt(&self, name: &str) -> bool {
        let removed = retain_changed(&self.catalog.prompts, |(p, _)| p.name != name);
        if removed {
            self.notify_prompt_list_changed();
        }
        removed
    }

    /// Sends `notifications/resources/updated` to every connected client
    /// subscribed to `uri` and returns how many received it. Over Streamable
    /// HTTP, only sessions with an open `GET` stream receive notifications.
    pub fn notify_resource_updated(&self, uri: &str) -> usize {
        let msg = protocol::notification(
            "notifications/resources/updated",
            Some(json!({ "uri": uri })),
        );
        self.catalog.hub.send_to_subscribers(uri, &msg)
    }

    /// Sends `notifications/resources/list_changed` to every connected
    /// client and returns how many received it.
    pub fn notify_resource_list_changed(&self) -> usize {
        let msg = protocol::notification("notifications/resources/list_changed", None);
        self.catalog.hub.broadcast(&msg)
    }

    /// Sends `notifications/prompts/list_changed` to every connected client
    /// and returns how many received it.
    pub fn notify_prompt_list_changed(&self) -> usize {
        let msg = protocol::notification("notifications/prompts/list_changed", None);
        self.catalog.hub.broadcast(&msg)
    }

    fn has_resources(&self) -> bool {
        !self.catalog.resources.read().unwrap().is_empty()
            || !self.catalog.templates.read().unwrap().is_empty()
    }

    fn has_prompts(&self) -> bool {
        !self.catalog.prompts.read().unwrap().is_empty()
    }

    #[cfg_attr(not(feature = "http-server"), allow(dead_code))]
    pub(crate) fn hub(&self) -> &Hub {
        &self.catalog.hub
    }

    /// A fresh peer id for a connection or session.
    #[cfg_attr(not(feature = "http-server"), allow(dead_code))]
    pub(crate) fn next_peer_id(&self, prefix: &str) -> String {
        let n = self.catalog.next_peer.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{n}")
    }

    /// Serves on this process's stdin/stdout until stdin closes. Nothing
    /// else may write to stdout; log to stderr.
    pub async fn serve_stdio(self) -> Result<()> {
        self.serve(tokio::io::stdin(), tokio::io::stdout()).await
    }

    /// Serves newline-delimited JSON-RPC on `reader`/`writer` until `reader`
    /// reaches EOF. `tools/call`, `resources/read` and `prompts/get` run
    /// concurrently and can be cancelled with `notifications/cancelled`;
    /// everything else is answered in order. Resource and list-changed
    /// notifications are written to `writer` as they are sent.
    pub async fn serve<R, W>(self, reader: R, writer: W) -> Result<()>
    where
        R: AsyncRead + Send + Unpin + 'static,
        W: AsyncWrite + Send + Unpin + 'static,
    {
        let server = Arc::new(self);
        let (tx, mut rx) = mpsc::unbounded_channel::<Value>();
        let writer_task = tokio::spawn(async move {
            let mut writer = writer;
            while let Some(msg) = rx.recv().await {
                let mut line = serde_json::to_vec(&msg)?;
                line.push(b'\n');
                writer.write_all(&line).await?;
                writer.flush().await?;
            }
            Ok::<_, crate::McpError>(())
        });
        let peer: Arc<str> = server.next_peer_id("stream").into();
        server.catalog.hub.add_peer(&peer, Some(tx.clone()));

        let running: Arc<Mutex<HashMap<String, AbortHandle>>> = Arc::default();
        let mut calls = JoinSet::new();
        let mut lines = BufReader::new(reader).lines();
        let read_result = loop {
            let line = match lines.next_line().await {
                Ok(Some(line)) => line,
                Ok(None) => break Ok(()),
                Err(e) => break Err(e),
            };
            if line.trim().is_empty() {
                continue;
            }
            let msg = match serde_json::from_str::<Value>(&line) {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.send(protocol::error_response(
                        Value::Null,
                        RpcError::new(error_code::PARSE_ERROR, format!("parse error: {e}")),
                    ));
                    continue;
                }
            };
            match classify(msg) {
                Ok(Incoming::Request { id, method, params })
                    if CONCURRENT_METHODS.contains(&method.as_str()) =>
                {
                    let (server, tx, running2, peer) =
                        (server.clone(), tx.clone(), running.clone(), peer.clone());
                    let key = id.to_string();
                    // Hold the lock while spawning so a fast call cannot try to
                    // deregister itself before it is registered.
                    let mut map = running.lock().unwrap();
                    let handle = calls.spawn(async move {
                        let reply = match server.handle(Some(&peer), &method, &params).await {
                            Ok(r) => protocol::response(id.clone(), r),
                            Err(e) => protocol::error_response(id.clone(), e),
                        };
                        running2.lock().unwrap().remove(&id.to_string());
                        let _ = tx.send(reply);
                    });
                    map.insert(key, handle);
                }
                Ok(Incoming::Request { id, method, params }) => {
                    let reply = match server.handle(Some(&peer), &method, &params).await {
                        Ok(r) => protocol::response(id, r),
                        Err(e) => protocol::error_response(id, e),
                    };
                    let _ = tx.send(reply);
                }
                Ok(Incoming::Notification { method, params }) => {
                    if method == "notifications/cancelled" {
                        if let Some(id) = params.get("requestId") {
                            // A cancelled request gets no response.
                            if let Some(h) = running.lock().unwrap().remove(&id.to_string()) {
                                h.abort();
                            }
                        }
                    }
                    // `notifications/initialized` and others need no action.
                }
                // We never send requests, so responses are unexpected; ignore them.
                Ok(Incoming::Response { .. }) => {}
                Err(e) => {
                    let _ = tx.send(protocol::error_response(Value::Null, e));
                }
            }
            // Reap finished calls so the set does not grow without bound.
            while calls.try_join_next().is_some() {}
        };
        // Input closed: let in-flight calls answer, then flush and stop.
        while calls.join_next().await.is_some() {}
        // The hub holds a sender too; drop it so the writer can finish.
        server.catalog.hub.remove_peer(&peer);
        drop(tx);
        read_result?;
        writer_task
            .await
            .map_err(|e| crate::McpError::Protocol(e.to_string()))?
    }

    /// Handles one request on behalf of `peer` (the connection or HTTP
    /// session that sent it, used for subscriptions).
    async fn handle(
        &self,
        peer: Option<&str>,
        method: &str,
        params: &Value,
    ) -> HandlerResult<Value> {
        match method {
            "initialize" => self.initialize(params),
            "ping" => Ok(json!({})),
            "tools/list" => {
                let tools: Vec<McpToolInfo> = self
                    .tools
                    .specs()
                    .iter()
                    .map(McpToolInfo::from_spec)
                    .collect();
                Ok(json!({ "tools": tools }))
            }
            "tools/call" => self.call_tool(params).await.map(|r| r.to_json()),
            "resources/list"
            | "resources/templates/list"
            | "resources/read"
            | "resources/subscribe"
            | "resources/unsubscribe"
                if self.has_resources() =>
            {
                self.handle_resources(peer, method, params).await
            }
            "prompts/list" if self.has_prompts() => {
                let prompts: Vec<McpPrompt> = self
                    .catalog
                    .prompts
                    .read()
                    .unwrap()
                    .iter()
                    .map(|(p, _)| p.clone())
                    .collect();
                self.page(prompts, params, "prompts")
            }
            "prompts/get" if self.has_prompts() => {
                self.get_prompt(params).await.map(|r| r.to_json())
            }
            other => Err(RpcError::new(
                error_code::METHOD_NOT_FOUND,
                format!("method not found: {other}"),
            )),
        }
    }

    fn initialize(&self, params: &Value) -> HandlerResult<Value> {
        let requested = params
            .get("protocolVersion")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                RpcError::new(error_code::INVALID_PARAMS, "missing `protocolVersion`")
            })?;
        // Echo a version we support, otherwise offer our latest and
        // let the client decide whether to continue.
        let version = if SUPPORTED_PROTOCOL_VERSIONS.contains(&requested) {
            requested
        } else {
            LATEST_PROTOCOL_VERSION
        };
        let mut capabilities = json!({ "tools": { "listChanged": false } });
        if self.has_resources() {
            capabilities["resources"] = json!({ "subscribe": true, "listChanged": true });
        }
        if self.has_prompts() {
            capabilities["prompts"] = json!({ "listChanged": true });
        }
        let mut result = json!({
            "protocolVersion": version,
            "capabilities": capabilities,
            "serverInfo": self.info,
        });
        if let Some(i) = &self.instructions {
            result["instructions"] = json!(i);
        }
        if let Some(id) = &self.identity {
            result["_meta"] = json!({ META_IDENTITY: id });
        }
        Ok(result)
    }

    /// One page of `items` starting at the request's `cursor` (an opaque
    /// offset), as `{key: [...], "nextCursor": ...}`.
    fn page<T: Serialize>(&self, items: Vec<T>, params: &Value, key: &str) -> HandlerResult<Value> {
        let total = items.len();
        let start = match params.get("cursor") {
            None | Some(Value::Null) => 0,
            Some(Value::String(c)) => c
                .parse::<usize>()
                .ok()
                .filter(|n| *n <= total)
                .ok_or_else(|| RpcError::invalid_params(format!("invalid cursor: {c}")))?,
            Some(_) => return Err(RpcError::invalid_params("`cursor` must be a string")),
        };
        let end = self.page_size.map_or(total, |n| (start + n).min(total));
        let page: Vec<T> = items.into_iter().skip(start).take(end - start).collect();
        let mut out = json!({ key: page });
        if end < total {
            out["nextCursor"] = json!(end.to_string());
        }
        Ok(out)
    }

    /// The handler and template variables for `uri`, if a resource matches.
    fn resolve(&self, uri: &str) -> Option<(ResourceHandler, HashMap<String, String>)> {
        let resources = self.catalog.resources.read().unwrap();
        if let Some((_, h)) = resources.iter().find(|(r, _)| r.uri == uri) {
            return Some((h.clone(), HashMap::new()));
        }
        drop(resources);
        self.catalog
            .templates
            .read()
            .unwrap()
            .iter()
            .find_map(|(_, m, h)| m.matches(uri).map(|vars| (h.clone(), vars)))
    }

    async fn handle_resources(
        &self,
        peer: Option<&str>,
        method: &str,
        params: &Value,
    ) -> HandlerResult<Value> {
        match method {
            "resources/list" => {
                let items: Vec<McpResource> = self
                    .catalog
                    .resources
                    .read()
                    .unwrap()
                    .iter()
                    .map(|(r, _)| r.clone())
                    .collect();
                self.page(items, params, "resources")
            }
            "resources/templates/list" => {
                let items: Vec<McpResourceTemplate> = self
                    .catalog
                    .templates
                    .read()
                    .unwrap()
                    .iter()
                    .map(|(t, _, _)| t.clone())
                    .collect();
                self.page(items, params, "resourceTemplates")
            }
            _ => {
                let uri = params
                    .get("uri")
                    .and_then(Value::as_str)
                    .ok_or_else(|| RpcError::invalid_params("missing resource `uri`"))?;
                let (handler, vars) = self
                    .resolve(uri)
                    .ok_or_else(|| RpcError::resource_not_found(uri))?;
                match (method, peer) {
                    ("resources/read", _) => {
                        let contents = handler(ResourceRequest {
                            uri: uri.to_string(),
                            params: vars,
                        })
                        .await?;
                        Ok(json!({ "contents": contents }))
                    }
                    ("resources/subscribe", Some(p)) => {
                        self.catalog.hub.subscribe(p, uri);
                        Ok(json!({}))
                    }
                    ("resources/unsubscribe", Some(p)) => {
                        self.catalog.hub.unsubscribe(p, uri);
                        Ok(json!({}))
                    }
                    // No connection to deliver notifications to (stateless HTTP,
                    // `handle_message`): accept and do nothing.
                    _ => Ok(json!({})),
                }
            }
        }
    }

    /// Handles `prompts/get`: unknown prompts, non-string arguments and
    /// missing required arguments are `-32602` errors.
    async fn get_prompt(&self, params: &Value) -> HandlerResult<GetPromptResult> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::invalid_params("missing prompt `name`"))?;
        let mut arguments = HashMap::new();
        match params.get("arguments") {
            None | Some(Value::Null) => {}
            Some(Value::Object(m)) => {
                for (k, v) in m {
                    let Value::String(s) = v else {
                        return Err(RpcError::invalid_params(format!(
                            "argument `{k}` must be a string"
                        )));
                    };
                    arguments.insert(k.clone(), s.clone());
                }
            }
            Some(_) => return Err(RpcError::invalid_params("`arguments` must be an object")),
        }
        let (prompt, handler) = self
            .catalog
            .prompts
            .read()
            .unwrap()
            .iter()
            .find(|(p, _)| p.name == name)
            .map(|(p, h)| (p.clone(), h.clone()))
            .ok_or_else(|| RpcError::invalid_params(format!("unknown prompt: {name}")))?;
        if let Some(missing) = prompt
            .arguments
            .iter()
            .find(|a| a.required && !arguments.contains_key(&a.name))
        {
            return Err(RpcError::invalid_params(format!(
                "missing required argument `{}` for prompt `{name}`",
                missing.name
            )));
        }
        handler(PromptRequest {
            name: name.to_string(),
            arguments,
        })
        .await
    }

    /// Handles `tools/call`. Unknown tools and bad params are JSON-RPC
    /// errors; a tool that fails returns a result with `isError: true`.
    async fn call_tool(&self, params: &Value) -> HandlerResult<CallToolResult> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::new(error_code::INVALID_PARAMS, "missing tool `name`"))?;
        let arguments = match params.get("arguments") {
            None | Some(Value::Null) => json!({}),
            Some(v @ Value::Object(_)) => v.clone(),
            Some(_) => {
                return Err(RpcError::new(
                    error_code::INVALID_PARAMS,
                    "`arguments` must be an object",
                ))
            }
        };
        if !self.tools.specs().iter().any(|s| s.name == name) {
            return Err(RpcError::new(
                error_code::INVALID_PARAMS,
                format!("unknown tool: {name}"),
            ));
        }
        let (ctx, credential) = self.identify(params.get("_meta"), name)?;
        let call = ToolCall {
            id: String::new(),
            name: name.into(),
            arguments,
        };
        Ok(
            match with_identity_scope(ctx, credential, self.tools.call(&call)).await {
                Ok(v) => CallToolResult::from_tool_value(v),
                Err(sinew_core::Error::Tool { message, .. }) => CallToolResult::error(message),
                Err(e) => CallToolResult::error(e.to_string()),
            },
        )
    }

    /// Answers a single JSON-RPC message without a transport: `None` for
    /// notifications and stray responses. Handy for custom transports.
    /// `resources/subscribe` succeeds but delivers nothing, since there is no
    /// connection to push notifications to.
    pub async fn handle_message(&self, msg: Value) -> Option<Value> {
        self.handle_message_from(None, msg).await
    }

    /// Like [`handle_message`](Self::handle_message), on behalf of a peer
    /// registered in the hub (an HTTP session).
    pub(crate) async fn handle_message_from(
        &self,
        peer: Option<&str>,
        msg: Value,
    ) -> Option<Value> {
        match classify(msg) {
            Ok(Incoming::Request { id, method, params }) => {
                Some(match self.handle(peer, &method, &params).await {
                    Ok(r) => protocol::response(id, r),
                    Err(e) => protocol::error_response(id, e),
                })
            }
            Ok(_) => None,
            Err(e) => Some(protocol::error_response(Value::Null, e)),
        }
    }
}

/// `retain` on a locked list; true if anything was removed.
fn retain_changed<T>(list: &RwLock<Vec<T>>, keep: impl FnMut(&T) -> bool) -> bool {
    let mut list = list.write().unwrap();
    let before = list.len();
    list.retain(keep);
    list.len() != before
}

#[cfg(test)]
mod tests {
    use super::*;
    use sinew_core::FnTool;

    fn server() -> McpServer {
        McpServer::new("test", "1.0").with_tool(FnTool::new(
            "echo",
            "Echo `text`",
            json!({"type": "object", "properties": {"text": {"type": "string"}}}),
            |a: Value| match a["text"].as_str() {
                Some(t) => Ok(json!(t)),
                None => Err(sinew_core::Error::Tool {
                    name: "echo".into(),
                    message: "`text` is required".into(),
                }),
            },
        ))
    }

    fn rich_server() -> McpServer {
        server()
            .with_page_size(2)
            .with_text_resource(
                McpResource::new("notes://a", "a").with_mime_type("text/plain"),
                "note a",
            )
            .with_text_resource(McpResource::new("notes://b", "b"), "note b")
            .with_resource(
                McpResource::new("notes://c", "c"),
                |r: ResourceRequest| async move {
                    Ok(vec![ResourceContents::blob_bytes(
                        r.uri,
                        "application/octet-stream",
                        b"c",
                    )])
                },
            )
            .with_resource_template(
                McpResourceTemplate::new("notes://{title}", "note"),
                |r: ResourceRequest| async move {
                    match r.params["title"].as_str() {
                        "missing" => Err(RpcError::resource_not_found(&r.uri)),
                        t => Ok(vec![ResourceContents::text(
                            r.uri.clone(),
                            format!("note {t}"),
                        )]),
                    }
                },
            )
            .with_prompt_template(
                "review",
                "Review code",
                PromptTemplate::new("Review this {lang} code:\n{code}").unwrap(),
            )
            .with_prompt(
                McpPrompt::new("greet").with_argument("name", "Who to greet", false),
                |r: PromptRequest| async move {
                    let name = r.arguments.get("name").cloned().unwrap_or("world".into());
                    Ok(GetPromptResult::user(format!("Hello, {name}")))
                },
            )
    }

    async fn ask(s: &McpServer, method: &str, params: Value) -> Value {
        s.handle_message(protocol::request(7, method, Some(params)))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn negotiates_protocol_version() {
        let s = server();
        let r = ask(&s, "initialize", json!({"protocolVersion": "2024-11-05"})).await;
        assert_eq!(r["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(r["result"]["serverInfo"]["name"], "test");
        let r = ask(&s, "initialize", json!({"protocolVersion": "1999-01-01"})).await;
        assert_eq!(r["result"]["protocolVersion"], LATEST_PROTOCOL_VERSION);
        let r = ask(&s, "initialize", json!({})).await;
        assert_eq!(r["error"]["code"], error_code::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn lists_and_calls_tools() {
        let s = server();
        let r = ask(&s, "tools/list", json!({})).await;
        assert_eq!(r["result"]["tools"][0]["name"], "echo");
        assert_eq!(r["result"]["tools"][0]["inputSchema"]["type"], "object");
        let r = ask(
            &s,
            "tools/call",
            json!({"name": "echo", "arguments": {"text": "hi"}}),
        )
        .await;
        assert_eq!(r["id"], 7);
        assert_eq!(r["result"]["content"][0]["text"], "hi");
        assert_eq!(r["result"]["isError"], false);
    }

    #[tokio::test]
    async fn reports_errors_correctly() {
        let s = server();
        // Tool failure: in-band.
        let r = ask(&s, "tools/call", json!({"name": "echo", "arguments": {}})).await;
        assert_eq!(r["result"]["isError"], true);
        assert_eq!(r["result"]["content"][0]["text"], "`text` is required");
        // Unknown tool and bad params: protocol errors.
        let r = ask(&s, "tools/call", json!({"name": "nope"})).await;
        assert_eq!(r["error"]["code"], error_code::INVALID_PARAMS);
        let r = ask(&s, "tools/call", json!({"name": "echo", "arguments": [1]})).await;
        assert_eq!(r["error"]["code"], error_code::INVALID_PARAMS);
        let r = ask(&s, "resources/list", json!({})).await;
        assert_eq!(r["error"]["code"], error_code::METHOD_NOT_FOUND);
        let r = ask(&s, "prompts/list", json!({})).await;
        assert_eq!(r["error"]["code"], error_code::METHOD_NOT_FOUND);
        assert!(s
            .handle_message(protocol::notification("notifications/initialized", None))
            .await
            .is_none());
    }

    #[tokio::test]
    async fn advertises_capabilities_only_when_registered() {
        let caps = |s: McpServer| async move {
            ask(&s, "initialize", json!({"protocolVersion": "2025-06-18"})).await["result"]
                ["capabilities"]
                .clone()
        };
        let plain = caps(server()).await;
        assert!(plain.get("resources").is_none() && plain.get("prompts").is_none());
        let rich = caps(rich_server()).await;
        assert_eq!(
            rich["resources"],
            json!({"subscribe": true, "listChanged": true})
        );
        assert_eq!(rich["prompts"], json!({"listChanged": true}));
    }

    #[tokio::test]
    async fn lists_resources_with_pagination() {
        let s = rich_server();
        let r = ask(&s, "resources/list", json!({})).await;
        assert_eq!(r["result"]["resources"].as_array().unwrap().len(), 2);
        assert_eq!(r["result"]["resources"][0]["mimeType"], "text/plain");
        assert_eq!(r["result"]["nextCursor"], "2");
        let r = ask(&s, "resources/list", json!({"cursor": "2"})).await;
        assert_eq!(r["result"]["resources"][0]["uri"], "notes://c");
        assert!(r["result"].get("nextCursor").is_none());
        let r = ask(&s, "resources/list", json!({"cursor": "nope"})).await;
        assert_eq!(r["error"]["code"], error_code::INVALID_PARAMS);
        let r = ask(&s, "resources/templates/list", json!({})).await;
        assert_eq!(
            r["result"]["resourceTemplates"][0]["uriTemplate"],
            "notes://{title}"
        );
    }

    #[tokio::test]
    async fn reads_static_and_templated_resources() {
        let s = rich_server();
        let r = ask(&s, "resources/read", json!({"uri": "notes://a"})).await;
        assert_eq!(
            r["result"]["contents"],
            json!([{"uri": "notes://a", "mimeType": "text/plain", "text": "note a"}])
        );
        let r = ask(&s, "resources/read", json!({"uri": "notes://c"})).await;
        assert_eq!(r["result"]["contents"][0]["blob"], "Yw==");
        let r = ask(&s, "resources/read", json!({"uri": "notes://zeta"})).await;
        assert_eq!(r["result"]["contents"][0]["text"], "note zeta");
        let r = ask(&s, "resources/read", json!({"uri": "notes://missing"})).await;
        assert_eq!(r["error"]["code"], error_code::RESOURCE_NOT_FOUND);
        let r = ask(&s, "resources/read", json!({"uri": "other://x"})).await;
        assert_eq!(r["error"]["code"], error_code::RESOURCE_NOT_FOUND);
        assert_eq!(r["error"]["data"]["uri"], "other://x");
        let r = ask(&s, "resources/read", json!({})).await;
        assert_eq!(r["error"]["code"], error_code::INVALID_PARAMS);
        let r = ask(&s, "resources/subscribe", json!({"uri": "notes://a"})).await;
        assert_eq!(r["result"], json!({}));
        let r = ask(&s, "resources/subscribe", json!({"uri": "other://x"})).await;
        assert_eq!(r["error"]["code"], error_code::RESOURCE_NOT_FOUND);
    }

    #[tokio::test]
    async fn gets_prompts_and_checks_arguments() {
        let s = rich_server();
        let r = ask(&s, "prompts/list", json!({})).await;
        let prompts = r["result"]["prompts"].as_array().unwrap();
        assert_eq!(prompts[0]["name"], "review");
        assert_eq!(
            prompts[0]["arguments"],
            json!([{"name": "lang", "required": true}, {"name": "code", "required": true}])
        );
        let r = ask(
            &s,
            "prompts/get",
            json!({"name": "review", "arguments": {"lang": "Rust", "code": "fn main() {}"}}),
        )
        .await;
        assert_eq!(
            r["result"]["messages"][0],
            json!({"role": "user", "content": {"type": "text", "text": "Review this Rust code:\nfn main() {}"}})
        );
        let r = ask(&s, "prompts/get", json!({"name": "greet"})).await;
        assert_eq!(
            r["result"]["messages"][0]["content"]["text"],
            "Hello, world"
        );
        for bad in [
            json!({"name": "review", "arguments": {"lang": "Rust"}}),
            json!({"name": "review", "arguments": {"lang": "Rust", "code": 1}}),
            json!({"name": "nope"}),
            json!({}),
        ] {
            let r = ask(&s, "prompts/get", bad).await;
            assert_eq!(r["error"]["code"], error_code::INVALID_PARAMS);
        }
        let r = ask(
            &s,
            "prompts/get",
            json!({"name": "review", "arguments": {"lang": "Rust"}}),
        )
        .await;
        assert!(r["error"]["message"].as_str().unwrap().contains("`code`"));
    }

    #[tokio::test]
    async fn runtime_changes_replace_and_remove() {
        let s = rich_server();
        let handle = s.clone();
        handle.add_resource(
            McpResource::new("notes://a", "a2"),
            |r: ResourceRequest| async move { Ok(vec![ResourceContents::text(r.uri, "replaced")]) },
        );
        let r = ask(&s, "resources/read", json!({"uri": "notes://a"})).await;
        assert_eq!(r["result"]["contents"][0]["text"], "replaced");
        assert!(handle.remove_resource("notes://b"));
        assert!(!handle.remove_resource("notes://b"));
        assert!(handle.remove_resource_template("notes://{title}"));
        let r = ask(&s, "resources/read", json!({"uri": "notes://zeta"})).await;
        assert_eq!(r["error"]["code"], error_code::RESOURCE_NOT_FOUND);
        assert!(handle.remove_prompt("greet"));
        assert!(handle
            .add_resource_template(McpResourceTemplate::new("x://{", "bad"), |_r| async {
                Ok(vec![])
            })
            .is_err());
    }
}
