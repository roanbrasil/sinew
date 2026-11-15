//! MCP server: serve a Sinew [`ToolBox`] (and optionally an agent graph as a
//! single tool) over stdio or any pair of byte streams.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use serde_json::{json, Value};
use sinew_core::{Tool, ToolBox, ToolCall};
use sinew_graph::{prebuilt::AgentState, CompiledGraph};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader},
    sync::mpsc,
    task::{AbortHandle, JoinSet},
};

use crate::{
    agent::AgentTool,
    error::Result,
    protocol::{
        self, classify, error_code, CallToolResult, Implementation, Incoming, McpToolInfo,
        RpcError, LATEST_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
    },
};

/// An MCP server exposing Sinew tools.
///
/// ```no_run
/// # use sinew_mcp::McpServer;
/// # use sinew_core::ToolBox;
/// # async fn demo(tools: ToolBox) -> sinew_mcp::Result<()> {
/// McpServer::new("calculator", "0.1.0").with_tools(tools).serve_stdio().await
/// # }
/// ```
#[derive(Clone)]
pub struct McpServer {
    info: Implementation,
    instructions: Option<String>,
    tools: ToolBox,
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
        }
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

    /// Serves on this process's stdin/stdout until stdin closes. Nothing
    /// else may write to stdout; log to stderr.
    pub async fn serve_stdio(self) -> Result<()> {
        self.serve(tokio::io::stdin(), tokio::io::stdout()).await
    }

    /// Serves newline-delimited JSON-RPC on `reader`/`writer` until `reader`
    /// reaches EOF. Tool calls run concurrently and can be cancelled with
    /// `notifications/cancelled`; everything else is answered in order.
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

        let running: Arc<Mutex<HashMap<String, AbortHandle>>> = Arc::default();
        let mut calls = JoinSet::new();
        let mut lines = BufReader::new(reader).lines();
        while let Some(line) = lines.next_line().await? {
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
                Ok(Incoming::Request { id, method, params }) if method == "tools/call" => {
                    let (server, tx, running2) = (server.clone(), tx.clone(), running.clone());
                    let key = id.to_string();
                    // Hold the lock while spawning so a fast call cannot try to
                    // deregister itself before it is registered.
                    let mut map = running.lock().unwrap();
                    let handle = calls.spawn(async move {
                        let reply = match server.call_tool(&params).await {
                            Ok(r) => protocol::response(id.clone(), r.to_json()),
                            Err(e) => protocol::error_response(id.clone(), e),
                        };
                        running2.lock().unwrap().remove(&id.to_string());
                        let _ = tx.send(reply);
                    });
                    map.insert(key, handle);
                }
                Ok(Incoming::Request { id, method, params }) => {
                    let reply = match server.handle_request(&method, &params) {
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
        }
        // Input closed: let in-flight calls answer, then flush and stop.
        while calls.join_next().await.is_some() {}
        drop(tx);
        writer_task
            .await
            .map_err(|e| crate::McpError::Protocol(e.to_string()))?
    }

    /// Handles one non-`tools/call` request and returns its result.
    fn handle_request(&self, method: &str, params: &Value) -> std::result::Result<Value, RpcError> {
        match method {
            "initialize" => {
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
                let mut result = json!({
                    "protocolVersion": version,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": self.info,
                });
                if let Some(i) = &self.instructions {
                    result["instructions"] = json!(i);
                }
                Ok(result)
            }
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
            other => Err(RpcError::new(
                error_code::METHOD_NOT_FOUND,
                format!("method not found: {other}"),
            )),
        }
    }

    /// Handles `tools/call`. Unknown tools and bad params are JSON-RPC
    /// errors; a tool that fails returns a result with `isError: true`.
    async fn call_tool(&self, params: &Value) -> std::result::Result<CallToolResult, RpcError> {
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
        let call = ToolCall {
            id: String::new(),
            name: name.into(),
            arguments,
        };
        Ok(match self.tools.call(&call).await {
            Ok(v) => CallToolResult::from_tool_value(v),
            Err(sinew_core::Error::Tool { message, .. }) => CallToolResult::error(message),
            Err(e) => CallToolResult::error(e.to_string()),
        })
    }

    /// Answers a single JSON-RPC message without a transport: `None` for
    /// notifications and stray responses. Handy for custom transports.
    pub async fn handle_message(&self, msg: Value) -> Option<Value> {
        match classify(msg) {
            Ok(Incoming::Request { id, method, params }) => {
                let result = if method == "tools/call" {
                    self.call_tool(&params).await.map(|r| r.to_json())
                } else {
                    self.handle_request(&method, &params)
                };
                Some(match result {
                    Ok(r) => protocol::response(id, r),
                    Err(e) => protocol::error_response(id, e),
                })
            }
            Ok(_) => None,
            Err(e) => Some(protocol::error_response(Value::Null, e)),
        }
    }
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
        assert!(s
            .handle_message(protocol::notification("notifications/initialized", None))
            .await
            .is_none());
    }
}
