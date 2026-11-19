//! JSON-RPC 2.0 framing and the subset of MCP message types Sinew uses
//! (tools, resources, prompts), plus the mapping between MCP results and
//! Sinew values.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sinew_core::{ContentPart, Message, Role, ToolSpec};

/// Protocol version this crate speaks by default.
pub const LATEST_PROTOCOL_VERSION: &str = "2025-06-18";

/// Every protocol version this crate accepts, newest first. The messages we
/// use (`initialize`, `ping`, tools, resources and prompts) are compatible
/// across these revisions; fields added later (such as `title`) are optional.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Standard JSON-RPC 2.0 error codes.
pub mod error_code {
    pub const PARSE_ERROR: i64 = -32700;
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
    pub const INTERNAL_ERROR: i64 = -32603;
    /// MCP: `resources/read` (or `resources/subscribe`) for an unknown URI.
    pub const RESOURCE_NOT_FOUND: i64 = -32002;
    /// Sinew extension: the caller's identity proof is missing or does not
    /// verify, or its identity does not allow the tool (see
    /// `McpServer::with_identity_verifier`).
    pub const IDENTITY_REJECTED: i64 = -32040;
}

/// A JSON-RPC error object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// `-32602`: the request parameters are invalid (missing prompt argument,
    /// unknown prompt name, bad cursor).
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self::new(error_code::INVALID_PARAMS, message)
    }

    /// `-32603`: the handler failed.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(error_code::INTERNAL_ERROR, message)
    }

    /// `-32002` with `data: {"uri": ...}`, the MCP error for an unknown resource.
    pub fn resource_not_found(uri: &str) -> Self {
        Self {
            code: error_code::RESOURCE_NOT_FOUND,
            message: format!("resource not found: {uri}"),
            data: Some(json!({ "uri": uri })),
        }
    }
}

/// Builds a request object.
pub fn request(id: u64, method: &str, params: Option<Value>) -> Value {
    let mut m = Map::new();
    m.insert("jsonrpc".into(), json!("2.0"));
    m.insert("id".into(), json!(id));
    m.insert("method".into(), json!(method));
    if let Some(p) = params {
        m.insert("params".into(), p);
    }
    Value::Object(m)
}

/// Builds a notification object (no id, no response expected).
pub fn notification(method: &str, params: Option<Value>) -> Value {
    let mut m = Map::new();
    m.insert("jsonrpc".into(), json!("2.0"));
    m.insert("method".into(), json!(method));
    if let Some(p) = params {
        m.insert("params".into(), p);
    }
    Value::Object(m)
}

/// Builds a success response.
pub fn response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// Builds an error response. `id` is `null` when the request id is unknown.
pub fn error_response(id: Value, error: RpcError) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": error })
}

/// What an incoming JSON-RPC message is.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Request {
        id: Value,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
    Response {
        id: Value,
        result: Result<Value, RpcError>,
    },
}

/// Classifies a JSON value as a JSON-RPC message.
pub fn classify(msg: Value) -> Result<Incoming, RpcError> {
    let Value::Object(mut m) = msg else {
        return Err(RpcError::new(
            error_code::INVALID_REQUEST,
            "message must be a JSON object (batches are not supported)",
        ));
    };
    if m.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(RpcError::new(
            error_code::INVALID_REQUEST,
            "`jsonrpc` must be \"2.0\"",
        ));
    }
    let id = m.remove("id");
    if let Some(method) = m.remove("method") {
        let Value::String(method) = method else {
            return Err(RpcError::new(
                error_code::INVALID_REQUEST,
                "`method` must be a string",
            ));
        };
        let params = m.remove("params").unwrap_or(Value::Null);
        return Ok(match id {
            Some(id) if id.is_string() || id.is_number() => {
                Incoming::Request { id, method, params }
            }
            Some(_) => {
                return Err(RpcError::new(
                    error_code::INVALID_REQUEST,
                    "`id` must be a string or a number",
                ))
            }
            None => Incoming::Notification { method, params },
        });
    }
    let id = id.ok_or_else(|| RpcError::new(error_code::INVALID_REQUEST, "missing `method`"))?;
    if let Some(err) = m.remove("error") {
        let err = serde_json::from_value(err)
            .unwrap_or_else(|_| RpcError::new(error_code::INTERNAL_ERROR, "malformed error"));
        return Ok(Incoming::Response {
            id,
            result: Err(err),
        });
    }
    match m.remove("result") {
        Some(result) => Ok(Incoming::Response {
            id,
            result: Ok(result),
        }),
        None => Err(RpcError::new(
            error_code::INVALID_REQUEST,
            "response has neither `result` nor `error`",
        )),
    }
}

/// Name and version of a client or server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Implementation {
    pub name: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// The server's answer to `initialize`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    #[serde(default)]
    pub capabilities: Value,
    pub server_info: Implementation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The result's `_meta` (a Sinew server puts its signed identity here,
    /// see `McpServer::with_identity`).
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A tool as described by `tools/list`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolInfo {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
}

impl McpToolInfo {
    /// Converts to a Sinew tool spec. A missing or non-object input schema
    /// becomes `{"type": "object"}`, which every provider accepts.
    pub fn to_spec(&self) -> ToolSpec {
        let parameters = if self.input_schema.is_object() {
            self.input_schema.clone()
        } else {
            json!({ "type": "object" })
        };
        ToolSpec {
            name: self.name.clone(),
            description: self
                .description
                .clone()
                .or_else(|| self.title.clone())
                .unwrap_or_default(),
            parameters,
        }
    }

    /// Builds the MCP description of a Sinew tool.
    pub fn from_spec(spec: &ToolSpec) -> Self {
        let mut schema = if spec.parameters.is_object() {
            spec.parameters.clone()
        } else {
            json!({})
        };
        // MCP requires `inputSchema.type == "object"`.
        if schema.get("type").is_none() {
            schema["type"] = json!("object");
        }
        Self {
            name: spec.name.clone(),
            title: None,
            description: Some(spec.description.clone()),
            input_schema: schema,
            output_schema: None,
            annotations: None,
        }
    }
}

/// One item of a tool result's `content` array. Unknown kinds are kept raw.
#[derive(Debug, Clone, PartialEq)]
pub enum Content {
    Text(String),
    Image { mime_type: String, data: String },
    Audio { mime_type: String, data: String },
    ResourceLink { uri: String, name: Option<String> },
    Resource(Value),
    Other(Value),
}

impl Content {
    pub fn text(s: impl Into<String>) -> Self {
        Content::Text(s.into())
    }

    pub fn from_json(v: &Value) -> Self {
        let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        match v.get("type").and_then(Value::as_str) {
            Some("text") => Content::Text(s("text")),
            Some("image") => Content::Image {
                mime_type: s("mimeType"),
                data: s("data"),
            },
            Some("audio") => Content::Audio {
                mime_type: s("mimeType"),
                data: s("data"),
            },
            Some("resource_link") => Content::ResourceLink {
                uri: s("uri"),
                name: v.get("name").and_then(Value::as_str).map(str::to_string),
            },
            Some("resource") => Content::Resource(v.get("resource").cloned().unwrap_or_default()),
            _ => Content::Other(v.clone()),
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            Content::Text(t) => json!({ "type": "text", "text": t }),
            Content::Image { mime_type, data } => {
                json!({ "type": "image", "mimeType": mime_type, "data": data })
            }
            Content::Audio { mime_type, data } => {
                json!({ "type": "audio", "mimeType": mime_type, "data": data })
            }
            Content::ResourceLink { uri, name } => {
                json!({ "type": "resource_link", "uri": uri, "name": name.clone().unwrap_or_else(|| uri.clone()) })
            }
            Content::Resource(r) => json!({ "type": "resource", "resource": r }),
            Content::Other(v) => v.clone(),
        }
    }

    /// Text a chat model can read. Binary payloads are summarised, not inlined.
    pub fn to_model_text(&self) -> String {
        match self {
            Content::Text(t) => t.clone(),
            Content::Image { mime_type, data } => {
                format!("[image {mime_type}, {} bytes base64]", data.len())
            }
            Content::Audio { mime_type, data } => {
                format!("[audio {mime_type}, {} bytes base64]", data.len())
            }
            Content::ResourceLink { uri, name } => match name {
                Some(n) if n != uri => format!("[resource {n}: {uri}]"),
                _ => format!("[resource {uri}]"),
            },
            Content::Resource(r) => {
                if let Some(t) = r.get("text").and_then(Value::as_str) {
                    t.to_string()
                } else {
                    let uri = r.get("uri").and_then(Value::as_str).unwrap_or("?");
                    let mime = r
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .unwrap_or("application/octet-stream");
                    format!("[resource {uri}, {mime}]")
                }
            }
            Content::Other(v) => v.to_string(),
        }
    }
}

/// Result of `tools/call`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CallToolResult {
    pub content: Vec<Content>,
    pub structured_content: Option<Value>,
    pub is_error: bool,
}

impl CallToolResult {
    pub fn from_json(v: &Value) -> Self {
        Self {
            content: v
                .get("content")
                .and_then(Value::as_array)
                .map(|items| items.iter().map(Content::from_json).collect())
                .unwrap_or_default(),
            structured_content: v.get("structuredContent").cloned().filter(|s| !s.is_null()),
            is_error: v.get("isError").and_then(Value::as_bool).unwrap_or(false),
        }
    }

    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert(
            "content".into(),
            Value::Array(self.content.iter().map(Content::to_json).collect()),
        );
        if let Some(s) = &self.structured_content {
            m.insert("structuredContent".into(), s.clone());
        }
        m.insert("isError".into(), json!(self.is_error));
        Value::Object(m)
    }

    /// All content as one string for a model: items joined by newlines.
    /// Falls back to `structuredContent` when there is no content.
    pub fn text(&self) -> String {
        if self.content.is_empty() {
            return self
                .structured_content
                .as_ref()
                .map(Value::to_string)
                .unwrap_or_default();
        }
        self.content
            .iter()
            .map(Content::to_model_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The Sinew view of a result: the text on success, the text as an error
    /// message when `isError` is set. A result with only structured content
    /// returns that JSON value unchanged.
    pub fn into_tool_value(self) -> Result<Value, String> {
        if self.is_error {
            let msg = self.text();
            return Err(if msg.is_empty() {
                "tool reported an error".into()
            } else {
                msg
            });
        }
        if self.content.is_empty() {
            if let Some(s) = self.structured_content {
                return Ok(s);
            }
        }
        Ok(Value::String(self.text()))
    }

    /// Builds the MCP result for a Sinew tool's output. Strings become one
    /// text item; other JSON is serialized as text, and objects are also sent
    /// as `structuredContent`.
    pub fn from_tool_value(value: Value) -> Self {
        match value {
            Value::String(s) => Self {
                content: vec![Content::Text(s)],
                ..Default::default()
            },
            Value::Object(_) => Self {
                content: vec![Content::Text(value.to_string())],
                structured_content: Some(value),
                is_error: false,
            },
            other => Self {
                content: vec![Content::Text(other.to_string())],
                ..Default::default()
            },
        }
    }

    /// A failed tool call reported in-band, so the model can see and recover.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![Content::Text(message.into())],
            structured_content: None,
            is_error: true,
        }
    }
}

/// A resource as described by `resources/list`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Size of the raw content in bytes, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
}

impl McpResource {
    pub fn new(uri: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            name: name.into(),
            title: None,
            description: None,
            mime_type: None,
            size: None,
            annotations: None,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    pub fn with_size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }
}

/// A parameterized resource as described by `resources/templates/list`.
/// `uri_template` is an RFC 6570 URI template such as `notes://{title}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpResourceTemplate {
    pub uri_template: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
}

impl McpResourceTemplate {
    pub fn new(uri_template: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            uri_template: uri_template.into(),
            name: name.into(),
            title: None,
            description: None,
            mime_type: None,
            annotations: None,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }
}

/// One item of a `resources/read` result: either `text` or `blob` (base64)
/// is set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceContents {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64-encoded binary content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

impl ResourceContents {
    /// Text contents.
    pub fn text(uri: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            mime_type: None,
            text: Some(text.into()),
            blob: None,
        }
    }

    /// Binary contents, already base64-encoded.
    pub fn blob(
        uri: impl Into<String>,
        mime_type: impl Into<String>,
        base64: impl Into<String>,
    ) -> Self {
        Self {
            uri: uri.into(),
            mime_type: Some(mime_type.into()),
            text: None,
            blob: Some(base64.into()),
        }
    }

    /// Binary contents from raw bytes (encoded as base64).
    pub fn blob_bytes(uri: impl Into<String>, mime_type: impl Into<String>, bytes: &[u8]) -> Self {
        Self::blob(uri, mime_type, sinew_core::message::base64_encode(bytes))
    }

    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    /// Whether this is binary (`blob`) content.
    pub fn is_blob(&self) -> bool {
        self.text.is_none() && self.blob.is_some()
    }

    /// Converts text contents into a `sinew_rag` [`Document`](sinew_rag::Document)
    /// whose id and `source` metadata are the URI, plus `mime_type` metadata
    /// when known. Blob contents return `None`. Requires the `rag` feature.
    #[cfg(feature = "rag")]
    pub fn to_document(&self) -> Option<sinew_rag::Document> {
        let text = self.text.as_ref()?;
        let mut doc = sinew_rag::Document::new(text.clone())
            .with_id(self.uri.clone())
            .with_metadata("source", self.uri.clone());
        if let Some(m) = &self.mime_type {
            doc = doc.with_metadata("mime_type", m.clone());
        }
        Some(doc)
    }
}

/// An argument a prompt accepts. Argument values are always strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptArgument {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// A prompt as described by `prompts/list`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpPrompt {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arguments: Vec<PromptArgument>,
}

impl McpPrompt {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            title: None,
            description: None,
            arguments: Vec::new(),
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adds an argument.
    pub fn with_argument(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        self.arguments.push(PromptArgument {
            name: name.into(),
            title: None,
            description: Some(description.into()),
            required,
        });
        self
    }
}

/// The role of a prompt message. MCP prompts only carry user and assistant
/// turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptRole {
    User,
    Assistant,
}

impl From<PromptRole> for Role {
    fn from(r: PromptRole) -> Self {
        match r {
            PromptRole::User => Role::User,
            PromptRole::Assistant => Role::Assistant,
        }
    }
}

/// One message of a `prompts/get` result.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptMessage {
    pub role: PromptRole,
    pub content: Content,
}

impl PromptMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: PromptRole::User,
            content: Content::Text(text.into()),
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: PromptRole::Assistant,
            content: Content::Text(text.into()),
        }
    }

    /// The message as a Sinew [`Message`]. An image in a user message becomes
    /// an image part; other content becomes text as in
    /// [`Content::to_model_text`].
    pub fn to_message(&self) -> Message {
        match (self.role, &self.content) {
            (PromptRole::User, Content::Image { mime_type, data }) => Message::user_with_parts(
                "",
                [ContentPart::image_base64(mime_type.clone(), data.clone())],
            ),
            (PromptRole::User, c) => Message::user(c.to_model_text()),
            (PromptRole::Assistant, c) => Message::assistant(c.to_model_text()),
        }
    }
}

/// Result of `prompts/get`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GetPromptResult {
    pub description: Option<String>,
    pub messages: Vec<PromptMessage>,
}

impl GetPromptResult {
    /// A result with the given messages.
    pub fn new(messages: impl IntoIterator<Item = PromptMessage>) -> Self {
        Self {
            description: None,
            messages: messages.into_iter().collect(),
        }
    }

    /// A result with one user text message.
    pub fn user(text: impl Into<String>) -> Self {
        Self::new([PromptMessage::user(text)])
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Parses a `prompts/get` result.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let items = v
            .get("messages")
            .and_then(Value::as_array)
            .ok_or("prompts/get result has no `messages`")?;
        let messages = items
            .iter()
            .map(|m| {
                let role = serde_json::from_value(m.get("role").cloned().unwrap_or_default())
                    .map_err(|e| format!("bad prompt message role: {e}"))?;
                let content = Content::from_json(m.get("content").unwrap_or(&Value::Null));
                Ok(PromptMessage { role, content })
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            description: v
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string),
            messages,
        })
    }

    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        if let Some(d) = &self.description {
            m.insert("description".into(), json!(d));
        }
        m.insert(
            "messages".into(),
            Value::Array(
                self.messages
                    .iter()
                    .map(|p| json!({ "role": p.role, "content": p.content.to_json() }))
                    .collect(),
            ),
        );
        Value::Object(m)
    }

    /// The messages as Sinew [`Message`]s, ready for a chat model.
    pub fn to_messages(&self) -> Vec<Message> {
        self.messages
            .iter()
            .map(PromptMessage::to_message)
            .collect()
    }
}

/// A notification sent by the server, as delivered by
/// [`McpClient::notifications`](crate::McpClient::notifications).
#[derive(Debug, Clone, PartialEq)]
pub enum McpNotification {
    /// `notifications/resources/updated`: a subscribed resource changed.
    ResourceUpdated { uri: String },
    /// `notifications/resources/list_changed`.
    ResourceListChanged,
    /// `notifications/prompts/list_changed`.
    PromptListChanged,
    /// `notifications/tools/list_changed`.
    ToolListChanged,
    /// Any other notification (logging, progress, ...), kept raw.
    Other { method: String, params: Value },
}

impl McpNotification {
    /// Classifies a notification by method name.
    pub fn from_message(method: &str, params: &Value) -> Self {
        let other = || McpNotification::Other {
            method: method.into(),
            params: params.clone(),
        };
        match method {
            "notifications/resources/updated" => match params.get("uri").and_then(Value::as_str) {
                Some(uri) => McpNotification::ResourceUpdated { uri: uri.into() },
                None => other(),
            },
            "notifications/resources/list_changed" => McpNotification::ResourceListChanged,
            "notifications/prompts/list_changed" => McpNotification::PromptListChanged,
            "notifications/tools/list_changed" => McpNotification::ToolListChanged,
            _ => other(),
        }
    }
}

/// A matcher for the RFC 6570 URI templates resource templates use.
///
/// Supported expressions: `{name}` (one value without `/`, `?` or `#`),
/// `{+name}` and `{#name}` (any characters, including `/`). Other operators
/// (`{?x}`, `{/x}`, `{.x}`, ...) and multi-variable expressions are accepted
/// but match one segment per variable name, without the operator prefix.
/// Matched values are percent-decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UriTemplate {
    template: String,
    parts: Vec<TemplatePart>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TemplatePart {
    Literal(String),
    Var { name: String, greedy: bool },
}

impl UriTemplate {
    /// Parses `template`. Fails on an unclosed `{` or an empty expression.
    pub fn new(template: impl Into<String>) -> Result<Self, String> {
        let template = template.into();
        let mut parts = Vec::new();
        let mut rest = template.as_str();
        while let Some(open) = rest.find('{') {
            if open > 0 {
                parts.push(TemplatePart::Literal(rest[..open].to_string()));
            }
            let close = rest[open..]
                .find('}')
                .ok_or_else(|| format!("unclosed `{{` in URI template `{template}`"))?;
            let expr = &rest[open + 1..open + close];
            let (greedy, names) = match expr.chars().next() {
                Some('+') | Some('#') => (true, &expr[1..]),
                Some('?' | '&' | '/' | '.' | ';') => (false, &expr[1..]),
                _ => (false, expr),
            };
            let names: Vec<&str> = names
                .split(',')
                .map(|n| n.trim().trim_end_matches('*'))
                .filter(|n| !n.is_empty())
                .collect();
            if names.is_empty() {
                return Err(format!("empty expression in URI template `{template}`"));
            }
            for (i, n) in names.iter().enumerate() {
                if i > 0 {
                    parts.push(TemplatePart::Literal(",".into()));
                }
                parts.push(TemplatePart::Var {
                    name: (*n).to_string(),
                    greedy,
                });
            }
            rest = &rest[open + close + 1..];
        }
        if !rest.is_empty() {
            parts.push(TemplatePart::Literal(rest.to_string()));
        }
        Ok(Self { template, parts })
    }

    /// The template string.
    pub fn as_str(&self) -> &str {
        &self.template
    }

    /// Variable names in order.
    pub fn variables(&self) -> Vec<&str> {
        self.parts
            .iter()
            .filter_map(|p| match p {
                TemplatePart::Var { name, .. } => Some(name.as_str()),
                TemplatePart::Literal(_) => None,
            })
            .collect()
    }

    /// Matches `uri` against the template and returns the variable values,
    /// or `None` if it does not match.
    pub fn matches(&self, uri: &str) -> Option<HashMap<String, String>> {
        let mut vars = HashMap::new();
        let mut failed = std::collections::HashSet::new();
        if match_parts(&self.parts, 0, uri, 0, &mut vars, &mut failed) {
            Some(vars)
        } else {
            None
        }
    }

    /// Expands the template with `vars` (values are inserted as given;
    /// missing variables expand to an empty string).
    pub fn expand(&self, vars: &HashMap<String, String>) -> String {
        let mut out = String::new();
        for p in &self.parts {
            match p {
                TemplatePart::Literal(l) => out.push_str(l),
                TemplatePart::Var { name, .. } => {
                    out.push_str(vars.get(name).map(String::as_str).unwrap_or(""))
                }
            }
        }
        out
    }
}

/// Backtracking matcher over `(part, offset)` states. States that already
/// failed are remembered, so the work is bounded by
/// `parts × positions × candidates` instead of growing exponentially with the
/// number of variables.
fn match_parts(
    parts: &[TemplatePart],
    pi: usize,
    uri: &str,
    at: usize,
    vars: &mut HashMap<String, String>,
    failed: &mut std::collections::HashSet<(usize, usize)>,
) -> bool {
    if failed.contains(&(pi, at)) {
        return false;
    }
    let s = &uri[at..];
    let ok = match parts.get(pi) {
        None => s.is_empty(),
        Some(TemplatePart::Literal(l)) => {
            s.starts_with(l.as_str()) && match_parts(parts, pi + 1, uri, at + l.len(), vars, failed)
        }
        Some(TemplatePart::Var { name, greedy }) => {
            // Try the longest candidate first; backtrack on failure.
            let limit = if *greedy {
                s.len()
            } else {
                s.find(['/', '?', '#']).unwrap_or(s.len())
            };
            let mut found = false;
            for end in (1..=limit).rev() {
                if !s.is_char_boundary(end) {
                    continue;
                }
                if match_parts(parts, pi + 1, uri, at + end, vars, failed) {
                    vars.insert(name.clone(), percent_decode(&s[..end]));
                    found = true;
                    break;
                }
            }
            found
        }
    };
    if !ok {
        failed.insert((pi, at));
    }
    ok
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // Exactly two hex digits; `from_str_radix` alone would also accept a
        // sign (`%+1`).
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && bytes[i + 1].is_ascii_hexdigit()
            && bytes[i + 2].is_ascii_hexdigit()
        {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("00");
            if let Ok(b) = u8::from_str_radix(hex, 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_decode_requires_two_hex_digits() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("%+1"), "%+1");
        assert_eq!(percent_decode("%-1x"), "%-1x");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%4"), "%4");
    }

    #[test]
    fn many_variables_match_in_bounded_time() {
        // 24 one-segment variables separated by `-`, against a URI with no
        // match at the end: exponential without memoization.
        let template: String = (0..24)
            .map(|i| format!("{{v{i}}}"))
            .collect::<Vec<_>>()
            .join("-");
        let t = UriTemplate::new(format!("x://{template}/end")).unwrap();
        let uri = format!("x://{}/nope", vec!["a-b"; 24].join("-"));
        let start = std::time::Instant::now();
        assert!(t.matches(&uri).is_none());
        assert!(
            start.elapsed() < std::time::Duration::from_secs(2),
            "{:?}",
            start.elapsed()
        );
        let ok = format!(
            "x://{}/end",
            (0..24).map(|i| i.to_string()).collect::<Vec<_>>().join("-")
        );
        assert_eq!(t.matches(&ok).unwrap()["v23"], "23");
    }

    #[test]
    fn uri_templates_match_and_expand() {
        let t = UriTemplate::new("notes://{title}").unwrap();
        assert_eq!(t.variables(), ["title"]);
        assert_eq!(t.matches("notes://standup").unwrap()["title"], "standup");
        assert_eq!(t.matches("notes://a%20b").unwrap()["title"], "a b");
        assert!(t.matches("notes://a/b").is_none());
        assert!(t.matches("notes://").is_none());
        assert!(t.matches("other://x").is_none());
        let g = UriTemplate::new("file:///{+path}").unwrap();
        assert_eq!(g.matches("file:///a/b.txt").unwrap()["path"], "a/b.txt");
        let two = UriTemplate::new("db://{table}/rows/{id}").unwrap();
        let m = two.matches("db://users/rows/42").unwrap();
        assert_eq!((m["table"].as_str(), m["id"].as_str()), ("users", "42"));
        assert_eq!(two.expand(&m), "db://users/rows/42");
        assert!(UriTemplate::new("x://{oops").is_err());
        assert!(UriTemplate::new("x://{}").is_err());
    }

    #[test]
    fn prompt_results_round_trip() {
        let r = GetPromptResult::new([
            PromptMessage::user("Review this"),
            PromptMessage::assistant("Sure"),
            PromptMessage {
                role: PromptRole::User,
                content: Content::Image {
                    mime_type: "image/png".into(),
                    data: "AAAA".into(),
                },
            },
        ])
        .with_description("code review");
        let wire = r.to_json();
        assert_eq!(wire["messages"][0]["role"], "user");
        assert_eq!(wire["messages"][1]["content"]["text"], "Sure");
        assert_eq!(GetPromptResult::from_json(&wire).unwrap(), r);
        let msgs = r.to_messages();
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert!(msgs[2].has_media());
        assert!(GetPromptResult::from_json(&json!({})).is_err());
        assert!(GetPromptResult::from_json(&json!({"messages": [{"role": "system"}]})).is_err());
    }

    #[test]
    fn resource_types_use_camel_case() {
        let r = McpResource::new("a://b", "b").with_mime_type("text/plain");
        let wire = serde_json::to_value(&r).unwrap();
        assert_eq!(
            wire,
            json!({"uri": "a://b", "name": "b", "mimeType": "text/plain"})
        );
        let t = serde_json::to_value(McpResourceTemplate::new("a://{x}", "x")).unwrap();
        assert_eq!(t["uriTemplate"], "a://{x}");
        let c = ResourceContents::blob_bytes("a://b", "application/octet-stream", b"hi");
        assert_eq!(c.blob.as_deref(), Some("aGk="));
        assert!(c.is_blob());
        let p = serde_json::to_value(McpPrompt::new("p").with_argument("x", "d", true)).unwrap();
        assert_eq!(
            p["arguments"][0],
            json!({"name": "x", "description": "d", "required": true})
        );
        assert_eq!(
            McpNotification::from_message(
                "notifications/resources/updated",
                &json!({"uri": "a://b"})
            ),
            McpNotification::ResourceUpdated {
                uri: "a://b".into()
            }
        );
    }

    #[test]
    fn classifies_messages() {
        let req = classify(request(1, "ping", None)).unwrap();
        assert!(matches!(req, Incoming::Request { ref method, .. } if method == "ping"));
        let n = classify(notification("notifications/initialized", None)).unwrap();
        assert!(matches!(n, Incoming::Notification { .. }));
        let ok = classify(response(json!(3), json!({}))).unwrap();
        assert_eq!(
            ok,
            Incoming::Response {
                id: json!(3),
                result: Ok(json!({}))
            }
        );
        let err = classify(error_response(
            json!("a"),
            RpcError::new(error_code::METHOD_NOT_FOUND, "nope"),
        ))
        .unwrap();
        assert!(
            matches!(err, Incoming::Response { result: Err(e), .. } if e.code == error_code::METHOD_NOT_FOUND)
        );
    }

    #[test]
    fn rejects_invalid_messages() {
        for bad in [
            json!([]),
            json!({"jsonrpc": "1.0", "id": 1, "method": "x"}),
            json!({"jsonrpc": "2.0", "id": {"x": 1}, "method": "x"}),
            json!({"jsonrpc": "2.0", "id": 1}),
            json!({"jsonrpc": "2.0"}),
        ] {
            assert_eq!(classify(bad).unwrap_err().code, error_code::INVALID_REQUEST);
        }
    }

    #[test]
    fn maps_text_and_fallback_content() {
        let v = json!({
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "mimeType": "image/png", "data": "AAAA"},
                {"type": "resource_link", "uri": "file:///a.txt", "name": "a.txt"},
                {"type": "resource", "resource": {"uri": "file:///b", "text": "inline"}},
                {"type": "resource", "resource": {"uri": "file:///c", "blob": "xx", "mimeType": "application/pdf"}},
                {"type": "something-new", "x": 1}
            ]
        });
        let r = CallToolResult::from_json(&v);
        assert!(!r.is_error);
        assert_eq!(
            r.text(),
            "hello\n[image image/png, 4 bytes base64]\n[resource a.txt: file:///a.txt]\ninline\n[resource file:///c, application/pdf]\n{\"type\":\"something-new\",\"x\":1}"
        );
        assert_eq!(
            r.content[5].to_json(),
            json!({"type": "something-new", "x": 1})
        );
    }

    #[test]
    fn is_error_becomes_err() {
        let r = CallToolResult::from_json(&json!({
            "content": [{"type": "text", "text": "boom"}], "isError": true
        }));
        assert_eq!(r.into_tool_value(), Err("boom".into()));
        let empty = CallToolResult::from_json(&json!({"content": [], "isError": true}));
        assert_eq!(
            empty.into_tool_value(),
            Err("tool reported an error".into())
        );
    }

    #[test]
    fn structured_only_result_keeps_json() {
        let r = CallToolResult::from_json(&json!({"content": [], "structuredContent": {"n": 1}}));
        assert_eq!(r.into_tool_value(), Ok(json!({"n": 1})));
    }

    #[test]
    fn tool_values_round_trip() {
        let s = CallToolResult::from_tool_value(json!("42"));
        assert_eq!(
            s.to_json(),
            json!({"content": [{"type": "text", "text": "42"}], "isError": false})
        );
        let n = CallToolResult::from_tool_value(json!(4.5));
        assert_eq!(n.text(), "4.5");
        let o = CallToolResult::from_tool_value(json!({"a": 1}));
        assert_eq!(o.structured_content, Some(json!({"a": 1})));
        let back = CallToolResult::from_json(&o.to_json());
        assert_eq!(back, o);
        assert_eq!(back.into_tool_value(), Ok(json!("{\"a\":1}")));
    }

    #[test]
    fn spec_conversion_forces_object_schema() {
        let spec = ToolSpec {
            name: "t".into(),
            description: "d".into(),
            parameters: json!({"properties": {}}),
        };
        let info = McpToolInfo::from_spec(&spec);
        assert_eq!(info.input_schema["type"], "object");
        let wire = serde_json::to_value(&info).unwrap();
        assert!(wire.get("inputSchema").is_some());
        let parsed: McpToolInfo =
            serde_json::from_value(json!({"name": "x", "title": "X tool", "inputSchema": null}))
                .unwrap();
        let spec = parsed.to_spec();
        assert_eq!(spec.parameters, json!({"type": "object"}));
        assert_eq!(spec.description, "X tool");
    }
}
