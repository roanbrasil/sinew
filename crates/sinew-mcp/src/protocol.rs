//! JSON-RPC 2.0 framing and the subset of MCP message types Sinew uses,
//! plus the mapping between MCP tool results and Sinew tool values.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sinew_core::ToolSpec;

/// Protocol version this crate speaks by default.
pub const LATEST_PROTOCOL_VERSION: &str = "2025-06-18";

/// Every protocol version this crate accepts, newest first. The tool surface
/// we use (`initialize`, `tools/list`, `tools/call`, `ping`) is unchanged
/// across these revisions.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Standard JSON-RPC 2.0 error codes.
pub mod error_code {
    pub const PARSE_ERROR: i64 = -32700;
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
    pub const INTERNAL_ERROR: i64 = -32603;
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

#[cfg(test)]
mod tests {
    use super::*;

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
