//! JSON-RPC 2.0 framing for the A2A binding: method names, error codes and
//! the error object.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A2A JSON-RPC method names.
pub mod method {
    pub const MESSAGE_SEND: &str = "message/send";
    pub const MESSAGE_STREAM: &str = "message/stream";
    pub const TASKS_GET: &str = "tasks/get";
    pub const TASKS_CANCEL: &str = "tasks/cancel";
    pub const TASKS_RESUBSCRIBE: &str = "tasks/resubscribe";
    pub const PUSH_CONFIG_SET: &str = "tasks/pushNotificationConfig/set";
    pub const PUSH_CONFIG_GET: &str = "tasks/pushNotificationConfig/get";
    pub const PUSH_CONFIG_LIST: &str = "tasks/pushNotificationConfig/list";
    pub const PUSH_CONFIG_DELETE: &str = "tasks/pushNotificationConfig/delete";
    pub const GET_EXTENDED_CARD: &str = "agent/getAuthenticatedExtendedCard";
}

/// Standard JSON-RPC codes and the A2A-specific ones (spec section 8).
pub mod error_code {
    pub const PARSE_ERROR: i64 = -32700;
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
    pub const INTERNAL_ERROR: i64 = -32603;
    pub const TASK_NOT_FOUND: i64 = -32001;
    pub const TASK_NOT_CANCELABLE: i64 = -32002;
    pub const PUSH_NOTIFICATION_NOT_SUPPORTED: i64 = -32003;
    pub const UNSUPPORTED_OPERATION: i64 = -32004;
    pub const CONTENT_TYPE_NOT_SUPPORTED: i64 = -32005;
    pub const INVALID_AGENT_RESPONSE: i64 = -32006;
    pub const AUTHENTICATED_EXTENDED_CARD_NOT_CONFIGURED: i64 = -32007;
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

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    pub fn parse_error(detail: impl std::fmt::Display) -> Self {
        Self::new(
            error_code::PARSE_ERROR,
            format!("Invalid JSON payload: {detail}"),
        )
    }
    pub fn invalid_request(detail: impl Into<String>) -> Self {
        Self::new(error_code::INVALID_REQUEST, detail)
    }
    pub fn method_not_found(method: &str) -> Self {
        Self::new(
            error_code::METHOD_NOT_FOUND,
            format!("Method not found: {method}"),
        )
    }
    pub fn invalid_params(detail: impl std::fmt::Display) -> Self {
        Self::new(
            error_code::INVALID_PARAMS,
            format!("Invalid method parameters: {detail}"),
        )
    }
    pub fn internal(detail: impl std::fmt::Display) -> Self {
        Self::new(
            error_code::INTERNAL_ERROR,
            format!("Internal error: {detail}"),
        )
    }
    pub fn task_not_found(id: &str) -> Self {
        Self::new(error_code::TASK_NOT_FOUND, format!("Task not found: {id}"))
    }
    pub fn task_not_cancelable(id: &str) -> Self {
        Self::new(
            error_code::TASK_NOT_CANCELABLE,
            format!("Task cannot be canceled: {id}"),
        )
    }
    pub fn push_not_supported() -> Self {
        Self::new(
            error_code::PUSH_NOTIFICATION_NOT_SUPPORTED,
            "Push Notification is not supported",
        )
    }
    pub fn unsupported(detail: impl Into<String>) -> Self {
        Self::new(error_code::UNSUPPORTED_OPERATION, detail)
    }
    pub fn extended_card_not_configured() -> Self {
        Self::new(
            error_code::AUTHENTICATED_EXTENDED_CARD_NOT_CONFIGURED,
            "Authenticated Extended Card not configured",
        )
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for RpcError {}

/// Builds a request object.
pub fn request(id: &Value, method: &str, params: &impl Serialize) -> serde_json::Result<Vec<u8>> {
    #[derive(Serialize)]
    struct Req<'a, P> {
        jsonrpc: &'static str,
        id: &'a Value,
        method: &'a str,
        params: &'a P,
    }
    serde_json::to_vec(&Req {
        jsonrpc: "2.0",
        id,
        method,
        params,
    })
}

/// Builds a success response body.
pub fn response(id: &Value, result: &impl Serialize) -> String {
    #[derive(Serialize)]
    struct Resp<'a, R> {
        jsonrpc: &'static str,
        id: &'a Value,
        result: &'a R,
    }
    serde_json::to_string(&Resp {
        jsonrpc: "2.0",
        id,
        result,
    })
    .unwrap_or_else(|e| error_response(id, &RpcError::internal(e)))
}

/// Builds a success response around an already-serialized `result`, without
/// re-encoding it (used to fan one streamed event out to many subscribers).
pub fn response_raw(id_json: &str, result_json: &str) -> String {
    let mut s = String::with_capacity(32 + id_json.len() + result_json.len());
    s.push_str(r#"{"jsonrpc":"2.0","id":"#);
    s.push_str(id_json);
    s.push_str(r#","result":"#);
    s.push_str(result_json);
    s.push('}');
    s
}

/// Builds an error response body. `id` is `null` when unknown.
pub fn error_response(id: &Value, error: &RpcError) -> String {
    json!({ "jsonrpc": "2.0", "id": id, "error": error }).to_string()
}

/// A decoded response: `Ok(result)` or the server's error.
pub fn decode_response(body: &[u8]) -> Result<Result<Value, RpcError>, serde_json::Error> {
    #[derive(Deserialize)]
    struct Resp {
        #[serde(default)]
        result: Option<Value>,
        #[serde(default)]
        error: Option<RpcError>,
    }
    let r: Resp = serde_json::from_slice(body)?;
    Ok(match (r.error, r.result) {
        (Some(e), _) => Err(e),
        (None, result) => Ok(result.unwrap_or(Value::Null)),
    })
}
