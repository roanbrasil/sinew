//! Streamable HTTP client transport (feature `http`).
//!
//! Every message is POSTed to the endpoint. The reply is either one JSON
//! body or an SSE stream whose `data:` events carry JSON-RPC messages; both
//! are fed to the client's dispatcher. The server-initiated GET stream is not
//! opened, since the tool surface does not need it.

use std::sync::Mutex;

use async_trait::async_trait;
use reqwest::header::{HeaderValue, ACCEPT, CONTENT_TYPE};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::{client::Outbound, error::Result, McpError};

const SESSION_HEADER: &str = "mcp-session-id";
const VERSION_HEADER: &str = "mcp-protocol-version";

pub(crate) struct HttpOut {
    http: reqwest::Client,
    url: String,
    session: Mutex<Option<String>>,
    version: Mutex<Option<String>>,
    incoming: mpsc::UnboundedSender<Value>,
    extra: Vec<(String, String)>,
}

impl HttpOut {
    pub(crate) fn new(
        url: String,
        incoming: mpsc::UnboundedSender<Value>,
        extra: Vec<(String, String)>,
    ) -> Self {
        Self {
            http: sinew_core::providers::shared_client(),
            url,
            session: Mutex::new(None),
            version: Mutex::new(None),
            incoming,
            extra,
        }
    }

    fn with_headers(&self, mut req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (k, v) in &self.extra {
            req = req.header(k.as_str(), v.as_str());
        }
        if let Some(s) = self.session.lock().unwrap().clone() {
            req = req.header(SESSION_HEADER, s);
        }
        if let Some(v) = self.version.lock().unwrap().clone() {
            req = req.header(VERSION_HEADER, v);
        }
        req
    }
}

/// Extracts the JSON payloads of an SSE body.
pub(crate) fn sse_messages(body: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut data = String::new();
    for line in body.lines().chain(std::iter::once("")) {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            if !data.is_empty() {
                if let Ok(v) = serde_json::from_str(&data) {
                    out.push(v);
                }
                data.clear();
            }
        } else if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    out
}

#[async_trait]
impl Outbound for HttpOut {
    async fn send(&self, msg: Value) -> Result<()> {
        let req = self
            .with_headers(self.http.post(&self.url))
            .header(ACCEPT, "application/json, text/event-stream")
            .header(CONTENT_TYPE, "application/json")
            .body(serde_json::to_vec(&msg)?);
        let resp = req.send().await?;
        if let Some(s) = resp
            .headers()
            .get(SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
        {
            *self.session.lock().unwrap() = Some(s.to_string());
        }
        let status = resp.status();
        if status == reqwest::StatusCode::ACCEPTED {
            return Ok(());
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(McpError::Protocol(format!("HTTP {status}: {body}")));
        }
        let is_sse = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v: &HeaderValue| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));
        let body = resp.text().await?;
        let messages = if is_sse {
            sse_messages(&body)
        } else if body.trim().is_empty() {
            Vec::new()
        } else {
            match serde_json::from_str::<Value>(&body)? {
                Value::Array(items) => items,
                v => vec![v],
            }
        };
        for m in messages {
            let _ = self.incoming.send(m);
        }
        Ok(())
    }

    fn set_protocol_version(&self, version: &str) {
        *self.version.lock().unwrap() = Some(version.to_string());
    }

    async fn close(&self) {
        let has_session = self.session.lock().unwrap().is_some();
        if has_session {
            let _ = self.with_headers(self.http.delete(&self.url)).send().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_sse_events() {
        let body = "event: message\r\ndata: {\"a\":1}\r\n\r\n: comment\ndata: {\"b\":\ndata: 2}\n";
        assert_eq!(sse_messages(body), vec![json!({"a": 1}), json!({"b": 2})]);
    }
}
