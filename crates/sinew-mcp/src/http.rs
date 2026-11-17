//! Streamable HTTP client transport (feature `http`).
//!
//! Every message is POSTed to the endpoint. The reply is either one JSON
//! body or an SSE stream whose `data:` events carry JSON-RPC messages; both
//! are fed to the client's dispatcher. The server-to-client GET stream is
//! opened on demand (resource subscriptions, notification receivers) and read
//! by a background task until the connection closes.

use std::sync::Mutex;

use async_trait::async_trait;
use reqwest::header::{HeaderValue, ACCEPT, CONTENT_TYPE};
use serde_json::Value;
use tokio::{sync::mpsc, task::JoinHandle};

use crate::{client::Outbound, error::Result, McpError};

const SESSION_HEADER: &str = "mcp-session-id";
const VERSION_HEADER: &str = "mcp-protocol-version";

/// State of the GET stream.
enum GetStream {
    /// Not requested yet.
    Closed,
    /// The server does not offer one (`405`).
    Unsupported,
    /// A task reads it.
    Open(JoinHandle<()>),
}

pub(crate) struct HttpOut {
    http: reqwest::Client,
    url: String,
    session: Mutex<Option<String>>,
    version: Mutex<Option<String>>,
    incoming: mpsc::UnboundedSender<Value>,
    extra: Vec<(String, String)>,
    stream: tokio::sync::Mutex<GetStream>,
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
            stream: tokio::sync::Mutex::new(GetStream::Closed),
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

/// Incremental SSE parsing: removes every complete event (terminated by a
/// blank line) from `buf` and returns its JSON payloads.
fn drain_sse_events(buf: &mut String) -> Vec<Value> {
    // A `\r` at the end of a chunk stays until its `\n` arrives.
    if buf.contains("\r\n") {
        *buf = buf.replace("\r\n", "\n");
    }
    match buf.rfind("\n\n") {
        Some(end) => {
            let complete: String = buf.drain(..end + 2).collect();
            sse_messages(&complete)
        }
        None => Vec::new(),
    }
}

/// Reads the GET stream and forwards its messages until it ends.
async fn read_stream(mut resp: reqwest::Response, incoming: mpsc::UnboundedSender<Value>) {
    let mut buf = String::new();
    while let Ok(Some(chunk)) = resp.chunk().await {
        buf.push_str(&String::from_utf8_lossy(&chunk));
        for m in drain_sse_events(&mut buf) {
            if incoming.send(m).is_err() {
                return;
            }
        }
    }
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

    /// Opens the GET stream once. Returns after the server accepted it, so a
    /// notification sent after this returns is delivered.
    async fn open_notification_stream(&self) -> Result<()> {
        let mut state = self.stream.lock().await;
        match &*state {
            GetStream::Unsupported => return Ok(()),
            GetStream::Open(task) if !task.is_finished() => return Ok(()),
            _ => {}
        }
        let resp = self
            .with_headers(self.http.get(&self.url))
            .header(ACCEPT, "text/event-stream")
            .send()
            .await?;
        let status = resp.status();
        if status == reqwest::StatusCode::METHOD_NOT_ALLOWED {
            *state = GetStream::Unsupported;
            return Ok(());
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(McpError::Protocol(format!(
                "HTTP {status} opening the notification stream: {body}"
            )));
        }
        *state = GetStream::Open(tokio::spawn(read_stream(resp, self.incoming.clone())));
        Ok(())
    }

    async fn close(&self) {
        if let GetStream::Open(task) = &*self.stream.lock().await {
            task.abort();
        }
        let has_session = self.session.lock().unwrap().is_some();
        if has_session {
            let _ = self.with_headers(self.http.delete(&self.url)).send().await;
        }
    }
}

impl Drop for HttpOut {
    fn drop(&mut self) {
        if let GetStream::Open(task) = self.stream.get_mut() {
            task.abort();
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

    #[test]
    fn drains_complete_events_only() {
        let mut buf = String::from("data: {\"a\":1}\r\n\r\n: keep-alive\n\ndata: {\"b\"");
        assert_eq!(drain_sse_events(&mut buf), vec![json!({"a": 1})]);
        assert_eq!(buf, "data: {\"b\"");
        buf.push_str(":2}\n\n");
        assert_eq!(drain_sse_events(&mut buf), vec![json!({"b": 2})]);
        assert!(buf.is_empty());
    }
}
