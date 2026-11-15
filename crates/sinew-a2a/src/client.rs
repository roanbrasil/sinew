//! A2A client: discover an agent from its card, send messages, stream
//! updates, and get, cancel or resubscribe to tasks.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use sinew_core::stream::{BoxStream, SseParser};

use crate::{
    error::{A2aError, Result},
    jsonrpc::{self, method},
    types::{
        AgentCard, Message, MessageSendParams, SendMessageResult, StreamResponse, Task,
        TaskIdParams, TaskQueryParams, AGENT_CARD_PATH,
    },
};

/// Stream of events from `message/stream` or `tasks/resubscribe`. It ends
/// after the final event (or when the server closes the connection).
pub type A2aStream = BoxStream<'static, Result<StreamResponse>>;

struct Inner {
    http: reqwest::Client,
    url: String,
    headers: HeaderMap,
    card: Option<AgentCard>,
    next_id: AtomicU64,
}

/// Talks to one A2A agent over JSON-RPC. Cheap to clone; clones share the
/// HTTP connection pool.
#[derive(Clone)]
pub struct A2aClient {
    inner: Arc<Inner>,
}

/// The default HTTP client: keep-alive pool, `TCP_NODELAY`.
pub fn default_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .tcp_nodelay(true)
        .build()
        .unwrap_or_default()
}

impl A2aClient {
    /// A client for the JSON-RPC endpoint `url` (the card's `url`).
    pub fn new(url: impl Into<String>) -> Self {
        Self::build(default_http_client(), url.into(), None)
    }

    /// Like [`new`](Self::new), reusing an existing `reqwest::Client` (and
    /// its connection pool, proxies, TLS settings and timeouts).
    pub fn with_http_client(http: reqwest::Client, url: impl Into<String>) -> Self {
        Self::build(http, url.into(), None)
    }

    /// A client for the agent described by `card`.
    pub fn from_card(card: AgentCard) -> Self {
        Self::from_card_with(default_http_client(), card)
    }

    pub fn from_card_with(http: reqwest::Client, card: AgentCard) -> Self {
        let url = card.url.clone();
        Self::build(http, url, Some(card))
    }

    /// Fetches the card from `base_url` + `/.well-known/agent-card.json` and
    /// connects to the endpoint it names.
    pub async fn discover(base_url: &str) -> Result<Self> {
        let http = default_http_client();
        let card = fetch_card_with(&http, base_url).await?;
        Ok(Self::from_card_with(http, card))
    }

    fn build(http: reqwest::Client, url: String, card: Option<AgentCard>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http,
                url,
                headers: HeaderMap::new(),
                card,
                next_id: AtomicU64::new(1),
            }),
        }
    }

    fn inner_mut(&mut self) -> &mut Inner {
        Arc::get_mut(&mut self.inner).expect("configure the client before cloning it")
    }

    /// Adds a header to every request (e.g. an API key).
    pub fn with_header(mut self, name: &str, value: &str) -> Result<Self> {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|e| A2aError::Protocol(format!("bad header name: {e}")))?;
        let value = HeaderValue::from_str(value)
            .map_err(|e| A2aError::Protocol(format!("bad header value: {e}")))?;
        self.inner_mut().headers.insert(name, value);
        Ok(self)
    }

    /// `Authorization: Bearer <token>` on every request.
    pub fn with_bearer_token(self, token: &str) -> Result<Self> {
        self.with_header("authorization", &format!("Bearer {token}"))
    }

    pub fn url(&self) -> &str {
        &self.inner.url
    }

    /// The card this client was built from, if any.
    pub fn card(&self) -> Option<&AgentCard> {
        self.inner.card.as_ref()
    }

    fn post(&self, method: &str, params: &impl Serialize) -> Result<reqwest::RequestBuilder> {
        let id = Value::from(self.inner.next_id.fetch_add(1, Ordering::Relaxed));
        let body = jsonrpc::request(&id, method, params)?;
        Ok(self
            .inner
            .http
            .post(&self.inner.url)
            .headers(self.inner.headers.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(body))
    }

    /// Sends a JSON-RPC request and decodes its result.
    pub async fn request<T: DeserializeOwned>(
        &self,
        method: &str,
        params: &impl Serialize,
    ) -> Result<T> {
        let resp = self
            .post(method, params)?
            .header(ACCEPT, "application/json")
            .send()
            .await?;
        let resp = check_status(resp, &self.inner.url).await?;
        let body = resp.bytes().await?;
        let result = jsonrpc::decode_response(&body)??;
        Ok(serde_json::from_value(result)?)
    }

    /// `message/send`. Blocks until the task finishes or needs input, unless
    /// the params ask for `blocking: false`.
    pub async fn send_message(&self, params: MessageSendParams) -> Result<SendMessageResult> {
        self.request(method::MESSAGE_SEND, &params).await
    }

    /// `message/send` with one text part.
    pub async fn send_text(&self, text: impl Into<String>) -> Result<SendMessageResult> {
        self.send_message(MessageSendParams::new(Message::user_text(text)))
            .await
    }

    /// `message/stream`: events arrive as the agent produces them.
    pub async fn stream_message(&self, params: MessageSendParams) -> Result<A2aStream> {
        self.open_stream(method::MESSAGE_STREAM, &params).await
    }

    /// `message/stream` with one text part.
    pub async fn stream_text(&self, text: impl Into<String>) -> Result<A2aStream> {
        self.stream_message(MessageSendParams::new(Message::user_text(text)))
            .await
    }

    /// `tasks/get`; `history_length` keeps only the last n history messages.
    pub async fn get_task(&self, id: &str, history_length: Option<usize>) -> Result<Task> {
        let params = TaskQueryParams {
            id: id.into(),
            history_length,
            metadata: None,
        };
        self.request(method::TASKS_GET, &params).await
    }

    /// `tasks/cancel`.
    pub async fn cancel_task(&self, id: &str) -> Result<Task> {
        let params = TaskIdParams {
            id: id.into(),
            metadata: None,
        };
        self.request(method::TASKS_CANCEL, &params).await
    }

    /// `tasks/resubscribe`: reattach to a running task's stream. The Sinew
    /// server first sends the current task snapshot (with everything
    /// produced so far), then the live updates.
    pub async fn resubscribe(&self, id: &str) -> Result<A2aStream> {
        let params = TaskIdParams {
            id: id.into(),
            metadata: None,
        };
        self.open_stream(method::TASKS_RESUBSCRIBE, &params).await
    }

    async fn open_stream(&self, method: &str, params: &impl Serialize) -> Result<A2aStream> {
        let resp = self
            .post(method, params)?
            .header(ACCEPT, "text/event-stream")
            .send()
            .await?;
        let resp = check_status(resp, &self.inner.url).await?;
        let is_sse = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));
        if !is_sse {
            // Errors (and non-streaming servers) answer with plain JSON.
            let body = resp.bytes().await?;
            let result = jsonrpc::decode_response(&body)??;
            let ev: StreamResponse = serde_json::from_value(result)?;
            return Ok(futures_util::stream::iter([Ok(ev)]).boxed());
        }
        Ok(sse_events(resp))
    }
}

async fn check_status(resp: reqwest::Response, url: &str) -> Result<reqwest::Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    Err(A2aError::Status {
        url: url.into(),
        status: status.as_u16(),
        body,
    })
}

/// Fetches an Agent Card. `base_url` is the agent's origin (the card path is
/// appended) or a full URL ending in `.json`.
pub async fn fetch_card(base_url: &str) -> Result<AgentCard> {
    fetch_card_with(&default_http_client(), base_url).await
}

pub async fn fetch_card_with(http: &reqwest::Client, base_url: &str) -> Result<AgentCard> {
    let url = if base_url.ends_with(".json") {
        base_url.to_string()
    } else {
        format!("{}{AGENT_CARD_PATH}", base_url.trim_end_matches('/'))
    };
    let resp = http
        .get(&url)
        .header(ACCEPT, "application/json")
        .send()
        .await?;
    let resp = check_status(resp, &url).await?;
    Ok(serde_json::from_slice(&resp.bytes().await?)?)
}

/// Decodes an SSE body of JSON-RPC responses into events, chunk by chunk.
fn sse_events(resp: reqwest::Response) -> A2aStream {
    struct St {
        resp: Option<reqwest::Response>,
        parser: SseParser,
        queue: VecDeque<Result<StreamResponse>>,
        finished: bool,
    }
    let st = St {
        resp: Some(resp),
        parser: SseParser::new(),
        queue: VecDeque::new(),
        finished: false,
    };
    futures_util::stream::unfold(st, |mut st| async move {
        loop {
            if let Some(item) = st.queue.pop_front() {
                let stop = match &item {
                    Ok(ev) => ev.is_final(),
                    Err(_) => true,
                };
                if stop {
                    st.finished = true;
                    st.queue.clear();
                    st.resp = None;
                }
                return Some((item, st));
            }
            if st.finished {
                return None;
            }
            let events = match st.resp.as_mut()?.chunk().await {
                Ok(Some(bytes)) => st.parser.push(&bytes),
                Ok(None) => {
                    st.resp = None;
                    st.finished = true;
                    st.parser.finish().into_iter().collect()
                }
                Err(e) => {
                    st.resp = None;
                    st.finished = true;
                    return Some((Err(e.into()), st));
                }
            };
            for ev in events {
                let decoded = jsonrpc::decode_response(ev.data.as_bytes())
                    .map_err(A2aError::from)
                    .and_then(|r| r.map_err(A2aError::Rpc))
                    .and_then(|v| serde_json::from_value(v).map_err(A2aError::from));
                st.queue.push_back(decoded);
            }
        }
    })
    .boxed()
}
