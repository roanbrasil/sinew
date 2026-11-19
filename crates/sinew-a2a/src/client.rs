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
use sinew_core::identity::{
    current_context, user_credential, AgentCredentials, IdentityError, MessageVerifier,
    SignedMessage, TrustStore, VerifiedIdentity, VerifiedMessage, META_CREDENTIAL, META_IDENTITY,
    META_MESSAGE,
};
use sinew_core::stream::{BoxStream, SseParser};

use crate::{
    error::{A2aError, Result},
    jsonrpc::{self, method},
    types::{
        task_text, AgentCard, Message, MessageSendParams, SendMessageResult, StreamResponse, Task,
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
    credentials: Option<AgentCredentials>,
    audience: Option<String>,
    verified: Option<VerifiedIdentity>,
    message_verifier: Option<MessageVerifier>,
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

    /// Like [`discover`](Self::discover), but only connects when the card
    /// [verifies](AgentCard::verify) against `trust` (a signed identity from
    /// a trusted issuer, and a card signature by that identity's key).
    pub async fn discover_verified(base_url: &str, trust: &TrustStore) -> Result<Self> {
        let http = default_http_client();
        let card = fetch_card_with(&http, base_url).await?;
        let verified = card.verify(trust)?;
        let mut client = Self::from_card_with(http, card);
        client.inner_mut().verified = Some(verified);
        Ok(client)
    }

    fn build(http: reqwest::Client, url: String, card: Option<AgentCard>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http,
                url,
                headers: HeaderMap::new(),
                card,
                next_id: AtomicU64::new(1),
                credentials: None,
                audience: None,
                verified: None,
                message_verifier: None,
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

    /// Proves this agent's identity on every request: a call assertion of
    /// the current [`CallContext`](sinew_core::identity::CallContext) (or a
    /// fresh one with this agent as the origin), signed with the agent's
    /// key, sent with its signed identity in the params' `metadata` under
    /// [`META_IDENTITY`]. An on-behalf-of credential in scope
    /// ([`user_credential`]) goes along under [`META_CREDENTIAL`].
    pub fn with_credentials(mut self, credentials: AgentCredentials) -> Self {
        self.inner_mut().credentials = Some(credentials);
        self
    }

    /// The audience of call assertions (default: the card's
    /// [`agent_id`](AgentCard::agent_id), else the endpoint URL).
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.inner_mut().audience = Some(audience.into());
        self
    }

    /// Verifies the signature on every `message/send` reply (see
    /// [`A2aServer::with_message_signing`](crate::A2aServer::with_message_signing),
    /// OWASP ASI07): signed by the remote agent (the identity in its card,
    /// when the client has one), addressed to this agent (the verifier's
    /// recipient, else the id of [`with_credentials`](Self::with_credentials)),
    /// in the reply's `contextId`, over the reply text, fresh and never seen
    /// before. A reply that fails is an [`A2aError::Identity`] error; an
    /// unsigned one too, unless the verifier was built with
    /// `required(false)`. Use [`send_message_verified`](Self::send_message_verified)
    /// to get the verified sender.
    pub fn with_message_verifier(mut self, verifier: MessageVerifier) -> Self {
        self.inner_mut().message_verifier = Some(verifier);
        self
    }

    pub fn message_verifier(&self) -> Option<&MessageVerifier> {
        self.inner.message_verifier.as_ref()
    }

    /// The remote agent's id as this client knows it: the verified
    /// identity, else the identity in its card, else the card name.
    pub fn remote_agent_id(&self) -> Option<String> {
        self.inner
            .verified
            .as_ref()
            .map(|v| v.identity.id.clone())
            .or_else(|| self.inner.card.as_ref().map(AgentCard::agent_id))
    }

    /// Checks the signature of a `message/send` reply with the message
    /// verifier; `Ok(None)` without a verifier, or for an unsigned reply
    /// when signatures are not required.
    pub fn verify_reply(&self, result: &SendMessageResult) -> Result<Option<VerifiedMessage>> {
        let Some(verifier) = &self.inner.message_verifier else {
            return Ok(None);
        };
        let (text, thread, metadata) = match result {
            SendMessageResult::Message(m) => (m.text(), m.context_id.clone(), m.metadata.as_ref()),
            SendMessageResult::Task(t) => (
                task_text(t),
                Some(t.context_id.clone()),
                t.metadata.as_ref(),
            ),
        };
        let signed = metadata.and_then(|m| m.get(META_MESSAGE));
        let Some(signed) = signed.filter(|v| !v.is_null()) else {
            return verifier
                .verify_value(None, &text, None)
                .map_err(A2aError::from);
        };
        let signed = SignedMessage::from_value(signed)?;
        // Only the identity in a card proves who the remote agent is; a
        // bare card name is not checked against the signature.
        let sender = self
            .inner
            .verified
            .as_ref()
            .map(|v| v.identity.id.clone())
            .or_else(|| {
                self.inner
                    .card
                    .as_ref()
                    .and_then(AgentCard::identity)
                    .map(|i| i.payload.id)
            });
        let recipient = verifier
            .configured_recipient()
            .map(str::to_owned)
            .or_else(|| {
                self.inner
                    .credentials
                    .as_ref()
                    .map(|c| c.agent_id().to_string())
            });
        if signed.envelope.payload.thread != thread {
            return Err(IdentityError::Mismatch(format!(
                "reply signed for thread {:?}, it belongs to {:?}",
                signed.envelope.payload.thread, thread
            ))
            .into());
        }
        Ok(Some(verifier.verify_for(
            &signed,
            &text,
            sender.as_deref(),
            recipient.as_deref(),
        )?))
    }

    /// The remote agent's identity, when the client was built with
    /// [`discover_verified`](Self::discover_verified).
    pub fn verified_identity(&self) -> Option<&VerifiedIdentity> {
        self.inner.verified.as_ref()
    }

    fn audience(&self) -> String {
        self.inner
            .audience
            .clone()
            .or_else(|| self.inner.card.as_ref().map(AgentCard::agent_id))
            .unwrap_or_else(|| self.inner.url.clone())
    }

    /// `params` with the identity proof and the on-behalf-of credential in
    /// its `metadata`.
    fn identified(&self, params: &impl Serialize) -> Result<Value> {
        let mut v = serde_json::to_value(params)?;
        let proof = match &self.inner.credentials {
            Some(c) => Some(c.prove(current_context().as_ref(), &self.audience())?),
            None => None,
        };
        let credential = user_credential();
        if proof.is_none() && credential.is_none() {
            return Ok(v);
        }
        let Some(obj) = v.as_object_mut() else {
            return Ok(v);
        };
        let meta = obj
            .entry("metadata")
            .or_insert_with(|| Value::Object(Default::default()));
        if !meta.is_object() {
            *meta = Value::Object(Default::default());
        }
        if let Some(p) = proof {
            meta[META_IDENTITY] = p.to_value();
        }
        if let Some(c) = credential {
            meta[META_CREDENTIAL] = serde_json::to_value(c)?;
        }
        Ok(v)
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
        let body = jsonrpc::request(&id, method, &self.identified(params)?)?;
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
    /// With a [message verifier](Self::with_message_verifier), the reply's
    /// signature is checked first.
    pub async fn send_message(&self, params: MessageSendParams) -> Result<SendMessageResult> {
        self.send_message_verified(params).await.map(|(r, _)| r)
    }

    /// `message/send`, returning the reply and its verified sender (`None`
    /// without a message verifier, or for an accepted unsigned reply).
    pub async fn send_message_verified(
        &self,
        params: MessageSendParams,
    ) -> Result<(SendMessageResult, Option<VerifiedMessage>)> {
        let result: SendMessageResult = self.request(method::MESSAGE_SEND, &params).await?;
        let verified = self.verify_reply(&result)?;
        Ok((result, verified))
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
