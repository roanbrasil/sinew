//! Typed Rust client for a `sinew-serve` server.

use std::pin::Pin;

use futures_core::Stream;
use futures_util::StreamExt;
use reqwest::{Method, RequestBuilder, Response};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{
    stream::{SseEvent, SseParser},
    Message,
};

use crate::api::{
    AssistantInfo, Run, RunCreate, RunStatus, StateUpdate, Thread, ThreadCreate, ThreadQuery,
    ThreadState,
};

/// Client-side failure.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("server returned HTTP {status}: {message}")]
    Api { status: u16, message: String },
    #[error("invalid response: {0}")]
    Json(#[from] serde_json::Error),
}

/// One event of a streamed run.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamPart {
    /// First event: ids of the run.
    Metadata {
        run_id: String,
        thread_id: Option<String>,
    },
    /// Full state after a node (`values` mode).
    Values(Value),
    /// `{node: changed keys}` (`updates` mode).
    Updates(Value),
    /// A model token (`tokens` mode).
    Token {
        node: String,
        delta: String,
    },
    /// Node lifecycle (`events` mode): `{"event": "node_start"|"node_end", "node", "step", "task"}`.
    Event(Value),
    /// A value emitted by a node (`custom` mode).
    Custom {
        node: String,
        value: Value,
    },
    /// The run paused before `next`.
    Interrupt {
        next: Vec<String>,
        values: Value,
    },
    Error(String),
    /// Last event.
    End {
        status: RunStatus,
    },
    /// Anything this client version does not know.
    Other {
        event: String,
        data: Value,
    },
}

impl StreamPart {
    fn from_sse(ev: SseEvent) -> Result<Self, ClientError> {
        let data: Value = serde_json::from_str(&ev.data)?;
        let s = |v: &Value, k: &str| v[k].as_str().unwrap_or_default().to_string();
        Ok(match ev.event.as_deref().unwrap_or("message") {
            "metadata" => StreamPart::Metadata {
                run_id: s(&data, "run_id"),
                thread_id: data["thread_id"].as_str().map(str::to_string),
            },
            "values" => StreamPart::Values(data),
            "updates" => StreamPart::Updates(data),
            "tokens" => StreamPart::Token {
                node: s(&data, "node"),
                delta: s(&data, "delta"),
            },
            "events" => StreamPart::Event(data),
            "custom" => StreamPart::Custom {
                node: s(&data, "node"),
                value: data["value"].clone(),
            },
            "interrupt" => StreamPart::Interrupt {
                next: serde_json::from_value(data["next"].clone()).unwrap_or_default(),
                values: data["values"].clone(),
            },
            "error" => StreamPart::Error(s(&data, "error")),
            "end" => StreamPart::End {
                status: serde_json::from_value(data["status"].clone())?,
            },
            other => StreamPart::Other {
                event: other.to_string(),
                data,
            },
        })
    }
}

/// Stream of [`StreamPart`]s; ends after [`StreamPart::End`].
pub type RunStream = Pin<Box<dyn Stream<Item = Result<StreamPart, ClientError>> + Send>>;

/// Client for the run/thread/assistant API and the OpenAI façade.
#[derive(Clone)]
pub struct ServeClient {
    base: String,
    api_key: Option<String>,
    http: reqwest::Client,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: Value,
}

impl ServeClient {
    /// `base_url` like `http://127.0.0.1:8123`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base: base_url.into().trim_end_matches('/').to_string(),
            api_key: None,
            http: reqwest::Client::new(),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Uses your own `reqwest::Client` (timeouts, proxies, pools).
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    fn req(&self, method: Method, path: &str) -> RequestBuilder {
        let r = self.http.request(method, format!("{}{}", self.base, path));
        match &self.api_key {
            Some(k) => r.bearer_auth(k),
            None => r,
        }
    }

    async fn check(resp: Response) -> Result<Response, ClientError> {
        if resp.status().is_success() {
            return Ok(resp);
        }
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        let message = match serde_json::from_str::<ErrorBody>(&text) {
            Ok(b) => match b.error {
                Value::String(s) => s,
                other => other["message"]
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| other.to_string()),
            },
            Err(_) => text,
        };
        Err(ClientError::Api { status, message })
    }

    async fn send<T: DeserializeOwned>(&self, rb: RequestBuilder) -> Result<T, ClientError> {
        let resp = Self::check(rb.send().await?).await?;
        Ok(resp.json().await?)
    }

    async fn post<B: Serialize + ?Sized, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, ClientError> {
        self.send(self.req(Method::POST, path).json(body)).await
    }

    async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, ClientError> {
        self.send(self.req(Method::GET, path)).await
    }

    fn runs_path(thread_id: Option<&str>) -> String {
        match thread_id {
            Some(t) => format!("/threads/{t}/runs"),
            None => "/runs".into(),
        }
    }

    /// `GET /ok`.
    pub async fn health(&self) -> Result<bool, ClientError> {
        let v: Value = self.get("/ok").await?;
        Ok(v["ok"] == json!(true))
    }

    pub async fn assistants(&self) -> Result<Vec<AssistantInfo>, ClientError> {
        self.get("/assistants").await
    }

    pub async fn assistant(&self, id: &str) -> Result<AssistantInfo, ClientError> {
        self.get(&format!("/assistants/{id}")).await
    }

    pub async fn create_thread(&self, req: &ThreadCreate) -> Result<Thread, ClientError> {
        self.post("/threads", req).await
    }

    pub async fn thread(&self, thread_id: &str) -> Result<Thread, ClientError> {
        self.get(&format!("/threads/{thread_id}")).await
    }

    pub async fn threads(&self, query: &ThreadQuery) -> Result<Vec<Thread>, ClientError> {
        self.post("/threads/search", query).await
    }

    pub async fn delete_thread(&self, thread_id: &str) -> Result<(), ClientError> {
        Self::check(
            self.req(Method::DELETE, &format!("/threads/{thread_id}"))
                .send()
                .await?,
        )
        .await?;
        Ok(())
    }

    pub async fn state(&self, thread_id: &str) -> Result<ThreadState, ClientError> {
        self.get(&format!("/threads/{thread_id}/state")).await
    }

    /// Shallow-merges `values` into the thread state.
    pub async fn update_state(
        &self,
        thread_id: &str,
        values: Value,
    ) -> Result<ThreadState, ClientError> {
        self.post(
            &format!("/threads/{thread_id}/state"),
            &StateUpdate { values },
        )
        .await
    }

    /// Checkpoints, newest first.
    pub async fn history(
        &self,
        thread_id: &str,
        limit: usize,
    ) -> Result<Vec<ThreadState>, ClientError> {
        self.get(&format!("/threads/{thread_id}/history?limit={limit}"))
            .await
    }

    /// Runs and waits for the outcome (`thread_id = None`: stateless).
    pub async fn wait(&self, thread_id: Option<&str>, req: &RunCreate) -> Result<Run, ClientError> {
        self.post(&format!("{}/wait", Self::runs_path(thread_id)), req)
            .await
    }

    /// Starts a background run and returns it immediately.
    pub async fn create_run(
        &self,
        thread_id: Option<&str>,
        req: &RunCreate,
    ) -> Result<Run, ClientError> {
        self.post(&Self::runs_path(thread_id), req).await
    }

    pub async fn run(&self, run_id: &str) -> Result<Run, ClientError> {
        self.get(&format!("/runs/{run_id}")).await
    }

    pub async fn thread_runs(&self, thread_id: &str) -> Result<Vec<Run>, ClientError> {
        self.get(&format!("/threads/{thread_id}/runs")).await
    }

    /// Cancels; with `wait`, returns once the run is over.
    pub async fn cancel_run(&self, run_id: &str, wait: bool) -> Result<Run, ClientError> {
        self.post(&format!("/runs/{run_id}/cancel?wait={wait}"), &json!({}))
            .await
    }

    /// Waits for a run to finish.
    pub async fn join_run(&self, run_id: &str) -> Result<Run, ClientError> {
        self.get(&format!("/runs/{run_id}/join")).await
    }

    /// Runs and streams its events.
    pub async fn stream(
        &self,
        thread_id: Option<&str>,
        req: &RunCreate,
    ) -> Result<RunStream, ClientError> {
        let rb = self
            .req(
                Method::POST,
                &format!("{}/stream", Self::runs_path(thread_id)),
            )
            .json(req);
        self.open_stream(rb).await
    }

    /// Streams a run that is already going (from now on).
    pub async fn join_stream(&self, run_id: &str) -> Result<RunStream, ClientError> {
        self.open_stream(self.req(Method::GET, &format!("/runs/{run_id}/stream")))
            .await
    }

    async fn open_stream(&self, rb: RequestBuilder) -> Result<RunStream, ClientError> {
        let resp = Self::check(rb.header("accept", "text/event-stream").send().await?).await?;
        let bytes = resp.bytes_stream();
        let s = futures_util::stream::unfold(
            (
                bytes,
                SseParser::new(),
                std::collections::VecDeque::new(),
                false,
            ),
            |(mut bytes, mut parser, mut queue, mut finished)| async move {
                loop {
                    if let Some(ev) = queue.pop_front() {
                        let part = StreamPart::from_sse(ev);
                        if matches!(part, Ok(StreamPart::End { .. })) {
                            finished = true;
                            queue.clear();
                        }
                        return Some((part, (bytes, parser, queue, finished)));
                    }
                    if finished {
                        return None;
                    }
                    match bytes.next().await {
                        Some(Ok(chunk)) => queue.extend(parser.push(&chunk)),
                        Some(Err(e)) => {
                            finished = true;
                            return Some((Err(e.into()), (bytes, parser, queue, finished)));
                        }
                        None => {
                            finished = true;
                            queue.extend(parser.finish());
                        }
                    }
                }
            },
        );
        Ok(Box::pin(s))
    }

    /// `POST /v1/chat/completions` (non-streaming): the assistant's reply text.
    pub async fn chat_completion(
        &self,
        model: &str,
        messages: &[Message],
    ) -> Result<String, ClientError> {
        let v: Value = self
            .post(
                "/v1/chat/completions",
                &json!({"model": model, "messages": messages}),
            )
            .await?;
        Ok(v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string())
    }

    /// `GET /v1/models`: ids usable as `model`.
    pub async fn models(&self) -> Result<Vec<String>, ClientError> {
        let v: Value = self.get("/v1/models").await?;
        Ok(v["data"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| m["id"].as_str().map(str::to_string))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stream_parts() {
        let ev = |e: &str, d: &str| SseEvent {
            event: Some(e.into()),
            data: d.into(),
        };
        assert_eq!(
            StreamPart::from_sse(ev("tokens", r#"{"node":"agent","delta":"hi"}"#)).unwrap(),
            StreamPart::Token {
                node: "agent".into(),
                delta: "hi".into()
            }
        );
        assert_eq!(
            StreamPart::from_sse(ev("end", r#"{"run_id":"r","status":"interrupted"}"#)).unwrap(),
            StreamPart::End {
                status: RunStatus::Interrupted
            }
        );
        assert!(matches!(
            StreamPart::from_sse(ev("weird", "1")).unwrap(),
            StreamPart::Other { .. }
        ));
    }
}
