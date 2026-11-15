//! Client for the `sinew-scope serve` API: datasets, experiments, pairwise
//! comparisons, feedback and annotation queues.

use std::time::Duration;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{Dataset, EvalError, Example, ExperimentReport, PairwiseReport};

/// Feedback on a run: a score and/or a value, plus an optional comment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Feedback {
    pub run_id: String,
    /// Feedback key, e.g. `correctness`, `thumbs`, `user_rating`.
    pub key: String,
    /// Numeric score (usually 0..1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Any JSON value (a label, a correction...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    /// Who or what gave it (`api` when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl Feedback {
    pub fn score(run_id: impl Into<String>, key: impl Into<String>, score: f64) -> Self {
        Self {
            run_id: run_id.into(),
            key: key.into(),
            score: Some(score),
            ..Default::default()
        }
    }

    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    pub fn with_value(mut self, value: Value) -> Self {
        self.value = Some(value);
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

/// Talks to a scope server.
#[derive(Clone)]
pub struct ScopeClient {
    base_url: String,
    token: Option<String>,
    client: reqwest::Client,
}

impl ScopeClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            token: None,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Reads `SINEW_SCOPE_URL` (default `http://127.0.0.1:7878`) and
    /// `SINEW_SCOPE_TOKEN`.
    pub fn from_env() -> Self {
        let url =
            std::env::var("SINEW_SCOPE_URL").unwrap_or_else(|_| "http://127.0.0.1:7878".into());
        let c = Self::new(url);
        match std::env::var("SINEW_SCOPE_TOKEN") {
            Ok(t) if !t.is_empty() => c.with_token(t),
            _ => c,
        }
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Request timeout (default 30 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_default();
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn send<T: DeserializeOwned>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<T, EvalError> {
        let mut req = self
            .client
            .request(method, format!("{}{path}", self.base_url));
        if let Some(b) = body {
            req = req.json(&b);
        }
        if let Some(t) = &self.token {
            req = req.bearer_auth(t);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| EvalError::Http(e.to_string()))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status == reqwest::StatusCode::NOT_FOUND {
            let msg = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["error"].as_str().map(str::to_string))
                .unwrap_or_else(|| path.to_string());
            return Err(EvalError::NotFound(msg));
        }
        if !status.is_success() {
            return Err(EvalError::Http(format!("{status}: {text}")));
        }
        let text = if text.is_empty() { "null" } else { &text };
        serde_json::from_str(text).map_err(|e| EvalError::Http(format!("bad response: {e}")))
    }

    /// `GET` a JSON resource (path starts with `/`).
    pub async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, EvalError> {
        self.send(reqwest::Method::GET, path, None).await
    }

    /// `POST` JSON and decode the JSON reply.
    pub async fn post_json<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<T, EvalError> {
        let body = serde_json::to_value(body).map_err(|e| EvalError::Http(e.to_string()))?;
        self.send(reqwest::Method::POST, path, Some(body)).await
    }

    /// `PUT` JSON and decode the JSON reply.
    pub async fn put_json<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<T, EvalError> {
        let body = serde_json::to_value(body).map_err(|e| EvalError::Http(e.to_string()))?;
        self.send(reqwest::Method::PUT, path, Some(body)).await
    }

    /// `DELETE` a resource.
    pub async fn delete(&self, path: &str) -> Result<(), EvalError> {
        self.send::<Value>(reqwest::Method::DELETE, path, None)
            .await
            .map(|_| ())
    }

    /// Creates or replaces a dataset (matched by name).
    pub async fn upload_dataset(&self, dataset: &Dataset) -> Result<(), EvalError> {
        self.post_json::<Value>("/api/datasets", dataset)
            .await
            .map(|_| ())
    }

    /// Appends one example to a dataset (created if missing).
    pub async fn add_example(&self, dataset: &str, example: &Example) -> Result<(), EvalError> {
        self.post_json::<Value>(
            &format!("/api/datasets/{}/examples", encode(dataset)),
            example,
        )
        .await
        .map(|_| ())
    }

    /// Uploads an experiment report (idempotent on the report id).
    pub async fn upload_report(&self, report: &ExperimentReport) -> Result<(), EvalError> {
        self.post_json::<Value>("/api/experiments", report)
            .await
            .map(|_| ())
    }

    /// Uploads the dataset, then the report, so the experiment shows up
    /// under its dataset in the UI.
    pub async fn upload_experiment(
        &self,
        dataset: &Dataset,
        report: &ExperimentReport,
    ) -> Result<(), EvalError> {
        self.upload_dataset(dataset).await?;
        self.upload_report(report).await
    }

    /// Uploads a pairwise comparison (idempotent on its id).
    pub async fn upload_pairwise(&self, report: &PairwiseReport) -> Result<(), EvalError> {
        self.post_json::<Value>("/api/pairwise", report)
            .await
            .map(|_| ())
    }

    /// Records feedback on a run; returns the feedback id.
    pub async fn create_feedback(&self, feedback: &Feedback) -> Result<String, EvalError> {
        let v: Value = self.post_json("/api/feedback", feedback).await?;
        Ok(v["id"].as_str().unwrap_or_default().to_string())
    }

    /// Every feedback entry of a run, oldest first.
    pub async fn feedback_for_run(&self, run_id: &str) -> Result<Vec<Value>, EvalError> {
        self.get_json(&format!("/api/feedback?run_id={}", encode(run_id)))
            .await
    }

    /// Creates an annotation queue; `keys` are the feedback keys reviewers
    /// score. Returns the queue id.
    pub async fn create_queue(
        &self,
        name: &str,
        description: &str,
        keys: &[&str],
    ) -> Result<String, EvalError> {
        let v: Value = self
            .post_json(
                "/api/queues",
                &json!({ "name": name, "description": description, "keys": keys }),
            )
            .await?;
        Ok(v["id"].as_str().unwrap_or_default().to_string())
    }

    /// Adds runs to an annotation queue.
    pub async fn add_to_queue(&self, queue_id: &str, run_ids: &[&str]) -> Result<(), EvalError> {
        self.post_json::<Value>(
            &format!("/api/queues/{}/items", encode(queue_id)),
            &json!({ "run_ids": run_ids }),
        )
        .await
        .map(|_| ())
    }
}

/// Percent-encodes a path segment or query value.
pub(crate) fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
