//! Minimal client for the `sinew-scope serve` dataset and experiment API.

use std::time::Duration;

use serde::Serialize;

use crate::{Dataset, EvalError, ExperimentReport};

/// Uploads datasets and experiment reports to a scope server.
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

    async fn post<T: Serialize + ?Sized>(&self, path: &str, body: &T) -> Result<(), EvalError> {
        let mut req = self
            .client
            .post(format!("{}{path}", self.base_url))
            .json(body);
        if let Some(t) = &self.token {
            req = req.bearer_auth(t);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| EvalError::Http(e.to_string()))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            Err(EvalError::Http(format!("{status}: {text}")))
        }
    }

    /// Creates or replaces a dataset (matched by name).
    pub async fn upload_dataset(&self, dataset: &Dataset) -> Result<(), EvalError> {
        self.post("/api/datasets", dataset).await
    }

    /// Uploads an experiment report (idempotent on the report id).
    pub async fn upload_report(&self, report: &ExperimentReport) -> Result<(), EvalError> {
        self.post("/api/experiments", report).await
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
}
