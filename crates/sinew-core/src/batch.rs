//! Batch APIs: submit many requests at once and collect the answers later,
//! at a discount (Anthropic Message Batches bill 50% of the normal price).
//! Good for evaluations, backfills and offline enrichment; not for anything
//! a user waits on (results can take up to 24 hours).

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{ChatModel, ChatRequest, ChatResponse, Error, Result};

/// Progress of a submitted batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchStatus {
    pub id: String,
    /// True once every request has a result (succeeded, errored, canceled or
    /// expired).
    pub ended: bool,
    /// Requests still processing.
    pub processing: u64,
    pub succeeded: u64,
    pub errored: u64,
}

/// A model whose provider offers a batch API.
#[async_trait]
pub trait BatchModel: ChatModel {
    /// Submits the requests; returns the provider's batch id.
    async fn submit_batch(&self, requests: Vec<ChatRequest>) -> Result<String>;

    async fn batch_status(&self, id: &str) -> Result<BatchStatus>;

    /// Results in submission order. Call after [`BatchStatus::ended`].
    async fn batch_results(&self, id: &str) -> Result<Vec<Result<ChatResponse>>>;
}

/// Submits `requests`, polls every `poll` until the batch ends, and returns
/// one result per request, in order.
///
/// ```no_run
/// # async fn demo() -> sinew_core::Result<()> {
/// use std::time::Duration;
/// use sinew_core::{batch::run_batch, providers::Anthropic, ChatRequest, Message};
///
/// let model = Anthropic::from_env("claude-haiku-5-5")?;
/// let questions = ["What is Rust?", "What is Tokio?"];
/// let requests = questions.iter().map(|q| ChatRequest::new(vec![Message::user(*q)])).collect();
/// for answer in run_batch(&model, requests, Duration::from_secs(30)).await? {
///     println!("{}", answer?.message.content);
/// }
/// # Ok(()) }
/// ```
pub async fn run_batch<M: BatchModel + ?Sized>(
    model: &M,
    requests: Vec<ChatRequest>,
    poll: Duration,
) -> Result<Vec<Result<ChatResponse>>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let id = model.submit_batch(requests).await?;
    loop {
        let status = model.batch_status(&id).await?;
        if status.ended {
            break;
        }
        tokio::time::sleep(poll).await;
    }
    model.batch_results(&id).await
}

/// Index of a request from the `custom_id` Sinew assigns (`"r<index>"`).
pub(crate) fn index_of(custom_id: &str) -> Result<usize> {
    custom_id
        .strip_prefix('r')
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| Error::InvalidResponse(format!("unexpected batch custom_id `{custom_id}`")))
}
