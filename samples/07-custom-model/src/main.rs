//! `ChatModel` is a one-method trait, so cross-cutting behavior composes as
//! wrappers. This sample builds two: `Retry` (re-tries transient failures with
//! backoff) and `Fallback` (switches to a second model when the first fails),
//! then uses them anywhere a model is accepted — here, the prebuilt agent.
//!
//! ```bash
//! cargo run -p sample-07-custom-model
//! cargo run -p sample-07-custom-model -- --ollama   # Ollama as the fallback
//! ```

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use sample_common::{pick_model, BoxError};
use sinew::core::{ChatResponse, Error, Result as CoreResult};
use sinew::prelude::*;

/// Retries errors up to `attempts` times with exponential backoff.
struct Retry<M> {
    inner: M,
    attempts: u32,
    base_delay: Duration,
}

#[async_trait]
impl<M: ChatModel> ChatModel for Retry<M> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    async fn chat(&self, req: ChatRequest) -> CoreResult<ChatResponse> {
        let mut last = None;
        for attempt in 0..self.attempts {
            match self.inner.chat(req.clone()).await {
                Ok(r) => return Ok(r),
                Err(e) => {
                    eprintln!("  attempt {} failed: {e}", attempt + 1);
                    last = Some(e);
                    tokio::time::sleep(self.base_delay * 2u32.pow(attempt)).await;
                }
            }
        }
        Err(last.expect("attempts > 0"))
    }
}

/// Uses `primary`, switching to `backup` if it fails.
struct Fallback {
    primary: Arc<dyn ChatModel>,
    backup: Arc<dyn ChatModel>,
}

#[async_trait]
impl ChatModel for Fallback {
    fn name(&self) -> &str {
        self.primary.name()
    }
    async fn chat(&self, req: ChatRequest) -> CoreResult<ChatResponse> {
        match self.primary.chat(req.clone()).await {
            Ok(r) => Ok(r),
            Err(e) => {
                eprintln!(
                    "  primary `{}` failed ({e}); using `{}`",
                    self.primary.name(),
                    self.backup.name()
                );
                self.backup.chat(req).await
            }
        }
    }
}

/// A stand-in for a flaky provider: always fails with an HTTP 503.
struct Flaky {
    calls: AtomicU32,
}

#[async_trait]
impl ChatModel for Flaky {
    fn name(&self) -> &str {
        "flaky-provider"
    }
    async fn chat(&self, _req: ChatRequest) -> CoreResult<ChatResponse> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(Error::Provider {
            provider: "flaky".into(),
            status: 503,
            body: "overloaded".into(),
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let flaky = Arc::new(Flaky {
        calls: AtomicU32::new(0),
    });
    let primary = Arc::new(Retry {
        inner: flaky.clone(),
        attempts: 3,
        base_delay: Duration::from_millis(50),
    });
    let backup = pick_model(vec![Message::assistant(
        "Backup model here: the answer is 42.",
    )])?;
    let model: Arc<dyn ChatModel> = Arc::new(Fallback { primary, backup });

    let agent = react_agent(model, ToolBox::new())?;
    let out = agent
        .invoke(AgentState::new(vec![Message::user(
            "What is 6 times 7? Answer briefly.",
        )]))
        .await?;

    println!(
        "\nflaky provider was called {} times",
        flaky.calls.load(Ordering::Relaxed)
    );
    println!(
        "answer: {}",
        out.last().map(|m| m.content.as_str()).unwrap_or_default()
    );
    Ok(())
}
