pub mod anthropic;
pub mod gemini;
pub mod openai;
pub mod presets;

use std::{sync::OnceLock, time::Duration};

pub use anthropic::{Anthropic, CacheTtl, PromptCaching};
pub use gemini::Gemini;
pub use openai::OpenAiCompat;
pub use presets::{
    capabilities_for, catalog, from_spec, provider_names, ProviderInfo, ProviderKind,
};

/// A token count from a provider's usage object: absent or non-integer
/// values are `None`, counts above `u32::MAX` saturate instead of wrapping.
pub(crate) fn token_count(v: &serde_json::Value) -> Option<u32> {
    v.as_u64().map(|n| u32::try_from(n).unwrap_or(u32::MAX))
}

/// Builds a [`reqwest::Client`] tuned for LLM APIs: connection pooling with
/// keep-alive, `TCP_NODELAY` (no Nagle delay on small streaming writes),
/// TCP keepalive, HTTP/2 via ALPN when the `http2` feature is on (default),
/// and a 10 s connect timeout. There is no overall request timeout, because
/// long generations and streams can legitimately take minutes; wrap the model
/// in [`with_timeout`](crate::resilience::ChatModelExt::with_timeout) for one.
pub fn tuned_client_builder() -> reqwest::ClientBuilder {
    let b = reqwest::Client::builder()
        .tcp_nodelay(true)
        .tcp_keepalive(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(64)
        .connect_timeout(Duration::from_secs(10))
        .user_agent(concat!("sinew/", env!("CARGO_PKG_VERSION")));
    #[cfg(feature = "http2")]
    let b = b
        .http2_adaptive_window(true)
        .http2_keep_alive_interval(Duration::from_secs(30))
        .http2_keep_alive_while_idle(true);
    b
}

/// The process-wide HTTP client every provider uses unless given another one
/// with `with_http_client`. Sharing it means one connection pool (and one
/// TLS session cache) for all models, so the second request to a host skips
/// the TCP and TLS handshakes. Cloning a `reqwest::Client` is a cheap `Arc`
/// clone.
pub fn shared_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            tuned_client_builder()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new())
        })
        .clone()
}

fn status_error(
    provider: &str,
    resp_headers: &reqwest::header::HeaderMap,
    status: u16,
    body: String,
) -> crate::Error {
    let retry_after = resp_headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(crate::error::parse_retry_after);
    if status == 429 || retry_after.is_some() {
        crate::Error::RateLimited {
            provider: provider.into(),
            status,
            retry_after,
            body,
        }
    } else {
        crate::Error::Provider {
            provider: provider.into(),
            status,
            body,
        }
    }
}

/// Serializes `body` once into a byte buffer and sends it.
fn with_json_body(
    req: reqwest::RequestBuilder,
    body: &serde_json::Value,
) -> crate::Result<reqwest::RequestBuilder> {
    let bytes = serde_json::to_vec(body)?;
    Ok(req
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(bytes))
}

pub(crate) async fn post_json(
    provider: &str,
    req: reqwest::RequestBuilder,
    body: &serde_json::Value,
) -> crate::Result<serde_json::Value> {
    let resp = with_json_body(req, body)?.send().await?;
    let status = resp.status();
    if !status.is_success() {
        let headers = resp.headers().clone();
        let text = resp.text().await.unwrap_or_default();
        return Err(status_error(provider, &headers, status.as_u16(), text));
    }
    // Parse straight from bytes: no intermediate String / UTF-8 pass.
    let bytes = resp.bytes().await?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Sends a streaming request and returns the response once the status is OK.
pub(crate) async fn post_sse(
    provider: &str,
    req: reqwest::RequestBuilder,
    body: &serde_json::Value,
) -> crate::Result<reqwest::Response> {
    let resp = with_json_body(req.header("accept", "text/event-stream"), body)?
        .send()
        .await?;
    let status = resp.status();
    if !status.is_success() {
        let headers = resp.headers().clone();
        let text = resp.text().await.unwrap_or_default();
        return Err(status_error(provider, &headers, status.as_u16(), text));
    }
    Ok(resp)
}
