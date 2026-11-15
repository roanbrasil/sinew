pub mod anthropic;
pub mod gemini;
pub mod openai;
pub mod presets;

pub use anthropic::Anthropic;
pub use gemini::Gemini;
pub use openai::OpenAiCompat;
pub use presets::{from_spec, provider_names};

pub(crate) async fn post_json(
    provider: &str,
    req: reqwest::RequestBuilder,
    body: &serde_json::Value,
) -> crate::Result<serde_json::Value> {
    let resp = req.json(body).send().await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        return Err(crate::Error::Provider {
            provider: provider.into(),
            status: status.as_u16(),
            body: text,
        });
    }
    Ok(serde_json::from_str(&text)?)
}

/// Sends a streaming request and returns the response once the status is OK.
pub(crate) async fn post_sse(
    provider: &str,
    req: reqwest::RequestBuilder,
    body: &serde_json::Value,
) -> crate::Result<reqwest::Response> {
    let resp = req
        .header("accept", "text/event-stream")
        .json(body)
        .send()
        .await?;
    let status = resp.status();
    if !status.is_success() {
        return Err(crate::Error::Provider {
            provider: provider.into(),
            status: status.as_u16(),
            body: resp.text().await.unwrap_or_default(),
        });
    }
    Ok(resp)
}
