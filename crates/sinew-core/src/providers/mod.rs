pub mod anthropic;
pub mod openai;

pub use anthropic::Anthropic;
pub use openai::OpenAiCompat;

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
