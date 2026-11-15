//! Model pricing: turns token usage into a cost in USD.
//!
//! The built-in table lists list prices (USD per million tokens) for common
//! OpenAI, Anthropic, Google Gemini, xAI and other hosted models. It is a
//! starting point: prices change, so override it with `--pricing file.json`
//! or `PUT /api/pricing`. Lookups normalize the model name (lowercase, any
//! `provider:` or `provider/` prefix removed) and pick the longest matching
//! prefix, so `gpt-4o-mini-2024-07-18` matches `gpt-4o-mini`, not `gpt-4o`.
//!
//! The table is kept behind [`Pricing`] so it can later be backed by a shared
//! table (for example one in `sinew-core`) without touching callers.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinew_trace::{Run, RunKind};

/// Price of one model family, in USD per million tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPrice {
    /// Model name or prefix, e.g. `gpt-4o-mini` or `claude-sonnet-4`.
    pub model: String,
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

/// A pricing table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pricing {
    pub models: Vec<ModelPrice>,
}

impl Default for Pricing {
    fn default() -> Self {
        Self::builtin()
    }
}

fn p(provider: &str, model: &str, input: f64, output: f64) -> ModelPrice {
    ModelPrice {
        model: model.into(),
        input_per_mtok: input,
        output_per_mtok: output,
        provider: Some(provider.into()),
    }
}

/// Strips a `provider:` / `provider/` prefix and lowercases.
pub fn normalize_model(model: &str) -> String {
    let m = model.trim().to_ascii_lowercase();
    let m = m.rsplit_once('/').map(|(_, r)| r.to_string()).unwrap_or(m);
    const PROVIDERS: &[&str] = &[
        "openai",
        "anthropic",
        "gemini",
        "google",
        "xai",
        "groq",
        "mistral",
        "deepseek",
        "openrouter",
        "azure",
        "bedrock",
        "vertex",
    ];
    // `openai:gpt-4o` → `gpt-4o`, but keep Ollama tags like `qwen3:4b`.
    match m.split_once(':') {
        Some((prov, rest)) if PROVIDERS.contains(&prov) => rest.to_string(),
        _ => m,
    }
}

impl Pricing {
    /// The built-in table (list prices, USD per million tokens).
    pub fn builtin() -> Self {
        let models = vec![
            // OpenAI
            p("openai", "gpt-5-nano", 0.05, 0.40),
            p("openai", "gpt-5-mini", 0.25, 2.00),
            p("openai", "gpt-5", 1.25, 10.00),
            p("openai", "gpt-4.1-nano", 0.10, 0.40),
            p("openai", "gpt-4.1-mini", 0.40, 1.60),
            p("openai", "gpt-4.1", 2.00, 8.00),
            p("openai", "gpt-4o-mini", 0.15, 0.60),
            p("openai", "gpt-4o", 2.50, 10.00),
            p("openai", "gpt-4-turbo", 10.00, 30.00),
            p("openai", "gpt-3.5-turbo", 0.50, 1.50),
            p("openai", "o4-mini", 1.10, 4.40),
            p("openai", "o3-mini", 1.10, 4.40),
            p("openai", "o3", 2.00, 8.00),
            p("openai", "o1-mini", 1.10, 4.40),
            p("openai", "o1", 15.00, 60.00),
            p("openai", "text-embedding-3-small", 0.02, 0.0),
            p("openai", "text-embedding-3-large", 0.13, 0.0),
            // Anthropic
            p("anthropic", "claude-fable-5", 10.00, 50.00),
            p("anthropic", "claude-mythos-5", 10.00, 50.00),
            p("anthropic", "claude-opus-5-5", 4.00, 20.00),
            p("anthropic", "claude-opus-5", 5.00, 25.00),
            p("anthropic", "claude-opus-4-8", 5.00, 25.00),
            p("anthropic", "claude-opus-4-7", 5.00, 25.00),
            p("anthropic", "claude-opus-4-6", 5.00, 25.00),
            p("anthropic", "claude-opus-4-5", 5.00, 25.00),
            p("anthropic", "claude-opus-4", 15.00, 75.00),
            p("anthropic", "claude-sonnet-5", 2.00, 10.00),
            p("anthropic", "claude-sonnet-4", 3.00, 15.00),
            p("anthropic", "claude-haiku-5-5", 0.10, 0.50),
            p("anthropic", "claude-haiku-4-5", 1.00, 5.00),
            p("anthropic", "claude-3-7-sonnet", 3.00, 15.00),
            p("anthropic", "claude-3-5-sonnet", 3.00, 15.00),
            p("anthropic", "claude-3-5-haiku", 0.80, 4.00),
            p("anthropic", "claude-3-opus", 15.00, 75.00),
            p("anthropic", "claude-3-haiku", 0.25, 1.25),
            // Google Gemini
            p("gemini", "gemini-2.5-pro", 1.25, 10.00),
            p("gemini", "gemini-2.5-flash-lite", 0.10, 0.40),
            p("gemini", "gemini-2.5-flash", 0.30, 2.50),
            p("gemini", "gemini-2.0-flash-lite", 0.075, 0.30),
            p("gemini", "gemini-2.0-flash", 0.10, 0.40),
            p("gemini", "gemini-1.5-pro", 1.25, 5.00),
            p("gemini", "gemini-1.5-flash", 0.075, 0.30),
            // xAI
            p("xai", "grok-4-fast", 0.20, 0.50),
            p("xai", "grok-4", 3.00, 15.00),
            p("xai", "grok-3-mini", 0.30, 0.50),
            p("xai", "grok-3", 3.00, 15.00),
            // Others
            p("mistral", "mistral-large", 2.00, 6.00),
            p("mistral", "mistral-small", 0.10, 0.30),
            p("deepseek", "deepseek-chat", 0.27, 1.10),
            p("deepseek", "deepseek-reasoner", 0.55, 2.19),
        ];
        Self { models }
    }

    /// The entry for a model: longest prefix of the normalized name.
    pub fn lookup(&self, model: &str) -> Option<&ModelPrice> {
        let m = normalize_model(model);
        self.models
            .iter()
            .filter(|p| m.starts_with(&p.model.to_ascii_lowercase()))
            .max_by_key(|p| p.model.len())
    }

    /// Cost in USD, or `None` for an unknown model.
    pub fn cost(&self, model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        self.lookup(model).map(|p| {
            (input_tokens as f64 * p.input_per_mtok + output_tokens as f64 * p.output_per_mtok)
                / 1_000_000.0
        })
    }

    /// Checks prices are finite and non-negative and names are non-empty.
    pub fn validate(&self) -> Result<(), String> {
        for m in &self.models {
            if m.model.trim().is_empty() {
                return Err("model name must not be empty".into());
            }
            for v in [m.input_per_mtok, m.output_per_mtok] {
                if !v.is_finite() || v < 0.0 {
                    return Err(format!("invalid price for {}", m.model));
                }
            }
        }
        Ok(())
    }
}

fn meta_str<'a>(meta: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| {
        meta.get(*k)
            .or_else(|| meta.get("attributes").and_then(|a| a.get(*k)))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    })
}

/// The model a run used: `metadata.model`, an OTLP `gen_ai.*.model`
/// attribute, or the run name for `llm` runs.
pub fn run_model(run: &Run) -> Option<String> {
    meta_str(
        &run.metadata,
        &["model", "gen_ai.response.model", "gen_ai.request.model"],
    )
    .map(str::to_string)
    .or_else(|| (run.kind == RunKind::Llm).then(|| run.name.clone()))
}

/// Cost of a run: an explicit `metadata.cost_usd` wins, else the pricing
/// table applied to the token usage of an `llm` run.
pub fn run_cost(pricing: &Pricing, run: &Run, model: Option<&str>) -> Option<f64> {
    if let Some(c) = run
        .metadata
        .get("cost_usd")
        .and_then(Value::as_f64)
        .filter(|c| c.is_finite() && *c >= 0.0)
    {
        return Some(c);
    }
    if run.kind != RunKind::Llm || (run.input_tokens.is_none() && run.output_tokens.is_none()) {
        return None;
    }
    pricing.cost(
        model?,
        u64::from(run.input_tokens.unwrap_or(0)),
        u64::from(run.output_tokens.unwrap_or(0)),
    )
}

/// Time to first token: `metadata.ttft_ms` (set by `traced_chat_stream`).
pub fn run_ttft(run: &Run) -> Option<i64> {
    run.metadata
        .get("ttft_ms")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(|v| v as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_prefix_and_normalization() {
        let p = Pricing::builtin();
        assert_eq!(
            p.lookup("gpt-4o-mini-2024-07-18").unwrap().model,
            "gpt-4o-mini"
        );
        assert_eq!(p.lookup("openai:gpt-4o").unwrap().model, "gpt-4o");
        assert_eq!(
            p.lookup("models/gemini-2.5-flash").unwrap().model,
            "gemini-2.5-flash"
        );
        assert_eq!(
            p.lookup("anthropic:claude-opus-5-5")
                .unwrap()
                .input_per_mtok,
            4.0
        );
        assert_eq!(
            p.lookup("claude-opus-4-1-20250805").unwrap().input_per_mtok,
            15.0
        );
        assert!(p.lookup("qwen3:4b").is_none());
        let c = p.cost("gpt-4o-mini", 1_000_000, 1_000_000).unwrap();
        assert!((c - 0.75).abs() < 1e-9);
        assert!(p.validate().is_ok());
    }
}
