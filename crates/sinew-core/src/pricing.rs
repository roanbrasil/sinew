//! Token prices and cost estimates.
//!
//! [`price_for`] looks a model up in a built-in table of list prices (USD per
//! million tokens) for common OpenAI, Anthropic, Google Gemini and xAI
//! models; [`Usage::cost`](crate::Usage::cost) uses it. Prices change, and
//! long-context or batch tiers are not modelled, so treat the result as an
//! estimate. Register your own (or override built-in) prices with
//! [`set_price`].
//!
//! Lookup is by longest prefix after stripping a `provider:` or `vendor/`
//! prefix, so `gpt-4o-mini-2024-07-18`, `openai/gpt-4o-mini` and
//! `openai:gpt-4o-mini` all resolve to `gpt-4o-mini`.

use std::{
    collections::HashMap,
    sync::{OnceLock, RwLock},
};

use crate::Usage;

/// USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPrice {
    pub input: f64,
    pub output: f64,
    /// Price of prompt tokens read from cache; defaults to `input`.
    pub cached_input: Option<f64>,
    /// Price of prompt tokens written to cache; defaults to `input`.
    pub cache_write: Option<f64>,
}

impl ModelPrice {
    pub const fn new(input: f64, output: f64) -> Self {
        Self {
            input,
            output,
            cached_input: None,
            cache_write: None,
        }
    }

    pub const fn with_cached(mut self, cached_input: f64) -> Self {
        self.cached_input = Some(cached_input);
        self
    }

    pub const fn with_cache_write(mut self, cache_write: f64) -> Self {
        self.cache_write = Some(cache_write);
        self
    }

    /// Cost of `usage` in USD.
    pub fn cost(&self, usage: &Usage) -> f64 {
        let cached = usage.cached_input_tokens.min(usage.input_tokens);
        let written = usage
            .cache_creation_input_tokens
            .min(usage.input_tokens - cached);
        let plain = usage.input_tokens - cached - written;
        let m = 1_000_000.0;
        f64::from(plain) * self.input / m
            + f64::from(cached) * self.cached_input.unwrap_or(self.input) / m
            + f64::from(written) * self.cache_write.unwrap_or(self.input) / m
            + f64::from(usage.output_tokens) * self.output / m
    }
}

/// Anthropic: cache reads cost 10% of input, 5-minute cache writes 125%.
const fn claude(input: f64, output: f64) -> ModelPrice {
    ModelPrice::new(input, output)
        .with_cached(input * 0.1)
        .with_cache_write(input * 1.25)
}

/// Built-in list prices (USD / 1M tokens), standard tier, short context.
/// More specific prefixes must not be shadowed: lookup picks the longest
/// matching prefix, so order does not matter.
pub const BUILTIN_PRICES: &[(&str, ModelPrice)] = &[
    // OpenAI
    ("gpt-5", ModelPrice::new(1.25, 10.0).with_cached(0.125)),
    ("gpt-5-mini", ModelPrice::new(0.25, 2.0).with_cached(0.025)),
    ("gpt-5-nano", ModelPrice::new(0.05, 0.40).with_cached(0.005)),
    ("gpt-4.1", ModelPrice::new(2.0, 8.0).with_cached(0.50)),
    (
        "gpt-4.1-mini",
        ModelPrice::new(0.40, 1.60).with_cached(0.10),
    ),
    (
        "gpt-4.1-nano",
        ModelPrice::new(0.10, 0.40).with_cached(0.025),
    ),
    ("gpt-4o", ModelPrice::new(2.50, 10.0).with_cached(1.25)),
    (
        "gpt-4o-mini",
        ModelPrice::new(0.15, 0.60).with_cached(0.075),
    ),
    ("gpt-4-turbo", ModelPrice::new(10.0, 30.0)),
    ("gpt-3.5-turbo", ModelPrice::new(0.50, 1.50)),
    ("o1", ModelPrice::new(15.0, 60.0).with_cached(7.50)),
    ("o1-mini", ModelPrice::new(1.10, 4.40).with_cached(0.55)),
    ("o3", ModelPrice::new(2.0, 8.0).with_cached(0.50)),
    ("o3-mini", ModelPrice::new(1.10, 4.40).with_cached(0.55)),
    ("o4-mini", ModelPrice::new(1.10, 4.40).with_cached(0.275)),
    // Anthropic
    ("claude-fable-5", claude(10.0, 50.0)),
    ("claude-mythos-5", claude(10.0, 50.0)),
    ("claude-opus-5-5", claude(4.0, 20.0)),
    ("claude-opus-5", claude(5.0, 25.0)),
    ("claude-opus-4-8", claude(5.0, 25.0)),
    ("claude-opus-4-7", claude(5.0, 25.0)),
    ("claude-opus-4-6", claude(5.0, 25.0)),
    ("claude-opus-4-5", claude(5.0, 25.0)),
    ("claude-opus-4", claude(15.0, 75.0)),
    ("claude-sonnet-5-5", claude(2.0, 10.0)),
    ("claude-sonnet-5", claude(2.0, 10.0)),
    ("claude-sonnet-4", claude(3.0, 15.0)),
    ("claude-3-7-sonnet", claude(3.0, 15.0)),
    ("claude-3-5-sonnet", claude(3.0, 15.0)),
    ("claude-haiku-5-5", claude(0.10, 0.50)),
    ("claude-haiku-4-5", claude(1.0, 5.0)),
    ("claude-3-5-haiku", claude(0.80, 4.0)),
    ("claude-3-haiku", claude(0.25, 1.25)),
    // Google Gemini (prompts up to 200k tokens)
    ("gemini-3-pro", ModelPrice::new(2.0, 12.0).with_cached(0.20)),
    (
        "gemini-2.5-pro",
        ModelPrice::new(1.25, 10.0).with_cached(0.31),
    ),
    (
        "gemini-2.5-flash",
        ModelPrice::new(0.30, 2.50).with_cached(0.075),
    ),
    (
        "gemini-2.5-flash-lite",
        ModelPrice::new(0.10, 0.40).with_cached(0.025),
    ),
    (
        "gemini-2.0-flash",
        ModelPrice::new(0.10, 0.40).with_cached(0.025),
    ),
    ("gemini-2.0-flash-lite", ModelPrice::new(0.075, 0.30)),
    ("gemini-1.5-pro", ModelPrice::new(1.25, 5.0)),
    ("gemini-1.5-flash", ModelPrice::new(0.075, 0.30)),
    // xAI
    ("grok-4", ModelPrice::new(3.0, 15.0).with_cached(0.75)),
    ("grok-4-fast", ModelPrice::new(0.20, 0.50).with_cached(0.05)),
    (
        "grok-code-fast",
        ModelPrice::new(0.20, 1.50).with_cached(0.02),
    ),
    ("grok-3", ModelPrice::new(3.0, 15.0).with_cached(0.75)),
    (
        "grok-3-mini",
        ModelPrice::new(0.30, 0.50).with_cached(0.075),
    ),
];

fn custom() -> &'static RwLock<HashMap<String, ModelPrice>> {
    static C: OnceLock<RwLock<HashMap<String, ModelPrice>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// Registers (or overrides) the price of every model whose id starts with
/// `prefix`. Takes precedence over built-in prices of the same length.
pub fn set_price(prefix: impl Into<String>, price: ModelPrice) {
    custom()
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(prefix.into().to_ascii_lowercase(), price);
}

fn normalise(model: &str) -> String {
    let m = model.trim().to_ascii_lowercase();
    // `provider:model` (from_spec) and `vendor/model` (OpenRouter).
    let m = match m.split_once(':') {
        Some((p, rest)) if !p.contains('/') && !rest.is_empty() => rest.to_string(),
        _ => m,
    };
    let m = m.rsplit('/').next().unwrap_or(&m).to_string();
    // Gemini returns `models/gemini-...`; Bedrock/Vertex add prefixes.
    m.trim_start_matches("anthropic.").to_string()
}

fn is_boundary(model: &str, prefix: &str) -> bool {
    // `gpt-4o` must not match `gpt-4o-mini`... but the longer prefix wins
    // anyway; this just stops `o1` matching `o1x`.
    model.len() == prefix.len()
        || matches!(
            model.as_bytes()[prefix.len()],
            b'-' | b'@' | b'.' | b':' | b'_'
        )
}

/// The price for `model`, if known.
pub fn price_for(model: &str) -> Option<ModelPrice> {
    let m = normalise(model);
    let mut best: Option<(usize, ModelPrice)> = None;
    {
        let custom = custom().read().unwrap_or_else(|e| e.into_inner());
        for (p, price) in custom.iter() {
            if m.starts_with(p.as_str())
                && is_boundary(&m, p)
                && best.is_none_or(|(l, _)| p.len() > l)
            {
                best = Some((p.len(), *price));
            }
        }
    }
    for (p, price) in BUILTIN_PRICES {
        if m.starts_with(p) && is_boundary(&m, p) && best.is_none_or(|(l, _)| p.len() > l) {
            best = Some((p.len(), *price));
        }
    }
    best.map(|(_, p)| p)
}

/// Rough token count for `text` without a tokenizer: about 4 characters per
/// token for English, never fewer than one token per 1.3 words. Good for
/// budgeting and rate limiting, not for billing.
pub fn estimate_tokens(text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }
    let chars = text.chars().count() as f64;
    let words = text.split_whitespace().count() as f64;
    (chars / 4.0).max(words * 1.3).ceil() as u32
}

/// Rough prompt size of a request: message text, a per-message overhead,
/// tool schemas and a flat allowance per image / document / audio part.
pub fn estimate_request_tokens(req: &crate::ChatRequest) -> u32 {
    use crate::message::ContentPart;
    let mut n = 3u32;
    for m in &req.messages {
        n += 4 + estimate_tokens(&m.content);
        for c in &m.tool_calls {
            n += estimate_tokens(&c.name) + estimate_tokens(&c.arguments.to_string());
        }
        for p in &m.parts {
            n += match p {
                ContentPart::Text { text } => estimate_tokens(text),
                ContentPart::Image { .. } => 800,
                ContentPart::Document { .. } => 1500,
                ContentPart::Audio { .. } => 1000,
            };
        }
    }
    for t in &req.tools {
        n += estimate_tokens(&t.name)
            + estimate_tokens(&t.description)
            + estimate_tokens(&t.parameters.to_string());
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_uses_longest_prefix_and_strips_prefixes() {
        assert_eq!(price_for("gpt-4o-mini-2024-07-18").unwrap().input, 0.15);
        assert_eq!(price_for("gpt-4o-2024-08-06").unwrap().input, 2.50);
        assert_eq!(price_for("openai:gpt-4o-mini").unwrap().input, 0.15);
        assert_eq!(price_for("openai/gpt-4.1-nano").unwrap().input, 0.10);
        assert_eq!(price_for("models/gemini-2.5-flash").unwrap().input, 0.30);
        assert_eq!(price_for("gemini-2.5-flash-lite").unwrap().input, 0.10);
        assert_eq!(price_for("claude-opus-5-5").unwrap().input, 4.0);
        assert_eq!(price_for("claude-opus-5").unwrap().input, 5.0);
        assert_eq!(price_for("xai:grok-4-0709").unwrap().output, 15.0);
        assert!(price_for("o1x").is_none());
        assert!(price_for("qwen3:4b").is_none());
    }

    #[test]
    fn cost_accounts_for_cached_and_written_tokens() {
        let u = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cached_input_tokens: 500_000,
            ..Default::default()
        };
        // gpt-4o: 0.5M * 2.5 + 0.5M * 1.25 + 1M * 10
        let c = u.cost("gpt-4o").unwrap();
        assert!((c - 11.875).abs() < 1e-9, "{c}");
        let u = Usage {
            input_tokens: 1_000_000,
            cache_creation_input_tokens: 1_000_000,
            ..Default::default()
        };
        assert!((u.cost("claude-sonnet-5-5").unwrap() - 2.5).abs() < 1e-9);
        assert!(Usage::new(10, 10).cost("my-local-model").is_none());
    }

    #[test]
    fn custom_prices_override() {
        set_price("my-finetune", ModelPrice::new(1.0, 2.0));
        let c = Usage::new(1_000_000, 500_000)
            .cost("my-finetune-v2")
            .unwrap();
        assert!((c - 2.0).abs() < 1e-9);
    }

    #[test]
    fn token_estimates_are_sane() {
        assert_eq!(estimate_tokens(""), 0);
        let n = estimate_tokens("The quick brown fox jumps over the lazy dog.");
        assert!((9..=14).contains(&n), "{n}");
    }
}
