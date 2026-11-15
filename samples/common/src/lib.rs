//! Picks a chat model from the command line so every sample runs three ways:
//!
//! - no flag: a scripted model, fully offline and deterministic;
//! - `--ollama`: a local Ollama model (`SINEW_MODEL`, default `qwen3:4b-instruct-2507-q4_K_M`);
//! - `--openai`: OpenAI (`OPENAI_API_KEY`, `SINEW_MODEL` default `gpt-4.1-mini`);
//! - `--anthropic`: Anthropic (`ANTHROPIC_API_KEY`, `SINEW_MODEL` default `claude-sonnet-5-5`).

use std::sync::Arc;

use sinew::core::{mock::ScriptedModel, ChatModel, Message};
use sinew::prelude::{Anthropic, OpenAiCompat};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn model_name(default: &str) -> String {
    std::env::var("SINEW_MODEL").unwrap_or_else(|_| default.to_string())
}

/// `scripted` is what the offline model answers, in order.
pub fn pick_model(scripted: Vec<Message>) -> Result<Arc<dyn ChatModel>, BoxError> {
    let flag = std::env::args()
        .skip(1)
        .find(|a| a.starts_with("--"))
        .unwrap_or_default();
    Ok(match flag.as_str() {
        "--ollama" => Arc::new(OpenAiCompat::ollama(model_name(
            "qwen3:4b-instruct-2507-q4_K_M",
        ))),
        "--openai" => Arc::new(OpenAiCompat::openai(model_name("gpt-4.1-mini"))?),
        "--anthropic" => Arc::new(Anthropic::from_env(model_name("claude-sonnet-5-5"))?),
        _ => Arc::new(ScriptedModel::new(scripted)),
    })
}

/// True when the sample runs against a real model.
pub fn is_live() -> bool {
    std::env::args().any(|a| matches!(a.as_str(), "--ollama" | "--openai" | "--anthropic"))
}
