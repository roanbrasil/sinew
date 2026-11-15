//! Picks a chat model from the command line so every sample runs offline or
//! against any provider:
//!
//! - no flag: a scripted model, fully offline and deterministic;
//! - `--ollama`: local Ollama (`SINEW_MODEL`, default `qwen3:4b-instruct-2507-q4_K_M`);
//! - `--openai`, `--anthropic`, `--gemini`, `--xai`: hosted APIs with a default model;
//! - `--model provider:model`: anything [`from_spec`] understands, e.g.
//!   `--model lmstudio:qwen3-8b`, `--model groq:llama-3.3-70b-versatile`,
//!   `--model vllm:Qwen/Qwen3-8B`.

use std::sync::Arc;

use sinew::core::{mock::ScriptedModel, providers::from_spec, ChatModel, Message};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn model_name(default: &str) -> String {
    std::env::var("SINEW_MODEL").unwrap_or_else(|_| default.to_string())
}

/// The `provider:model` the command line asks for, if any.
fn requested_spec() -> Option<String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--model") {
        return args.get(i + 1).cloned();
    }
    let flag = args.iter().find(|a| a.starts_with("--"))?;
    let (provider, default) = match flag.as_str() {
        "--ollama" => ("ollama", "qwen3:4b-instruct-2507-q4_K_M"),
        "--openai" => ("openai", "gpt-4.1-mini"),
        "--anthropic" => ("anthropic", "claude-sonnet-5-5"),
        "--gemini" => ("gemini", "gemini-2.5-flash"),
        "--xai" => ("xai", "grok-4"),
        _ => return None,
    };
    Some(format!("{provider}:{}", model_name(default)))
}

/// `scripted` is what the offline model answers, in order.
pub fn pick_model(scripted: Vec<Message>) -> Result<Arc<dyn ChatModel>, BoxError> {
    Ok(match requested_spec() {
        Some(spec) => from_spec(&spec)?,
        None => Arc::new(ScriptedModel::new(scripted)),
    })
}

/// True when the sample runs against a real model.
pub fn is_live() -> bool {
    requested_spec().is_some()
}
