//! Live smoke tests against real providers. Ignored by default; the nightly
//! workflow runs them with `SINEW_LIVE_SPECS` set to a comma-separated list of
//! `provider:model` specs whose API keys are available, e.g.
//! `SINEW_LIVE_SPECS=anthropic:claude-haiku-5-5,openai:gpt-4.1-mini cargo test -p sinew-core --test live_providers -- --ignored`.
//!
//! Each spec checks a plain chat, a stream, a tool call and usage reporting,
//! so a provider changing its wire format is caught within a day.

use futures_util::StreamExt;
use serde_json::json;
use sinew_core::{
    providers::{capabilities_for, from_spec},
    ChatRequest, Message, StreamEvent, ToolChoice, ToolSpec,
};

fn specs() -> Vec<String> {
    std::env::var("SINEW_LIVE_SPECS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

#[tokio::test]
#[ignore = "calls real providers (SINEW_LIVE_SPECS)"]
async fn chat_stream_and_tools_work_for_every_spec() {
    let specs = specs();
    assert!(!specs.is_empty(), "set SINEW_LIVE_SPECS");
    let mut failures = Vec::new();
    for spec in &specs {
        if let Err(e) = check(spec).await {
            failures.push(format!("{spec}: {e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "live failures:\n{}",
        failures.join("\n")
    );
}

async fn check(spec: &str) -> Result<(), String> {
    let model = from_spec(spec).map_err(|e| e.to_string())?;
    let (provider, name) = spec.split_once(':').unwrap_or((spec, ""));

    // Plain chat with usage.
    let r = model
        .chat(
            ChatRequest::new(vec![Message::user("Reply with the single word: pong")])
                .with_max_tokens(20),
        )
        .await
        .map_err(|e| format!("chat: {e}"))?;
    if !r.message.content.to_lowercase().contains("pong") {
        return Err(format!("chat: unexpected answer {:?}", r.message.content));
    }
    if r.usage.input_tokens == 0 {
        return Err("chat: no input token usage reported".into());
    }

    // Streaming: text deltas and a final Done.
    let mut s = model
        .chat_stream(
            ChatRequest::new(vec![Message::user("Count from 1 to 5.")]).with_max_tokens(40),
        )
        .await
        .map_err(|e| format!("stream: {e}"))?;
    let (mut deltas, mut done) = (0, false);
    while let Some(ev) = s.next().await {
        match ev.map_err(|e| format!("stream: {e}"))? {
            StreamEvent::TextDelta { .. } => deltas += 1,
            StreamEvent::Done { .. } => done = true,
            _ => {}
        }
    }
    if deltas == 0 || !done {
        return Err(format!("stream: {deltas} deltas, done = {done}"));
    }

    // Tool call, when the provider supports tools.
    if capabilities_for(provider, name).is_some_and(|c| c.tools) {
        let tool = ToolSpec {
            name: "get_weather".into(),
            description: "Current weather for a city".into(),
            parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        };
        let r = model
            .chat(
                ChatRequest::new(vec![Message::user(
                    "What is the weather in Montreal? Use the tool.",
                )])
                .with_tools(vec![tool])
                .with_tool_choice(ToolChoice::Required)
                .with_max_tokens(200),
            )
            .await
            .map_err(|e| format!("tools: {e}"))?;
        let call = r.message.tool_calls.first().ok_or("tools: no tool call")?;
        if call.name != "get_weather" || call.arguments["city"].as_str().is_none() {
            return Err(format!("tools: unexpected call {call:?}"));
        }
    }
    Ok(())
}
