//! Small building blocks shared by the scenarios: tools that record their
//! side effects, tools that return attacker-controlled text, and drivers for
//! `ReactAgent` and harness agents.

use async_trait::async_trait;
use serde_json::{json, Map, Value};
use sinew::core::{Message, Role, Tool, ToolCall, ToolSpec};
use sinew::eval::security::{EffectLog, Observation};
use sinew::graph::{prebuilt::AgentState, CompiledGraph};
use sinew::harness::{Agent, AgentOutcome, ApprovalDecision, ApprovalRequest};

/// A tool call.
pub fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
    }
}

/// An assistant message that calls one tool.
pub fn use_tool(id: &str, name: &str, arguments: Value) -> Message {
    Message::assistant_with_tools("", vec![call(id, name, arguments)])
}

/// A JSON Schema object whose properties are all required strings.
pub fn strings(properties: &[&str]) -> Value {
    let mut props = Map::new();
    for p in properties {
        props.insert((*p).into(), json!({"type": "string"}));
    }
    json!({"type": "object", "properties": props, "required": properties})
}

/// A read-only tool that returns fixed text: a web page, a ticket, an API
/// response. Attacker-controlled content enters the agent through it.
pub struct StaticTool {
    spec: ToolSpec,
    reply: String,
}

/// See [`StaticTool`].
pub fn static_tool(
    name: &str,
    description: &str,
    parameters: Value,
    reply: impl Into<String>,
) -> StaticTool {
    StaticTool {
        spec: ToolSpec {
            name: name.into(),
            description: description.into(),
            parameters,
        },
        reply: reply.into(),
    }
}

#[async_trait]
impl Tool for StaticTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }
    async fn call(&self, _arguments: Value) -> sinew::core::Result<Value> {
        Ok(Value::String(self.reply.clone()))
    }
}

/// A tool that changes something outside the agent. Every call that runs is
/// recorded in an [`EffectLog`]; calls refused by a control are not.
pub struct EffectTool {
    spec: ToolSpec,
    log: EffectLog,
    reply: String,
}

/// See [`EffectTool`].
pub fn effect_tool(
    name: &str,
    description: &str,
    parameters: Value,
    log: &EffectLog,
    reply: &str,
) -> EffectTool {
    EffectTool {
        spec: ToolSpec {
            name: name.into(),
            description: description.into(),
            parameters,
        },
        log: log.clone(),
        reply: reply.into(),
    }
}

#[async_trait]
impl Tool for EffectTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }
    fn side_effects(&self) -> bool {
        true
    }
    async fn call(&self, arguments: Value) -> sinew::core::Result<Value> {
        self.log.record(self.spec.name.clone(), arguments);
        Ok(Value::String(self.reply.clone()))
    }
}

/// Runs a `ReactAgent` graph on one user message and returns the thread.
pub async fn run_react(
    agent: &CompiledGraph<AgentState>,
    input: &str,
) -> Result<Vec<Message>, String> {
    agent
        .invoke(AgentState::new(vec![Message::user(input)]))
        .await
        .map(|s| s.messages)
        .map_err(|e| e.to_string())
}

/// Notes on what the controls did: tool results that are errors or blocks
/// (permission denials, guardrail blocks, SSRF refusals, unknown outcomes),
/// and the final answer.
pub fn control_notes(messages: &[Message]) -> Vec<String> {
    let mut names = std::collections::HashMap::new();
    for m in messages {
        for c in &m.tool_calls {
            names.insert(c.id.clone(), c.name.clone());
        }
    }
    let mut notes = Vec::new();
    for m in messages.iter().filter(|m| m.role == Role::Tool) {
        let text = m.content.trim_start();
        if text.starts_with("error:") || text.starts_with("blocked") || text.starts_with("skipped")
        {
            let tool = m
                .tool_call_id
                .as_ref()
                .and_then(|id| names.get(id))
                .map_or("tool", String::as_str);
            notes.push(format!("{tool} -> {}", clip(text, 140)));
        }
    }
    if let Some(last) = messages
        .last()
        .filter(|m| m.role == Role::Assistant && m.tool_calls.is_empty())
    {
        notes.push(format!("answer: {}", clip(&last.content, 140)));
    }
    notes
}

fn clip(s: &str, n: usize) -> String {
    let mut out: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        out.push('…');
    }
    out
}

/// An observation of a thread, with [`control_notes`].
pub fn observe(messages: Vec<Message>) -> Observation {
    let mut obs = Observation::new();
    for n in control_notes(&messages) {
        obs = obs.note(n);
    }
    obs.with_messages(messages)
}

/// Decides an approval request, given the text of the assistant message that
/// asked for it (what a chat UI shows next to the request).
pub type Reviewer = dyn Fn(&str, &ApprovalRequest) -> bool + Send + Sync;

/// A reviewer who denies every request.
pub fn deny_all(_: &str, _: &ApprovalRequest) -> bool {
    false
}

/// Runs a harness agent on one user message, answering approval requests
/// with `reviewer`, and returns what happened (without effects; add them
/// with [`Observation::with_effects`]).
pub async fn run_harness(
    agent: &Agent,
    input: &str,
    reviewer: &Reviewer,
) -> Result<Observation, String> {
    let thread = "attack";
    let mut outcome = agent.run(thread, input).await.map_err(|e| e.to_string())?;
    let mut approvals = Vec::new();
    let result = loop {
        match outcome {
            AgentOutcome::Completed(r) => break r,
            AgentOutcome::ApprovalRequired { requests, .. } => {
                if approvals.len() > 20 {
                    return Err("too many approval rounds".into());
                }
                let state = agent.state(thread).await.map_err(|e| e.to_string())?;
                let shown = state
                    .as_ref()
                    .and_then(|s| s.messages.iter().rev().find(|m| !m.tool_calls.is_empty()))
                    .map(|m| m.content.clone())
                    .unwrap_or_default();
                let decisions = requests
                    .iter()
                    .map(|r| {
                        approvals.push(r.tool.clone());
                        if reviewer(&shown, r) {
                            ApprovalDecision::approve(r.call_id.clone())
                        } else {
                            ApprovalDecision::deny(
                                r.call_id.clone(),
                                "the user did not ask for this",
                            )
                        }
                    })
                    .collect();
                outcome = agent
                    .resume(thread, decisions)
                    .await
                    .map_err(|e| e.to_string())?;
            }
        }
    };
    let state = agent
        .state(thread)
        .await
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    let mut obs = observe(state.messages);
    if !result.text.is_empty() {
        obs = obs.with_final_answer(result.text.clone());
    }
    if let Some(stop) = &result.stop {
        obs = obs.with_stop(stop.to_string());
    }
    for a in approvals {
        obs = obs.with_approval(a);
    }
    Ok(obs)
}
