//! Hooks (middleware) around model and tool calls.
//!
//! Every method has a no-op default, so a hook implements only what it needs.
//! Hooks run in registration order.

use async_trait::async_trait;
use sinew_core::{ChatRequest, ChatResponse, ToolCall};

use crate::Result;

/// Where a hook is running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookContext {
    pub thread_id: String,
    /// Model calls already made in this turn.
    pub step: u32,
    /// Subagent nesting level (0 = top-level agent).
    pub depth: usize,
    /// Name of the agent (or subagent) running.
    pub agent: String,
}

/// A `before_tool` verdict. When several hooks and the permission policy
/// disagree, the most restrictive verdict wins (deny > ask > allow).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolDecision {
    Allow,
    /// Pause the run for human approval (see [`Agent::resume`](crate::Agent::resume)).
    Ask(String),
    /// Do not run the tool; the model sees `error: permission denied: <reason>`.
    Deny(String),
}

#[async_trait]
pub trait Hook: Send + Sync {
    /// Inspect or rewrite the request (messages, tools, temperature...).
    /// An error aborts the run.
    async fn before_model(&self, _ctx: &HookContext, _request: &mut ChatRequest) -> Result<()> {
        Ok(())
    }

    /// Inspect or rewrite the model's reply before it enters the state.
    async fn after_model(&self, _ctx: &HookContext, _response: &mut ChatResponse) -> Result<()> {
        Ok(())
    }

    /// Allow, deny or ask for approval; may rewrite the call's arguments.
    async fn before_tool(&self, _ctx: &HookContext, _call: &mut ToolCall) -> ToolDecision {
        ToolDecision::Allow
    }

    /// Inspect or rewrite a tool's output before the model sees it.
    async fn after_tool(&self, _ctx: &HookContext, _call: &ToolCall, _output: &mut String) {}
}
