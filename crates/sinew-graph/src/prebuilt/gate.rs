//! [`AgentGate`]: an authorization check before every model call and every
//! tool call of a [`ReactAgent`](super::ReactAgent).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sinew_core::ToolCall;

use crate::NodeContext;

/// The reply an agent stores as its final message when a gate stops the
/// run, followed by the gate's reason.
pub const GATE_STOP_PREFIX: &str = "The run was stopped: ";

/// What an [`AgentGate`] says about the next model call or tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", content = "reason", rename_all = "snake_case")]
#[non_exhaustive]
pub enum GateVerdict {
    /// Go ahead.
    Allow,
    /// Skip this tool call; the model sees `error: denied: <reason>` as its
    /// result and the run continues. Before a model call it stops the run,
    /// like [`Stop`](Self::Stop).
    Deny(String),
    /// Pause the run with an interrupt (`{"approval_required": {...}}`) and
    /// wait for a human: resume with `true` to go ahead, `false` to deny.
    /// Needs a checkpointer, like every interrupt.
    Ask(String),
    /// End the run now: remaining tool calls are answered with
    /// `skipped: <reason>` and the agent's final message is
    /// [`GATE_STOP_PREFIX`] followed by the reason.
    Stop(String),
}

/// An authorization check that a [`ReactAgent`](super::ReactAgent) runs
/// before each model call and before each tool call (see
/// [`ReactAgent::gate`](super::ReactAgent::gate)). Policy engines and
/// control planes (the `sinew-govern` crate) implement it to enforce
/// permissions and kill switches at the next step of a running agent.
///
/// Both methods default to [`GateVerdict::Allow`].
#[async_trait]
pub trait AgentGate: Send + Sync {
    /// Called in the `agent` node before the model is called. `model` is
    /// the model's [`name`](sinew_core::ChatModel::name).
    async fn before_model(&self, _ctx: &NodeContext, _model: &str) -> GateVerdict {
        GateVerdict::Allow
    }

    /// Called in the `tools` node before `call` runs. The gate may rewrite
    /// the call's arguments (for example to mask a field); the rewritten
    /// call is the one that runs. `side_effects` is
    /// [`Tool::side_effects`](sinew_core::Tool::side_effects) of the tool
    /// (`false` for unknown tools and handoffs).
    async fn before_tool(
        &self,
        _ctx: &NodeContext,
        _call: &mut ToolCall,
        _side_effects: bool,
    ) -> GateVerdict {
        GateVerdict::Allow
    }
}
