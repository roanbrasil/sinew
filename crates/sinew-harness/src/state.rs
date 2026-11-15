//! The graph state of a harness thread and the types around it.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinew_core::{Message, ToolCall, Usage};

use crate::budget::Pricing;

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Token and cost totals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageTotals {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub model_calls: u64,
    /// Estimated cost in USD (0 unless [`Pricing`] is configured).
    pub cost_usd: f64,
}

impl UsageTotals {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    pub(crate) fn add(&mut self, usage: Usage, pricing: Option<&Pricing>) {
        self.input_tokens += u64::from(usage.input_tokens);
        self.output_tokens += u64::from(usage.output_tokens);
        self.model_calls += 1;
        if let Some(p) = pricing {
            self.cost_usd += p.cost(usage);
        }
    }
}

/// Why a turn stopped before the model gave a final answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum StopReason {
    MaxSteps,
    TokenBudget,
    CostBudget,
    Timeout,
    /// A guardrail blocked the input or the final answer.
    Guardrail(String),
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::MaxSteps => f.write_str("max steps reached"),
            StopReason::TokenBudget => f.write_str("token budget exhausted"),
            StopReason::CostBudget => f.write_str("cost budget exhausted"),
            StopReason::Timeout => f.write_str("wall-clock timeout"),
            StopReason::Guardrail(r) => write!(f, "blocked by guardrail ({r})"),
        }
    }
}

/// Per-turn bookkeeping (reset for every new user message).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TurnState {
    /// Model calls made in this turn.
    pub steps: u32,
    pub usage: UsageTotals,
    /// Unix-epoch milliseconds after which no new model call starts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopReason>,
}

/// Permission outcome for one pending tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum CallDecision {
    Allow,
    Ask { reason: String },
    Deny { reason: String },
}

/// A tool call the model requested, with its permission outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingCall {
    pub call: ToolCall,
    pub decision: CallDecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// One entry of the agent's plan (`write_todos`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Todo {
    pub content: String,
    pub status: TodoStatus,
}

/// Graph state of a thread. Checkpointed after every node, so threads
/// survive interrupts (and restarts, with a durable checkpointer).
///
/// The system prompt is not stored here: it is rebuilt for every model call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HarnessState {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub todos: Vec<Todo>,
    /// Totals for the whole thread.
    #[serde(default)]
    pub usage: UsageTotals,
    #[serde(default)]
    pub turn: TurnState,
    /// Tool calls of the last assistant message, waiting to run.
    #[serde(default)]
    pub pending: Vec<PendingCall>,
    /// Tools the user approved "always" in this thread.
    #[serde(default)]
    pub always_allow: Vec<String>,
    /// How many times the history was compacted.
    #[serde(default)]
    pub compactions: u32,
}

impl HarnessState {
    /// Content of the last assistant message without tool calls, if the
    /// conversation ends with one.
    pub fn final_text(&self) -> Option<&str> {
        self.messages
            .last()
            .filter(|m| m.role == sinew_core::Role::Assistant)
            .map(|m| m.content.as_str())
    }

    /// Calls waiting for a human decision.
    pub fn approval_requests(&self) -> Vec<ApprovalRequest> {
        self.pending
            .iter()
            .filter_map(|p| match &p.decision {
                CallDecision::Ask { reason } => Some(ApprovalRequest {
                    call_id: p.call.id.clone(),
                    tool: p.call.name.clone(),
                    arguments: p.call.arguments.clone(),
                    reason: reason.clone(),
                }),
                _ => None,
            })
            .collect()
    }
}

/// A tool call that needs a human decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub call_id: String,
    pub tool: String,
    pub arguments: Value,
    pub reason: String,
}

/// A human decision for one [`ApprovalRequest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalDecision {
    pub call_id: String,
    pub approve: bool,
    /// Also allow every later call of this tool in the thread.
    #[serde(default)]
    pub always: bool,
    /// Shown to the model when the call is denied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl ApprovalDecision {
    pub fn approve(call_id: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            approve: true,
            always: false,
            reason: None,
        }
    }

    pub fn approve_always(call_id: impl Into<String>) -> Self {
        Self {
            always: true,
            ..Self::approve(call_id)
        }
    }

    pub fn deny(call_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            approve: false,
            always: false,
            reason: Some(reason.into()),
        }
    }

    /// Approves every request.
    pub fn approve_all(requests: &[ApprovalRequest]) -> Vec<Self> {
        requests
            .iter()
            .map(|r| Self::approve(r.call_id.clone()))
            .collect()
    }
}
