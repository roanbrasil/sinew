//! Run results and streaming events.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinew_core::BoxStream;

use crate::{
    context::CompactionInfo,
    state::{ApprovalRequest, StopReason, Todo, UsageTotals},
};

/// Outcome of a finished turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    pub thread_id: String,
    /// The model's final answer (empty if the turn stopped early before
    /// one was produced).
    pub text: String,
    /// Parsed structured output, when an output schema is configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// Set when a budget or guardrail ended the turn early.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopReason>,
    /// Model calls in this turn.
    pub steps: u32,
    /// Usage of this turn.
    pub usage: UsageTotals,
    /// Usage of the whole thread so far.
    pub thread_usage: UsageTotals,
    /// The plan at the end of the turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub todos: Vec<Todo>,
}

/// What [`Agent::run`](crate::Agent::run) and [`Agent::resume`](crate::Agent::resume) return.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentOutcome {
    Completed(RunResult),
    /// The run paused before running tools that need approval. Answer with
    /// [`Agent::resume`](crate::Agent::resume).
    ApprovalRequired {
        thread_id: String,
        requests: Vec<ApprovalRequest>,
    },
}

impl AgentOutcome {
    /// The result, if the turn completed.
    pub fn completed(self) -> Option<RunResult> {
        match self {
            AgentOutcome::Completed(r) => Some(r),
            AgentOutcome::ApprovalRequired { .. } => None,
        }
    }

    pub fn is_approval_required(&self) -> bool {
        matches!(self, AgentOutcome::ApprovalRequired { .. })
    }
}

/// Something that happened during a streamed turn. A stream ends with
/// exactly one of `Done`, `ApprovalRequired` or `Error`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// A model call is starting (1-based within the turn).
    Step { step: u32 },
    /// A chunk of the model's text.
    Token { text: String },
    /// A tool is about to run (or be refused, see `ToolEnd::is_error`).
    ToolStart {
        id: String,
        name: String,
        arguments: Value,
    },
    /// A tool finished; `output` is what the model will see.
    ToolEnd {
        id: String,
        name: String,
        output: String,
        is_error: bool,
    },
    /// Older history was summarized to fit the context budget.
    Compacted(CompactionInfo),
    /// The turn paused: these calls need a decision.
    ApprovalRequired { requests: Vec<ApprovalRequest> },
    /// The turn finished.
    Done { result: RunResult },
    /// The turn failed.
    Error { message: String },
}

impl AgentEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            AgentEvent::ApprovalRequired { .. }
                | AgentEvent::Done { .. }
                | AgentEvent::Error { .. }
        )
    }
}

/// Stream returned by [`Agent::stream`](crate::Agent::stream). Dropping it
/// cancels the turn; checkpoints written so far stay.
pub type AgentStream = BoxStream<'static, AgentEvent>;
