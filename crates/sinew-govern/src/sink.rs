//! Decision and control events, the [`DecisionSink`] trait they are sent
//! to, and their trace runs.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::json;
use sinew_trace::{RunKind, RunRef, Tracer};

use crate::{
    decision::{Decision, Verdict},
    request::Request,
};

/// One authorization decision, as recorded for audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionEvent {
    /// Unique id (UUID v4).
    pub id: String,
    /// When the decision was made (milliseconds since the Unix epoch).
    pub time_ms: u64,
    /// The request, with the verdict's redacted fields already masked.
    pub request: Request,
    pub verdict: Verdict,
}

/// What an operator did to the control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ControlAction {
    Register,
    Unregister,
    Suspend,
    Quarantine,
    Reinstate,
    DisableTool,
    EnableTool,
    /// Policies were added or replaced (`target` names them).
    UpdatePolicies,
}

/// One operator action (kill switch, quarantine, registration), as
/// recorded for audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlEvent {
    pub id: String,
    pub time_ms: u64,
    pub action: ControlAction,
    /// The agent id or tool pattern acted on.
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Receives every decision and operator action of a
/// [`ControlPlane`](crate::ControlPlane): an audit log, a SIEM forwarder,
/// metrics. Calls are synchronous and on the request path, so slow sinks
/// should hand events to a background task.
pub trait DecisionSink: Send + Sync {
    fn record(&self, event: &DecisionEvent);

    /// Operator actions; ignored by default.
    fn record_control(&self, _event: &ControlEvent) {}
}

/// Keeps every event in memory (tests, dashboards).
#[derive(Debug, Default)]
pub struct MemoryDecisionSink {
    decisions: Mutex<Vec<DecisionEvent>>,
    controls: Mutex<Vec<ControlEvent>>,
}

impl MemoryDecisionSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn decisions(&self) -> Vec<DecisionEvent> {
        self.decisions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn controls(&self) -> Vec<ControlEvent> {
        self.controls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl DecisionSink for MemoryDecisionSink {
    fn record(&self, event: &DecisionEvent) {
        self.decisions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event.clone());
    }

    fn record_control(&self, event: &ControlEvent) {
        self.controls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event.clone());
    }
}

/// Records a decision as a finished `chain` run named
/// `policy:<tool or model>`, with the request as `inputs`, the decision as
/// `outputs` and the whole event in `metadata.governance`. Denials are
/// recorded as failed runs with the reason as the error, so they show up
/// as errors in Sinew Scope and OpenTelemetry backends.
pub fn record_decision(tracer: &Tracer, parent: Option<&RunRef>, event: &DecisionEvent) {
    let mut run = tracer.start(
        format!("policy:{}", event.request.target()),
        RunKind::Chain,
        parent,
        json!(event.request),
    );
    run.set_metadata("governance", json!(event));
    match &event.verdict.decision {
        Decision::Deny { reason } => run.fail(reason),
        d => run.finish(json!(d)),
    }
}
