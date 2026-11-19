//! [`GovernHook`]: enforces a [`ControlPlane`] in `sinew-harness` agents
//! (feature `harness`).

use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use sinew_core::ToolCall;
use sinew_harness::{Hook, HookContext, RiskLevel, StepPhase, ToolDecision};

use crate::{
    control::ControlPlane,
    decision::Decision,
    integrations::apply,
    pattern::glob_match,
    request::{Identity, Request},
};

impl ControlPlane {
    /// A harness [`Hook`] acting as `identity`; register it with
    /// [`AgentBuilder::hook`](sinew_harness::AgentBuilder::hook).
    ///
    /// ```no_run
    /// # fn demo() -> sinew_harness::Result<()> {
    /// use sinew_govern::{AgentProfile, ControlPlane, Identity, Rule};
    ///
    /// let cp = ControlPlane::builder()
    ///     .agent(AgentProfile::new("coder").allow_tools(["read_file", "grep", "write_file"]))
    ///     .policy(Rule::ask("writes").tools(["write_file"]))
    ///     .tool("write_file", true)
    ///     .build();
    /// let agent = sinew_harness::Agent::builder()
    ///     .model_spec("ollama:qwen3:4b")
    ///     .workspace(".")
    ///     .filesystem()
    ///     .hook(cp.hook(Identity::agent("coder").user("ana")))
    ///     .build()?;
    /// # Ok(()) }
    /// ```
    pub fn hook(&self, identity: impl Into<Identity>) -> GovernHook {
        GovernHook {
            cp: self.clone(),
            identity: identity.into(),
            model: None,
            risks: Vec::new(),
        }
    }
}

/// Enforces a [`ControlPlane`] in a `sinew-harness` agent:
///
/// - **Tool calls** (`before_tool`): `Allow` → allow, `Redact` → allow with
///   the fields masked in the call, `Ask` → the harness approval flow
///   ([`Agent::resume`](sinew_harness::Agent::resume) with
///   `ApprovalDecision`s), `Deny` → the model sees
///   `error: permission denied: <reason>`. The harness combines this with
///   its own `PermissionPolicy` and other hooks; the most restrictive
///   verdict wins. `auto_approve` (the CLI's `--yes`) turns every `Ask`
///   into `Allow`, the control plane's included: use `Deny` for limits that
///   must hold without a human.
/// - **Kill switch** (`should_stop`): before every model call and before
///   the tools of every step, a suspended (or unregistered) agent ends the
///   turn with `StopReason::Guardrail(reason)`. The stop is recorded as a
///   decision (target `*` before tools).
/// - **Model calls**: with [`model`](Self::model) set, every model call is
///   checked before it is made; a denial ends the turn the same way.
///
/// The hook knows a tool's side effects from the control plane's tool
/// catalog ([`ControlPlaneBuilder::tool`](crate::ControlPlaneBuilder::tool)
/// or [`toolbox`](crate::ControlPlaneBuilder::toolbox)); unknown tools are
/// treated as having none. Subagents inherit the hook and act under the
/// same identity; their name is recorded in `metadata.subagent`.
#[derive(Clone)]
pub struct GovernHook {
    cp: ControlPlane,
    identity: Identity,
    model: Option<String>,
    risks: Vec<(String, RiskLevel)>,
}

impl GovernHook {
    /// Risk level of the tools matching `tool_pattern` (glob), reported to
    /// the harness through [`Hook::tool_risk`]. It is carried in the
    /// harness `ApprovalRequest`, and calls at or above the agent's
    /// `ApprovalConfig::ask_at` (default `critical`) need approval. When
    /// several patterns match, the highest level wins; the harness then
    /// takes the highest of this and its own `ApprovalConfig` level.
    ///
    /// ```
    /// use sinew_govern::{ControlPlane, Identity};
    /// use sinew_harness::RiskLevel;
    ///
    /// let cp = ControlPlane::builder().build();
    /// let hook = cp
    ///     .hook(Identity::agent("bank-bot"))
    ///     .risk("transfer_*", RiskLevel::Critical)
    ///     .risk("send_*", RiskLevel::High);
    /// assert_eq!(hook.risk_for("transfer_funds"), Some(RiskLevel::Critical));
    /// assert_eq!(hook.risk_for("lookup"), None);
    /// ```
    pub fn risk(mut self, tool_pattern: impl Into<String>, level: RiskLevel) -> Self {
        self.risks.push((tool_pattern.into(), level));
        self
    }

    /// The risk level set for `tool`, if any pattern matches.
    pub fn risk_for(&self, tool: &str) -> Option<RiskLevel> {
        self.risks
            .iter()
            .filter(|(p, _)| glob_match(p, tool))
            .map(|(_, r)| *r)
            .max()
    }

    /// Checks every model call against this model id (the `provider:model`
    /// spec the agent uses).
    pub fn model(mut self, spec: impl Into<String>) -> Self {
        self.model = Some(spec.into());
        self
    }

    fn request(&self, ctx: &HookContext, req: Request) -> Request {
        let mut req = req
            .identity(&self.identity)
            .thread(ctx.thread_id.clone())
            .meta("step", json!(ctx.step));
        if ctx.depth > 0 {
            req = req.meta("subagent", json!(ctx.agent));
        }
        req
    }
}

#[async_trait]
impl Hook for GovernHook {
    async fn before_tool(&self, ctx: &HookContext, call: &mut ToolCall) -> ToolDecision {
        let req = self.request(
            ctx,
            Request::tool("", call.name.clone(), call.arguments.clone()),
        );
        let v = self.cp.check(req).await;
        apply(&v, call);
        match v.decision {
            Decision::Allow | Decision::Redact { .. } => ToolDecision::Allow,
            Decision::Ask { reason } => ToolDecision::Ask(reason),
            Decision::Deny { reason } => ToolDecision::Deny(reason),
        }
    }

    async fn tool_risk(&self, _ctx: &HookContext, call: &ToolCall) -> Option<RiskLevel> {
        self.risk_for(&call.name)
    }

    async fn should_stop(&self, ctx: &HookContext, phase: StepPhase) -> Option<String> {
        let step = match phase {
            StepPhase::BeforeModel => {
                Request::model("", self.model.clone().unwrap_or_else(|| "*".into()))
            }
            _ => Request::tool("", "*", Value::Null),
        };
        let step = self
            .request(ctx, step)
            .meta("phase", json!(format!("{phase:?}")));
        if let Some(reason) = self.cp.check_halt(&step, None, None) {
            return Some(reason);
        }
        if phase != StepPhase::BeforeModel || self.model.is_none() {
            return None;
        }
        match self.cp.check(step).await.decision {
            Decision::Allow | Decision::Redact { .. } => None,
            Decision::Deny { reason } => Some(reason),
            Decision::Ask { reason } => Some(format!(
                "{reason} (model calls cannot wait for approval in the harness)"
            )),
        }
    }
}
