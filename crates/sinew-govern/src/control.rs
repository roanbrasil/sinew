//! [`ControlPlane`]: the agent registry, the policies, the kill switch and
//! the decision events, in one place.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
};

use sinew_core::ToolBox;
use sinew_trace::{RunRef, Tracer};

use crate::{
    decision::{Decision, Verdict},
    file::PolicyFile,
    pattern::glob_match,
    policy::{Defaults, Policy, PolicySet},
    registry::{AgentProfile, AgentStatus, AgentUsage},
    request::{now_ms, Request, RequestKind},
    sink::{record_decision, ControlAction, ControlEvent, DecisionEvent, DecisionSink},
};

/// Policy id of decisions made because the agent is not registered.
pub const POLICY_REGISTRY: &str = "control-plane/registry";
/// Policy id of decisions made by the kill switch (suspended agent or
/// disabled tool).
pub const POLICY_KILL_SWITCH: &str = "control-plane/kill-switch";
/// Policy id of decisions made because the agent is quarantined.
pub const POLICY_QUARANTINE: &str = "control-plane/quarantine";
/// Policy id of decisions made by the agent's allowed tools or models.
pub const POLICY_ALLOWED: &str = "control-plane/allowed";
/// Policy id of decisions made by the agent's budget.
pub const POLICY_BUDGET: &str = "control-plane/budget";
/// Policy id of decisions made because the request is for another tenant.
pub const POLICY_TENANT: &str = "control-plane/tenant";

struct Inner {
    agents: RwLock<HashMap<String, AgentProfile>>,
    usage: Mutex<HashMap<String, AgentUsage>>,
    /// Disabled tool patterns and why.
    disabled_tools: RwLock<Vec<(String, String)>>,
    /// Known tools and whether they have side effects.
    catalog: RwLock<HashMap<String, bool>>,
    policies: RwLock<PolicySet>,
    sinks: RwLock<Vec<Arc<dyn DecisionSink>>>,
    tracer: Option<Tracer>,
    require_registration: bool,
}

/// Governs agents in one place: a registry of agents (owner, tenant,
/// allowed tools and models, budget, status), the [`Policy`]s evaluated
/// before every tool call and model call, a kill switch, and an event for
/// every decision (to a [`Tracer`] and to [`DecisionSink`]s).
///
/// Cheap to clone; clones share everything, so an operator can suspend an
/// agent from one task while turns run in others. Changes take effect at
/// the next check, which integrations make at every step: before each model
/// call and before each tool call.
///
/// Order of checks for a request:
///
/// 1. Registry: unregistered agents are denied (unless
///    [`require_registration(false)`](ControlPlaneBuilder::require_registration)),
///    requests for another tenant are denied.
/// 2. Kill switch: a suspended agent is denied with
///    [`Verdict::stop`] set; a disabled tool is denied; a quarantined agent
///    may not call tools with side effects.
/// 3. The agent's allowed tools or models, then its budget.
/// 4. The policies, combined (`Deny` > `Ask` > `Redact` > `Allow`), then
///    the [`Defaults`].
///
/// Integrations: [`gate`](Self::gate) for `ReactAgent`,
/// [`governed_toolbox`](Self::governed_toolbox) for a plain `ToolBox`,
/// [`govern_model`](Self::govern_model) for any `ChatModel`, and
/// `hook` (feature `harness`) for `sinew-harness` agents.
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use std::sync::Arc;
/// use serde_json::json;
/// use sinew_govern::{AgentProfile, ControlPlane, MemoryDecisionSink, Request, Rule};
///
/// let audit = Arc::new(MemoryDecisionSink::new());
/// let cp = ControlPlane::builder()
///     .agent(AgentProfile::new("billing-bot").owner("payments"))
///     .policy(Rule::deny("no-deletes").tools(["delete_*"]))
///     .sink(audit.clone())
///     .build();
///
/// let ok = cp.check(Request::tool("billing-bot", "get_invoice", json!({"id": 7}))).await;
/// assert!(ok.decision.is_allowed());
/// let no = cp.check(Request::tool("billing-bot", "delete_invoice", json!({"id": 7}))).await;
/// assert_eq!(no.policy_id.as_deref(), Some("no-deletes"));
///
/// cp.suspend("billing-bot", "incident 42");
/// let stopped = cp.check(Request::tool("billing-bot", "get_invoice", json!({}))).await;
/// assert!(stopped.stop && stopped.decision.is_denied());
/// assert_eq!(audit.decisions().len(), 3);
/// # });
/// ```
#[derive(Clone)]
pub struct ControlPlane {
    inner: Arc<Inner>,
}

/// Builds a [`ControlPlane`].
#[derive(Default)]
pub struct ControlPlaneBuilder {
    agents: Vec<AgentProfile>,
    policies: PolicySet,
    sinks: Vec<Arc<dyn DecisionSink>>,
    tracer: Option<Tracer>,
    allow_unregistered: bool,
    disabled_tools: Vec<(String, String)>,
    catalog: HashMap<String, bool>,
}

impl ControlPlaneBuilder {
    /// Registers an agent.
    pub fn agent(mut self, profile: AgentProfile) -> Self {
        self.agents.push(profile);
        self
    }

    /// Adds a policy (evaluated in the order added).
    pub fn policy(mut self, policy: impl Policy + 'static) -> Self {
        self.policies.push(Arc::new(policy));
        self
    }

    pub fn policy_arc(mut self, policy: Arc<dyn Policy>) -> Self {
        self.policies.push(policy);
        self
    }

    /// What happens when no policy decides (see [`Defaults`]).
    pub fn defaults(mut self, defaults: Defaults) -> Self {
        self.policies.set_defaults(defaults);
        self
    }

    /// Agents, policies, defaults and disabled tools from a policy file.
    pub fn policy_file(mut self, file: PolicyFile) -> Self {
        let (agents, policies, defaults, disabled) = file.into_parts();
        self.agents.extend(agents);
        for p in policies {
            self.policies.push(p);
        }
        if let Some(d) = defaults {
            self.policies.set_defaults(d);
        }
        self.disabled_tools.extend(disabled);
        self
    }

    /// Sends every decision and operator action to `sink`.
    pub fn sink(mut self, sink: Arc<dyn DecisionSink>) -> Self {
        self.sinks.push(sink);
        self
    }

    /// Records every decision as a trace run (see
    /// [`record_decision`]) when the integration does not
    /// pass its own tracer.
    pub fn tracer(mut self, tracer: Tracer) -> Self {
        self.tracer = Some(tracer);
        self
    }

    /// Whether agents must be registered (default `true`: requests from
    /// unknown agents are denied).
    pub fn require_registration(mut self, yes: bool) -> Self {
        self.allow_unregistered = !yes;
        self
    }

    /// Records whether a tool has side effects, for integrations that do
    /// not know (the harness hook).
    pub fn tool(mut self, name: impl Into<String>, side_effects: bool) -> Self {
        self.catalog.insert(name.into(), side_effects);
        self
    }

    /// Records every tool of `toolbox` with its `side_effects()`.
    pub fn toolbox(mut self, toolbox: &ToolBox) -> Self {
        for t in toolbox.tools() {
            self.catalog.insert(t.spec().name, t.side_effects());
        }
        self
    }

    pub fn build(self) -> ControlPlane {
        let agents = self.agents.into_iter().map(|a| (a.id.clone(), a)).collect();
        ControlPlane {
            inner: Arc::new(Inner {
                agents: RwLock::new(agents),
                usage: Mutex::new(HashMap::new()),
                disabled_tools: RwLock::new(self.disabled_tools),
                catalog: RwLock::new(self.catalog),
                policies: RwLock::new(self.policies),
                sinks: RwLock::new(self.sinks),
                tracer: self.tracer,
                require_registration: !self.allow_unregistered,
            }),
        }
    }
}

fn read<T>(l: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(|e| e.into_inner())
}

fn write<T>(l: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    l.write().unwrap_or_else(|e| e.into_inner())
}

/// Fills the request's user, tenant and trace from the current call
/// context, when it is for the same agent.
fn fill_from_context(request: &mut Request) {
    let Some(ctx) = sinew_core::identity::current_context().filter(|c| c.agent == request.agent)
    else {
        return;
    };
    if request.user.is_none() {
        request.user = ctx.user.clone();
    }
    if request.tenant.is_none() {
        request.tenant = ctx.tenant.clone();
    }
    if request.trace_id.is_none() {
        request.trace_id = ctx.trace_id.clone();
    }
    if ctx.chain.len() > 1 {
        let chain = serde_json::json!(ctx.chain_agents());
        if !request.metadata.is_object() {
            request.metadata = serde_json::Value::Object(Default::default());
        }
        request.metadata["delegation_chain"] = chain;
    }
}

impl ControlPlane {
    pub fn builder() -> ControlPlaneBuilder {
        ControlPlaneBuilder::default()
    }

    // ---- registry ----

    /// Registers (or replaces) an agent.
    pub fn register(&self, profile: AgentProfile) {
        let id = profile.id.clone();
        write(&self.inner.agents).insert(id.clone(), profile);
        self.control(ControlAction::Register, &id, None);
    }

    /// Registers the agent described by a signed identity document, after
    /// verifying it against `trust`: the profile takes its id, owner,
    /// tenant and allowed tools from the document
    /// ([`AgentProfile::from_identity`]), so the registry enforces exactly
    /// what the issuer signed. Feature `identity`.
    ///
    /// ```
    /// use serde_json::json;
    /// use sinew_core::identity::{AgentIdentity, AgentKey, TrustStore};
    /// use sinew_govern::{ControlPlane, Request};
    ///
    /// let issuer = AgentKey::from_bytes(&[1; 32]);
    /// let agent_key = AgentKey::from_bytes(&[2; 32]);
    /// let signed = AgentIdentity::new("billing-bot", agent_key.public_key())
    ///     .owner("payments-team")
    ///     .allow_tools(["get_*"])
    ///     .sign(&issuer)
    ///     .unwrap();
    ///
    /// let cp = ControlPlane::builder().require_registration(true).build();
    /// let profile = cp.register_verified(&signed, &TrustStore::new().trust(issuer.public_key())).unwrap();
    /// assert_eq!(profile.owner.as_deref(), Some("payments-team"));
    /// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
    /// let v = cp.check(Request::tool("billing-bot", "delete_invoice", json!({}))).await;
    /// assert!(!v.decision.is_allowed());
    /// # });
    /// ```
    #[cfg(feature = "identity")]
    pub fn register_verified(
        &self,
        identity: &sinew_core::identity::SignedAgentIdentity,
        trust: &sinew_core::identity::TrustStore,
    ) -> Result<AgentProfile, sinew_core::identity::IdentityError> {
        let verified = trust.verify(identity)?;
        let profile = AgentProfile::from_identity(&verified.identity);
        self.register(profile.clone());
        Ok(profile)
    }

    /// Removes an agent; its later requests are denied when registration is
    /// required.
    pub fn unregister(&self, agent: &str) -> Option<AgentProfile> {
        let removed = write(&self.inner.agents).remove(agent);
        if removed.is_some() {
            self.control(ControlAction::Unregister, agent, None);
        }
        removed
    }

    /// The registered profile of `agent`.
    pub fn agent(&self, agent: &str) -> Option<AgentProfile> {
        read(&self.inner.agents).get(agent).cloned()
    }

    /// Every registered agent, sorted by id.
    pub fn agents(&self) -> Vec<AgentProfile> {
        let mut v: Vec<_> = read(&self.inner.agents).values().cloned().collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    /// The status of `agent` (`None` when unregistered).
    pub fn status(&self, agent: &str) -> Option<AgentStatus> {
        read(&self.inner.agents).get(agent).map(|a| a.status)
    }

    /// Calls the control plane allowed for `agent`.
    pub fn usage(&self, agent: &str) -> AgentUsage {
        self.inner
            .usage
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(agent)
            .copied()
            .unwrap_or_default()
    }

    /// Resets the counters behind [`AgentBudget`](crate::AgentBudget).
    pub fn reset_usage(&self, agent: &str) {
        self.inner
            .usage
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(agent);
    }

    /// Records whether a tool has side effects (see
    /// [`ControlPlaneBuilder::tool`]).
    pub fn register_tool(&self, name: impl Into<String>, side_effects: bool) {
        write(&self.inner.catalog).insert(name.into(), side_effects);
    }

    /// Records every tool of `toolbox` with its `side_effects()`.
    pub fn register_toolbox(&self, toolbox: &ToolBox) {
        let mut c = write(&self.inner.catalog);
        for t in toolbox.tools() {
            c.insert(t.spec().name, t.side_effects());
        }
    }

    // ---- policies and sinks ----

    /// Adds a policy at the end of the list.
    pub fn add_policy(&self, policy: Arc<dyn Policy>) {
        let id = policy.id().to_string();
        write(&self.inner.policies).push(policy);
        self.control(ControlAction::UpdatePolicies, &id, Some("added".into()));
    }

    /// Replaces every policy and the defaults.
    pub fn set_policies(&self, policies: PolicySet) {
        let ids: Vec<&str> = policies.policies().iter().map(|p| p.id()).collect();
        let target = ids.join(",");
        *write(&self.inner.policies) = policies;
        self.control(
            ControlAction::UpdatePolicies,
            &target,
            Some("replaced".into()),
        );
    }

    /// Replaces the policies, defaults and disabled tools with a policy
    /// file's, and registers (or replaces) its agents.
    pub fn apply_file(&self, file: PolicyFile) {
        let (agents, policies, defaults, disabled) = file.into_parts();
        let mut set = PolicySet::new().defaults(defaults.unwrap_or_default());
        for p in policies {
            set.push(p);
        }
        self.set_policies(set);
        *write(&self.inner.disabled_tools) = disabled;
        for a in agents {
            self.register(a);
        }
    }

    pub fn add_sink(&self, sink: Arc<dyn DecisionSink>) {
        write(&self.inner.sinks).push(sink);
    }

    // ---- kill switch ----

    fn set_status(&self, agent: &str, status: AgentStatus, reason: Option<String>) -> bool {
        let found = match write(&self.inner.agents).get_mut(agent) {
            Some(a) => {
                a.status = status;
                a.status_reason.clone_from(&reason);
                true
            }
            None => false,
        };
        if found {
            let action = match status {
                AgentStatus::Active => ControlAction::Reinstate,
                AgentStatus::Suspended => ControlAction::Suspend,
                AgentStatus::Quarantined => ControlAction::Quarantine,
            };
            self.control(action, agent, reason);
        }
        found
    }

    /// Kill switch for one agent: every later model call and tool call is
    /// denied and running turns end at their next step. Returns `false`
    /// when the agent is not registered.
    pub fn suspend(&self, agent: &str, reason: impl Into<String>) -> bool {
        self.set_status(agent, AgentStatus::Suspended, Some(reason.into()))
    }

    /// Contains one agent: tools with side effects are denied from its next
    /// step on. Returns `false` when the agent is not registered.
    pub fn quarantine(&self, agent: &str, reason: impl Into<String>) -> bool {
        self.set_status(agent, AgentStatus::Quarantined, Some(reason.into()))
    }

    /// Makes a suspended or quarantined agent active again.
    pub fn reinstate(&self, agent: &str) -> bool {
        self.set_status(agent, AgentStatus::Active, None)
    }

    /// Kill switch for a tool, for every agent: calls of tools matching
    /// `pattern` (a glob, e.g. `mcp_github_*`) are denied.
    pub fn disable_tool(&self, pattern: impl Into<String>, reason: impl Into<String>) {
        let (pattern, reason) = (pattern.into(), reason.into());
        {
            let mut d = write(&self.inner.disabled_tools);
            d.retain(|(p, _)| *p != pattern);
            d.push((pattern.clone(), reason.clone()));
        }
        self.control(ControlAction::DisableTool, &pattern, Some(reason));
    }

    /// Removes a pattern added with [`disable_tool`](Self::disable_tool).
    pub fn enable_tool(&self, pattern: &str) -> bool {
        let removed = {
            let mut d = write(&self.inner.disabled_tools);
            let before = d.len();
            d.retain(|(p, _)| p != pattern);
            d.len() != before
        };
        if removed {
            self.control(ControlAction::EnableTool, pattern, None);
        }
        removed
    }

    /// Disabled tool patterns and their reasons.
    pub fn disabled_tools(&self) -> Vec<(String, String)> {
        read(&self.inner.disabled_tools).clone()
    }

    /// Why `agent` must stop now, if it must: it is suspended, or it is not
    /// registered while registration is required. Integrations check this
    /// at every step.
    pub fn stop_reason(&self, agent: &str) -> Option<String> {
        match read(&self.inner.agents).get(agent) {
            Some(a) if a.status == AgentStatus::Suspended => Some(suspended_reason(a)),
            Some(_) => None,
            None if self.inner.require_registration => {
                Some(format!("agent `{agent}` is not registered"))
            }
            None => None,
        }
    }

    /// The kill-switch part of a check, for runtimes that check between
    /// steps: when `request.agent` must stop ([`stop_reason`](Self::stop_reason)),
    /// records a `Deny` decision for `request` (with [`Verdict::stop`] set)
    /// and returns the reason. Records nothing otherwise.
    pub fn check_halt(
        &self,
        request: &Request,
        tracer: Option<&Tracer>,
        parent: Option<&RunRef>,
    ) -> Option<String> {
        let reason = self.stop_reason(&request.agent)?;
        let id = if self.status(&request.agent).is_some() {
            POLICY_KILL_SWITCH
        } else {
            POLICY_REGISTRY
        };
        let mut v = Verdict::new(Decision::deny(reason.clone()), Some(id.into()));
        v.stop = true;
        self.emit(request, &v, tracer, parent);
        Some(reason)
    }

    fn control(&self, action: ControlAction, target: &str, reason: Option<String>) {
        let event = ControlEvent {
            id: uuid::Uuid::new_v4().to_string(),
            time_ms: now_ms(),
            action,
            target: target.to_string(),
            reason,
        };
        for s in read(&self.inner.sinks).iter() {
            s.record_control(&event);
        }
    }

    // ---- evaluation ----

    /// Whether `tool` has side effects according to the catalog.
    pub fn tool_side_effects(&self, tool: &str) -> Option<bool> {
        read(&self.inner.catalog).get(tool).copied()
    }

    /// Evaluates `request`, records the decision (control-plane tracer and
    /// sinks) and counts allowed calls against the agent's budget.
    pub async fn check(&self, request: Request) -> Verdict {
        self.check_traced(request, None, None).await
    }

    /// Like [`check`](Self::check), recording the trace run with `tracer`
    /// (falling back to the control plane's) under `parent`.
    ///
    /// Inside a [`CallContext`](sinew_core::identity::CallContext) for the
    /// same agent (set with `with_context`, or by `sinew-a2a` / `sinew-mcp`
    /// for a verified remote caller), the request's missing user, tenant and
    /// trace id come from the context, and a delegated request carries the
    /// chain as `metadata.delegation_chain` (agent ids, first to last).
    pub async fn check_traced(
        &self,
        mut request: Request,
        tracer: Option<&Tracer>,
        parent: Option<&RunRef>,
    ) -> Verdict {
        fill_from_context(&mut request);
        if request.kind == RequestKind::Tool && !request.side_effects {
            request.side_effects = request
                .tool
                .as_deref()
                .and_then(|t| self.tool_side_effects(t))
                .unwrap_or(false);
        }
        let verdict = match self.precheck(&mut request) {
            Some(v) => v,
            None => {
                let policies = read(&self.inner.policies).clone();
                policies.evaluate(&request).await
            }
        };
        if verdict.decision.is_allowed() {
            let mut usage = self.inner.usage.lock().unwrap_or_else(|e| e.into_inner());
            let u = usage.entry(request.agent.clone()).or_default();
            match request.kind {
                RequestKind::Tool => u.tool_calls += 1,
                RequestKind::Model => u.model_calls += 1,
            }
        }
        self.emit(&request, &verdict, tracer, parent);
        verdict
    }

    /// Registry, kill switch, allowed tools/models and budget.
    fn precheck(&self, request: &mut Request) -> Option<Verdict> {
        let deny =
            |id: &str, reason: String| Some(Verdict::new(Decision::deny(reason), Some(id.into())));
        let agents = read(&self.inner.agents);
        let Some(profile) = agents.get(&request.agent) else {
            return if self.inner.require_registration {
                deny(
                    POLICY_REGISTRY,
                    format!("agent `{}` is not registered", request.agent),
                )
            } else {
                self.disabled(request)
            };
        };
        if let Some(t) = &profile.tenant {
            match &request.tenant {
                None => request.tenant = Some(t.clone()),
                Some(rt) if rt != t => {
                    return deny(
                        POLICY_TENANT,
                        format!(
                            "agent `{}` belongs to tenant `{t}`, not `{rt}`",
                            request.agent
                        ),
                    )
                }
                Some(_) => {}
            }
        }
        match profile.status {
            AgentStatus::Suspended => {
                let mut v = Verdict::new(
                    Decision::deny(suspended_reason(profile)),
                    Some(POLICY_KILL_SWITCH.into()),
                );
                v.stop = true;
                return Some(v);
            }
            AgentStatus::Quarantined
                if request.kind == RequestKind::Tool && request.side_effects =>
            {
                let why = profile
                    .status_reason
                    .as_deref()
                    .map(|r| format!(": {r}"))
                    .unwrap_or_default();
                return deny(
                    POLICY_QUARANTINE,
                    format!(
                        "agent `{}` is quarantined{why}; tools with side effects are blocked",
                        request.agent
                    ),
                );
            }
            _ => {}
        }
        if let Some(v) = self.disabled(request) {
            return Some(v);
        }
        let target = request.target().to_string();
        let usage = self.usage(&request.agent);
        match request.kind {
            RequestKind::Tool => {
                if !profile.tool_allowed(&target) {
                    return deny(
                        POLICY_ALLOWED,
                        format!(
                            "tool `{target}` is not in the allowed tools of agent `{}`",
                            request.agent
                        ),
                    );
                }
                if profile
                    .budget
                    .max_tool_calls
                    .is_some_and(|max| usage.tool_calls >= max)
                {
                    return deny(
                        POLICY_BUDGET,
                        format!("agent `{}` used its tool call budget", request.agent),
                    );
                }
            }
            RequestKind::Model => {
                if !profile.model_allowed(&target) {
                    return deny(
                        POLICY_ALLOWED,
                        format!(
                            "model `{target}` is not in the allowed models of agent `{}`",
                            request.agent
                        ),
                    );
                }
                if profile
                    .budget
                    .max_model_calls
                    .is_some_and(|max| usage.model_calls >= max)
                {
                    return deny(
                        POLICY_BUDGET,
                        format!("agent `{}` used its model call budget", request.agent),
                    );
                }
            }
        }
        None
    }

    fn disabled(&self, request: &Request) -> Option<Verdict> {
        if request.kind != RequestKind::Tool {
            return None;
        }
        let tool = request.target();
        read(&self.inner.disabled_tools)
            .iter()
            .find(|(p, _)| glob_match(p, tool))
            .map(|(_, reason)| {
                Verdict::new(
                    Decision::deny(format!("tool `{tool}` is disabled: {reason}")),
                    Some(POLICY_KILL_SWITCH.into()),
                )
            })
    }

    fn emit(
        &self,
        request: &Request,
        verdict: &Verdict,
        tracer: Option<&Tracer>,
        parent: Option<&RunRef>,
    ) {
        let mut request = request.clone();
        if !verdict.redact.is_empty() {
            request.arguments = verdict.redacted(&request.arguments);
        }
        let event = DecisionEvent {
            id: uuid::Uuid::new_v4().to_string(),
            time_ms: now_ms(),
            request,
            verdict: verdict.clone(),
        };
        if let Some(t) = tracer.or(self.inner.tracer.as_ref()) {
            record_decision(t, parent, &event);
        }
        for s in read(&self.inner.sinks).iter() {
            s.record(&event);
        }
    }
}

fn suspended_reason(a: &AgentProfile) -> String {
    match &a.status_reason {
        Some(r) => format!("agent `{}` is suspended: {r}", a.id),
        None => format!("agent `{}` is suspended", a.id),
    }
}
