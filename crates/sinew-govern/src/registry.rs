//! The agent registry: [`AgentProfile`], [`AgentStatus`], [`AgentBudget`].

use serde::{Deserialize, Serialize};

/// Whether an agent may act.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    /// Policies decide.
    #[default]
    Active,
    /// Kill switch: every model call and tool call is denied, and running
    /// turns end at their next step.
    Suspended,
    /// Contained: tools with side effects are denied; tools without side
    /// effects and model calls still go through the policies, so the agent
    /// can be observed without acting on the outside world.
    Quarantined,
}

/// Limits on what an agent may consume, counted by the control plane over
/// its lifetime (or since [`ControlPlane::reset_usage`](crate::ControlPlane::reset_usage)).
/// Per-turn limits (steps, tokens, cost, time) stay with the runtime: the
/// harness [`Budget`](https://docs.rs/sinew-harness/latest/sinew_harness/struct.Budget.html)
/// and `ReactAgent::max_iterations`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentBudget {
    /// Allowed tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u64>,
    /// Allowed model calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_model_calls: Option<u64>,
}

/// Calls an agent made that the control plane allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AgentUsage {
    pub tool_calls: u64,
    pub model_calls: u64,
}

/// A registered agent: identity, owner, what it may use, budget and status.
///
/// ```
/// use sinew_govern::{AgentBudget, AgentProfile};
///
/// let profile = AgentProfile::new("billing-bot")
///     .owner("payments-team")
///     .tenant("acme")
///     .allow_tools(["get_*", "refund"])
///     .allow_models(["anthropic:*"])
///     .budget(AgentBudget { max_tool_calls: Some(500), max_model_calls: None });
/// assert!(profile.tool_allowed("get_invoice"));
/// assert!(!profile.tool_allowed("delete_invoice"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentProfile {
    pub id: String,
    /// Team or person accountable for the agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The tenant the agent belongs to. Requests for another tenant are
    /// denied; requests without a tenant get this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Tool name patterns the agent may call (`None`: no restriction from
    /// the registry; policies still apply).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    /// Model patterns the agent may call (`None`: no restriction).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_models: Option<Vec<String>>,
    #[serde(default)]
    pub budget: AgentBudget,
    #[serde(default)]
    pub status: AgentStatus,
    /// Why the status was last changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_reason: Option<String>,
}

impl AgentProfile {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    /// The profile a signed identity document describes: id, owner,
    /// tenant, description (its version and capabilities) and allowed tools
    /// (an empty list allows none). Verify the document first, or use
    /// [`ControlPlane::register_verified`](crate::ControlPlane::register_verified).
    /// Feature `identity`.
    #[cfg(feature = "identity")]
    pub fn from_identity(identity: &sinew_core::identity::AgentIdentity) -> Self {
        let mut description = format!("version {}", identity.version);
        if !identity.capabilities.is_empty() {
            description.push_str(&format!(
                "; capabilities: {}",
                identity.capabilities.join(", ")
            ));
        }
        Self {
            id: identity.id.clone(),
            owner: identity.owner.clone(),
            description: Some(description),
            tenant: identity.tenant.clone(),
            allowed_tools: Some(identity.allowed_tools.clone()),
            ..Self::default()
        }
    }

    pub fn owner(mut self, owner: impl Into<String>) -> Self {
        self.owner = Some(owner.into());
        self
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.description = Some(text.into());
        self
    }

    pub fn tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    /// Restricts the agent to tools matching these patterns.
    pub fn allow_tools<I: IntoIterator<Item = S>, S: Into<String>>(mut self, p: I) -> Self {
        self.allowed_tools
            .get_or_insert_with(Vec::new)
            .extend(p.into_iter().map(Into::into));
        self
    }

    /// Restricts the agent to models matching these patterns.
    pub fn allow_models<I: IntoIterator<Item = S>, S: Into<String>>(mut self, p: I) -> Self {
        self.allowed_models
            .get_or_insert_with(Vec::new)
            .extend(p.into_iter().map(Into::into));
        self
    }

    pub fn budget(mut self, budget: AgentBudget) -> Self {
        self.budget = budget;
        self
    }

    pub fn status(mut self, status: AgentStatus) -> Self {
        self.status = status;
        self
    }

    /// Whether the registry lets the agent call `tool`.
    pub fn tool_allowed(&self, tool: &str) -> bool {
        self.allowed_tools
            .as_ref()
            .is_none_or(|p| crate::pattern::any_match(p, Some(tool)) && !p.is_empty())
    }

    /// Whether the registry lets the agent call `model`.
    pub fn model_allowed(&self, model: &str) -> bool {
        self.allowed_models
            .as_ref()
            .is_none_or(|p| crate::pattern::any_match(p, Some(model)) && !p.is_empty())
    }
}
