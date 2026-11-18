//! [`Request`]: what a policy sees about one tool call or model call.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Milliseconds since the Unix epoch, now.
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// What is being authorized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    /// A tool call ([`Request::tool`] and [`Request::arguments`] are set).
    #[default]
    Tool,
    /// A model call ([`Request::model`] is set).
    Model,
}

/// Who acts: the agent, and the user and tenant it acts for.
///
/// ```
/// use sinew_govern::Identity;
/// let id = Identity::agent("billing-bot").user("ana").tenant("acme");
/// assert_eq!(id.user.as_deref(), Some("ana"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Identity {
    /// The agent's id in the [`ControlPlane`](crate::ControlPlane) registry.
    pub agent: String,
    /// The end user the agent acts for, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// The tenant (customer, organization) the call belongs to, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

impl Identity {
    /// An identity with only an agent id.
    pub fn agent(id: impl Into<String>) -> Self {
        Self {
            agent: id.into(),
            user: None,
            tenant: None,
        }
    }

    /// Sets the user.
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Sets the tenant.
    pub fn tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }
}

impl From<&str> for Identity {
    fn from(agent: &str) -> Self {
        Self::agent(agent)
    }
}

impl From<String> for Identity {
    fn from(agent: String) -> Self {
        Self::agent(agent)
    }
}

/// The context a [`Policy`](crate::Policy) evaluates: one tool call or one
/// model call, by whom, for whom, and where it happens.
///
/// ```
/// use serde_json::json;
/// use sinew_govern::Request;
///
/// let req = Request::tool("billing-bot", "refund", json!({"amount": 40}))
///     .user("ana")
///     .tenant("acme")
///     .side_effects(true)
///     .thread("thread-1");
/// assert_eq!(req.tool.as_deref(), Some("refund"));
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Request {
    pub kind: RequestKind,
    /// The calling agent's id.
    pub agent: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Tool name, for tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    /// Tool arguments (JSON), for tool calls; `null` otherwise.
    #[serde(default)]
    pub arguments: Value,
    /// Whether the tool changes something outside the program
    /// ([`Tool::side_effects`](sinew_core::Tool::side_effects)).
    #[serde(default)]
    pub side_effects: bool,
    /// Model name or `provider:model` spec, for model calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Trace the call belongs to, when traced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Conversation thread, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    /// Evaluation time in milliseconds since the Unix epoch (used by time
    /// windows and rate limits). Set to the current time by the
    /// constructors.
    pub time_ms: u64,
    /// Extra facts for custom policies (e.g. `{"subagent": "researcher"}`).
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

impl Request {
    /// A tool call request.
    pub fn tool(agent: impl Into<String>, tool: impl Into<String>, arguments: Value) -> Self {
        Self {
            kind: RequestKind::Tool,
            agent: agent.into(),
            tool: Some(tool.into()),
            arguments,
            time_ms: now_ms(),
            ..Self::default()
        }
    }

    /// A model call request.
    pub fn model(agent: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            kind: RequestKind::Model,
            agent: agent.into(),
            model: Some(model.into()),
            time_ms: now_ms(),
            ..Self::default()
        }
    }

    /// Takes agent, user and tenant from `identity`.
    pub fn identity(mut self, identity: &Identity) -> Self {
        self.agent.clone_from(&identity.agent);
        self.user.clone_from(&identity.user);
        self.tenant.clone_from(&identity.tenant);
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub fn tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    pub fn side_effects(mut self, yes: bool) -> Self {
        self.side_effects = yes;
        self
    }

    pub fn thread(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    pub fn trace(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Sets the evaluation time (milliseconds since the Unix epoch).
    pub fn at(mut self, time_ms: u64) -> Self {
        self.time_ms = time_ms;
        self
    }

    /// Sets one metadata field.
    pub fn meta(mut self, key: &str, value: Value) -> Self {
        if !self.metadata.is_object() {
            self.metadata = Value::Object(Default::default());
        }
        self.metadata[key] = value;
        self
    }

    /// What is being called: the tool name or the model.
    pub fn target(&self) -> &str {
        match self.kind {
            RequestKind::Tool => self.tool.as_deref().unwrap_or_default(),
            RequestKind::Model => self.model.as_deref().unwrap_or_default(),
        }
    }
}
