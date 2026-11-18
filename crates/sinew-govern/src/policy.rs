//! The [`Policy`] trait, the built-in [`Rule`] and [`RateLimit`]
//! policies, and [`PolicySet`], which evaluates several policies and
//! applies [`Defaults`].

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    condition::{Condition, Op},
    decision::{Decision, PolicyMatch, Verdict},
    pattern::any_match,
    request::{Request, RequestKind},
    time::TimeWindow,
    GovernError,
};

/// A policy: looks at a [`Request`] and returns a [`Decision`], or `None`
/// when it does not apply (it abstains).
///
/// Policies are evaluated in a [`PolicySet`]; the most restrictive
/// decision wins (`Deny` > `Ask` > `Redact` > `Allow`) and the set's
/// [`Defaults`] apply when every policy abstains. An `Err` counts as a
/// `Deny` (fail closed).
///
/// ```
/// use async_trait::async_trait;
/// use sinew_govern::{Decision, GovernError, Policy, Request};
///
/// /// Denies tools whose name starts with `admin_` for every agent but `ops`.
/// struct AdminOnly;
///
/// #[async_trait]
/// impl Policy for AdminOnly {
///     fn id(&self) -> &str { "admin-only" }
///     async fn evaluate(&self, req: &Request) -> Result<Option<Decision>, GovernError> {
///         let admin = req.tool.as_deref().is_some_and(|t| t.starts_with("admin_"));
///         Ok((admin && req.agent != "ops").then(|| Decision::deny("admin tools are for ops")))
///     }
/// }
/// ```
#[async_trait]
pub trait Policy: Send + Sync {
    /// Stable id, recorded with every decision this policy makes.
    fn id(&self) -> &str;

    /// The decision for `request`, or `None` to abstain.
    async fn evaluate(&self, request: &Request) -> Result<Option<Decision>, GovernError>;
}

/// A default effect, used when no policy decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    Allow,
    Ask,
    Deny,
}

/// What happens when every policy abstains.
///
/// The default allows tools without side effects and model calls, and
/// denies tools with side effects ([`Tool::side_effects`](sinew_core::Tool::side_effects)),
/// so a tool that charges, sends or writes runs only when a policy allows
/// it (or asks a human).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Defaults {
    /// Tools without side effects.
    #[serde(default = "Defaults::allow")]
    pub tools: Effect,
    /// Tools with side effects.
    #[serde(default = "Defaults::deny")]
    pub side_effect_tools: Effect,
    /// Model calls.
    #[serde(default = "Defaults::allow")]
    pub models: Effect,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            tools: Effect::Allow,
            side_effect_tools: Effect::Deny,
            models: Effect::Allow,
        }
    }
}

impl Defaults {
    fn allow() -> Effect {
        Effect::Allow
    }
    fn deny() -> Effect {
        Effect::Deny
    }

    /// Everything not decided by a policy is allowed.
    pub fn allow_all() -> Self {
        Self {
            tools: Effect::Allow,
            side_effect_tools: Effect::Allow,
            models: Effect::Allow,
        }
    }

    /// Everything not decided by a policy is denied (allow-list mode).
    pub fn deny_all() -> Self {
        Self {
            tools: Effect::Deny,
            side_effect_tools: Effect::Deny,
            models: Effect::Deny,
        }
    }

    /// The default decision for `request`.
    pub fn decide(&self, request: &Request) -> Decision {
        let (effect, what) = match request.kind {
            RequestKind::Model => (self.models, format!("model `{}`", request.target())),
            RequestKind::Tool if request.side_effects => (
                self.side_effect_tools,
                format!("tool `{}` (it has side effects)", request.target()),
            ),
            RequestKind::Tool => (self.tools, format!("tool `{}`", request.target())),
        };
        match effect {
            Effect::Allow => Decision::Allow,
            Effect::Ask => Decision::ask(format!("no policy allows {what} without approval")),
            Effect::Deny => Decision::deny(format!("no policy allows {what}")),
        }
    }
}

/// An ordered list of policies plus [`Defaults`].
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use serde_json::json;
/// use sinew_govern::{Condition, Decision, PolicySet, Request, Rule};
///
/// let set = PolicySet::new()
///     .with(Rule::allow("small-refunds").tools(["refund"]).when(Condition::parse("amount <= 100").unwrap()))
///     .with(Rule::ask("large-refunds").tools(["refund"]).when(Condition::parse("amount > 100").unwrap()));
///
/// let small = Request::tool("bot", "refund", json!({"amount": 40})).side_effects(true);
/// assert_eq!(set.evaluate(&small).await.decision, Decision::Allow);
/// let large = Request::tool("bot", "refund", json!({"amount": 400})).side_effects(true);
/// assert!(matches!(set.evaluate(&large).await.decision, Decision::Ask { .. }));
/// // No policy matches and the tool has side effects: denied by default.
/// let other = Request::tool("bot", "send_email", json!({})).side_effects(true);
/// let v = set.evaluate(&other).await;
/// assert!(v.decision.is_denied() && v.policy_id.is_none());
/// # });
/// ```
#[derive(Clone, Default)]
pub struct PolicySet {
    policies: Vec<Arc<dyn Policy>>,
    defaults: Defaults,
}

impl PolicySet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a policy.
    pub fn with(mut self, policy: impl Policy + 'static) -> Self {
        self.policies.push(Arc::new(policy));
        self
    }

    /// Adds a shared policy.
    pub fn with_arc(mut self, policy: Arc<dyn Policy>) -> Self {
        self.policies.push(policy);
        self
    }

    pub fn push(&mut self, policy: Arc<dyn Policy>) {
        self.policies.push(policy);
    }

    pub fn defaults(mut self, defaults: Defaults) -> Self {
        self.defaults = defaults;
        self
    }

    pub fn set_defaults(&mut self, defaults: Defaults) {
        self.defaults = defaults;
    }

    pub fn get_defaults(&self) -> Defaults {
        self.defaults
    }

    pub fn policies(&self) -> &[Arc<dyn Policy>] {
        &self.policies
    }

    pub fn len(&self) -> usize {
        self.policies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.policies.is_empty()
    }

    /// Evaluates every policy and combines their decisions. A policy that
    /// fails counts as a `Deny` with the error as reason.
    pub async fn evaluate(&self, request: &Request) -> Verdict {
        let mut matched = Vec::new();
        for p in &self.policies {
            let decision = match p.evaluate(request).await {
                Ok(Some(d)) => d,
                Ok(None) => continue,
                Err(e) => Decision::deny(format!("policy `{}` failed: {e}", p.id())),
            };
            matched.push(PolicyMatch {
                policy_id: p.id().to_string(),
                decision,
            });
        }
        Verdict::combine(matched)
            .unwrap_or_else(|| Verdict::new(self.defaults.decide(request), None))
    }
}

/// The built-in policy: a decision for the requests that match its
/// filters. Every filter left empty matches everything.
///
/// - `agents`, `users`, `tenants`, `tools`, `models`: glob patterns
///   ([`glob_match`](crate::glob_match)); a rule with `tools` only applies
///   to tool calls, a rule with `models` only to model calls.
/// - [`side_effects`](Self::side_effects): only tools with (or without)
///   side effects.
/// - [`when`](Self::when) / [`arg`](Self::arg): conditions on the tool
///   arguments, all of which must hold.
/// - [`during`](Self::during) / [`outside`](Self::outside): a
///   [`TimeWindow`].
///
/// ```
/// use sinew_govern::{Op, Rule, TimeWindow};
///
/// // Payments over 1 000 are never made by agents.
/// let cap = Rule::deny("payment-cap").tools(["pay_*"]).arg("amount", Op::Gt(1_000.0))
///     .reason("payments over 1000 are made by people");
/// // Every side-effecting tool needs approval outside business hours.
/// let after_hours = Rule::ask("after-hours").side_effects(true)
///     .outside(TimeWindow::workdays("09:00", "17:00").unwrap());
/// // Card numbers never reach the tool in clear.
/// let mask = Rule::redact("mask-card", ["/card/number"]).tools(["*"]);
/// ```
#[derive(Debug, Clone)]
pub struct Rule {
    id: String,
    decision: Decision,
    agents: Vec<String>,
    users: Vec<String>,
    tenants: Vec<String>,
    tools: Vec<String>,
    models: Vec<String>,
    kind: Option<RequestKind>,
    side_effects: Option<bool>,
    conditions: Vec<Condition>,
    during: Option<TimeWindow>,
    outside: Option<TimeWindow>,
}

impl Rule {
    /// A rule that returns `decision` for matching requests.
    pub fn new(id: impl Into<String>, decision: Decision) -> Self {
        Self {
            id: id.into(),
            decision,
            agents: Vec::new(),
            users: Vec::new(),
            tenants: Vec::new(),
            tools: Vec::new(),
            models: Vec::new(),
            kind: None,
            side_effects: None,
            conditions: Vec::new(),
            during: None,
            outside: None,
        }
    }

    pub fn allow(id: impl Into<String>) -> Self {
        Self::new(id, Decision::Allow)
    }

    /// Asks a human; the reason defaults to ``policy `ID` requires approval``.
    pub fn ask(id: impl Into<String>) -> Self {
        let id = id.into();
        let reason = format!("policy `{id}` requires approval");
        Self::new(id, Decision::ask(reason))
    }

    /// Denies; the reason defaults to ``denied by policy `ID` ``.
    pub fn deny(id: impl Into<String>) -> Self {
        let id = id.into();
        let reason = format!("denied by policy `{id}`");
        Self::new(id, Decision::deny(reason))
    }

    /// Allows with these argument fields redacted.
    pub fn redact<I, S>(id: impl Into<String>, fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new(id, Decision::redact(fields))
    }

    /// Every tool with side effects needs approval (`id`
    /// `side-effects-need-approval`).
    pub fn side_effects_need_approval() -> Self {
        Self::ask("side-effects-need-approval")
            .side_effects(true)
            .reason("tools with side effects need approval")
    }

    /// The reason given with `Ask` and `Deny` decisions.
    pub fn reason(mut self, reason: impl Into<String>) -> Self {
        match &mut self.decision {
            Decision::Ask { reason: r } | Decision::Deny { reason: r } => {
                *r = reason.into();
            }
            _ => {}
        }
        self
    }

    pub fn agents<I: IntoIterator<Item = S>, S: Into<String>>(mut self, p: I) -> Self {
        self.agents.extend(p.into_iter().map(Into::into));
        self
    }

    pub fn users<I: IntoIterator<Item = S>, S: Into<String>>(mut self, p: I) -> Self {
        self.users.extend(p.into_iter().map(Into::into));
        self
    }

    pub fn tenants<I: IntoIterator<Item = S>, S: Into<String>>(mut self, p: I) -> Self {
        self.tenants.extend(p.into_iter().map(Into::into));
        self
    }

    /// Tool name patterns; the rule then applies to tool calls only.
    pub fn tools<I: IntoIterator<Item = S>, S: Into<String>>(mut self, p: I) -> Self {
        self.tools.extend(p.into_iter().map(Into::into));
        self
    }

    /// Model patterns (`anthropic:*`, `*gpt-4*`); the rule then applies to
    /// model calls only.
    pub fn models<I: IntoIterator<Item = S>, S: Into<String>>(mut self, p: I) -> Self {
        self.models.extend(p.into_iter().map(Into::into));
        self
    }

    /// Restricts the rule to tool calls or model calls.
    pub fn only(mut self, kind: RequestKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Only tools whose `side_effects()` equals `yes`.
    pub fn side_effects(mut self, yes: bool) -> Self {
        self.side_effects = Some(yes);
        self
    }

    /// A condition on the arguments (see [`Condition::parse`]).
    pub fn when(mut self, condition: Condition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// A condition on one argument field.
    pub fn arg(self, field: impl Into<String>, op: Op) -> Self {
        self.when(Condition::new(field, op))
    }

    /// Applies only inside `window`.
    pub fn during(mut self, window: TimeWindow) -> Self {
        self.during = Some(window);
        self
    }

    /// Applies only outside `window`.
    pub fn outside(mut self, window: TimeWindow) -> Self {
        self.outside = Some(window);
        self
    }

    pub fn rule_id(&self) -> &str {
        &self.id
    }

    pub fn decision(&self) -> &Decision {
        &self.decision
    }

    /// Whether the rule applies to `request`.
    pub fn matches(&self, r: &Request) -> bool {
        let kind = self.kind.or(
            if !self.tools.is_empty() || self.side_effects.is_some() || !self.conditions.is_empty()
            {
                Some(RequestKind::Tool)
            } else if !self.models.is_empty() {
                Some(RequestKind::Model)
            } else {
                None
            },
        );
        if kind.is_some_and(|k| k != r.kind) {
            return false;
        }
        let target = |patterns: &[String], kind: RequestKind| {
            patterns.is_empty() || (r.kind == kind && any_match(patterns, Some(r.target())))
        };
        any_match(&self.agents, Some(&r.agent))
            && any_match(&self.users, r.user.as_deref())
            && any_match(&self.tenants, r.tenant.as_deref())
            && target(&self.tools, RequestKind::Tool)
            && target(&self.models, RequestKind::Model)
            && self.side_effects.is_none_or(|s| s == r.side_effects)
            && self.conditions.iter().all(|c| c.matches(&r.arguments))
            && self.during.as_ref().is_none_or(|w| w.contains(r.time_ms))
            && self.outside.as_ref().is_none_or(|w| !w.contains(r.time_ms))
    }
}

#[async_trait]
impl Policy for Rule {
    fn id(&self) -> &str {
        &self.id
    }

    async fn evaluate(&self, request: &Request) -> Result<Option<Decision>, GovernError> {
        Ok(self.matches(request).then(|| self.decision.clone()))
    }
}

/// What a [`RateLimit`] counts separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateKey {
    Agent,
    User,
    Tenant,
    /// The tool name (or the model, for model calls).
    Tool,
}

/// Denies a request once `max` matching requests were allowed through
/// this policy within the sliding `window`. Counts are kept per
/// combination of [`RateKey`]s (default: per agent and tool), in memory,
/// and use [`Request::time_ms`].
///
/// A request this policy lets through is counted even when another policy
/// denies it, so the limit is on attempts the limiter saw, not on calls
/// that ran.
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use std::time::Duration;
/// use serde_json::json;
/// use sinew_govern::{Policy, RateLimit, Request};
///
/// let limit = RateLimit::new("refund-rate", 2, Duration::from_secs(60)).tools(["refund"]);
/// let req = |t| Request::tool("bot", "refund", json!({})).at(t);
/// assert!(limit.evaluate(&req(0)).await.unwrap().is_none());
/// assert!(limit.evaluate(&req(1_000)).await.unwrap().is_none());
/// assert!(limit.evaluate(&req(2_000)).await.unwrap().unwrap().is_denied());
/// assert!(limit.evaluate(&req(61_000)).await.unwrap().is_none());
/// # });
/// ```
#[derive(Debug)]
pub struct RateLimit {
    id: String,
    max: u32,
    window_ms: u64,
    key: Vec<RateKey>,
    kind: RequestKind,
    agents: Vec<String>,
    tools: Vec<String>,
    hits: Mutex<HashMap<String, VecDeque<u64>>>,
}

impl RateLimit {
    /// At most `max` tool calls per `window`, per agent and tool.
    pub fn new(id: impl Into<String>, max: u32, window: Duration) -> Self {
        Self {
            id: id.into(),
            max,
            window_ms: window.as_millis() as u64,
            key: vec![RateKey::Agent, RateKey::Tool],
            kind: RequestKind::Tool,
            agents: Vec::new(),
            tools: Vec::new(),
            hits: Mutex::new(HashMap::new()),
        }
    }

    /// What to count separately (default `[Agent, Tool]`; an empty list
    /// counts every matching request together).
    pub fn per(mut self, key: impl IntoIterator<Item = RateKey>) -> Self {
        self.key = key.into_iter().collect();
        self
    }

    /// Agent patterns this limit applies to (default: all).
    pub fn agents<I: IntoIterator<Item = S>, S: Into<String>>(mut self, p: I) -> Self {
        self.agents.extend(p.into_iter().map(Into::into));
        self
    }

    /// Tool (or model) patterns this limit applies to (default: all).
    pub fn tools<I: IntoIterator<Item = S>, S: Into<String>>(mut self, p: I) -> Self {
        self.tools.extend(p.into_iter().map(Into::into));
        self
    }

    /// Counts model calls instead of tool calls.
    pub fn model_calls(mut self) -> Self {
        self.kind = RequestKind::Model;
        self
    }

    fn key_for(&self, r: &Request) -> String {
        let mut parts = Vec::with_capacity(self.key.len());
        for k in &self.key {
            parts.push(match k {
                RateKey::Agent => r.agent.as_str(),
                RateKey::User => r.user.as_deref().unwrap_or(""),
                RateKey::Tenant => r.tenant.as_deref().unwrap_or(""),
                RateKey::Tool => r.target(),
            });
        }
        parts.join("\u{1f}")
    }
}

#[async_trait]
impl Policy for RateLimit {
    fn id(&self) -> &str {
        &self.id
    }

    async fn evaluate(&self, r: &Request) -> Result<Option<Decision>, GovernError> {
        if r.kind != self.kind
            || !any_match(&self.agents, Some(&r.agent))
            || !any_match(&self.tools, Some(r.target()))
        {
            return Ok(None);
        }
        let key = self.key_for(r);
        let mut hits = self.hits.lock().unwrap_or_else(|e| e.into_inner());
        let q = hits.entry(key).or_default();
        while q
            .front()
            .is_some_and(|t| t.saturating_add(self.window_ms) <= r.time_ms)
        {
            q.pop_front();
        }
        if q.len() >= self.max as usize {
            return Ok(Some(Decision::deny(format!(
                "rate limit `{}`: at most {} calls of `{}` per {} s",
                self.id,
                self.max,
                r.target(),
                self.window_ms / 1000
            ))));
        }
        q.push_back(r.time_ms);
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeWindow;
    use serde_json::json;

    const MONDAY_10H: u64 = 1_794_787_200_000 + 10 * 3_600_000;

    fn tool(name: &str, args: serde_json::Value) -> Request {
        Request::tool("billing-bot", name, args).at(MONDAY_10H)
    }

    #[tokio::test]
    async fn rule_filters() {
        let r = Rule::deny("x")
            .agents(["billing-*"])
            .users(["ana"])
            .tenants(["acme"])
            .tools(["refund", "pay_*"]);
        let req = tool("pay_invoice", json!({})).user("ana").tenant("acme");
        assert!(r.matches(&req));
        assert!(!r.matches(&req.clone().user("bob")));
        assert!(!r.matches(&tool("pay_invoice", json!({})).tenant("acme"))); // no user
        assert!(!r.matches(
            &Request::tool("other", "refund", json!({}))
                .user("ana")
                .tenant("acme")
        ));
        assert!(!r.matches(
            &Request::model("billing-bot", "pay_x")
                .user("ana")
                .tenant("acme")
        ));

        let m = Rule::deny("m").models(["openai:*"]);
        assert!(m.matches(&Request::model("a", "openai:gpt-4.1")));
        assert!(!m.matches(&Request::model("a", "anthropic:claude")));
        assert!(!m.matches(&Request::tool("a", "openai:x", json!({}))));

        // No target filter: tools and models.
        let any = Rule::deny("any").agents(["rogue"]);
        assert!(any.matches(&Request::model("rogue", "x")));
        assert!(any.matches(&Request::tool("rogue", "t", json!({}))));
        assert!(!any
            .only(RequestKind::Model)
            .matches(&Request::tool("rogue", "t", json!({}))));
    }

    #[tokio::test]
    async fn side_effects_conditions_and_time() {
        let r = Rule::side_effects_need_approval();
        assert!(r.matches(&tool("send", json!({})).side_effects(true)));
        assert!(!r.matches(&tool("read", json!({}))));

        let c = Rule::allow("small")
            .arg("amount", Op::Le(100.0))
            .arg("currency", Op::Eq(json!("USD")));
        assert!(c.matches(&tool("refund", json!({"amount": 5, "currency": "USD"}))));
        assert!(!c.matches(&tool("refund", json!({"amount": 5, "currency": "EUR"}))));
        assert!(!c.matches(&tool("refund", json!({"currency": "USD"}))));

        let hours = TimeWindow::workdays("09:00", "17:00").unwrap();
        let during = Rule::allow("d").during(hours.clone());
        let outside = Rule::deny("o").outside(hours);
        assert!(during.matches(&tool("t", json!({}))));
        assert!(!outside.matches(&tool("t", json!({}))));
        let night = tool("t", json!({})).at(MONDAY_10H + 12 * 3_600_000);
        assert!(!during.matches(&night));
        assert!(outside.matches(&night));
    }

    #[tokio::test]
    async fn defaults_apply_when_every_policy_abstains() {
        let set = PolicySet::new().with(Rule::deny("never").tools(["drop_db"]));
        let v = set.evaluate(&tool("read", json!({}))).await;
        assert_eq!((v.decision, v.policy_id), (Decision::Allow, None));
        let v = set
            .evaluate(&tool("send", json!({})).side_effects(true))
            .await;
        assert_eq!(
            v.decision,
            Decision::deny("no policy allows tool `send` (it has side effects)")
        );
        let v = set.evaluate(&tool("drop_db", json!({}))).await;
        assert_eq!(v.policy_id.as_deref(), Some("never"));
        assert_eq!(v.decision, Decision::deny("denied by policy `never`"));

        let strict = PolicySet::new().defaults(Defaults::deny_all());
        assert!(strict
            .evaluate(&Request::model("a", "m"))
            .await
            .decision
            .is_denied());
        let ask = PolicySet::new().defaults(Defaults {
            side_effect_tools: Effect::Ask,
            ..Defaults::default()
        });
        assert!(matches!(
            ask.evaluate(&tool("send", json!({})).side_effects(true))
                .await
                .decision,
            Decision::Ask { .. }
        ));
    }

    struct Failing;
    #[async_trait]
    impl Policy for Failing {
        fn id(&self) -> &str {
            "failing"
        }
        async fn evaluate(&self, _: &Request) -> Result<Option<Decision>, GovernError> {
            Err(GovernError::Evaluation {
                policy: "failing".into(),
                message: "down".into(),
            })
        }
    }

    #[tokio::test]
    async fn failing_policies_deny() {
        let set = PolicySet::new().with(Rule::allow("all")).with(Failing);
        let v = set.evaluate(&tool("read", json!({}))).await;
        assert!(v.decision.is_denied());
        assert_eq!(v.policy_id.as_deref(), Some("failing"));
        assert_eq!(v.matched.len(), 2);
    }

    #[tokio::test]
    async fn rate_limit_keys_and_window() {
        let rl = RateLimit::new("rl", 1, Duration::from_secs(10)).per([RateKey::User]);
        let r = |user: &str, t: u64| Request::tool("a", "x", json!({})).user(user).at(t);
        assert!(rl.evaluate(&r("ana", 100_000)).await.unwrap().is_none());
        assert!(rl.evaluate(&r("ana", 105_000)).await.unwrap().is_some());
        assert!(rl.evaluate(&r("bob", 105_000)).await.unwrap().is_none());
        assert!(rl.evaluate(&r("ana", 110_001)).await.unwrap().is_none());
        // Model calls are not counted by a tool limiter.
        assert!(rl
            .evaluate(&Request::model("a", "m").user("ana").at(110_002))
            .await
            .unwrap()
            .is_none());
        let models = RateLimit::new("m", 1, Duration::from_secs(10)).model_calls();
        assert!(models
            .evaluate(&Request::model("a", "m").at(1))
            .await
            .unwrap()
            .is_none());
        assert!(models
            .evaluate(&Request::model("a", "m").at(2))
            .await
            .unwrap()
            .is_some());
    }
}
