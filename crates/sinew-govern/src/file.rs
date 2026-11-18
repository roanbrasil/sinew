//! Declarative policy files (YAML or JSON).
//!
//! ```yaml
//! version: 1
//! defaults:
//!   tools: allow
//!   side_effect_tools: deny
//!   models: allow
//! agents:
//!   - id: billing-bot
//!     owner: payments-team
//!     tenant: acme
//!     allowed_tools: ["get_*", "refund"]
//!     allowed_models: ["anthropic:*"]
//!     budget: { max_tool_calls: 500 }
//! disabled_tools:
//!   - { tool: "shell", reason: "incident 42" }
//! policies:
//!   - id: small-refunds
//!     effect: allow
//!     tools: [refund]
//!     when: ["amount <= 100", "currency in [\"USD\", \"CAD\"]"]
//!   - id: large-refunds
//!     effect: ask
//!     reason: refunds over 100 need a human
//!     tools: [refund]
//!     when: ["amount > 100"]
//!   - id: mask-card
//!     effect: redact
//!     fields: ["/card/number"]
//!   - id: business-hours
//!     effect: deny
//!     side_effects: true
//!     outside: { days: [mon, tue, wed, thu, fri], from: "09:00", to: "17:00", utc_offset: "-05:00" }
//!   - id: refund-rate
//!     rate_limit: { max: 5, per_seconds: 60, key: [agent, tool] }
//!     tools: [refund]
//! ```
//!
//! Each policy entry is a [`Rule`] (`effect` is `allow`, `ask`, `deny` or
//! `redact`) or, with `rate_limit`, a [`RateLimit`]. Unknown fields,
//! unknown effects, malformed conditions, times and globs are reported with
//! the entry they are in (`policies[1] (id "large-refunds")`) or the line
//! and column.

use std::{path::Path, sync::Arc, time::Duration};

use serde::Deserialize;

use crate::{
    condition::Condition,
    decision::Decision,
    policy::{Defaults, Policy, RateKey, RateLimit, Rule},
    registry::AgentProfile,
    request::RequestKind,
    time::{parse_hhmm, parse_offset, TimeWindow, Weekday},
    GovernError,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFile {
    version: u32,
    #[serde(default)]
    defaults: Option<Defaults>,
    #[serde(default)]
    agents: Vec<AgentProfile>,
    #[serde(default)]
    disabled_tools: Vec<RawDisabled>,
    #[serde(default)]
    policies: Vec<RawPolicy>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDisabled {
    tool: String,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicy {
    id: String,
    #[serde(default)]
    effect: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    fields: Vec<String>,
    #[serde(default)]
    agents: Vec<String>,
    #[serde(default)]
    users: Vec<String>,
    #[serde(default)]
    tenants: Vec<String>,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    applies_to: Option<RequestKind>,
    #[serde(default)]
    side_effects: Option<bool>,
    #[serde(default)]
    when: Vec<String>,
    #[serde(default)]
    during: Option<RawWindow>,
    #[serde(default)]
    outside: Option<RawWindow>,
    #[serde(default)]
    rate_limit: Option<RawRate>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWindow {
    #[serde(default)]
    days: Vec<Weekday>,
    from: String,
    to: String,
    #[serde(default)]
    utc_offset: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRate {
    max: u32,
    per_seconds: u64,
    #[serde(default)]
    key: Option<Vec<RateKey>>,
}

/// A parsed and validated policy file. Turn it into a control plane with
/// [`ControlPlaneBuilder::policy_file`](crate::ControlPlaneBuilder::policy_file),
/// or swap it into a running one with
/// [`ControlPlane::apply_file`](crate::ControlPlane::apply_file). See the
/// [module docs](self) for the format.
///
/// ```
/// use sinew_govern::PolicyFile;
///
/// let file = PolicyFile::from_json_str(r#"{
///   "version": 1,
///   "agents": [{ "id": "billing-bot" }],
///   "policies": [{ "id": "small-refunds", "effect": "allow", "tools": ["refund"], "when": ["amount <= 100"] }]
/// }"#).unwrap();
/// assert_eq!(file.policy_ids(), ["small-refunds"]);
///
/// let err = PolicyFile::from_json_str(r#"{"version": 1, "policies": [{"id": "x", "effect": "alow"}]}"#)
///     .unwrap_err();
/// assert_eq!(
///     err.to_string(),
///     "policy file: policies[0] (id \"x\"): unknown effect `alow` (expected allow, ask, deny or redact)"
/// );
/// ```
pub struct PolicyFile {
    agents: Vec<AgentProfile>,
    policies: Vec<Arc<dyn Policy>>,
    defaults: Option<Defaults>,
    disabled_tools: Vec<(String, String)>,
}

impl std::fmt::Debug for PolicyFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyFile")
            .field("agents", &self.agents)
            .field("policies", &self.policy_ids())
            .field("defaults", &self.defaults)
            .field("disabled_tools", &self.disabled_tools)
            .finish()
    }
}

impl PolicyFile {
    /// Parses a JSON policy file.
    pub fn from_json_str(text: &str) -> Result<Self, GovernError> {
        let raw: RawFile = serde_json::from_str(text).map_err(|e| {
            GovernError::file(
                format!("line {} column {}", e.line(), e.column()),
                strip_position(&e.to_string()),
            )
        })?;
        Self::from_raw(raw)
    }

    /// Parses a YAML policy file (feature `yaml`, on by default).
    #[cfg(feature = "yaml")]
    pub fn from_yaml_str(text: &str) -> Result<Self, GovernError> {
        let raw: RawFile = serde_norway::from_str(text).map_err(|e| {
            let location = e
                .location()
                .map(|l| format!("line {} column {}", l.line(), l.column()))
                .unwrap_or_else(|| "document".into());
            GovernError::file(location, strip_position(&e.to_string()))
        })?;
        Self::from_raw(raw)
    }

    /// Reads a policy file: `.json` as JSON, `.yaml` / `.yml` as YAML.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, GovernError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        match path.extension().and_then(|e| e.to_str()) {
            Some("json") => Self::from_json_str(&text),
            #[cfg(feature = "yaml")]
            Some("yaml" | "yml") => Self::from_yaml_str(&text),
            other => Err(GovernError::file(
                path.display().to_string(),
                format!(
                    "unsupported extension {:?} (expected .json{})",
                    other.unwrap_or(""),
                    if cfg!(feature = "yaml") {
                        ", .yaml or .yml"
                    } else {
                        "; enable the `yaml` feature for YAML"
                    }
                ),
            )),
        }
    }

    /// Ids of the policies, in order.
    pub fn policy_ids(&self) -> Vec<String> {
        self.policies.iter().map(|p| p.id().to_string()).collect()
    }

    /// Registered agents.
    pub fn agents(&self) -> &[AgentProfile] {
        &self.agents
    }

    /// The defaults, when the file sets them.
    pub fn defaults(&self) -> Option<Defaults> {
        self.defaults
    }

    /// Disabled tool patterns and their reasons.
    pub fn disabled_tools(&self) -> &[(String, String)] {
        &self.disabled_tools
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(
        self,
    ) -> (
        Vec<AgentProfile>,
        Vec<Arc<dyn Policy>>,
        Option<Defaults>,
        Vec<(String, String)>,
    ) {
        (
            self.agents,
            self.policies,
            self.defaults,
            self.disabled_tools,
        )
    }

    fn from_raw(raw: RawFile) -> Result<Self, GovernError> {
        if raw.version != 1 {
            return Err(GovernError::file(
                "version",
                format!("unsupported version {} (expected 1)", raw.version),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for (i, a) in raw.agents.iter().enumerate() {
            if a.id.trim().is_empty() {
                return Err(GovernError::file(format!("agents[{i}]"), "empty id"));
            }
            if !seen.insert(a.id.clone()) {
                return Err(GovernError::file(
                    format!("agents[{i}] (id {:?})", a.id),
                    "duplicate agent id",
                ));
            }
        }
        let mut ids = std::collections::HashSet::new();
        let mut policies: Vec<Arc<dyn Policy>> = Vec::new();
        for (i, p) in raw.policies.into_iter().enumerate() {
            let at = format!("policies[{i}] (id {:?})", p.id);
            if p.id.trim().is_empty() {
                return Err(GovernError::file(format!("policies[{i}]"), "empty id"));
            }
            if !ids.insert(p.id.clone()) {
                return Err(GovernError::file(at, "duplicate policy id"));
            }
            policies.push(build_policy(p).map_err(|m| GovernError::file(&at, m))?);
        }
        let disabled_tools = raw
            .disabled_tools
            .into_iter()
            .map(|d| {
                let reason = d.reason.unwrap_or_else(|| "disabled by policy file".into());
                (d.tool, reason)
            })
            .collect();
        Ok(Self {
            agents: raw.agents,
            policies,
            defaults: raw.defaults,
            disabled_tools,
        })
    }
}

fn build_policy(p: RawPolicy) -> Result<Arc<dyn Policy>, String> {
    if let Some(rate) = p.rate_limit {
        if p.effect.is_some() || p.reason.is_some() || !p.fields.is_empty() {
            return Err("a `rate_limit` entry takes no `effect`, `reason` or `fields`".into());
        }
        if !p.users.is_empty()
            || !p.tenants.is_empty()
            || !p.models.is_empty()
            || p.side_effects.is_some()
            || !p.when.is_empty()
            || p.during.is_some()
            || p.outside.is_some()
        {
            return Err(
                "a `rate_limit` entry filters only on `agents`, `tools` and `applies_to`".into(),
            );
        }
        if rate.max == 0 || rate.per_seconds == 0 {
            return Err("`rate_limit.max` and `rate_limit.per_seconds` must be positive".into());
        }
        let mut rl = RateLimit::new(p.id, rate.max, Duration::from_secs(rate.per_seconds))
            .agents(p.agents)
            .tools(p.tools);
        if let Some(key) = rate.key {
            rl = rl.per(key);
        }
        if p.applies_to == Some(RequestKind::Model) {
            rl = rl.model_calls();
        }
        return Ok(Arc::new(rl));
    }
    let effect = p
        .effect
        .as_deref()
        .ok_or("missing `effect` (allow, ask, deny or redact) or `rate_limit`")?;
    let mut rule = match effect {
        "allow" => Rule::allow(p.id),
        "ask" => Rule::ask(p.id),
        "deny" => Rule::deny(p.id),
        "redact" => {
            if p.fields.is_empty() {
                return Err("`effect: redact` needs `fields` (JSON pointers)".into());
            }
            Rule::new(p.id, Decision::redact(p.fields.clone()))
        }
        other => {
            return Err(format!(
                "unknown effect `{other}` (expected allow, ask, deny or redact)"
            ))
        }
    };
    if effect != "redact" && !p.fields.is_empty() {
        return Err("`fields` is only valid with `effect: redact`".into());
    }
    if let Some(reason) = p.reason {
        if matches!(effect, "allow" | "redact") {
            return Err(format!(
                "`reason` is only valid with `ask` or `deny`, not `{effect}`"
            ));
        }
        rule = rule.reason(reason);
    }
    for (name, list) in [
        ("agents", &p.agents),
        ("users", &p.users),
        ("tenants", &p.tenants),
        ("tools", &p.tools),
        ("models", &p.models),
    ] {
        if list.iter().any(|s| s.trim().is_empty()) {
            return Err(format!("`{name}` contains an empty pattern"));
        }
    }
    if !p.tools.is_empty() && !p.models.is_empty() {
        return Err("a rule matches `tools` or `models`, not both; split it in two".into());
    }
    rule = rule
        .agents(p.agents)
        .users(p.users)
        .tenants(p.tenants)
        .tools(p.tools)
        .models(p.models);
    if let Some(kind) = p.applies_to {
        rule = rule.only(kind);
    }
    if let Some(s) = p.side_effects {
        rule = rule.side_effects(s);
    }
    for (j, c) in p.when.iter().enumerate() {
        let c = Condition::parse(c).map_err(|e| format!("when[{j}]: {e}"))?;
        rule = rule.when(c);
    }
    if let Some(w) = p.during {
        rule = rule.during(window(w).map_err(|e| format!("during: {e}"))?);
    }
    if let Some(w) = p.outside {
        rule = rule.outside(window(w).map_err(|e| format!("outside: {e}"))?);
    }
    Ok(Arc::new(rule))
}

fn window(w: RawWindow) -> Result<TimeWindow, GovernError> {
    Ok(TimeWindow {
        days: w.days,
        from_minute: parse_hhmm(&w.from)?,
        to_minute: parse_hhmm(&w.to)?,
        utc_offset_minutes: match &w.utc_offset {
            Some(o) => parse_offset(o)?,
            None => 0,
        },
    })
}

/// serde messages end with " at line X column Y"; the location is reported
/// separately.
fn strip_position(msg: &str) -> String {
    match msg.rfind(" at line ") {
        Some(i) => msg[..i].to_string(),
        None => msg.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ControlPlane, Request};
    use serde_json::json;

    const EXAMPLE: &str = r#"
version: 1
defaults:
  side_effect_tools: ask
agents:
  - id: billing-bot
    owner: payments-team
    tenant: acme
    allowed_tools: ["get_*", "refund"]
    budget: { max_tool_calls: 500 }
disabled_tools:
  - { tool: "shell", reason: "incident 42" }
policies:
  - id: small-refunds
    effect: allow
    tools: [refund]
    when: ["amount <= 100"]
  - id: large-refunds
    effect: ask
    reason: refunds over 100 need a human
    tools: [refund]
    when: ["amount > 100"]
  - id: mask-card
    effect: redact
    fields: ["/card/number"]
  - id: business-hours
    effect: deny
    side_effects: true
    outside: { days: [mon, tue, wed, thu, fri], from: "00:00", to: "00:00" }
  - id: refund-rate
    rate_limit: { max: 2, per_seconds: 60, key: [agent, tool] }
    tools: [refund]
"#;

    // 2026-11-16, a Monday, 10:00 UTC.
    const MONDAY_10H: u64 = 1_794_787_200_000 + 10 * 3_600_000;

    #[cfg(feature = "yaml")]
    #[tokio::test]
    async fn yaml_file_drives_a_control_plane() {
        let file = PolicyFile::from_yaml_str(EXAMPLE).unwrap();
        assert_eq!(
            file.policy_ids(),
            [
                "small-refunds",
                "large-refunds",
                "mask-card",
                "business-hours",
                "refund-rate"
            ]
        );
        assert_eq!(
            file.defaults().unwrap().side_effect_tools,
            crate::Effect::Ask
        );
        let cp = ControlPlane::builder().policy_file(file).build();
        let req = |args| {
            Request::tool("billing-bot", "refund", args)
                .side_effects(true)
                .at(MONDAY_10H)
        };
        let v = cp
            .check(req(json!({"amount": 40, "card": {"number": "4111"}})))
            .await;
        assert_eq!(v.decision, Decision::redact(["/card/number"]));
        let v = cp.check(req(json!({"amount": 400}))).await;
        assert_eq!(v.decision, Decision::ask("refunds over 100 need a human"));
        // Third refund in a minute: rate limited.
        let v = cp.check(req(json!({"amount": 1}))).await;
        assert_eq!(v.policy_id.as_deref(), Some("refund-rate"));
        // Saturday: outside the window.
        let v = cp
            .check(req(json!({"amount": 1})).at(MONDAY_10H + 5 * 86_400_000))
            .await;
        assert_eq!(v.policy_id.as_deref(), Some("business-hours"));
        // Registry: tenant and allowed tools.
        let v = cp
            .check(Request::tool("billing-bot", "delete_all", json!({})))
            .await;
        assert_eq!(v.policy_id.as_deref(), Some(crate::POLICY_ALLOWED));
        let v = cp
            .check(Request::tool("billing-bot", "get_x", json!({})).tenant("globex"))
            .await;
        assert_eq!(v.policy_id.as_deref(), Some(crate::POLICY_TENANT));
        assert_eq!(
            cp.disabled_tools(),
            [("shell".into(), "incident 42".into())]
        );
    }

    fn err(text: &str) -> String {
        PolicyFile::from_json_str(text).unwrap_err().to_string()
    }

    #[test]
    fn json_errors_name_the_entry() {
        let p = |entry: &str| format!(r#"{{"version": 1, "policies": [{entry}]}}"#);
        assert_eq!(
            err(r#"{"version": 2}"#),
            "policy file: version: unsupported version 2 (expected 1)"
        );
        assert!(err(r#"{"version": 1, "polices": []}"#).starts_with("policy file: line 1 column "));
        assert!(err(r#"{"version": 1, "polices": []}"#).contains("unknown field `polices`"));
        assert_eq!(
            err(&p(r#"{"id": "a"}"#)),
            "policy file: policies[0] (id \"a\"): missing `effect` (allow, ask, deny or redact) or `rate_limit`"
        );
        assert_eq!(
            err(&p(r#"{"id": "a", "effect": "allow", "when": ["amount <== 1"]}"#)),
            "policy file: policies[0] (id \"a\"): when[0]: invalid condition `amount <== 1`: unknown operator `<==` (expected ==, !=, <, <=, >, >=, in, not in, matches, exists or missing)"
        );
        assert!(err(&p(r#"{"id": "a", "effect": "redact"}"#)).contains("needs `fields`"));
        assert!(err(&p(r#"{"id": "a", "effect": "allow", "reason": "x"}"#))
            .contains("`reason` is only valid with `ask` or `deny`"));
        assert!(
            err(&p(r#"{"id": "a", "effect": "deny", "fields": ["/x"]}"#))
                .contains("only valid with `effect: redact`")
        );
        assert!(err(&p(
            r#"{"id": "a", "effect": "deny", "during": {"from": "9am", "to": "17:00"}}"#
        ))
        .contains("during: invalid time: expected HH:MM, got `9am`"));
        assert!(err(&p(
            r#"{"id": "a", "rate_limit": {"max": 0, "per_seconds": 1}}"#
        ))
        .contains("must be positive"));
        assert!(err(&p(
            r#"{"id": "a", "effect": "deny", "rate_limit": {"max": 1, "per_seconds": 1}}"#
        ))
        .contains("takes no `effect`"));
        assert!(err(&p(
            r#"{"id": "a", "effect": "deny", "tools": ["x"], "models": ["y"]}"#
        ))
        .contains("not both"));
        assert!(err(&format!(
            r#"{{"version": 1, "policies": [{0}, {0}]}}"#,
            r#"{"id": "a", "effect": "deny"}"#
        ))
        .contains("policies[1] (id \"a\"): duplicate policy id"));
        assert!(
            err(r#"{"version": 1, "agents": [{"id": "a"}, {"id": "a"}]}"#)
                .contains("duplicate agent id")
        );
        assert!(
            err(r#"{"version": 1, "agents": [{"id": "a", "statu": "active"}]}"#)
                .contains("unknown field `statu`")
        );
        assert!(err(r#"{"version": 1, "defaults": {"tools": "maybe"}}"#)
            .contains("unknown variant `maybe`"));
    }

    #[cfg(feature = "yaml")]
    #[test]
    fn yaml_errors_have_positions() {
        let e = PolicyFile::from_yaml_str("version: 1\npolicies:\n  - id: a\n    efect: deny\n")
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("policy file: line 4 column 5: "), "{e}");
        assert!(e.contains("unknown field `efect`"), "{e}");
    }

    #[test]
    fn reads_files_by_extension() {
        let dir = std::env::temp_dir().join(format!("sinew-govern-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let json_path = dir.join("p.json");
        std::fs::write(&json_path, r#"{"version": 1}"#).unwrap();
        assert!(PolicyFile::from_path(&json_path).is_ok());
        let txt = dir.join("p.txt");
        std::fs::write(&txt, "").unwrap();
        assert!(PolicyFile::from_path(&txt)
            .unwrap_err()
            .to_string()
            .contains("unsupported extension"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
