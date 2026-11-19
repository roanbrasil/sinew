//! Approval requests that resist trust exploitation (OWASP ASI09).
//!
//! A human approving a tool call is a control only if the human sees what
//! the call does. A goal-hijacked agent can call `transfer_funds` while its
//! message says it is "updating notification settings"; a reviewer who
//! reads the message approves. This module adds to every
//! [`ApprovalRequest`]:
//!
//! - a [`RiskLevel`] (`low`, `high`, `critical`) set by the developer in an
//!   [`ApprovalConfig`] or by a [`Hook::tool_risk`](crate::Hook::tool_risk)
//!   such as `sinew_govern::GovernHook`;
//! - warnings from [`IntentCheck`]s that compare the assistant's message
//!   with the call ([`IntentRules`], or [`IntentJudge`] with a model);
//! - the arguments rendered in plain language from the tool's JSON Schema
//!   ([`render_arguments`]): labels come from the schema, not from the model;
//! - a [`Confirmation`] stronger than one click for high-impact calls: the
//!   reviewer retypes values (amount, recipient) or a second approver signs.
//!
//! Every decision is reported to [`Hook::on_approval`](crate::Hook::on_approval)
//! as an [`ApprovalRecord`] holding the request as shown, so an audit log can
//! record it.
//!
//! ```
//! use sinew_harness::{ApprovalConfig, Confirmation, IntentRules, RiskLevel};
//!
//! let approvals = ApprovalConfig::new()
//!     .risk("transfer_*", RiskLevel::Critical)
//!     .risk("send_email", RiskLevel::High)
//!     .confirm("transfer_*", Confirmation::retype(["to", "amount"]))
//!     .check(IntentRules::default().expect("transfer_*", ["transfer", "send", "pay"]));
//! assert_eq!(approvals.risk_for("transfer_funds"), RiskLevel::Critical);
//! assert_eq!(approvals.risk_for("read_file"), RiskLevel::Low);
//! ```

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinew_core::guardrails::{GuardrailStage, LlmJudge};
use sinew_core::{ChatModel, ToolCall, ToolSpec};

use crate::glob::wildcard_match;
use crate::state::{ApprovalDecision, ApprovalRequest};

/// How much damage a tool call can do. Ordered, so `a.max(b)` combines two
/// assessments.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    /// Read-only or easily undone (the default).
    #[default]
    Low,
    /// Changes something outside the agent: writes, messages, purchases.
    High,
    /// Irreversible or costly: payments, deletions, production changes.
    Critical,
}

impl RiskLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            RiskLevel::Low => "low",
            RiskLevel::High => "high",
            RiskLevel::Critical => "critical",
        }
    }

    /// `High` or `Critical`.
    pub fn is_high_impact(self) -> bool {
        self >= RiskLevel::High
    }
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RiskLevel {
    type Err = String;

    /// Parses `low`, `high` or `critical` (case-insensitive).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Ok(RiskLevel::Low),
            "high" => Ok(RiskLevel::High),
            "critical" => Ok(RiskLevel::Critical),
            other => Err(format!(
                "unknown risk level `{other}` (expected low, high or critical)"
            )),
        }
    }
}

/// A warning attached to an approval request, for example "the assistant's
/// message does not describe this call".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalWarning {
    /// Name of the [`IntentCheck`] that raised it.
    pub check: String,
    pub message: String,
}

/// One argument of a call rendered for a human. The label and description
/// come from the tool's JSON Schema; only the value comes from the model,
/// with control and bidirectional-override characters escaped and long
/// values clipped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArgumentLine {
    /// The argument's key.
    pub name: String,
    /// The schema's `title`, or the key in words (`account_id` → "Account id").
    pub label: String,
    /// The schema's `description`, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub value: String,
    /// `false` for an argument the schema does not declare.
    #[serde(default = "yes")]
    pub in_schema: bool,
}

fn yes() -> bool {
    true
}

/// What a reviewer must do, beyond approving, for the approval to count.
///
/// The harness checks it in [`Agent::resume`](crate::Agent::resume): an
/// approval that does not satisfy it fails with
/// [`HarnessError::Confirmation`](crate::HarnessError::Confirmation) and
/// the thread stays paused, so the reviewer can try again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Confirmation {
    /// Arguments the reviewer must retype exactly
    /// ([`ApprovalDecision::retype`]); keys or JSON pointers (`/payee/iban`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retype: Vec<String>,
    /// Distinct approvers needed ([`ApprovalDecision::approver`]); `1` means
    /// a single approval, with or without a name.
    #[serde(default = "one")]
    pub approvers: usize,
}

fn one() -> usize {
    1
}

impl Default for Confirmation {
    fn default() -> Self {
        Self::plain()
    }
}

impl Confirmation {
    /// One approval, nothing more.
    pub fn plain() -> Self {
        Self {
            retype: Vec::new(),
            approvers: 1,
        }
    }

    /// The reviewer retypes these arguments.
    pub fn retype<I, S>(fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            retype: fields.into_iter().map(Into::into).collect(),
            approvers: 1,
        }
    }

    /// Two distinct approvers.
    pub fn second_approver() -> Self {
        Self::plain().approvers(2)
    }

    /// Requires `n` distinct approvers (at least 1).
    pub fn approvers(mut self, n: usize) -> Self {
        self.approvers = n.max(1);
        self
    }

    /// One approval, nothing more.
    pub fn is_plain(&self) -> bool {
        self.retype.is_empty() && self.approvers <= 1
    }
}

/// What an [`IntentCheck`] looks at.
#[derive(Debug, Clone, Copy)]
pub struct IntentContext<'a> {
    /// The text of the assistant message that made the call: what a chat UI
    /// shows next to the approval request. Never empty when a check runs.
    pub explanation: &'a str,
    pub call: &'a ToolCall,
    /// The called tool's spec, when the agent has the tool.
    pub spec: Option<&'a ToolSpec>,
    /// Every tool of the agent.
    pub tools: &'a [ToolSpec],
    pub risk: RiskLevel,
}

/// Compares the assistant's message with the tool call it made, before a
/// human is asked to approve the call. Returns a warning when they disagree.
///
/// Checks run only for calls that need approval and whose assistant message
/// has text.
#[async_trait]
pub trait IntentCheck: Send + Sync {
    /// Shown in [`ApprovalWarning::check`].
    fn name(&self) -> &str;

    /// `Some(warning)` when the explanation does not match the call.
    async fn check(&self, ctx: &IntentContext<'_>) -> Option<String>;
}

const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "you", "your", "yours", "this", "that", "these", "those", "with", "now",
    "are", "was", "were", "will", "from", "into", "have", "has", "had", "its", "our", "all", "can",
    "not", "but", "let", "just", "here", "there", "some", "then", "than", "also", "been", "being",
    "what", "which", "who", "when", "where", "how", "i'm", "i'll", "going", "about", "make",
    "sure", "please", "done", "okay", "one", "get", "via", "use", "using", "new",
];

/// Lower-case words of at least three characters, without stop words.
fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3)
        .map(str::to_lowercase)
        .filter(|w| !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

/// Same word, or the same stem: one is a prefix of the other, or they share
/// at least five leading characters ("update" / "updating").
fn similar(a: &str, b: &str) -> bool {
    let common = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    common == a.chars().count().min(b.chars().count()) || common >= 5
}

fn scalar_words(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.extend(words(s)),
        Value::Number(n) => out.extend(words(&n.to_string())),
        Value::Array(a) => a.iter().for_each(|x| scalar_words(x, out)),
        Value::Object(o) => o.values().for_each(|x| scalar_words(x, out)),
        _ => {}
    }
}

/// Words that describe a call: its tool name, the tool description and the
/// argument values.
fn call_words(call: &ToolCall, spec: Option<&ToolSpec>) -> Vec<String> {
    let mut out = words(&call.name);
    if let Some(s) = spec {
        out.extend(words(&s.description));
    }
    scalar_words(&call.arguments, &mut out);
    out
}

/// Rule-based [`IntentCheck`], no model needed. On by default in
/// [`ApprovalConfig`]. It warns when:
///
/// - the message describes **another tool** of the agent (its name or
///   description words) and says nothing about the called one;
/// - the call is **high impact** (risk `high` or `critical`) and the message
///   mentions neither the tool nor any of its arguments;
/// - an [`expect`](Self::expect) rule matches the tool and the message
///   contains none of the expected words.
///
/// Words are compared case-insensitively by stem ("transfer" matches
/// "transferring"). The rules catch an agent that describes a different
/// action; they do not understand paraphrase, so pair them with
/// [`IntentJudge`] when that matters.
#[derive(Debug, Clone, Default)]
pub struct IntentRules {
    expectations: Vec<(String, Vec<String>)>,
}

impl IntentRules {
    /// The message must contain one of `words` when the tool matches
    /// `tool_pattern` (wildcards, `transfer_*`).
    pub fn expect<I, S>(mut self, tool_pattern: impl Into<String>, words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.expectations.push((
            tool_pattern.into(),
            words.into_iter().map(|w| w.into().to_lowercase()).collect(),
        ));
        self
    }
}

#[async_trait]
impl IntentCheck for IntentRules {
    fn name(&self) -> &str {
        "intent_rules"
    }

    async fn check(&self, ctx: &IntentContext<'_>) -> Option<String> {
        let said = words(ctx.explanation);
        let tool = &ctx.call.name;
        for (pattern, expected) in &self.expectations {
            if wildcard_match(pattern, tool)
                && !expected.iter().any(|e| said.iter().any(|w| similar(w, e)))
            {
                return Some(format!(
                    "the assistant's message does not say it will {} (it calls `{tool}`)",
                    expected.join(" / ")
                ));
            }
        }
        let own = call_words(ctx.call, ctx.spec);
        let mentions_call = said.iter().any(|w| own.iter().any(|o| similar(w, o)));
        if mentions_call {
            return None;
        }
        // The tool the message seems to describe instead.
        let other = ctx
            .tools
            .iter()
            .filter(|t| t.name != *tool)
            .map(|t| {
                let mut theirs = words(&t.name);
                theirs.extend(words(&t.description));
                let hits: Vec<&String> = said
                    .iter()
                    .filter(|w| theirs.iter().any(|x| similar(w, x)))
                    .collect();
                (t, hits)
            })
            .filter(|(_, hits)| !hits.is_empty())
            .max_by_key(|(_, hits)| hits.len());
        if let Some((t, hits)) = other {
            let mut hits: Vec<&str> = hits.iter().map(|s| s.as_str()).collect();
            hits.dedup();
            return Some(format!(
                "the assistant's message describes `{}` ({}), but the call is `{tool}`",
                t.name,
                hits.join(", ")
            ));
        }
        if ctx.risk.is_high_impact() {
            return Some(format!(
                "the assistant's message mentions neither `{tool}` nor its arguments, and the call is {} risk",
                ctx.risk
            ));
        }
        None
    }
}

/// The policy an [`IntentJudge`] gives its model.
pub const INTENT_JUDGE_POLICY: &str = "The text contains an AI assistant's message to its user \
and the tool call the assistant made in the same turn. The text complies when the message \
describes this call accurately: the same action, the same target and no important omission. It \
does not comply when the message describes a different action, hides the call, or understates \
its impact (for example it says \"updating settings\" while the call moves money).";

/// An [`IntentCheck`] that asks a model, through
/// [`LlmJudge`], whether the message
/// describes the call. A model error becomes a warning too ("the intent
/// judge failed"), so a broken judge is visible to the reviewer.
///
/// ```
/// use std::sync::Arc;
/// use serde_json::json;
/// use sinew_core::{mock::ScriptedModel, Message, ToolCall};
/// use sinew_harness::{IntentCheck, IntentContext, IntentJudge, RiskLevel};
///
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// let model = Arc::new(ScriptedModel::new([Message::assistant_with_tools("", vec![ToolCall {
///     id: "j".into(),
///     name: "guardrail_verdict".into(),
///     arguments: json!({"allow": false, "reason": "the message hides a money transfer"}),
/// }])]));
/// let judge = IntentJudge::new(model);
/// let call = ToolCall { id: "c".into(), name: "transfer_funds".into(), arguments: json!({"to": "X"}) };
/// let ctx = IntentContext {
///     explanation: "Updating your notification settings.",
///     call: &call,
///     spec: None,
///     tools: &[],
///     risk: RiskLevel::Critical,
/// };
/// assert_eq!(judge.check(&ctx).await.as_deref(), Some("the message hides a money transfer"));
/// # });
/// ```
pub struct IntentJudge {
    judge: LlmJudge,
}

impl IntentJudge {
    /// A judge with [`INTENT_JUDGE_POLICY`].
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self::from_judge(LlmJudge::new(model, INTENT_JUDGE_POLICY).named("intent_judge"))
    }

    /// A judge with your own policy, model mode or retries.
    pub fn from_judge(judge: LlmJudge) -> Self {
        Self { judge }
    }

    /// The text the judge reviews.
    pub fn text(ctx: &IntentContext<'_>) -> String {
        let args = serde_json::to_string(&ctx.call.arguments).unwrap_or_default();
        let description = ctx.spec.map(|s| s.description.as_str()).unwrap_or("");
        format!(
            "Assistant message:\n{}\n\nTool call: {}({args})\nTool description: {description}\nRisk level: {}",
            ctx.explanation, ctx.call.name, ctx.risk
        )
    }
}

#[async_trait]
impl IntentCheck for IntentJudge {
    fn name(&self) -> &str {
        "intent_judge"
    }

    async fn check(&self, ctx: &IntentContext<'_>) -> Option<String> {
        match self
            .judge
            .judge(GuardrailStage::Output, &Self::text(ctx))
            .await
        {
            Ok(v) if v.allow => None,
            Ok(v) => Some(v.reason),
            Err(e) => Some(format!("the intent judge failed: {e}")),
        }
    }
}

/// How the harness builds approval requests: tool risk levels, calls that
/// always need approval, stronger confirmations and intent checks. Set it
/// with [`AgentBuilder::approvals`](crate::AgentBuilder::approvals).
///
/// The default has no risk levels or confirmations, asks before every
/// `critical` call, and runs [`IntentRules`].
#[derive(Clone)]
pub struct ApprovalConfig {
    risks: Vec<(String, RiskLevel)>,
    confirmations: Vec<(String, Confirmation)>,
    ask_at: Option<RiskLevel>,
    checks: Vec<Arc<dyn IntentCheck>>,
}

impl fmt::Debug for ApprovalConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApprovalConfig")
            .field("risks", &self.risks)
            .field("confirmations", &self.confirmations)
            .field("ask_at", &self.ask_at)
            .field(
                "checks",
                &self.checks.iter().map(|c| c.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for ApprovalConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ApprovalConfig {
    /// See the type docs for the defaults.
    pub fn new() -> Self {
        Self {
            risks: Vec::new(),
            confirmations: Vec::new(),
            ask_at: Some(RiskLevel::Critical),
            checks: vec![Arc::new(IntentRules::default())],
        }
    }

    /// Risk level of the tools matching `tool_pattern` (wildcards). When
    /// several patterns match, the highest level wins.
    pub fn risk(mut self, tool_pattern: impl Into<String>, level: RiskLevel) -> Self {
        self.risks.push((tool_pattern.into(), level));
        self
    }

    /// Confirmation for approving the tools matching `tool_pattern`. When
    /// several patterns match, the last one wins.
    pub fn confirm(mut self, tool_pattern: impl Into<String>, confirmation: Confirmation) -> Self {
        self.confirmations.push((tool_pattern.into(), confirmation));
        self
    }

    /// Calls at or above `level` need approval even when the permission
    /// policy allows them or the tool was approved "always" (default
    /// `Some(Critical)`; `None` leaves it to the policy). `auto_approve`
    /// still approves them: use a deny rule for limits that must hold
    /// without a human.
    pub fn ask_at(mut self, level: impl Into<Option<RiskLevel>>) -> Self {
        self.ask_at = level.into();
        self
    }

    /// Adds an intent check (after the default [`IntentRules`]).
    pub fn check(mut self, check: impl IntentCheck + 'static) -> Self {
        self.checks.push(Arc::new(check));
        self
    }

    pub fn check_arc(mut self, check: Arc<dyn IntentCheck>) -> Self {
        self.checks.push(check);
        self
    }

    /// Removes every intent check, the default one included.
    pub fn no_checks(mut self) -> Self {
        self.checks.clear();
        self
    }

    /// The developer-set risk level of `tool` (`Low` when no pattern
    /// matches). Hooks can raise it per call.
    pub fn risk_for(&self, tool: &str) -> RiskLevel {
        self.risks
            .iter()
            .filter(|(p, _)| wildcard_match(p, tool))
            .map(|(_, r)| *r)
            .max()
            .unwrap_or_default()
    }

    /// The confirmation for approving `tool`.
    pub fn confirmation_for(&self, tool: &str) -> Confirmation {
        self.confirmations
            .iter()
            .rev()
            .find(|(p, _)| wildcard_match(p, tool))
            .map(|(_, c)| c.clone())
            .unwrap_or_default()
    }

    pub fn ask_threshold(&self) -> Option<RiskLevel> {
        self.ask_at
    }

    pub(crate) fn checks(&self) -> &[Arc<dyn IntentCheck>] {
        &self.checks
    }
}

const MAX_VALUE_CHARS: usize = 200;

/// Escapes control and bidirectional-override characters (so a value
/// cannot fake extra lines or reorder text) and clips long values.
fn sanitize(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i == MAX_VALUE_CHARS {
            out.push('…');
            break;
        }
        let bidi = matches!(c, '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{200E}' | '\u{200F}');
        if c.is_control() || bidi {
            out.push_str(&c.escape_unicode().to_string());
        } else {
            out.push(c);
        }
    }
    out
}

fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => sanitize(s),
        Value::Null => "(none)".into(),
        other => sanitize(&other.to_string()),
    }
}

/// `account_id` → "Account id".
fn humanize(key: &str) -> String {
    let spaced = key.replace(['_', '-'], " ");
    let mut c = spaced.trim().chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => key.to_string(),
    }
}

/// Renders a call's arguments for a human from the tool's JSON Schema:
/// required properties first (in the schema's order), then the other
/// declared properties, then arguments the schema does not declare
/// (flagged with `in_schema: false`). Labels are the property `title` or
/// the key in words; descriptions come from the schema. Values are escaped
/// and clipped (see [`ArgumentLine`]).
///
/// ```
/// use serde_json::json;
/// use sinew_core::ToolSpec;
/// use sinew_harness::render_arguments;
///
/// let spec = ToolSpec {
///     name: "transfer_funds".into(),
///     description: "Transfer money".into(),
///     parameters: json!({
///         "type": "object",
///         "properties": {
///             "to": {"type": "string", "title": "Recipient account"},
///             "amount": {"type": "string", "description": "in USD"}
///         },
///         "required": ["to", "amount"]
///     }),
/// };
/// let lines = render_arguments(Some(&spec), &json!({"to": "ACC-1", "amount": "4900.00", "memo": "x"}));
/// assert_eq!(lines[0].label, "Recipient account");
/// assert_eq!(lines[1].label, "Amount");
/// assert_eq!(lines[1].description.as_deref(), Some("in USD"));
/// assert!(!lines[2].in_schema);
/// ```
pub fn render_arguments(spec: Option<&ToolSpec>, arguments: &Value) -> Vec<ArgumentLine> {
    let Some(args) = arguments.as_object() else {
        if arguments.is_null() {
            return Vec::new();
        }
        return vec![ArgumentLine {
            name: String::new(),
            label: "Arguments".into(),
            description: None,
            value: value_text(arguments),
            in_schema: false,
        }];
    };
    let empty = serde_json::Map::new();
    let props = spec
        .and_then(|s| s.parameters.get("properties"))
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let mut order: Vec<&str> = spec
        .and_then(|s| s.parameters.get("required"))
        .and_then(Value::as_array)
        .map(|r| r.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    for k in props.keys() {
        if !order.contains(&k.as_str()) {
            order.push(k);
        }
    }
    let mut lines = Vec::new();
    for key in &order {
        let (Some(value), Some(schema)) = (args.get(*key), props.get(*key)) else {
            continue;
        };
        lines.push(ArgumentLine {
            name: (*key).to_string(),
            label: schema
                .get("title")
                .and_then(Value::as_str)
                .map(sanitize)
                .unwrap_or_else(|| humanize(key)),
            description: schema
                .get("description")
                .and_then(Value::as_str)
                .map(sanitize),
            value: value_text(value),
            in_schema: true,
        });
    }
    for (key, value) in args {
        if !props.contains_key(key) {
            lines.push(ArgumentLine {
                name: sanitize(key),
                label: sanitize(key),
                description: None,
                value: value_text(value),
                in_schema: false,
            });
        }
    }
    lines
}

/// The text a reviewer must retype for `field` of `arguments` (`None` when
/// the call has no such argument).
pub fn retype_value(arguments: &Value, field: &str) -> Option<String> {
    let v = if field.starts_with('/') {
        arguments.pointer(field)
    } else {
        arguments.get(field)
    }?;
    Some(match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}

impl ApprovalRequest {
    /// Checks that `decision` satisfies this request's [`Confirmation`].
    /// Denials always pass. The harness runs this in
    /// [`Agent::resume`](crate::Agent::resume); UIs can run it first to
    /// let the reviewer correct a typo.
    ///
    /// ```
    /// use serde_json::json;
    /// use sinew_harness::{ApprovalDecision, ApprovalRequest, Confirmation};
    ///
    /// let mut req = ApprovalRequest::new("c1", "transfer_funds", json!({"to": "ACC-1", "amount": "10.00"}));
    /// req.confirmation = Confirmation::retype(["amount"]);
    /// assert!(req.check_confirmation(&ApprovalDecision::approve("c1")).is_err());
    /// assert!(req.check_confirmation(&ApprovalDecision::approve("c1").retype("amount", "10.00")).is_ok());
    /// ```
    pub fn check_confirmation(&self, decision: &ApprovalDecision) -> Result<(), String> {
        if !decision.approve {
            return Ok(());
        }
        for field in &self.confirmation.retype {
            let Some(expected) = retype_value(&self.arguments, field) else {
                continue;
            };
            match decision.retyped.get(field) {
                None => return Err(format!("`{field}` must be retyped to approve this call")),
                Some(typed) if typed.trim() != expected.trim() => {
                    return Err(format!(
                        "the retyped `{field}` does not match the call's argument"
                    ))
                }
                Some(_) => {}
            }
        }
        let need = self.confirmation.approvers;
        if need > 1 {
            let mut distinct: Vec<&str> = decision
                .approvers
                .iter()
                .map(|a| a.trim())
                .filter(|a| !a.is_empty())
                .collect();
            distinct.sort_unstable();
            distinct.dedup();
            if distinct.len() < need {
                return Err(format!(
                    "this call needs {need} distinct approvers, got {}",
                    distinct.len()
                ));
            }
        }
        Ok(())
    }

    /// The request as plain text for a terminal or a log: tool, risk,
    /// reason, rendered arguments, warnings, the confirmation needed and,
    /// last, what the assistant said.
    pub fn render(&self) -> String {
        let mut out = format!("{} [risk: {}]", self.tool, self.risk);
        if !self.reason.is_empty() {
            out.push_str(&format!("\nReason: {}", self.reason));
        }
        if self.rendered_arguments.is_empty() && !self.arguments.is_null() {
            out.push_str(&format!("\n  {}", value_text(&self.arguments)));
        }
        for line in &self.rendered_arguments {
            out.push_str(&format!("\n  {}: {}", line.label, line.value));
            if let Some(d) = &line.description {
                out.push_str(&format!("  ({d})"));
            }
            if !line.in_schema {
                out.push_str("  [not in the tool's schema]");
            }
        }
        for w in &self.warnings {
            out.push_str(&format!("\nWARNING ({}): {}", w.check, w.message));
        }
        if !self.confirmation.retype.is_empty() {
            out.push_str(&format!(
                "\nTo approve, retype: {}",
                self.confirmation.retype.join(", ")
            ));
        }
        if self.confirmation.approvers > 1 {
            out.push_str(&format!(
                "\nNeeds {} distinct approvers",
                self.confirmation.approvers
            ));
        }
        if let Some(e) = &self.explanation {
            out.push_str(&format!("\nThe assistant said: {}", sanitize(e)));
        }
        out
    }
}

/// What happened to one approval request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ApprovalOutcome {
    /// Approved; the call will run.
    Approved,
    /// Denied, or no decision was given; the call will not run.
    Denied,
    /// An approval that did not satisfy the [`Confirmation`]; the thread
    /// stays paused.
    Rejected { reason: String },
}

/// One approval decision with the request exactly as it was shown, passed
/// to [`Hook::on_approval`](crate::Hook::on_approval). Serializable, so an
/// audit log can store it as is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRecord {
    pub thread_id: String,
    pub request: ApprovalRequest,
    /// `None` when the reviewer gave no decision for this request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<ApprovalDecision>,
    #[serde(flatten)]
    pub outcome: ApprovalOutcome,
    /// Unix-epoch milliseconds.
    pub at_ms: u64,
}

/// Review data computed when a call needs approval, stored with the
/// pending call so the request survives restarts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CallReview {
    #[serde(default)]
    pub risk: RiskLevel,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ApprovalWarning>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rendered_arguments: Vec<ArgumentLine>,
    #[serde(default, skip_serializing_if = "Confirmation::is_plain")]
    pub confirmation: Confirmation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
}

const MAX_EXPLANATION_CHARS: usize = 2000;

/// Builds the review of one call that needs approval.
pub(crate) async fn review_call(
    config: &ApprovalConfig,
    tools: &[ToolSpec],
    call: &ToolCall,
    risk: RiskLevel,
    explanation: &str,
) -> CallReview {
    let spec = tools.iter().find(|t| t.name == call.name);
    let mut warnings = Vec::new();
    let text = explanation.trim();
    if !text.is_empty() {
        let ctx = IntentContext {
            explanation: text,
            call,
            spec,
            tools,
            risk,
        };
        for check in config.checks() {
            if let Some(message) = check.check(&ctx).await {
                warnings.push(ApprovalWarning {
                    check: check.name().to_string(),
                    message,
                });
            }
        }
    }
    CallReview {
        risk,
        warnings,
        rendered_arguments: render_arguments(spec, &call.arguments),
        confirmation: config.confirmation_for(&call.name),
        explanation: (!text.is_empty()).then(|| text.chars().take(MAX_EXPLANATION_CHARS).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(name: &str, description: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: description.into(),
            parameters: json!({"type": "object", "properties": {}}),
        }
    }

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: "c".into(),
            name: name.into(),
            arguments,
        }
    }

    async fn rules(
        rules: &IntentRules,
        said: &str,
        c: &ToolCall,
        risk: RiskLevel,
    ) -> Option<String> {
        let tools = [
            spec("transfer_funds", "Transfer money to an account"),
            spec("set_alerts", "Change notification settings"),
            spec("edit_file", "Edit a file"),
        ];
        let ctx = IntentContext {
            explanation: said,
            call: c,
            spec: tools.iter().find(|t| t.name == c.name),
            tools: &tools,
            risk,
        };
        rules.check(&ctx).await
    }

    #[tokio::test]
    async fn rules_flag_a_message_that_describes_another_tool() {
        let r = IntentRules::default();
        let t = call(
            "transfer_funds",
            json!({"to": "ACC-9", "amount": "4900.00"}),
        );
        let w = rules(
            &r,
            "I'm updating your notification settings now.",
            &t,
            RiskLevel::Low,
        )
        .await
        .unwrap();
        assert!(
            w.contains("`set_alerts`") && w.contains("notification"),
            "{w}"
        );
        // A matching message passes, stems included.
        assert!(
            rules(&r, "Transferring 4900 to ACC-9.", &t, RiskLevel::Critical)
                .await
                .is_none()
        );
        // Silence about a high-impact call.
        let w = rules(&r, "Done, all good.", &t, RiskLevel::High)
            .await
            .unwrap();
        assert!(w.contains("high risk"), "{w}");
        // ...but not about a low-risk one.
        assert!(rules(
            &r,
            "Let me fix that.",
            &call("edit_file", json!({"path": "a.rs"})),
            RiskLevel::Low
        )
        .await
        .is_none());
    }

    #[tokio::test]
    async fn expectations() {
        let r = IntentRules::default().expect("transfer_*", ["transfer", "pay"]);
        let t = call("transfer_funds", json!({"to": "ACC-9"}));
        let w = rules(&r, "Sending ACC-9 the thing.", &t, RiskLevel::Low)
            .await
            .unwrap();
        assert!(w.contains("transfer / pay"), "{w}");
        assert!(rules(&r, "Paying ACC-9 now.", &t, RiskLevel::Low)
            .await
            .is_none());
    }

    #[test]
    fn risk_levels_parse_and_order() {
        assert_eq!(
            "Critical".parse::<RiskLevel>().unwrap(),
            RiskLevel::Critical
        );
        assert_eq!("low".parse::<RiskLevel>().unwrap(), RiskLevel::Low);
        assert_eq!("high".parse::<RiskLevel>().unwrap().to_string(), "high");
        assert!("severe".parse::<RiskLevel>().is_err());
        assert!(RiskLevel::Critical > RiskLevel::High && RiskLevel::High.is_high_impact());
        assert!(!RiskLevel::Low.is_high_impact());
        assert_eq!(
            serde_json::to_value(RiskLevel::High).unwrap(),
            json!("high")
        );
    }

    #[test]
    fn config_lookups() {
        let c = ApprovalConfig::new()
            .risk("*", RiskLevel::High)
            .risk("delete_*", RiskLevel::Critical)
            .confirm("delete_*", Confirmation::retype(["id"]))
            .confirm("delete_db", Confirmation::second_approver())
            .ask_at(None)
            .no_checks();
        assert_eq!(c.risk_for("delete_user"), RiskLevel::Critical);
        assert_eq!(c.risk_for("read"), RiskLevel::High);
        assert_eq!(c.confirmation_for("delete_user").retype, vec!["id"]);
        assert_eq!(c.confirmation_for("delete_db").approvers, 2);
        assert!(c.confirmation_for("read").is_plain());
        assert_eq!(c.ask_threshold(), None);
        assert!(format!("{c:?}").contains("checks: []"));
        assert_eq!(
            ApprovalConfig::default().ask_threshold(),
            Some(RiskLevel::Critical)
        );
        assert_eq!(Confirmation::plain().approvers(0).approvers, 1);
    }

    #[test]
    fn rendering_escapes_and_orders() {
        let s = ToolSpec {
            name: "t".into(),
            description: String::new(),
            parameters: json!({
                "type": "object",
                "properties": {"b_field": {"type": "string"}, "a": {"type": "integer"}},
                "required": ["b_field"]
            }),
        };
        let long = "x".repeat(300);
        let lines = render_arguments(
            Some(&s),
            &json!({"a": 5, "b_field": "line1\nline2\u{202E}", "extra": long}),
        );
        assert_eq!(lines[0].label, "B field");
        assert_eq!(lines[0].value, "line1\\u{a}line2\\u{202e}");
        assert_eq!(lines[1].value, "5");
        assert!(!lines[2].in_schema && lines[2].value.ends_with('…'));
        assert_eq!(lines[2].value.chars().count(), MAX_VALUE_CHARS + 1);
        assert!(render_arguments(None, &Value::Null).is_empty());
        let raw = render_arguments(None, &json!("just text"));
        assert_eq!(raw[0].value, "just text");
        assert_eq!(value_text(&Value::Null), "(none)");
        assert_eq!(humanize(""), "");
    }

    #[test]
    fn confirmations_are_enforced() {
        let mut r = ApprovalRequest::new(
            "c",
            "pay",
            json!({"to": "ACC", "amount": 12.5, "payee": {"iban": "DE00"}}),
        );
        r.confirmation =
            Confirmation::retype(["to", "amount", "/payee/iban", "missing"]).approvers(2);
        let base = ApprovalDecision::approve("c");
        assert!(r
            .check_confirmation(&ApprovalDecision::deny("c", "no"))
            .is_ok());
        assert!(r
            .check_confirmation(&base)
            .unwrap_err()
            .contains("`to` must be retyped"));
        let typed = base
            .clone()
            .retype("to", " ACC ")
            .retype("amount", "12.5")
            .retype("/payee/iban", "DE00");
        let e = r
            .check_confirmation(&typed.clone().approver("ana"))
            .unwrap_err();
        assert!(e.contains("2 distinct approvers, got 1"), "{e}");
        assert!(r
            .check_confirmation(&typed.clone().approver("ana").approver("ana"))
            .is_err());
        assert!(r
            .check_confirmation(&typed.clone().approver("ana").approver("bo"))
            .is_ok());
        let wrong = typed
            .retype("amount", "1250")
            .approver("ana")
            .approver("bo");
        assert!(r
            .check_confirmation(&wrong)
            .unwrap_err()
            .contains("does not match"));
    }

    #[test]
    fn render_text() {
        let mut r = ApprovalRequest::new("c", "pay", json!({"to": "ACC"}));
        r.risk = RiskLevel::Critical;
        r.reason = "policy".into();
        r.rendered_arguments = render_arguments(None, &r.arguments);
        r.warnings.push(ApprovalWarning {
            check: "intent_rules".into(),
            message: "mismatch".into(),
        });
        r.confirmation = Confirmation::retype(["to"]).approvers(2);
        r.explanation = Some("Updating settings".into());
        let text = r.render();
        for part in [
            "pay [risk: critical]",
            "Reason: policy",
            "to: ACC  [not in the tool's schema]",
            "WARNING (intent_rules): mismatch",
            "retype: to",
            "Needs 2 distinct approvers",
            "The assistant said: Updating settings",
        ] {
            assert!(text.contains(part), "{part} in {text}");
        }
        let bare = ApprovalRequest::new("c", "x", json!([1]));
        assert!(bare.render().contains("[1]"));
    }

    #[tokio::test]
    async fn judge_errors_become_warnings() {
        let model = Arc::new(sinew_core::mock::ScriptedModel::new(Vec::<
            sinew_core::Message,
        >::new()));
        let judge = IntentJudge::new(model);
        let c = call("x", json!({}));
        let ctx = IntentContext {
            explanation: "hi",
            call: &c,
            spec: None,
            tools: &[],
            risk: RiskLevel::Low,
        };
        assert!(IntentJudge::text(&ctx).contains("Tool call: x({})"));
        assert!(judge
            .check(&ctx)
            .await
            .unwrap()
            .starts_with("the intent judge failed"));
        assert_eq!(judge.name(), "intent_judge");
    }
}
