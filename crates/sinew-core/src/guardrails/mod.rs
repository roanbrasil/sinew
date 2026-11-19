//! Guardrails: input and output checks that wrap any model, chain, graph
//! node or harness agent.
//!
//! A guardrail looks at a piece of text at one [`GuardrailStage`] (the
//! user's input, a tool's output, or the model's answer) and returns a
//! [`GuardrailVerdict`]: pass, rewrite the text, or block it. Synchronous
//! checks implement [`Guardrail`]; checks that need I/O, such as the
//! [`LlmJudge`], implement [`AsyncGuardrail`]. Every `Guardrail` is also an
//! `AsyncGuardrail`.
//!
//! Built-in guardrails:
//!
//! | Guardrail | Finds | Verdict |
//! |---|---|---|
//! | [`PiiRedactor`] | e-mail addresses, phone numbers, Luhn-checked card numbers | rewrite with `[REDACTED_*]` |
//! | [`SecretRedactor`] | API keys and tokens with a known shape ([`patterns::SECRET_PATTERNS`]) | rewrite with `[REDACTED_SECRET]` |
//! | [`InjectionDetector`] | prompt-injection phrasing in tool output | rewrite with a warning prefix, or block |
//! | [`DenyList`] | denied keywords, topics or regular expressions | block, or rewrite with a placeholder |
//! | [`JsonSchemaGuard`] | text that is not JSON matching a schema | block |
//! | [`MaxLength`] | text longer than a character limit | block, or truncate |
//! | [`LlmJudge`] | whatever a policy prompt describes, judged by any [`ChatModel`](crate::ChatModel) | block |
//!
//! # Actions
//!
//! A [`GuardRule`] pairs a guardrail with a [`GuardAction`] that decides what
//! happens when the guardrail does not pass:
//!
//! - [`GuardAction::Redact`] (default): use the guardrail's rewrite; a
//!   guardrail that blocks still blocks.
//! - [`GuardAction::Block`]: fail with [`Error::Guardrail`].
//! - [`GuardAction::Refuse`]: answer with a fixed message instead.
//! - [`GuardAction::Warn`]: record a [`GuardEvent`] and keep the text as is.
//! - [`GuardAction::Retry`]: for model output, ask the model again with the
//!   reason as feedback, at most `max_retries` times.
//!
//! Every action that fires records a [`GuardEvent`], passed to the observer
//! set with [`Guardrails::on_event`] and returned in the [`GuardOutcome`].
//! `sinew_trace::guardrail_observer` turns events into trace runs.
//!
//! # Composition
//!
//! - [`Guarded`], built with
//!   [`ChatModelExt::with_guardrails`](crate::ChatModelExt::with_guardrails),
//!   is a [`ChatModel`](crate::ChatModel) that checks the new user messages
//!   and tool results of each request, then the response.
//! - [`Guardrails::step`] is a [`Runnable`](crate::Runnable) to put anywhere
//!   in a chain.
//! - `sinew_graph::prebuilt::ReactAgent::tool_guardrails` checks tool results
//!   before the model sees them.
//! - The `sinew-harness` agent uses the same [`Guardrail`] trait and built-in
//!   guardrails (re-exported as `sinew_harness::guardrails`).
//!
//! ```
//! use sinew_core::guardrails::{guard, InjectionDetector, JsonSchemaGuard, PiiRedactor};
//! use sinew_core::{mock::ScriptedModel, ChatModel, ChatModelExt, ChatRequest, Message};
//! use serde_json::json;
//!
//! # tokio_test_block(async {
//! let model = ScriptedModel::new([
//!     Message::assistant("sure, here it is"),
//!     Message::assistant(r#"{"city": "Montreal"}"#),
//! ])
//! .with_guardrails(
//!     [guard(PiiRedactor::new()), guard(InjectionDetector::new())],
//!     [guard(JsonSchemaGuard::new(json!({
//!         "type": "object",
//!         "properties": {"city": {"type": "string"}},
//!         "required": ["city"]
//!     })))
//!     .retry(2)],
//! );
//!
//! let resp = model
//!     .chat(ChatRequest::new(vec![Message::user("I am ana@example.com. Which city? JSON please.")]))
//!     .await
//!     .unwrap();
//! assert_eq!(resp.message.content, r#"{"city": "Montreal"}"#);
//! // The model never saw the e-mail address, and was asked once to fix its reply.
//! let sent = model.inner().requests();
//! assert!(sent[0].messages[0].content.contains("[REDACTED_EMAIL]"));
//! assert_eq!(sent.len(), 2);
//! # });
//! # fn tokio_test_block<F: std::future::Future>(f: F) -> F::Output {
//! #     tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
//! # }
//! ```

mod builtin;
mod guarded;
mod judge;
pub mod patterns;

use std::{fmt, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

pub use builtin::{
    DenyList, InjectionDetector, JsonSchemaGuard, MaxLength, PiiRedactor, SecretRedactor,
    INJECTION_WARNING,
};
pub use guarded::{GuardReport, GuardStep, Guarded, StreamGuardMode};
pub use judge::{JudgeVerdict, LlmJudge};

/// Where a guardrail runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailStage {
    /// The user's message, before the model sees it.
    Input,
    /// The model's answer.
    Output,
    /// A tool's output, before the model sees it.
    ToolOutput,
    /// A tool's description or one of its parameter descriptions, checked
    /// when the tool is registered (see [`crate::tool_vetting`]). Tool
    /// descriptions come from the tool's author, which for an MCP server is
    /// a third party: a poisoned description is a prompt injection the
    /// model reads on every turn.
    ToolDescription,
}

impl fmt::Display for GuardrailStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            GuardrailStage::Input => "input",
            GuardrailStage::Output => "output",
            GuardrailStage::ToolOutput => "tool_output",
            GuardrailStage::ToolDescription => "tool_description",
        })
    }
}

/// What a guardrail decided about a piece of text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailVerdict {
    /// The text is fine.
    Pass,
    /// Replace the text (redaction, annotation, truncation).
    Modify(String),
    /// Refuse the text, with a reason.
    Block(String),
}

/// A synchronous guardrail. Implement this for checks that need no I/O;
/// every `Guardrail` is also an [`AsyncGuardrail`].
///
/// ```
/// use sinew_core::guardrails::{Guardrail, GuardrailStage, GuardrailVerdict};
///
/// struct NoPasswords;
///
/// impl Guardrail for NoPasswords {
///     fn name(&self) -> &str {
///         "no_passwords"
///     }
///     fn check(&self, stage: GuardrailStage, text: &str) -> GuardrailVerdict {
///         if stage == GuardrailStage::Input && text.contains("password") {
///             GuardrailVerdict::Block("do not send passwords".into())
///         } else {
///             GuardrailVerdict::Pass
///         }
///     }
/// }
/// ```
pub trait Guardrail: Send + Sync {
    /// Name used in errors, events and traces.
    fn name(&self) -> &str;
    /// Checks `text` at `stage`.
    fn check(&self, stage: GuardrailStage, text: &str) -> GuardrailVerdict;
}

/// A guardrail that may do I/O, such as calling a model ([`LlmJudge`]).
///
/// An `Err` from [`check`](Self::check) is not a verdict: it propagates to
/// the caller (the request fails), so a judge that cannot be reached does
/// not let text through.
#[async_trait]
pub trait AsyncGuardrail: Send + Sync {
    /// Name used in errors, events and traces.
    fn name(&self) -> &str;
    /// Checks `text` at `stage`.
    async fn check(&self, stage: GuardrailStage, text: &str) -> Result<GuardrailVerdict>;
}

#[async_trait]
impl<G: Guardrail> AsyncGuardrail for G {
    fn name(&self) -> &str {
        Guardrail::name(self)
    }
    async fn check(&self, stage: GuardrailStage, text: &str) -> Result<GuardrailVerdict> {
        Ok(Guardrail::check(self, stage, text))
    }
}

/// Runs synchronous guardrails in order; rewrites chain. `Err(reason)`,
/// prefixed with the guardrail's name, when one blocks.
///
/// This is the [`GuardAction::Redact`] behavior without events; the
/// `sinew-harness` agent uses it for its input, output and tool-output
/// checks.
pub fn apply_guardrails(
    guardrails: &[Arc<dyn Guardrail>],
    stage: GuardrailStage,
    text: &str,
) -> std::result::Result<String, String> {
    let mut current = text.to_string();
    for g in guardrails {
        match g.check(stage, &current) {
            GuardrailVerdict::Pass => {}
            GuardrailVerdict::Modify(t) => current = t,
            GuardrailVerdict::Block(reason) => return Err(format!("{}: {reason}", g.name())),
        }
    }
    Ok(current)
}

/// What to do when a guardrail does not pass. See the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardAction {
    /// Apply the guardrail's rewrite ([`GuardrailVerdict::Modify`]); a
    /// [`GuardrailVerdict::Block`] still blocks.
    #[default]
    Redact,
    /// Fail with [`Error::Guardrail`], whatever the verdict.
    Block,
    /// Replace the text with this message. On model input the model is not
    /// called and the message is the answer; on output it replaces the
    /// answer; on a tool result it replaces the result.
    Refuse(String),
    /// Record a [`GuardEvent`] and continue with the text unchanged.
    Warn,
    /// On model output, append the rejected answer and a user message with
    /// the reason, and call the model again, at most `max_retries` times per
    /// rule; then fail with [`Error::Guardrail`]. On input and tool results,
    /// where there is nothing to retry, it behaves like [`Block`](Self::Block).
    Retry {
        /// Extra model calls allowed for this rule in one request.
        max_retries: usize,
    },
}

/// A guardrail with the action to take when it does not pass, and
/// optionally the stages it is limited to.
#[derive(Clone)]
pub struct GuardRule {
    guard: Arc<dyn AsyncGuardrail>,
    action: GuardAction,
    stages: Option<Vec<GuardrailStage>>,
}

impl fmt::Debug for GuardRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuardRule")
            .field("guard", &self.guard.name())
            .field("action", &self.action)
            .field("stages", &self.stages)
            .finish()
    }
}

/// Shorthand for [`GuardRule::new`]: `guard(PiiRedactor::new()).warn()`.
pub fn guard(guardrail: impl AsyncGuardrail + 'static) -> GuardRule {
    GuardRule::new(guardrail)
}

impl GuardRule {
    /// A rule with the default action, [`GuardAction::Redact`].
    pub fn new(guardrail: impl AsyncGuardrail + 'static) -> Self {
        Self::from_arc(Arc::new(guardrail))
    }

    /// A rule for a shared guardrail.
    pub fn from_arc(guardrail: Arc<dyn AsyncGuardrail>) -> Self {
        Self {
            guard: guardrail,
            action: GuardAction::Redact,
            stages: None,
        }
    }

    /// Sets the action.
    pub fn action(mut self, action: GuardAction) -> Self {
        self.action = action;
        self
    }

    /// [`GuardAction::Redact`].
    pub fn redact(self) -> Self {
        self.action(GuardAction::Redact)
    }

    /// [`GuardAction::Block`].
    pub fn block(self) -> Self {
        self.action(GuardAction::Block)
    }

    /// [`GuardAction::Refuse`] with `message`.
    pub fn refuse(self, message: impl Into<String>) -> Self {
        self.action(GuardAction::Refuse(message.into()))
    }

    /// [`GuardAction::Warn`].
    pub fn warn(self) -> Self {
        self.action(GuardAction::Warn)
    }

    /// [`GuardAction::Retry`] with at most `max_retries` extra model calls.
    pub fn retry(self, max_retries: usize) -> Self {
        self.action(GuardAction::Retry { max_retries })
    }

    /// Runs the rule only at these stages (by default it runs at every
    /// stage it is given; some guardrails also filter stages themselves,
    /// such as [`InjectionDetector`], which only looks at tool output).
    pub fn at(mut self, stages: impl IntoIterator<Item = GuardrailStage>) -> Self {
        self.stages = Some(stages.into_iter().collect());
        self
    }

    /// The guardrail's name.
    pub fn name(&self) -> &str {
        self.guard.name()
    }

    fn applies_to(&self, stage: GuardrailStage) -> bool {
        self.stages.as_ref().is_none_or(|s| s.contains(&stage))
    }
}

impl<G: AsyncGuardrail + 'static> From<G> for GuardRule {
    fn from(g: G) -> Self {
        GuardRule::new(g)
    }
}

/// A record of a guardrail that did not pass and the action taken.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardEvent {
    /// The guardrail's name.
    pub guard: String,
    /// Where the text was checked.
    pub stage: GuardrailStage,
    /// The rule's action.
    pub action: GuardAction,
    /// The guardrail's reason ([`GuardrailVerdict::Block`]), or a note that
    /// it rewrote the text ([`GuardrailVerdict::Modify`]).
    pub reason: String,
    /// Whether the guardrail returned a rewrite.
    pub rewrote: bool,
}

/// The final decision of a [`Guardrails::check`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardDecision {
    /// Continue with [`GuardOutcome::text`] (possibly rewritten).
    Allow,
    /// A rule blocked the text.
    Block {
        /// The guardrail's name.
        guard: String,
        /// Its reason.
        reason: String,
    },
    /// A [`GuardAction::Refuse`] rule fired; answer with `message`.
    Refuse {
        /// The guardrail's name.
        guard: String,
        /// The refusal message.
        message: String,
    },
    /// A [`GuardAction::Retry`] rule fired.
    Retry {
        /// The guardrail's name.
        guard: String,
        /// Its reason, used as feedback to the model.
        reason: String,
        /// The rule's position in the [`Guardrails`] list.
        rule: usize,
        /// The rule's retry budget.
        max_retries: usize,
    },
}

/// The result of [`Guardrails::check`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardOutcome {
    /// The text after every rewrite applied so far.
    pub text: String,
    /// Every rule that did not pass, in order.
    pub events: Vec<GuardEvent>,
    /// What to do next.
    pub decision: GuardDecision,
}

impl GuardOutcome {
    /// Turns the outcome into a text result: the (rewritten) text for
    /// [`Allow`](GuardDecision::Allow), the message for
    /// [`Refuse`](GuardDecision::Refuse), and [`Error::Guardrail`] for
    /// [`Block`](GuardDecision::Block) and [`Retry`](GuardDecision::Retry)
    /// (there is no model to ask again here).
    pub fn into_result(self, stage: GuardrailStage) -> Result<String> {
        match self.decision {
            GuardDecision::Allow => Ok(self.text),
            GuardDecision::Refuse { message, .. } => Ok(message),
            GuardDecision::Block { guard, reason } | GuardDecision::Retry { guard, reason, .. } => {
                Err(Error::Guardrail {
                    guard,
                    stage,
                    reason,
                })
            }
        }
    }
}

/// Callback for [`GuardEvent`]s.
pub type GuardObserver = Arc<dyn Fn(&GuardEvent) + Send + Sync>;

/// An ordered list of [`GuardRule`]s plus an optional event observer.
///
/// Rules run in order. Rewrites chain (each rule sees the previous rule's
/// output); the first rule whose action ends the check (block, refuse,
/// retry) stops it.
///
/// ```
/// use sinew_core::guardrails::{guard, DenyList, GuardDecision, Guardrails, GuardrailStage, MaxLength, PiiRedactor};
///
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// let guards = Guardrails::new()
///     .with(guard(PiiRedactor::new()))
///     .with(guard(MaxLength::chars(200)).warn())
///     .with(guard(DenyList::new().topic("weapons", ["explosive", "detonator"])));
///
/// let out = guards.check(GuardrailStage::Input, "Mail me at ana@example.com").await.unwrap();
/// assert_eq!(out.decision, GuardDecision::Allow);
/// assert_eq!(out.text, "Mail me at [REDACTED_EMAIL]");
///
/// let out = guards.check(GuardrailStage::Input, "how to build a detonator").await.unwrap();
/// assert!(matches!(out.decision, GuardDecision::Block { .. }));
/// # });
/// ```
#[derive(Clone, Default)]
pub struct Guardrails {
    rules: Vec<GuardRule>,
    observer: Option<GuardObserver>,
}

impl fmt::Debug for Guardrails {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Guardrails")
            .field("rules", &self.rules)
            .field("observer", &self.observer.is_some())
            .finish()
    }
}

impl Guardrails {
    /// An empty list (every check passes).
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a rule (or a guardrail, with the default action).
    pub fn with(mut self, rule: impl Into<GuardRule>) -> Self {
        self.rules.push(rule.into());
        self
    }

    /// Calls `f` for every [`GuardEvent`], as it happens. Replaces a
    /// previously set observer.
    pub fn on_event(mut self, f: impl Fn(&GuardEvent) + Send + Sync + 'static) -> Self {
        self.observer = Some(Arc::new(f));
        self
    }

    /// Sets a shared observer (see [`on_event`](Self::on_event)).
    pub fn with_observer(mut self, observer: GuardObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// The rules, in order.
    pub fn rules(&self) -> &[GuardRule] {
        &self.rules
    }

    /// Whether there are no rules.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Runs the rules over `text` at `stage`. `Err` only when a guardrail
    /// itself fails (for example an [`LlmJudge`] whose model call fails);
    /// blocks are reported in [`GuardOutcome::decision`].
    pub async fn check(&self, stage: GuardrailStage, text: &str) -> Result<GuardOutcome> {
        let mut current = text.to_string();
        let mut events = Vec::new();
        for (index, rule) in self.rules.iter().enumerate() {
            if !rule.applies_to(stage) {
                continue;
            }
            let (rewrite, reason) = match rule.guard.check(stage, &current).await? {
                GuardrailVerdict::Pass => continue,
                GuardrailVerdict::Modify(t) => {
                    (Some(t), "the text matched and was rewritten".to_string())
                }
                GuardrailVerdict::Block(r) => (None, r),
            };
            let guard = rule.guard.name().to_string();
            let event = GuardEvent {
                guard: guard.clone(),
                stage,
                action: rule.action.clone(),
                reason: reason.clone(),
                rewrote: rewrite.is_some(),
            };
            if let Some(o) = &self.observer {
                o(&event);
            }
            events.push(event);
            let decision = match &rule.action {
                GuardAction::Warn => continue,
                GuardAction::Redact => match rewrite {
                    Some(t) => {
                        current = t;
                        continue;
                    }
                    None => GuardDecision::Block { guard, reason },
                },
                GuardAction::Block => GuardDecision::Block { guard, reason },
                GuardAction::Refuse(message) => GuardDecision::Refuse {
                    guard,
                    message: message.clone(),
                },
                GuardAction::Retry { max_retries } => GuardDecision::Retry {
                    guard,
                    reason,
                    rule: index,
                    max_retries: *max_retries,
                },
            };
            return Ok(GuardOutcome {
                text: current,
                events,
                decision,
            });
        }
        Ok(GuardOutcome {
            text: current,
            events,
            decision: GuardDecision::Allow,
        })
    }

    /// A [`Runnable`](crate::Runnable) that checks its input at `stage`. See
    /// [`GuardStep`].
    pub fn step(&self, stage: GuardrailStage) -> GuardStep {
        GuardStep::new(self.clone(), stage)
    }
}

impl From<Vec<GuardRule>> for Guardrails {
    fn from(rules: Vec<GuardRule>) -> Self {
        Self {
            rules,
            observer: None,
        }
    }
}

impl<const N: usize> From<[GuardRule; N]> for Guardrails {
    fn from(rules: [GuardRule; N]) -> Self {
        Vec::from(rules).into()
    }
}

impl<R: Into<GuardRule>> FromIterator<R> for Guardrails {
    fn from_iter<T: IntoIterator<Item = R>>(iter: T) -> Self {
        iter.into_iter()
            .map(Into::into)
            .collect::<Vec<GuardRule>>()
            .into()
    }
}

#[cfg(test)]
mod tests;
