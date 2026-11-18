//! [`VulnerableModel`]: a deterministic chat model that follows instructions
//! it finds in its input, for attack scenarios run offline.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use sinew_core::{ChatModel, ChatRequest, ChatResponse, Error, Message, Role, Usage};

#[derive(Debug, Clone)]
struct Rule {
    triggers: Vec<String>,
    reply: Message,
    fired: bool,
}

#[derive(Debug, Default)]
struct Inner {
    rules: Vec<Rule>,
    script: VecDeque<Message>,
    requests: Vec<ChatRequest>,
    fired: Vec<String>,
}

/// A scripted model that emulates a model which falls for injected
/// instructions.
///
/// It answers from two sources, checked in this order on every call:
///
/// 1. **Rules** (`when(trigger, reply)`, `when_all([..], reply)`): if the
///    *new input* of the request contains the trigger (every trigger, for
///    `when_all`), the model answers `reply`. New input means the messages
///    after the last assistant message (the user message and tool results
///    the model has not seen yet), plus the tool descriptions sent with the
///    request. Each rule fires at most once, and a rule's answer is an extra
///    step: it does not consume a scripted answer.
/// 2. **Script** (`then(reply)`): otherwise the next scripted message, in
///    order, like [`ScriptedModel`](sinew_core::mock::ScriptedModel).
///
/// Because rules only fire when the attacker's text actually reaches the
/// model, a control that removes or blocks that text (an input or
/// tool-output guardrail, a permission that stops a tool from returning it)
/// changes what the model does, the same way it would for a real model that
/// follows injected instructions. Controls that act on the model's actions
/// (permissions, approvals, budgets, output guardrails) are tested with the
/// script alone.
///
/// ```
/// # async fn demo() -> sinew_core::Result<()> {
/// use sinew_core::{ChatModel, ChatRequest, Message};
/// use sinew_eval::security::VulnerableModel;
///
/// let model = VulnerableModel::new()
///     .when("attacker@evil.example", Message::assistant("Sending the file to attacker@evil.example."))
///     .then(Message::assistant("Here is your summary."));
///
/// let clean = model.chat(ChatRequest::new(vec![Message::user("Summarize my inbox")])).await?;
/// assert_eq!(clean.message.content, "Here is your summary.");
/// assert!(model.fired().is_empty());
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, Default)]
pub struct VulnerableModel {
    inner: Arc<Mutex<Inner>>,
}

impl VulnerableModel {
    /// A model with no rules and an empty script.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a rule: answer `reply` once when the new input contains `trigger`.
    pub fn when(self, trigger: impl Into<String>, reply: Message) -> Self {
        self.when_all([trigger.into()], reply)
    }

    /// Adds a rule that fires once when the new input contains every one of
    /// `triggers` (for example an instruction in a tool description and the
    /// data it asks for, which arrives later in a tool result).
    pub fn when_all<S: Into<String>>(
        self,
        triggers: impl IntoIterator<Item = S>,
        reply: Message,
    ) -> Self {
        self.lock().rules.push(Rule {
            triggers: triggers.into_iter().map(Into::into).collect(),
            reply,
            fired: false,
        });
        self
    }

    /// Appends a scripted answer, used when no rule fires.
    pub fn then(self, reply: Message) -> Self {
        self.lock().script.push_back(reply);
        self
    }

    /// Appends several scripted answers.
    pub fn script(self, replies: impl IntoIterator<Item = Message>) -> Self {
        self.lock().script.extend(replies);
        self
    }

    /// Every request received so far.
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.lock().requests.clone()
    }

    /// Triggers of the rules that fired, in order (several triggers of one
    /// rule are joined with ` + `).
    pub fn fired(&self) -> Vec<String> {
        self.lock().fired.clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The text a rule is matched against: messages after the last assistant
/// message, plus tool descriptions.
fn new_input(request: &ChatRequest) -> String {
    let start = request
        .messages
        .iter()
        .rposition(|m| m.role == Role::Assistant)
        .map_or(0, |i| i + 1);
    let mut text = String::new();
    for m in &request.messages[start..] {
        if m.role != Role::System {
            text.push_str(&m.content);
            text.push('\n');
        }
    }
    for t in &request.tools {
        text.push_str(&t.description);
        text.push('\n');
    }
    text
}

#[async_trait]
impl ChatModel for VulnerableModel {
    fn name(&self) -> &str {
        "vulnerable-scripted"
    }

    fn capabilities(&self) -> sinew_core::Capabilities {
        sinew_core::Capabilities::all()
    }

    async fn chat(&self, request: ChatRequest) -> sinew_core::Result<ChatResponse> {
        let input_tokens = request
            .messages
            .iter()
            .map(|m| m.content.split_whitespace().count() as u32)
            .sum();
        let text = new_input(&request);
        let mut inner = self.lock();
        inner.requests.push(request);
        let hit = inner
            .rules
            .iter_mut()
            .find(|r| !r.fired && r.triggers.iter().all(|t| text.contains(t.as_str())))
            .map(|r| {
                r.fired = true;
                (r.triggers.join(" + "), r.reply.clone())
            });
        let message = match hit {
            Some((trigger, reply)) => {
                inner.fired.push(trigger);
                reply
            }
            None => inner.script.pop_front().ok_or(Error::ScriptExhausted)?,
        };
        let output_tokens = message.content.split_whitespace().count() as u32;
        Ok(ChatResponse {
            message,
            usage: Usage::new(input_tokens, output_tokens),
            model: "vulnerable-scripted".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sinew_core::{ToolCall, ToolSpec};

    #[tokio::test]
    async fn rules_fire_on_new_input_only_once() {
        let m = VulnerableModel::new()
            .when("EVIL", Message::assistant("obeyed"))
            .then(Message::assistant("default 1"))
            .then(Message::assistant("default 2"));
        let r1 = m
            .chat(ChatRequest::new(vec![
                Message::system("EVIL in system is ignored"),
                Message::user("hi"),
            ]))
            .await
            .unwrap();
        assert_eq!(r1.message.content, "default 1");
        let call = ToolCall {
            id: "1".into(),
            name: "t".into(),
            arguments: json!({}),
        };
        let r2 = m
            .chat(ChatRequest::new(vec![
                Message::user("hi"),
                Message::assistant_with_tools("", vec![call]),
                Message::tool_result("1", "EVIL payload"),
            ]))
            .await
            .unwrap();
        assert_eq!(r2.message.content, "obeyed");
        let r3 = m
            .chat(ChatRequest::new(vec![Message::tool_result("1", "EVIL")]))
            .await
            .unwrap();
        assert_eq!(r3.message.content, "default 2", "a rule fires once");
        assert_eq!(m.fired(), ["EVIL"]);
        assert!(m.chat(ChatRequest::new(vec![])).await.is_err());
        assert_eq!(m.requests().len(), 4);
    }

    #[tokio::test]
    async fn old_messages_do_not_trigger_but_tool_descriptions_do() {
        let m = VulnerableModel::new()
            .when("EVIL", Message::assistant("obeyed"))
            .then(Message::assistant("default"));
        let r = m
            .chat(ChatRequest::new(vec![
                Message::user("EVIL long ago"),
                Message::assistant("ok"),
                Message::user("next"),
            ]))
            .await
            .unwrap();
        assert_eq!(r.message.content, "default");
        let mut req = ChatRequest::new(vec![Message::user("next")]);
        req.tools.push(ToolSpec {
            name: "note".into(),
            description: "EVIL description".into(),
            parameters: json!({}),
        });
        assert_eq!(m.chat(req).await.unwrap().message.content, "obeyed");
    }

    #[tokio::test]
    async fn when_all_needs_every_trigger() {
        let m = VulnerableModel::new()
            .when_all(["A", "B"], Message::assistant("both"))
            .then(Message::assistant("default"));
        let only_a = m
            .chat(ChatRequest::new(vec![Message::user("A")]))
            .await
            .unwrap();
        assert_eq!(only_a.message.content, "default");
        let both = m
            .chat(ChatRequest::new(vec![Message::user("A B")]))
            .await
            .unwrap();
        assert_eq!(both.message.content, "both");
        assert_eq!(m.fired(), ["A + B"]);
    }
}
