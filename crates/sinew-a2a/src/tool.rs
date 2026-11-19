//! [`A2aAgentTool`]: a remote A2A agent as a Sinew [`Tool`], so a local
//! agent can delegate work to it.

use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{
    identity::{peer_data_block, PeerLedger, PeerRecord},
    Error, Result, Tool, ToolSpec,
};

use crate::{
    client::A2aClient,
    error::A2aError,
    types::{task_text, AgentCard, Message, MessageSendParams, SendMessageResult, TaskState},
};

/// Calls a remote agent with `message/send` and returns its answer as text.
///
/// Arguments: `{"input": "...", "task_id"?: "..."}`. When the remote agent
/// asks for more input (`input-required`), the tool result says so and
/// includes the task id; the model answers by calling the tool again with
/// that `task_id`. Calls share one A2A `contextId` by default, so the remote
/// agent sees them as one conversation; see [`with_shared_context`](Self::with_shared_context).
///
/// When the client has a [message verifier](A2aClient::with_message_verifier)
/// or the tool a [peer ledger](Self::with_peer_ledger), the answer is
/// treated as a peer message (OWASP ASI07): it is recorded in the ledger
/// with its verified sender (or as unsigned, or rejected) and the model gets
/// it fenced by [`peer_data_block`], marked as data with the sender metadata
/// outside the fence. A reply that fails verification never reaches the
/// model.
pub struct A2aAgentTool {
    client: A2aClient,
    spec: ToolSpec,
    shared_context: bool,
    context_id: Mutex<Option<String>>,
    ledger: Option<PeerLedger>,
}

impl A2aAgentTool {
    pub fn new(name: &str, description: &str, client: A2aClient) -> Self {
        Self {
            client,
            spec: ToolSpec {
                name: name.into(),
                description: description.into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Request or question for the remote agent, self-contained"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Only to answer a remote agent that asked for more input: the task id it gave"
                        }
                    },
                    "required": ["input"]
                }),
            },
            shared_context: true,
            context_id: Mutex::new(None),
            ledger: None,
        }
    }

    /// Records every answer in `ledger` (shared with a
    /// `sinew_govern::PeerApprovalPolicy`) and fences it as peer data.
    pub fn with_peer_ledger(mut self, ledger: PeerLedger) -> Self {
        self.ledger = Some(ledger);
        self
    }

    fn peer_mode(&self) -> bool {
        self.ledger.is_some() || self.client.message_verifier().is_some()
    }

    fn remote(&self) -> String {
        self.client
            .remote_agent_id()
            .unwrap_or_else(|| self.spec.name.clone())
    }

    /// The answer as the model sees it, recorded when in peer mode.
    fn present(
        &self,
        text: String,
        verified: Option<&sinew_core::identity::VerifiedMessage>,
    ) -> Value {
        if !self.peer_mode() {
            return Value::String(text);
        }
        let mut record = match verified {
            Some(v) => v.record(),
            None => PeerRecord::unsigned(self.remote(), &text),
        };
        if record.thread.is_none() {
            record.thread = self.context_id.lock().unwrap().clone();
        }
        if let Some(l) = &self.ledger {
            l.record(record.clone());
        }
        Value::String(peer_data_block(&record, &text))
    }

    /// Builds the tool from the client's Agent Card: the name is the card
    /// name made tool-safe, the description is the card description plus
    /// its skills.
    pub fn from_client(client: A2aClient) -> Self {
        let (name, description) = match client.card() {
            Some(card) => (tool_name(&card.name), describe(card)),
            None => ("remote_agent".to_string(), "A remote A2A agent".to_string()),
        };
        Self::new(&name, &description, client)
    }

    /// Discovers the agent at `base_url` and builds the tool from its card.
    pub async fn discover(base_url: &str) -> crate::Result<Self> {
        Ok(Self::from_client(A2aClient::discover(base_url).await?))
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.spec.name = name.into();
        self
    }

    pub fn with_description(mut self, description: &str) -> Self {
        self.spec.description = description.into();
        self
    }

    /// `false`: every call starts a new remote conversation.
    pub fn with_shared_context(mut self, shared: bool) -> Self {
        self.shared_context = shared;
        self
    }

    pub fn client(&self) -> &A2aClient {
        &self.client
    }
}

/// Lowercase, `[a-z0-9_]`, as model APIs require for tool names.
fn tool_name(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    while s.contains("__") {
        s = s.replace("__", "_");
    }
    let s = s.trim_matches('_').to_string();
    if s.is_empty() {
        "remote_agent".into()
    } else {
        s.chars().take(64).collect()
    }
}

fn describe(card: &AgentCard) -> String {
    let mut d = card.description.clone();
    if !card.skills.is_empty() {
        d.push_str(" Skills: ");
        let skills: Vec<String> = card
            .skills
            .iter()
            .map(|s| format!("{} ({})", s.name, s.description))
            .collect();
        d.push_str(&skills.join("; "));
        d.push('.');
    }
    d
}

#[async_trait]
impl Tool for A2aAgentTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value) -> Result<Value> {
        let err = |message: String| Error::Tool {
            name: self.spec.name.clone(),
            message,
        };
        let input = arguments
            .get("input")
            .and_then(Value::as_str)
            .ok_or_else(|| err("`input` (string) is required".into()))?;
        let mut message = Message::user_text(input);
        if let Some(t) = arguments
            .get("task_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            message.task_id = Some(t.into());
        } else if self.shared_context {
            message.context_id = self.context_id.lock().unwrap().clone();
        }
        let (result, verified) = match self
            .client
            .send_message_verified(MessageSendParams::new(message))
            .await
        {
            Ok(r) => r,
            Err(A2aError::Identity(e)) => {
                if let Some(l) = &self.ledger {
                    l.record(PeerRecord::rejected(self.remote(), "", e.to_string()));
                }
                return Err(err(format!(
                    "the reply from `{}` failed authentication and was dropped: {e}",
                    self.remote()
                )));
            }
            Err(e) => return Err(err(e.to_string())),
        };
        let verified = verified.as_ref();
        match result {
            SendMessageResult::Message(m) => {
                if let Some(c) = m.context_id.clone().filter(|_| self.shared_context) {
                    *self.context_id.lock().unwrap() = Some(c);
                }
                Ok(self.present(m.text(), verified))
            }
            SendMessageResult::Task(t) => {
                if self.shared_context {
                    *self.context_id.lock().unwrap() = Some(t.context_id.clone());
                }
                let text = task_text(&t);
                match t.status.state {
                    TaskState::Completed => Ok(self.present(text, verified)),
                    s if s.is_interrupted() => Ok(self.present(
                        format!(
                            "The remote agent needs more input (task_id: {}): {}",
                            t.id,
                            t.status.message.as_ref().map(Message::text).unwrap_or(text)
                        ),
                        verified,
                    )),
                    TaskState::Working | TaskState::Submitted => Ok(self.present(
                        format!(
                            "The remote agent is still working on task {}. Partial output: {text}",
                            t.id
                        ),
                        verified,
                    )),
                    s => Err(err(format!(
                        "remote task {} ended `{}`: {text}",
                        t.id,
                        serde_json::to_value(s)
                            .ok()
                            .and_then(|v| v.as_str().map(str::to_owned))
                            .unwrap_or_default()
                    ))),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names_are_safe() {
        assert_eq!(tool_name("Currency Agent!"), "currency_agent");
        assert_eq!(tool_name("  "), "remote_agent");
        assert_eq!(tool_name("a--b"), "a_b");
    }
}
