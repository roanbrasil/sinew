//! [`A2aAgentTool`]: a remote A2A agent as a Sinew [`Tool`], so a local
//! agent can delegate work to it.

use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{Error, Result, Tool, ToolSpec};

use crate::{
    client::A2aClient,
    types::{task_text, AgentCard, Message, MessageSendParams, SendMessageResult, TaskState},
};

/// Calls a remote agent with `message/send` and returns its answer as text.
///
/// Arguments: `{"input": "...", "task_id"?: "..."}`. When the remote agent
/// asks for more input (`input-required`), the tool result says so and
/// includes the task id; the model answers by calling the tool again with
/// that `task_id`. Calls share one A2A `contextId` by default, so the remote
/// agent sees them as one conversation; see [`with_shared_context`](Self::with_shared_context).
pub struct A2aAgentTool {
    client: A2aClient,
    spec: ToolSpec,
    shared_context: bool,
    context_id: Mutex<Option<String>>,
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
        }
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
        let result = self
            .client
            .send_message(MessageSendParams::new(message))
            .await
            .map_err(|e| err(e.to_string()))?;
        match result {
            SendMessageResult::Message(m) => {
                if let Some(c) = m.context_id.clone().filter(|_| self.shared_context) {
                    *self.context_id.lock().unwrap() = Some(c);
                }
                Ok(Value::String(m.text()))
            }
            SendMessageResult::Task(t) => {
                if self.shared_context {
                    *self.context_id.lock().unwrap() = Some(t.context_id.clone());
                }
                let text = task_text(&t);
                match t.status.state {
                    TaskState::Completed => Ok(Value::String(text)),
                    s if s.is_interrupted() => Ok(Value::String(format!(
                        "The remote agent needs more input (task_id: {}): {}",
                        t.id,
                        t.status.message.as_ref().map(Message::text).unwrap_or(text)
                    ))),
                    TaskState::Working | TaskState::Submitted => Ok(Value::String(format!(
                        "The remote agent is still working on task {}. Partial output: {text}",
                        t.id
                    ))),
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
