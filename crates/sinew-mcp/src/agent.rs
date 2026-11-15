//! Wraps an agent graph as a single Sinew [`Tool`].

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{Error, Message, Result, Role, Tool, ToolSpec};
use sinew_graph::{prebuilt::AgentState, CompiledGraph};

/// An agent (any `CompiledGraph<AgentState>`, typically from `react_agent`)
/// exposed as a tool taking `{"input": "..."}` and returning the content of
/// the last assistant message.
#[derive(Clone)]
pub struct AgentTool {
    spec: ToolSpec,
    graph: CompiledGraph<AgentState>,
    system: Option<String>,
}

impl AgentTool {
    pub fn new(name: &str, description: &str, graph: CompiledGraph<AgentState>) -> Self {
        Self {
            spec: ToolSpec {
                name: name.into(),
                description: description.into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "input": { "type": "string", "description": "Task or question for the agent" }
                    },
                    "required": ["input"]
                }),
            },
            graph,
            system: None,
        }
    }

    /// System prompt placed before every input.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system = Some(prompt.into());
        self
    }
}

#[async_trait]
impl Tool for AgentTool {
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
        let mut messages = Vec::new();
        if let Some(s) = &self.system {
            messages.push(Message::system(s.clone()));
        }
        messages.push(Message::user(input));
        let out = self
            .graph
            .invoke(AgentState::new(messages))
            .await
            .map_err(|e| err(e.to_string()))?;
        let answer = out
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .map(|m| m.content.clone())
            .unwrap_or_default();
        Ok(Value::String(answer))
    }
}
