use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Error, Message, Result, ToolCall};

/// Tool description sent to the model. `parameters` is a JSON Schema object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call(&self, arguments: Value) -> Result<Value>;
}

/// Wraps a synchronous closure as a [`Tool`].
pub struct FnTool<F> {
    spec: ToolSpec,
    f: F,
}

impl<F> FnTool<F>
where
    F: Fn(Value) -> Result<Value> + Send + Sync,
{
    pub fn new(name: &str, description: &str, parameters: Value, f: F) -> Self {
        Self {
            spec: ToolSpec {
                name: name.into(),
                description: description.into(),
                parameters,
            },
            f,
        }
    }
}

#[async_trait]
impl<F> Tool for FnTool<F>
where
    F: Fn(Value) -> Result<Value> + Send + Sync,
{
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }
    async fn call(&self, arguments: Value) -> Result<Value> {
        (self.f)(arguments)
    }
}

/// Registry of tools, keyed by name. Cheap to clone.
#[derive(Clone, Default)]
pub struct ToolBox {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolBox {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, tool: impl Tool + 'static) -> Self {
        self.register(tool);
        self
    }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        let tool: Arc<dyn Tool> = Arc::new(tool);
        self.tools.insert(tool.spec().name, tool);
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub async fn call(&self, call: &ToolCall) -> Result<Value> {
        let tool = self
            .tools
            .get(&call.name)
            .ok_or_else(|| Error::ToolNotFound(call.name.clone()))?;
        tool.call(call.arguments.clone()).await
    }

    /// Runs a tool call and turns the outcome into a tool-result message.
    /// Failures become an `error: ...` message so the model can recover.
    pub async fn run(&self, call: &ToolCall) -> Message {
        let content = match self.call(call).await {
            Ok(Value::String(s)) => s,
            Ok(v) => v.to_string(),
            Err(e) => format!("error: {e}"),
        };
        Message::tool_result(call.id.clone(), content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn adder() -> FnTool<impl Fn(Value) -> Result<Value> + Send + Sync> {
        FnTool::new("add", "Adds a and b", json!({"type": "object"}), |args| {
            let a = args["a"].as_f64().ok_or(Error::Tool {
                name: "add".into(),
                message: "a".into(),
            })?;
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!(a + b))
        })
    }

    #[tokio::test]
    async fn runs_registered_tool() {
        let tb = ToolBox::new().with(adder());
        let call = ToolCall {
            id: "c1".into(),
            name: "add".into(),
            arguments: json!({"a": 2, "b": 3}),
        };
        let msg = tb.run(&call).await;
        assert_eq!(msg.content, "5.0");
        assert_eq!(msg.tool_call_id.as_deref(), Some("c1"));
    }

    #[tokio::test]
    async fn unknown_tool_becomes_error_message() {
        let tb = ToolBox::new();
        let call = ToolCall {
            id: "c1".into(),
            name: "nope".into(),
            arguments: json!({}),
        };
        assert!(tb
            .run(&call)
            .await
            .content
            .starts_with("error: tool `nope`"));
    }
}
