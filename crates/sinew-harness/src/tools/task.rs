//! `task`: delegate work to a subagent with its own instructions, tools and
//! context window. Only the subagent's final answer comes back, so long
//! searches or file reads don't fill the parent's context.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{ChatModel, Tool, ToolSpec};

use super::{arg_str, opt_str, tool_err};
use crate::agent::SubagentContext;

pub const NAME: &str = "task";

/// Which tools a subagent gets.
#[derive(Clone, Default)]
pub enum SubagentTools {
    /// The parent's tools (minus `task` once the depth limit is reached).
    #[default]
    Inherit,
    /// Only the parent tools with these names.
    Only(Vec<String>),
    /// No inherited tools (only [`SubagentSpec::tool`] extras).
    None,
}

/// A subagent the `task` tool can start.
#[derive(Clone)]
pub struct SubagentSpec {
    pub name: String,
    /// Shown to the parent model so it knows when to delegate.
    pub description: String,
    /// The subagent's system prompt.
    pub instructions: String,
    pub tools: SubagentTools,
    pub extra_tools: Vec<Arc<dyn Tool>>,
    /// Defaults to the parent's model.
    pub model: Option<Arc<dyn ChatModel>>,
    /// Defaults to the parent's budget.
    pub max_steps: Option<u32>,
}

impl SubagentSpec {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        instructions: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            instructions: instructions.into(),
            tools: SubagentTools::Inherit,
            extra_tools: Vec::new(),
            model: None,
            max_steps: None,
        }
    }

    /// The built-in general-purpose subagent: the parent's tools, generic
    /// instructions.
    pub fn general_purpose() -> Self {
        Self::new(
            "general-purpose",
            "General agent with the same tools as you. Use it for research, searching files and multi-step subtasks whose details you don't need in your own context.",
            "You are a subagent. Complete the task you are given using your tools, then reply with a concise, self-contained final report: the answer, key facts and file paths, and anything left undone. Your reply is all the caller sees.",
        )
    }

    /// Restricts the inherited tools to these names.
    pub fn only_tools<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tools = SubagentTools::Only(names.into_iter().map(Into::into).collect());
        self
    }

    /// No inherited tools.
    pub fn no_inherited_tools(mut self) -> Self {
        self.tools = SubagentTools::None;
        self
    }

    /// Adds a tool only this subagent has.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.extra_tools.push(Arc::new(tool));
        self
    }

    pub fn model(mut self, model: Arc<dyn ChatModel>) -> Self {
        self.model = Some(model);
        self
    }

    pub fn max_steps(mut self, n: u32) -> Self {
        self.max_steps = Some(n);
        self
    }
}

/// The `task` tool. Built by the [`AgentBuilder`](crate::AgentBuilder) when
/// subagents are configured and the depth limit allows it.
pub struct TaskTool {
    pub(crate) ctx: Arc<SubagentContext>,
}

#[async_trait]
impl Tool for TaskTool {
    fn spec(&self) -> ToolSpec {
        let names: Vec<&str> = self.ctx.specs.iter().map(|s| s.name.as_str()).collect();
        let list: Vec<String> = self
            .ctx
            .specs
            .iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect();
        ToolSpec {
            name: NAME.into(),
            description: format!(
                "Delegate a self-contained task to a subagent that works in its own context and returns only its final answer. Give it everything it needs in `prompt`; it cannot see this conversation. Several task calls in one reply run in parallel. Available subagents:\n{}",
                list.join("\n")
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Short (3-5 word) label for the task"},
                    "prompt": {"type": "string", "description": "Complete instructions for the subagent"},
                    "subagent_type": {"type": "string", "enum": names, "description": "Which subagent to use"}
                },
                "required": ["prompt"]
            }),
        }
    }

    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        let prompt = arg_str(NAME, &args, "prompt")?;
        let wanted = opt_str(&args, "subagent_type");
        let spec = match wanted {
            Some(w) => self.ctx.specs.iter().find(|s| s.name == w),
            None => self.ctx.specs.first(),
        }
        .ok_or_else(|| {
            tool_err(
                NAME,
                format!("unknown subagent_type `{}`", wanted.unwrap_or_default()),
            )
        })?;
        let agent = crate::agent::build_subagent(&self.ctx, spec).map_err(|e| tool_err(NAME, e))?;
        let result = agent.ask(prompt).await.map_err(|e| tool_err(NAME, e))?;
        let text = if result.text.trim().is_empty() {
            "(the subagent returned no answer)".to_string()
        } else {
            result.text
        };
        Ok(Value::String(match result.stop {
            Some(stop) => format!("[subagent stopped early: {stop}]\n{text}"),
            None => text,
        }))
    }
}
