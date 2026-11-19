//! Built-in tools. Each is opt-in on the [`AgentBuilder`](crate::AgentBuilder).

pub mod fs;
pub mod http;
pub mod shell;
pub mod skills;
pub mod task;
pub mod think;
pub mod todo;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use sinew_core::{Error, Tool, ToolBox, ToolCall, ToolSpec};

pub use fs::{fs_tools, Workspace};
pub use http::{HttpFetchTool, HttpOptions};
pub use shell::{ShellOptions, ShellTool};
pub use skills::{load_skills, LoadSkillTool, Skill};
pub use task::{SubagentSpec, SubagentTools, TaskTool};
pub use think::ThinkTool;
pub use todo::WriteTodosTool;

pub(crate) fn tool_err(name: &str, message: impl std::fmt::Display) -> Error {
    Error::Tool {
        name: name.into(),
        message: message.to_string(),
    }
}

pub(crate) fn arg_str<'a>(tool: &str, args: &'a Value, key: &str) -> sinew_core::Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| tool_err(tool, format!("`{key}` (string) is required")))
}

pub(crate) fn opt_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

pub(crate) fn arg_u64(args: &Value, key: &str) -> Option<u64> {
    let v = args.get(key)?;
    v.as_u64()
        .or_else(|| v.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64))
        .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
}

pub(crate) fn arg_bool(args: &Value, key: &str) -> Option<bool> {
    let v = args.get(key)?;
    v.as_bool().or_else(|| match v.as_str()? {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    })
}

/// Keeps the head and tail of a long text.
pub(crate) fn truncate_middle(text: &str, max_chars: usize) -> String {
    let n = text.chars().count();
    if n <= max_chars {
        return text.to_string();
    }
    let head = max_chars * 2 / 3;
    let tail = max_chars - head;
    let h: String = text.chars().take(head).collect();
    let t: String = text.chars().skip(n - tail).collect();
    format!("{h}\n… [{} characters omitted] …\n{t}", n - head - tail)
}

/// Exposes one tool of a [`ToolBox`] as a standalone [`Tool`], so tool boxes
/// (e.g. from `McpClient::toolbox`) can be handed to the harness.
pub struct ToolBoxEntry {
    toolbox: ToolBox,
    spec: ToolSpec,
}

/// Every tool of a [`ToolBox`] as individual [`Tool`]s.
pub fn from_toolbox(toolbox: &ToolBox) -> Vec<Arc<dyn Tool>> {
    toolbox
        .specs()
        .into_iter()
        .map(|spec| {
            Arc::new(ToolBoxEntry {
                toolbox: toolbox.clone(),
                spec,
            }) as Arc<dyn Tool>
        })
        .collect()
}

#[async_trait]
impl Tool for ToolBoxEntry {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value) -> sinew_core::Result<Value> {
        self.toolbox
            .call(&ToolCall {
                id: String::new(),
                name: self.spec.name.clone(),
                arguments,
            })
            .await
    }

    fn source(&self) -> Option<sinew_core::ToolSource> {
        self.toolbox.get(&self.spec.name).and_then(|t| t.source())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_in_the_middle() {
        let s: String = (0..100)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let t = truncate_middle(&s, 30);
        assert!(t.starts_with(&s[..20]));
        assert!(t.ends_with(&s[90..]));
        assert!(t.contains("70 characters omitted"));
        assert_eq!(truncate_middle("short", 30), "short");
    }
}
