//! `write_todos`: the agent's plan, kept in [`HarnessState::todos`](crate::HarnessState).
//!
//! The tool itself only carries the spec: the harness intercepts its calls
//! and updates the thread state, so the plan is checkpointed with the rest
//! of the conversation.

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{Tool, ToolSpec};

use super::tool_err;
use crate::state::{Todo, TodoStatus};

pub const NAME: &str = "write_todos";

/// Spec-only tool; calls are handled by the harness.
pub struct WriteTodosTool;

#[async_trait]
impl Tool for WriteTodosTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: NAME.into(),
            description: "Create or update your task plan. Send the complete list every time (it replaces the previous one). Use it for multi-step tasks: mark one item in_progress while working on it and completed as soon as it is done.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }),
        }
    }

    async fn call(&self, _args: Value) -> sinew_core::Result<Value> {
        Err(tool_err(
            NAME,
            "write_todos must run inside a harness Agent",
        ))
    }
}

/// Parses `write_todos` arguments (lenient about status spelling and about
/// plain strings as items).
pub fn parse_todos(args: &Value) -> Result<Vec<Todo>, String> {
    let mut items = args.get("todos").cloned().ok_or("`todos` is required")?;
    if let Value::String(s) = &items {
        items = serde_json::from_str(s).map_err(|e| format!("`todos` is not a list: {e}"))?;
    }
    let list = items.as_array().ok_or("`todos` must be an array")?;
    list.iter()
        .map(|item| {
            if let Some(s) = item.as_str() {
                return Ok(Todo {
                    content: s.to_string(),
                    status: TodoStatus::Pending,
                });
            }
            let content = item
                .get("content")
                .or_else(|| item.get("task"))
                .and_then(Value::as_str)
                .ok_or("every todo needs `content`")?
                .to_string();
            let status = match item
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("pending")
                .to_ascii_lowercase()
                .replace(['-', ' '], "_")
                .as_str()
            {
                "in_progress" | "doing" | "active" => TodoStatus::InProgress,
                "completed" | "done" | "complete" => TodoStatus::Completed,
                _ => TodoStatus::Pending,
            };
            Ok(Todo { content, status })
        })
        .collect()
}

/// Renders the plan as a checklist.
pub fn render_todos(todos: &[Todo]) -> String {
    if todos.is_empty() {
        return "(no todos)".into();
    }
    todos
        .iter()
        .map(|t| {
            let mark = match t.status {
                TodoStatus::Pending => "[ ]",
                TodoStatus::InProgress => "[~]",
                TodoStatus::Completed => "[x]",
            };
            format!("{mark} {}", t.content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_leniently() {
        let t = parse_todos(&json!({"todos": [
            {"content": "a", "status": "in-progress"},
            {"content": "b", "status": "done"},
            "c"
        ]}))
        .unwrap();
        assert_eq!(render_todos(&t), "[~] a\n[x] b\n[ ] c");
        let t = parse_todos(&json!({"todos": "[{\"content\": \"x\", \"status\": \"pending\"}]"}))
            .unwrap();
        assert_eq!(t.len(), 1);
        assert!(parse_todos(&json!({})).is_err());
    }
}
