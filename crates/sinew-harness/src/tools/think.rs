//! `think`: a scratchpad. The thought is recorded in the conversation (as
//! the tool call's arguments) and nothing else happens, which gives the
//! model a place to reason between tool calls.

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{Tool, ToolSpec};

use super::arg_str;

pub struct ThinkTool;

#[async_trait]
impl Tool for ThinkTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "think".into(),
            description: "Think through a problem step by step: plan, check results so far, or decide what to do next. It does not change anything or fetch new information.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "Your reasoning"}
                },
                "required": ["thought"]
            }),
        }
    }

    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        arg_str("think", &args, "thought")?;
        Ok(json!("Thought recorded."))
    }
}
