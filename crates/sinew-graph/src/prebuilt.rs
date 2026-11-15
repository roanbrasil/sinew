//! Ready-made graphs.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sinew_core::{ChatModel, ChatRequest, Message, ToolBox};
use sinew_trace::{traced_chat, traced_tool};

use crate::{CompiledGraph, GraphError, StateGraph, END};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentState {
    pub messages: Vec<Message>,
}

impl AgentState {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }
    pub fn last(&self) -> Option<&Message> {
        self.messages.last()
    }
}

/// Classic tool-calling loop: `agent` calls the model; if it asked for tools,
/// `tools` runs them and control returns to `agent`; otherwise the run ends.
pub fn react_agent(
    model: Arc<dyn ChatModel>,
    tools: ToolBox,
) -> Result<CompiledGraph<AgentState>, GraphError> {
    let specs = tools.specs();
    let mut g = StateGraph::<AgentState>::new();

    g.add_node_with_context("agent", move |mut s: AgentState, ctx| {
        let model = model.clone();
        let req = ChatRequest::new(s.messages.clone()).with_tools(specs.clone());
        async move {
            let resp =
                traced_chat(model.as_ref(), req, ctx.tracer.as_ref(), ctx.run.as_ref()).await?;
            s.messages.push(resp.message);
            Ok(s)
        }
    });

    g.add_node_with_context("tools", move |mut s: AgentState, ctx| {
        let tools = tools.clone();
        async move {
            let calls = s.last().map(|m| m.tool_calls.clone()).unwrap_or_default();
            for call in &calls {
                let result = traced_tool(&tools, call, ctx.tracer.as_ref(), ctx.run.as_ref()).await;
                s.messages.push(result);
            }
            Ok(s)
        }
    });

    g.add_conditional_edges(
        "agent",
        |s: &AgentState| {
            if s.last().is_some_and(Message::has_tool_calls) {
                "tools".into()
            } else {
                END.into()
            }
        },
        ["tools", END],
    );
    g.add_edge("tools", "agent");
    g.set_entry_point("agent");
    g.compile()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use sinew_core::{mock::ScriptedModel, FnTool, Role, ToolCall};

    #[tokio::test]
    async fn loops_through_tools_and_answers() {
        let model = Arc::new(ScriptedModel::new([
            Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "c1".into(),
                    name: "add".into(),
                    arguments: json!({"a": 2, "b": 40}),
                }],
            ),
            Message::assistant("The answer is 42."),
        ]));
        let tools = ToolBox::new().with(FnTool::new(
            "add",
            "add numbers",
            json!({"type": "object"}),
            |a: Value| {
                Ok(json!(
                    a["a"].as_i64().unwrap_or(0) + a["b"].as_i64().unwrap_or(0)
                ))
            },
        ));
        let graph = react_agent(model.clone(), tools).unwrap();
        let out = graph
            .invoke(AgentState::new(vec![Message::user("2 + 40?")]))
            .await
            .unwrap();

        let roles: Vec<Role> = out.messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![Role::User, Role::Assistant, Role::Tool, Role::Assistant]
        );
        assert_eq!(out.messages[2].content, "42");
        assert_eq!(out.last().unwrap().content, "The answer is 42.");
        // The second model call saw the tool result.
        assert_eq!(model.requests()[1].messages.len(), 3);
        assert_eq!(model.requests()[0].tools[0].name, "add");
    }
}
