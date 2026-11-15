use sinew_core::ToolSpec;

use super::empty_tool;

/// A tool that transfers control to another agent (`transfer_to_<agent>`).
///
/// Give it to an agent with [`ReactAgent::handoff`](super::ReactAgent::handoff):
/// when the model calls it, the agent answers the call, leaves its graph and
/// its parent continues at `agent` (a [`Command::to_parent`](crate::Command::to_parent)
/// under the hood).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handoff {
    /// Node name of the target agent in the parent graph.
    pub agent: String,
    pub tool_name: String,
    pub description: String,
}

impl Handoff {
    /// A handoff to `agent`, described to the model by `description`.
    pub fn to(agent: impl Into<String>, description: impl Into<String>) -> Self {
        let agent = agent.into();
        Self {
            tool_name: format!("transfer_to_{agent}"),
            agent,
            description: description.into(),
        }
    }

    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }

    /// The tool as presented to the model.
    pub fn spec(&self) -> ToolSpec {
        empty_tool(self.tool_name.clone(), self.description.clone())
    }
}

/// Shorthand for [`Handoff::to`] (LangGraph's `create_handoff_tool`).
pub fn create_handoff_tool(agent: impl Into<String>, description: impl Into<String>) -> Handoff {
    Handoff::to(agent, description)
}
