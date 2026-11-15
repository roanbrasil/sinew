use serde::{Deserialize, Serialize};
use sinew_core::Message;

use super::AgentState;
use crate::{CompiledGraph, GraphError, StateGraph, END, START};

/// State of a [`Swarm`]: the shared conversation and the agent that spoke
/// last (the next user turn starts there).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SwarmState {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub active_agent: Option<String>,
}

impl SwarmState {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            active_agent: None,
        }
    }
}

/// Agents that hand the conversation to each other (LangGraph's
/// `create_swarm`).
///
/// Each agent is typically a [`ReactAgent`](super::ReactAgent) with
/// [`Handoff`](super::Handoff) tools naming its peers. A run starts at the
/// `active_agent` (or the default agent), and a handoff moves the
/// conversation to the target agent with
/// [`Command::to_parent`](crate::Command::to_parent). An agent that answers
/// without handing off ends the turn; its name is kept in `active_agent`,
/// so pass the returned state back in (with the new user message) to
/// continue the conversation with the same agent.
#[derive(Default)]
pub struct Swarm {
    agents: Vec<(String, CompiledGraph<AgentState>)>,
    default_agent: Option<String>,
}

impl Swarm {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn agent(mut self, name: impl Into<String>, graph: CompiledGraph<AgentState>) -> Self {
        self.agents.push((name.into(), graph));
        self
    }

    /// Where a conversation without `active_agent` starts (default: the
    /// first agent added).
    pub fn default_agent(mut self, name: impl Into<String>) -> Self {
        self.default_agent = Some(name.into());
        self
    }

    /// The swarm graph, not yet compiled.
    pub fn graph(self) -> StateGraph<SwarmState> {
        let names: Vec<String> = self.agents.iter().map(|(n, _)| n.clone()).collect();
        let default = self
            .default_agent
            .or_else(|| names.first().cloned())
            .unwrap_or_default();
        let mut g = StateGraph::<SwarmState>::new();
        let known = names.clone();
        g.add_conditional_edges(
            START,
            move |s: &SwarmState| {
                s.active_agent
                    .clone()
                    .filter(|a| known.contains(a))
                    .unwrap_or_else(|| default.clone())
            },
            names.clone(),
        );
        for (name, graph) in self.agents {
            let me = name.clone();
            g.add_subgraph_mapped(
                name.clone(),
                graph,
                |s: &SwarmState| AgentState::new(s.messages.clone()),
                move |mut parent: SwarmState, child: AgentState| {
                    parent.messages = child.messages;
                    parent.active_agent = Some(me.clone());
                    parent
                },
            );
            g.set_node_destinations(name.clone(), names.iter().filter(|n| **n != name).cloned());
            g.add_edge(name, END);
        }
        g
    }

    pub fn compile(self) -> Result<CompiledGraph<SwarmState>, GraphError> {
        self.graph().compile()
    }
}

/// Builds a [`Swarm`]; the first agent is the default.
pub fn create_swarm(
    agents: Vec<(String, CompiledGraph<AgentState>)>,
) -> Result<CompiledGraph<SwarmState>, GraphError> {
    let mut s = Swarm::new();
    s.agents = agents;
    s.compile()
}
