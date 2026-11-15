//! Prebuilt multi-agent graphs and ReactAgent options.

use std::sync::Arc;

use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, FnTool, Message, Role, StructuredOutput, ToolBox, ToolCall};
use sinew_graph::{
    prebuilt::{
        AgentState, Handoff, ReactAgent, Supervisor, Swarm, SwarmState, MAX_ITERATIONS_REPLY,
    },
    MemoryCheckpointer, StateGraph, END,
};

fn call(id: &str, name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args,
    }
}

fn add_tool() -> ToolBox {
    ToolBox::new().with(FnTool::new(
        "add",
        "add numbers",
        json!({"type": "object"}),
        |a: Value| {
            Ok(json!(
                a["a"].as_i64().unwrap_or(0) + a["b"].as_i64().unwrap_or(0)
            ))
        },
    ))
}

#[tokio::test]
async fn supervisor_routes_to_workers_and_answers() {
    let sup_model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools("", vec![call("h1", "transfer_to_math", json!({}))]),
        Message::assistant("The answer is 42."),
    ]));
    let math_model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools("", vec![call("c1", "add", json!({"a": 2, "b": 40}))]),
        Message::assistant("2 + 40 = 42"),
    ]));
    let writer_model = Arc::new(ScriptedModel::new(Vec::<Message>::new()));
    let math = ReactAgent::new(math_model, add_tool()).compile().unwrap();
    let writer = ReactAgent::new(writer_model, ToolBox::new())
        .compile()
        .unwrap();
    let graph = Supervisor::new(sup_model.clone())
        .agent("math", "does arithmetic", math)
        .agent("writer", "writes prose", writer)
        .compile()
        .unwrap();

    let out = graph
        .invoke(AgentState::new(vec![Message::user("what is 2 + 40?")]))
        .await
        .unwrap();
    let contents: Vec<&str> = out.messages.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(out.last().unwrap().content, "The answer is 42.");
    assert!(contents.contains(&"Successfully transferred to math"));
    assert!(contents.contains(&"2 + 40 = 42"));
    assert!(contents.contains(&"Successfully transferred back to supervisor"));
    // The supervisor saw both handoff tools and, on its second call, the
    // worker's result.
    let reqs = sup_model.requests();
    let tools: Vec<&str> = reqs[0].tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tools, vec!["transfer_to_math", "transfer_to_writer"]);
    assert_eq!(reqs[0].messages[0].role, Role::System);
    assert!(reqs[1].messages.iter().any(|m| m.content == "2 + 40 = 42"));
}

#[tokio::test]
async fn supervisor_last_message_mode_keeps_history_short() {
    let sup_model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools("", vec![call("h1", "transfer_to_math", json!({}))]),
        Message::assistant("done"),
    ]));
    let math_model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools("", vec![call("c1", "add", json!({"a": 1, "b": 1}))]),
        Message::assistant("2"),
    ]));
    let graph = Supervisor::new(sup_model)
        .agent(
            "math",
            "math",
            ReactAgent::new(math_model, add_tool()).compile().unwrap(),
        )
        .output_mode(sinew_graph::prebuilt::OutputMode::LastMessage)
        .compile()
        .unwrap();
    let out = graph
        .invoke(AgentState::new(vec![Message::user("1+1")]))
        .await
        .unwrap();
    // No `add` tool call or result from the worker in the shared history.
    assert!(!out
        .messages
        .iter()
        .any(|m| m.tool_calls.iter().any(|c| c.name == "add")));
    assert!(out.messages.iter().any(|m| m.content == "2"));
}

#[tokio::test]
async fn swarm_hands_off_and_remembers_the_active_agent() {
    let alice_model = Arc::new(ScriptedModel::new([Message::assistant_with_tools(
        "",
        vec![call("h1", "transfer_to_bob", json!({}))],
    )]));
    let bob_model = Arc::new(ScriptedModel::new([
        Message::assistant("Bob here, how can I help?"),
        Message::assistant("Still Bob."),
    ]));
    let alice = ReactAgent::new(alice_model, ToolBox::new())
        .handoff(Handoff::to("bob", "Transfer to Bob for billing"))
        .compile()
        .unwrap();
    let bob = ReactAgent::new(bob_model.clone(), ToolBox::new())
        .handoff(Handoff::to("alice", "Transfer to Alice for anything else"))
        .compile()
        .unwrap();
    let swarm = Swarm::new()
        .agent("alice", alice)
        .agent("bob", bob)
        .default_agent("alice")
        .compile()
        .unwrap()
        .with_checkpointer(MemoryCheckpointer::new());

    let out = swarm
        .run(
            "t",
            Some(SwarmState::new(vec![Message::user("billing question")])),
        )
        .await
        .unwrap()
        .into_state();
    assert_eq!(out.active_agent.as_deref(), Some("bob"));
    assert_eq!(
        out.messages.last().unwrap().content,
        "Bob here, how can I help?"
    );
    assert!(out
        .messages
        .iter()
        .any(|m| m.content == "Successfully transferred to bob"));
    // Bob saw Alice's handoff in the conversation.
    assert!(bob_model.requests()[0]
        .messages
        .iter()
        .any(|m| m.content == "Successfully transferred to bob"));

    // Next turn starts with Bob directly.
    let mut next = out.clone();
    next.messages.push(Message::user("and another thing"));
    let out = swarm.run("t", Some(next)).await.unwrap().into_state();
    assert_eq!(out.messages.last().unwrap().content, "Still Bob.");
}

#[tokio::test]
async fn react_agent_hooks_iterations_and_structured_response() {
    let model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools("", vec![call("c1", "add", json!({"a": 1, "b": 2}))]),
        Message::assistant_with_tools("", vec![call("c2", "add", json!({"a": 3, "b": 4}))]),
        // Structured response (tool mode: the model calls the schema tool).
        Message::assistant_with_tools("", vec![call("s1", "answer", json!({"value": 3}))]),
    ]));
    let schema = json!({
        "type": "object",
        "properties": { "value": { "type": "integer" } },
        "required": ["value"]
    });
    let agent = ReactAgent::new(model.clone(), add_tool())
        .system_prompt("You add numbers.")
        .pre_model_hook(|s: &AgentState| {
            // Only the last 3 messages go to the model.
            let n = s.messages.len();
            s.messages[n.saturating_sub(3)..].to_vec()
        })
        .post_model_hook(|mut s: AgentState, _ctx| async move {
            // Guardrail: tag every model reply.
            if let Some(m) = s.messages.last_mut() {
                m.content.push_str("[checked]");
            }
            Ok(s)
        })
        .max_iterations(2)
        .response_format(StructuredOutput::new("answer", schema))
        .compile()
        .unwrap();
    let out = agent
        .invoke(AgentState::new(vec![
            Message::user("old 1"),
            Message::user("old 2"),
            Message::user("1+2, then 3+4"),
        ]))
        .await
        .unwrap();
    // Two model calls, then the iteration limit answered instead.
    let last_ai = out
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant)
        .unwrap();
    assert!(last_ai.content.starts_with(MAX_ITERATIONS_REPLY));
    assert_eq!(out.structured_response, Some(json!({"value": 3})));
    let reqs = model.requests();
    assert_eq!(reqs[0].messages[0].role, Role::System);
    assert_eq!(reqs[0].messages.len(), 4); // system + last 3
    assert!(out
        .messages
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .all(|m| m.content.ends_with("[checked]")));
}

#[tokio::test]
async fn handoff_outside_a_parent_graph_is_an_error() {
    let model = Arc::new(ScriptedModel::new([Message::assistant_with_tools(
        "",
        vec![call("h", "transfer_to_x", json!({}))],
    )]));
    let agent = ReactAgent::new(model, ToolBox::new())
        .handoff(Handoff::to("x", "x"))
        .compile()
        .unwrap();
    let err = agent
        .invoke(AgentState::new(vec![Message::user("hi")]))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("parent graph"), "{err}");
}

#[tokio::test]
async fn agent_as_subgraph_in_a_custom_graph() {
    let model = Arc::new(ScriptedModel::new([Message::assistant("hello")]));
    let agent = ReactAgent::new(model, ToolBox::new()).compile().unwrap();
    let mut g = StateGraph::<AgentState>::new();
    g.add_subgraph("assistant", agent);
    g.add_node("audit", |mut s: AgentState| async move {
        s.messages.push(Message::assistant("audited"));
        Ok(s)
    });
    g.add_edge("assistant", "audit");
    g.add_edge("audit", END);
    g.set_entry_point("assistant");
    let out = g
        .compile()
        .unwrap()
        .invoke(AgentState::new(vec![Message::user("hi")]))
        .await
        .unwrap();
    let c: Vec<&str> = out.messages.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(c, vec!["hi", "hello", "audited"]);
}
