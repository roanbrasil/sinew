//! Graph diagrams: `view`, `to_mermaid`, `to_dot`.

use std::sync::Arc;

use sinew_core::{mock::ScriptedModel, ToolBox};
use sinew_graph::{
    prebuilt::ReactAgent, EdgeKind, GraphView, NodeKind, RetryPolicy, SendTo, StateGraph, END,
    START,
};
use sinew_trace::{MemorySink, RunKind, Tracer};

#[test]
fn react_agent_mermaid_snapshot() {
    let agent = ReactAgent::new(Arc::new(ScriptedModel::new([])), ToolBox::new())
        .compile()
        .unwrap()
        .interrupt_before(["tools"]);
    let m = agent.to_mermaid();
    let expected = r#"flowchart TD
    __start__(["START"]):::terminal
    __end__(["END"]):::terminal
    agent["agent"]
    tools{{"⏸ tools"}}:::paused
    __start__ --> agent
    agent -.-> tools
    agent -.-> __end__
    tools -. goto .-> agent
    tools -. goto .-> __end__
    classDef terminal fill:#eee,stroke:#999,color:#333
    classDef paused stroke:#d97706,stroke-width:2px
"#;
    assert_eq!(m, expected, "\n{m}");

    let v = agent.view();
    assert_eq!(v.nodes[1].kind, NodeKind::Command);
    assert!(v.nodes[1].interrupt_before);
    // Serializable for UIs, round trip.
    let json = serde_json::to_value(&v).unwrap();
    let back: GraphView = serde_json::from_value(json).unwrap();
    assert_eq!(back, v);
}

#[test]
fn fan_out_send_retry_and_subgraph() {
    let mut inner = StateGraph::<u32>::new();
    inner.add_node("step", |s: u32| async move { Ok(s + 1) });
    inner.set_entry_point("step");
    inner.add_edge("step", END);
    let inner = inner.compile().unwrap();

    let mut g = StateGraph::<u32>::new();
    g.add_node("plan", |s: u32| async move { Ok(s) });
    g.add_node("work \"A\"", |s: u32| async move { Ok(s) });
    g.add_subgraph("nested", inner);
    g.set_entry_point("plan");
    g.add_send_edges(
        "plan",
        |_: &u32| vec![SendTo::new("work \"A\"", 1)],
        ["work \"A\""],
    );
    g.add_edge("work \"A\"", "nested");
    g.add_edge("nested", END);
    g.with_reducer(|a: u32, b: u32| a + b);
    g.set_node_retry("work \"A\"", RetryPolicy::default());
    let graph = g.compile().unwrap();

    let v = graph.view();
    let names: Vec<&str> = v.nodes.iter().map(|n| n.name.as_str()).collect();
    assert_eq!(names, ["plan", "work \"A\"", "nested"], "definition order");
    assert!(v.nodes[1].retry);
    assert_eq!(v.nodes[2].kind, NodeKind::Subgraph);
    assert_eq!(v.nodes[2].subgraph.as_ref().unwrap().nodes[0].name, "step");
    assert!(v.edges.iter().any(|e| e.from == START && e.to == "plan"));
    assert!(v
        .edges
        .iter()
        .any(|e| e.from == "plan" && e.kind == EdgeKind::Send));

    let m = graph.to_mermaid();
    assert!(
        m.contains("work_20__22_A_22_[\"work #quot;A#quot;<br/><small>retry</small>\"]"),
        "{m}"
    );
    assert!(m.contains("plan -. send .-> work_20__22_A_22_"), "{m}");
    assert!(m.contains("subgraph nested[\"nested\"]"), "{m}");
    assert!(m.contains("nested__step[\"step\"]"), "{m}");
    assert!(m.contains("nested____start__ --> nested__step"), "{m}");

    let d = graph.to_dot();
    assert!(d.starts_with("digraph G {"));
    assert!(
        d.contains("\"work \\\"A\\\"\" [label=\"work \\\"A\\\"\"];"),
        "{d}"
    );
    assert!(d.contains("subgraph \"cluster_nested\""), "{d}");
    // Edges into a subgraph point at its inner start node.
    assert!(
        d.contains("\"work \\\"A\\\"\" -> \"nested/__start__\";"),
        "{d}"
    );
    assert!(d.contains("\"nested/__end__\" -> \"__end__\";"), "{d}");
    assert!(
        d.contains("\"plan\" -> \"work \\\"A\\\"\" [style=dashed, label=\"send\"];"),
        "{d}"
    );
}

#[tokio::test]
async fn root_trace_run_carries_the_graph_shape() {
    let mut g = StateGraph::<u32>::new();
    g.add_node("a", |s: u32| async move { Ok(s + 1) });
    g.set_entry_point("a");
    g.add_edge("a", END);
    let sink = Arc::new(MemorySink::new());
    let graph = g
        .compile()
        .unwrap()
        .with_tracer(Tracer::new().with_shared_sink(sink.clone()));
    graph.invoke(1).await.unwrap();
    let root = sink
        .runs()
        .into_iter()
        .find(|r| r.kind == RunKind::Graph)
        .unwrap();
    let view: GraphView = serde_json::from_value(root.metadata["graph"].clone()).unwrap();
    assert_eq!(view.nodes[0].name, "a");
}
