//! A whole ReAct agent recorded once and replayed offline from a cassette.

use std::sync::Arc;

use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, vcr::Vcr, FnTool, Message, ToolBox, ToolCall};
use sinew_graph::prebuilt::{react_agent, AgentState};

fn tools(live: bool) -> ToolBox {
    ToolBox::new().with(FnTool::new(
        "weather",
        "weather in a city",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        move |a: Value| {
            assert!(live, "tools must not run during replay");
            Ok(json!({"city": a["city"], "celsius": -3}))
        },
    ))
}

#[tokio::test]
async fn agent_replays_from_cassette() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("agent.json");
    let input = || AgentState::new(vec![Message::user("Weather in Montreal?")]);

    let vcr = Vcr::record(&path);
    let live_model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "weather".into(),
                arguments: json!({"city": "Montreal"}),
            }],
        ),
        Message::assistant("It is -3 °C in Montreal."),
    ]));
    let agent = react_agent(vcr.model(live_model), vcr.toolbox(&tools(true))).unwrap();
    let recorded = agent.invoke(input()).await.unwrap();
    vcr.save().unwrap();

    let vcr = Vcr::replay(&path).unwrap();
    let agent = react_agent(
        vcr.model(Arc::new(ScriptedModel::new([]))),
        vcr.toolbox(&tools(false)),
    )
    .unwrap();
    let replayed = agent.invoke(input()).await.unwrap();
    assert_eq!(replayed, recorded);
    vcr.assert_all_played().unwrap();

    // Changing the prompt is caught instead of answered with a stale response.
    let vcr = Vcr::replay(&path).unwrap();
    let agent = react_agent(
        vcr.model(Arc::new(ScriptedModel::new([]))),
        vcr.toolbox(&tools(false)),
    )
    .unwrap();
    let err = agent
        .invoke(AgentState::new(vec![Message::user("Weather in Quebec?")]))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("differs from the recording"),
        "{err}"
    );
}
