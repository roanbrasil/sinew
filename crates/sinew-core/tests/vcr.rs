//! Record/replay cassettes.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use futures_util::StreamExt;
use serde_json::{json, Value};
use sinew_core::{
    collect_stream,
    mock::ScriptedModel,
    vcr::{Cassette, MatchRequests, Vcr, VcrMode},
    ChatModel, ChatRequest, Error, FnTool, Message, ToolBox, ToolCall, ToolSpec,
};

fn empty() -> Arc<dyn ChatModel> {
    Arc::new(ScriptedModel::new([]))
}

fn req(text: &str) -> ChatRequest {
    ChatRequest::new(vec![Message::user(text)])
}

#[tokio::test]
async fn chat_round_trip_and_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c/chat.json");

    let vcr = Vcr::record(&path);
    let m = vcr.model(Arc::new(ScriptedModel::new([
        Message::assistant("one"),
        Message::assistant("two"),
    ])));
    assert_eq!(m.chat(req("a")).await.unwrap().message.content, "one");
    assert_eq!(m.chat(req("b")).await.unwrap().message.content, "two");
    vcr.save().unwrap();
    assert_eq!(Cassette::load(&path).unwrap().interactions.len(), 2);

    let vcr = Vcr::replay(&path).unwrap();
    assert_eq!(vcr.mode(), VcrMode::Replay);
    let m = vcr.model(empty());
    assert_eq!(m.chat(req("a")).await.unwrap().message.content, "one");
    assert!(vcr.assert_all_played().is_err());
    let err = m.chat(req("changed")).await.unwrap_err().to_string();
    assert!(
        err.contains("model call #2 differs") && err.contains("message 0 changed"),
        "{err}"
    );
    assert_eq!(m.chat(req("b")).await.unwrap().message.content, "two");
    vcr.assert_all_played().unwrap();
    let err = m.chat(req("c")).await.unwrap_err().to_string();
    assert!(err.contains("no recorded model call #3"), "{err}");
}

#[tokio::test]
async fn matching_modes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.json");
    let vcr = Vcr::record(&path);
    vcr.model(Arc::new(ScriptedModel::new([Message::assistant("x")])))
        .chat(req("a"))
        .await
        .unwrap();
    vcr.save().unwrap();

    let tool = ToolSpec {
        name: "t".into(),
        description: "d".into(),
        parameters: json!({"type": "object"}),
    };
    let with_tool = req("a").with_tools(vec![tool]);
    let exact = Vcr::replay(&path).unwrap();
    let e = exact
        .model(empty())
        .chat(with_tool.clone())
        .await
        .unwrap_err();
    assert!(e.to_string().contains("`tools` changed"), "{e}");

    let lenient = Vcr::replay(&path)
        .unwrap()
        .matching(MatchRequests::Messages);
    assert!(lenient.model(empty()).chat(with_tool).await.is_ok());

    let seq = Vcr::replay(&path)
        .unwrap()
        .matching(MatchRequests::Sequential);
    assert!(seq.model(empty()).chat(req("anything")).await.is_ok());
}

#[tokio::test]
async fn streams_are_recorded_event_by_event() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.json");
    let vcr = Vcr::record(&path);
    let m = vcr.model(Arc::new(ScriptedModel::new([Message::assistant(
        "hello big world",
    )])));
    let live: Vec<_> = m.chat_stream(req("a")).await.unwrap().collect().await;
    vcr.save().unwrap();
    let c = Cassette::load(&path).unwrap();
    let rec = &c.interactions[0];
    assert_eq!(rec.stream.as_ref().unwrap().len(), live.len());
    assert_eq!(
        rec.response.as_ref().unwrap().message.content,
        "hello big world"
    );

    // Replayed as a stream: same events. As a chat: folded response.
    let vcr = Vcr::replay(&path).unwrap();
    let replayed: Vec<_> = vcr
        .model(empty())
        .chat_stream(req("a"))
        .await
        .unwrap()
        .collect()
        .await;
    assert_eq!(
        format!(
            "{:?}",
            replayed
                .iter()
                .map(|e| e.as_ref().unwrap())
                .collect::<Vec<_>>()
        ),
        format!(
            "{:?}",
            live.iter().map(|e| e.as_ref().unwrap()).collect::<Vec<_>>()
        )
    );
    let vcr = Vcr::replay(&path).unwrap();
    let r = vcr.model(empty()).chat(req("a")).await.unwrap();
    assert_eq!(r.message.content, "hello big world");

    // A chat recording replays as a stream too.
    let path2 = dir.path().join("s2.json");
    let vcr = Vcr::record(&path2);
    vcr.model(Arc::new(ScriptedModel::new([Message::assistant("hi")])))
        .chat(req("a"))
        .await
        .unwrap();
    vcr.save().unwrap();
    let vcr = Vcr::replay(&path2).unwrap();
    let s = vcr.model(empty()).chat_stream(req("a")).await.unwrap();
    assert_eq!(collect_stream(s).await.unwrap().message.content, "hi");
}

#[tokio::test]
async fn errors_are_recorded_and_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("e.json");
    let vcr = Vcr::record(&path);
    let m = vcr.model(empty()); // no responses: ScriptExhausted
    assert!(m.chat(req("a")).await.is_err());
    vcr.save().unwrap();
    let vcr = Vcr::replay(&path).unwrap();
    let e = vcr.model(empty()).chat(req("a")).await.unwrap_err();
    assert!(e.to_string().contains("no responses left"), "{e}");
}

fn counting_tool(
    n: Arc<AtomicUsize>,
) -> FnTool<impl Fn(Value) -> sinew_core::Result<Value> + Send + Sync> {
    FnTool::new(
        "double",
        "doubles x",
        json!({"type": "object"}),
        move |a: Value| {
            n.fetch_add(1, Ordering::SeqCst);
            match a["x"].as_i64() {
                Some(x) => Ok(json!(x * 2)),
                None => Err(Error::Tool {
                    name: "double".into(),
                    message: "x missing".into(),
                }),
            }
        },
    )
}

#[tokio::test]
async fn tools_round_trip_in_any_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.json");
    let n = Arc::new(AtomicUsize::new(0));
    let vcr = Vcr::record(&path);
    let tb = vcr.toolbox(&ToolBox::new().with(counting_tool(n.clone())));
    let call = |id: &str, args: Value| ToolCall {
        id: id.into(),
        name: "double".into(),
        arguments: args,
    };
    assert_eq!(
        tb.call(&call("1", json!({"x": 1}))).await.unwrap(),
        json!(2)
    );
    assert_eq!(
        tb.call(&call("2", json!({"x": 5}))).await.unwrap(),
        json!(10)
    );
    assert!(tb.call(&call("3", json!({}))).await.is_err());
    vcr.save().unwrap();
    assert_eq!(n.load(Ordering::SeqCst), 3);

    let n2 = Arc::new(AtomicUsize::new(0));
    let vcr = Vcr::replay(&path).unwrap();
    let tb = vcr.toolbox(&ToolBox::new().with(counting_tool(n2.clone())));
    // Out of order (parallel tool execution) still matches.
    assert_eq!(
        tb.call(&call("2", json!({"x": 5}))).await.unwrap(),
        json!(10)
    );
    assert_eq!(
        tb.call(&call("1", json!({"x": 1}))).await.unwrap(),
        json!(2)
    );
    let e = tb.call(&call("3", json!({}))).await.unwrap_err();
    assert_eq!(e.to_string(), "tool `double` failed: x missing");
    assert_eq!(n2.load(Ordering::SeqCst), 0, "replay never runs the tool");
    vcr.assert_all_played().unwrap();
    let e = tb.call(&call("4", json!({"x": 9}))).await.unwrap_err();
    assert!(e.to_string().contains("no recorded call"), "{e}");
    assert_eq!(vcr.tool(Arc::new(counting_tool(n2))).spec().name, "double");
}

#[tokio::test]
async fn from_env_auto_records_then_replays() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("auto.json");
    let vcr = Vcr::from_env(&path).unwrap();
    assert!(vcr.is_recording());
    vcr.model(Arc::new(ScriptedModel::new([Message::assistant("x")])))
        .chat(req("a"))
        .await
        .unwrap();
    vcr.save().unwrap();
    let vcr = Vcr::from_env(&path).unwrap();
    assert_eq!(vcr.mode(), VcrMode::Replay);
    assert!(Vcr::replay(dir.path().join("missing.json")).is_err());
}
