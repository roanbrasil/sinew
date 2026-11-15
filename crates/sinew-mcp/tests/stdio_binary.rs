//! Spawns the `sinew-mcp` binary and drives it over stdio, both with this
//! crate's client and with raw JSON-RPC lines.

use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
};

use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, FnTool, Message, ToolBox, ToolCall};
use sinew_graph::prebuilt::{react_agent, AgentState};
use sinew_mcp::{protocol::error_code, McpClient, McpError};
use sinew_trace::{JsonlSink, Tracer};

const BIN: &str = env!("CARGO_BIN_EXE_sinew-mcp");

/// Writes a real trace file by running a traced agent whose tool fails once.
async fn write_traces(name: &str) -> (PathBuf, String) {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.jsonl"));
    let _ = std::fs::remove_file(&path);
    let model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "",
            vec![
                ToolCall {
                    id: "c1".into(),
                    name: "divide".into(),
                    arguments: json!({"a": 1, "b": 0}),
                },
                ToolCall {
                    id: "c2".into(),
                    name: "divide".into(),
                    arguments: json!({"a": 9, "b": 3}),
                },
            ],
        ),
        Message::assistant("9 / 3 = 3; dividing by zero is undefined."),
    ]));
    let tools = ToolBox::new().with(FnTool::new(
        "divide",
        "a / b",
        json!({"type": "object"}),
        |a: Value| {
            let b = a["b"].as_f64().unwrap_or(0.0);
            if b == 0.0 {
                return Err(sinew_core::Error::Tool {
                    name: "divide".into(),
                    message: "division by zero".into(),
                });
            }
            Ok(json!(a["a"].as_f64().unwrap_or(0.0) / b))
        },
    ));
    let tracer = Tracer::new().with_sink(JsonlSink::open(&path).unwrap());
    let agent = react_agent(model, tools).unwrap().with_tracer(tracer);
    agent
        .invoke(AgentState::new(vec![Message::user("1/0 and 9/3?")]))
        .await
        .unwrap();
    let first: Value = serde_json::from_str(
        std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    (path, first["trace_id"].as_str().unwrap().to_string())
}

#[tokio::test]
async fn client_drives_trace_server() {
    let (path, trace_id) = write_traces("client_drives").await;
    let client = McpClient::connect_stdio(BIN, ["--traces".as_ref(), path.as_os_str()])
        .await
        .unwrap();
    let info = client.server_info();
    assert_eq!(info.server_info.name, "sinew-mcp");
    assert_eq!(info.protocol_version, "2025-06-18");
    assert!(info
        .instructions
        .as_deref()
        .unwrap()
        .contains("list_traces"));
    client.ping().await.unwrap();

    // tools/list
    let mut names: Vec<String> = client
        .list_tools()
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    names.sort();
    assert_eq!(names, ["get_trace", "list_traces", "search_runs"]);

    // tools/call: list, get, search
    let list = client.call_tool("list_traces", json!({})).await.unwrap();
    assert!(!list.is_error);
    let text = list.text();
    assert!(text.starts_with("1 traces in"), "{text}");
    assert!(
        text.contains(&trace_id[..8])
            && text.contains("1 errors: tool `divide` failed: division by zero")
    );

    let tree = client
        .call_tool("get_trace", json!({"id": &trace_id[..6], "io": true}))
        .await
        .unwrap()
        .text();
    assert!(tree.contains(&format!("trace {trace_id}")), "{tree}");
    assert!(tree.contains("├─ node tools"), "{tree}");
    assert!(
        tree.contains("tool divide")
            && tree.contains("ERROR: tool `divide` failed: division by zero")
    );
    assert!(tree.contains("in : {\"a\":9,\"b\":3}"), "{tree}");

    let failing = client
        .call_tool("search_runs", json!({"kind": "tool", "error": true}))
        .await
        .unwrap()
        .text();
    assert!(failing.starts_with("1 matching runs"), "{failing}");
    let all_tools = client
        .call_tool("search_runs", json!({"name": "DIV"}))
        .await
        .unwrap()
        .text();
    assert!(all_tools.starts_with("2 matching runs"), "{all_tools}");

    // Error path 1: tool-level failure comes back in-band ...
    let missing = client
        .call_tool("get_trace", json!({"id": "zzzz"}))
        .await
        .unwrap();
    assert!(missing.is_error);
    assert_eq!(missing.text(), "no trace matches `zzzz`");
    // ... and through the Sinew toolbox it becomes an error message.
    let tb = client.toolbox().await.unwrap();
    let msg = tb
        .run(&ToolCall {
            id: "x".into(),
            name: "search_runs".into(),
            arguments: json!({"kind": "banana"}),
        })
        .await;
    assert!(
        msg.content
            .starts_with("error: tool `search_runs` failed: unknown kind `banana`"),
        "{}",
        msg.content
    );

    // Error path 2: protocol errors.
    match client.call_tool("nope", json!({})).await {
        Err(McpError::Rpc(e)) => assert_eq!(e.code, error_code::INVALID_PARAMS),
        other => panic!("expected an RPC error, got {other:?}"),
    }
    match client.request("resources/list", None).await {
        Err(McpError::Rpc(e)) => assert_eq!(e.code, error_code::METHOD_NOT_FOUND),
        other => panic!("expected an RPC error, got {other:?}"),
    }

    client.close().await;
    assert!(matches!(client.ping().await, Err(McpError::Closed)));
}

#[test]
fn raw_json_rpc_over_stdio() {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("raw_missing.jsonl");
    let _ = std::fs::remove_file(&path);
    let mut child = Command::new(BIN)
        .arg("--traces")
        .arg(&path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let input = [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"raw","version":"0"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":"p","method":"ping"}"#,
        r#"this is not json"#,
        r#"[{"jsonrpc":"2.0","id":9,"method":"ping"}]"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_traces","arguments":{}}}"#,
    ];
    for line in input {
        writeln!(stdin, "{line}").unwrap();
    }
    drop(stdin); // EOF: the server answers in-flight calls and exits.
    let out: Vec<Value> = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    assert!(child.wait().unwrap().success());
    assert_eq!(out.len(), 5, "{out:#?}");
    let by_id = |id: Value| out.iter().find(|m| m["id"] == id).unwrap();
    assert_eq!(by_id(json!(1))["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(by_id(json!("p"))["result"], json!({}));
    let errors: Vec<i64> = out
        .iter()
        .filter(|m| m["id"].is_null())
        .map(|m| m["error"]["code"].as_i64().unwrap())
        .collect();
    assert_eq!(
        errors,
        [error_code::PARSE_ERROR, error_code::INVALID_REQUEST]
    );
    // The trace file does not exist: a tool error, not a crash.
    let call = by_id(json!(2));
    assert_eq!(call["result"]["isError"], true);
    assert!(call["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .starts_with("cannot read"));
}
