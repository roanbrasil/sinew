//! `#[tool]` and `tools!` (feature `macros`).
#![cfg(feature = "macros")]

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{tool, tools, Error, Tool, ToolCall};

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Status {
    Pending,
    Shipped,
}

/// Updates the status of an order.
///
/// Returns the new status.
#[tool]
async fn update_status(
    /// Order id, e.g. "A-1001".
    id: String,
    /// New status.
    status: Status,
    /// Free-text note for the audit log.
    note: Option<String>,
) -> Result<Value, String> {
    if id.starts_with('A') {
        Ok(json!({ "id": id, "status": status, "note": note }))
    } else {
        Err(format!("unknown order `{id}`"))
    }
}

/// Multiplies two numbers.
#[tool(name = "multiply")]
fn mul(a: i64, b: i64) -> i64 {
    a * b
}

#[tool(description = "Returns the server time.")]
fn now() -> &'static str {
    "12:00"
}

/// Repeats a word.
#[tool]
fn repeat(
    word: String,
    #[serde(default = "two")]
    /// How many times (default 2).
    times: usize,
) -> String {
    word.repeat(times)
}
fn two() -> usize {
    2
}

#[test]
fn spec_comes_from_signature_and_docs() {
    let spec = update_status::tool().spec();
    assert_eq!(spec.name, "update_status");
    assert_eq!(
        spec.description,
        "Updates the status of an order.\n\nReturns the new status."
    );
    let p = &spec.parameters;
    assert_eq!(p["type"], "object");
    assert_eq!(p["required"], json!(["id", "status"]));
    assert_eq!(
        p["properties"]["id"]["description"],
        "Order id, e.g. \"A-1001\"."
    );
    assert_eq!(
        p["properties"]["note"]["description"],
        "Free-text note for the audit log."
    );
    assert!(p.get("$schema").is_none() && p.get("title").is_none());
    assert_eq!(p["additionalProperties"], false);
}

#[test]
fn name_and_description_overrides() {
    assert_eq!(mul::tool().spec().name, "multiply");
    let spec = now::tool().spec();
    assert_eq!(spec.description, "Returns the server time.");
    assert_eq!(spec.parameters["properties"], json!({}));
}

#[test]
fn functions_stay_callable() {
    assert_eq!(mul(6, 7), 42);
    assert_eq!(now(), "12:00");
}

#[tokio::test]
async fn calls_async_and_sync_tools() {
    let out = update_status::tool()
        .call(json!({ "id": "A-1", "status": "shipped" }))
        .await
        .unwrap();
    assert_eq!(
        out,
        json!({ "id": "A-1", "status": "shipped", "note": null })
    );
    assert_eq!(
        mul::tool().call(json!({ "a": 6, "b": 7 })).await.unwrap(),
        json!(42)
    );
    assert_eq!(now::tool().call(Value::Null).await.unwrap(), json!("12:00"));
    assert_eq!(
        repeat::tool().call(json!({ "word": "ab" })).await.unwrap(),
        json!("abab")
    );
}

#[tokio::test]
async fn errors_become_tool_errors() {
    let err = update_status::tool()
        .call(json!({ "id": "B-9", "status": "pending" }))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, Error::Tool { name, message } if name == "update_status" && message == "unknown order `B-9`")
    );

    let bad = mul::tool().call(json!({ "a": 1 })).await.unwrap_err();
    assert!(matches!(&bad, Error::Tool { message, .. } if message.contains("missing field `b`")));

    let unknown = mul::tool()
        .call(json!({ "a": 1, "b": 2, "c": 3 }))
        .await
        .unwrap_err();
    assert!(
        matches!(&unknown, Error::Tool { message, .. } if message.contains("unknown field `c`"))
    );
}

#[tokio::test]
async fn tools_macro_builds_a_toolbox() {
    let tb = tools![update_status, mul, now, repeat,];
    let names: Vec<String> = tb.specs().into_iter().map(|s| s.name).collect();
    assert_eq!(names.len(), 4);
    assert!(names.contains(&"multiply".to_string()));
    let call = ToolCall {
        id: "1".into(),
        name: "multiply".into(),
        arguments: json!({ "a": 2, "b": 3 }),
    };
    assert_eq!(tb.call(&call).await.unwrap(), json!(6));
}

#[test]
fn compile_errors_are_helpful() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}

#[derive(Clone, Default)]
struct Db(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

/// Saves a note.
#[tool]
async fn save_note(#[state] db: Db, text: String) -> sinew_core::Result<usize> {
    let mut v = db.0.lock().unwrap();
    v.push(text);
    Ok(v.len())
}

#[tokio::test]
async fn state_is_injected_and_hidden_from_the_schema() {
    let db = Db::default();
    let t = save_note::tool(db.clone());
    assert_eq!(t.spec().parameters["required"], json!(["text"]));
    assert!(t.spec().parameters["properties"].get("db").is_none());
    assert_eq!(t.call(json!({ "text": "a" })).await.unwrap(), json!(1));
    let tb = sinew_core::ToolBox::new().with(t);
    let call = ToolCall {
        id: "1".into(),
        name: "save_note".into(),
        arguments: json!({ "text": "b" }),
    };
    assert_eq!(tb.call(&call).await.unwrap(), json!(2));
    assert_eq!(db.0.lock().unwrap().len(), 2);
}
