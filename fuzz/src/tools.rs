//! Tool argument parsing in the `#[tool]` expansion path.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{tool, Error, Tool, ToolBox, ToolCall};

use crate::block_on;

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Status {
    Pending,
    Shipped { carrier: String },
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct Address {
    street: String,
    #[serde(default)]
    unit: Option<u16>,
}

/// Every common argument shape: strings, integers of several widths,
/// floats, booleans, enums with data, nested structs, lists and maps.
#[tool]
#[allow(clippy::too_many_arguments)]
fn everything(
    /// A name.
    name: String,
    /// A small count.
    count: u8,
    /// A signed offset.
    offset: Option<i64>,
    /// A ratio.
    ratio: f64,
    /// A flag.
    flag: Option<bool>,
    /// A status.
    status: Status,
    /// An address.
    address: Option<Address>,
    /// Tags.
    #[serde(default)]
    tags: Vec<String>,
    /// Labels.
    #[serde(default)]
    labels: BTreeMap<String, u32>,
) -> Value {
    json!({
        "name": name, "count": count, "offset": offset, "ratio": ratio, "flag": flag,
        "status": status, "address": address, "tags": tags, "labels": labels,
    })
}

/// No arguments at all.
#[tool]
fn nothing() -> &'static str {
    "ok"
}

/// An async tool with a renamed argument and a fallible body.
#[tool(name = "lookup")]
async fn lookup(
    /// The key.
    #[serde(rename = "key-name")]
    key: String,
) -> Result<String, String> {
    if key.is_empty() {
        Err("empty key".into())
    } else {
        Ok(key)
    }
}

/// Arbitrary JSON (or, failing that, the raw text as a string) as the
/// arguments of every tool: the call returns a value or `Error::Tool`, never
/// panics, and a `ToolBox` dispatch agrees with the direct call.
pub fn tool_args(data: &[u8]) {
    let arguments = serde_json::from_slice::<Value>(data)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(data).into_owned()));
    block_on(async {
        let tools: [Box<dyn Tool>; 3] = [
            Box::new(everything::tool()),
            Box::new(nothing::tool()),
            Box::new(lookup::tool()),
        ];
        let toolbox = ToolBox::new()
            .with(everything::tool())
            .with(nothing::tool())
            .with(lookup::tool());
        for t in &tools {
            let spec = t.spec();
            assert!(spec.parameters.is_object());
            let direct = t.call(arguments.clone()).await;
            match &direct {
                Ok(_) => {}
                Err(Error::Tool { name, .. }) => assert_eq!(name, &spec.name),
                Err(e) => panic!("unexpected error kind from {}: {e:?}", spec.name),
            }
            let call = ToolCall {
                id: "c".into(),
                name: spec.name.clone(),
                arguments: arguments.clone(),
            };
            let via_box = toolbox.call(&call).await;
            assert_eq!(direct.is_ok(), via_box.is_ok(), "ToolBox disagrees");
        }
    });
}
