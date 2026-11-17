//! Two ways to define tools — a plain function with `#[sinew::tool]` (schema
//! from the arguments, description from the doc comment) and a struct
//! implementing the async `Tool` trait by hand — then hand them to the
//! prebuilt tool-calling agent.
//!
//! ```bash
//! cargo run -p sample-03-tools-agent
//! cargo run -p sample-03-tools-agent -- --ollama
//! ```

use std::collections::HashMap;

use async_trait::async_trait;
use sample_common::{pick_model, BoxError};
use serde_json::{json, Value};
use sinew::core::{Error, Result as CoreResult};
use sinew::prelude::*;

/// A tool with state: an in-memory inventory lookup.
struct Inventory {
    stock: HashMap<&'static str, u32>,
}

#[async_trait]
impl Tool for Inventory {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "check_stock".into(),
            description: "Returns how many units of a product are in stock".into(),
            parameters: json!({
                "type": "object",
                "properties": { "product": { "type": "string" } },
                "required": ["product"]
            }),
        }
    }

    async fn call(&self, args: Value) -> CoreResult<Value> {
        let product = args["product"].as_str().ok_or(Error::Tool {
            name: "check_stock".into(),
            message: "missing `product`".into(),
        })?;
        match self.stock.get(product.to_lowercase().as_str()) {
            Some(n) => Ok(json!({ "product": product, "units": n })),
            None => Err(Error::Tool {
                name: "check_stock".into(),
                message: format!("unknown product `{product}`"),
            }),
        }
    }
}

/// Returns the unit price in dollars of a product.
#[tool]
fn unit_price(
    /// Product name, e.g. "keyboard".
    product: String,
) -> f64 {
    match product.to_lowercase().as_str() {
        "keyboard" => 89.0,
        "mouse" => 35.0,
        _ => 0.0,
    }
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let tools = ToolBox::new().with(unit_price::tool()).with(Inventory {
        stock: HashMap::from([("keyboard", 12), ("mouse", 0)]),
    });

    let model = pick_model(vec![
        Message::assistant_with_tools(
            "",
            vec![
                ToolCall {
                    id: "1".into(),
                    name: "check_stock".into(),
                    arguments: json!({"product": "keyboard"}),
                },
                ToolCall {
                    id: "2".into(),
                    name: "unit_price".into(),
                    arguments: json!({"product": "keyboard"}),
                },
            ],
        ),
        Message::assistant(
            "We have 12 keyboards in stock at $89 each, so all of them are worth $1,068.",
        ),
    ])?;

    let agent = react_agent(model, tools)?;
    let out = agent
        .invoke(AgentState::new(vec![
            Message::system("Use the tools. Answer in one sentence."),
            Message::user(
                "How many keyboards do we have and what is the total value of that stock?",
            ),
        ]))
        .await?;

    for m in &out.messages[1..] {
        match m.role {
            Role::Assistant if m.has_tool_calls() => {
                for c in &m.tool_calls {
                    println!("→ call  {}({})", c.name, c.arguments);
                }
            }
            Role::Tool => println!("← result {}", m.content),
            _ => println!("{:?}: {}", m.role, m.content),
        }
    }
    Ok(())
}
