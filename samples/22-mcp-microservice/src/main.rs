//! An "orders" microservice that also speaks MCP.
//!
//! The service keeps its normal REST route (`GET /orders`) and mounts an MCP
//! endpoint (`/mcp`, Streamable HTTP) with bearer auth next to it. Its tools
//! are plain functions with `#[sinew::tool]`: the JSON Schema comes from the
//! arguments, the description from the doc comment, and the database handle
//! is injected with `#[state]` (the model never sees it). Any MCP client can use them:
//! Claude Code, Cursor, another service, or a Sinew agent.
//!
//! ```bash
//! cargo run -p sample-22-mcp-microservice                     # service + agent, offline model
//! cargo run -p sample-22-mcp-microservice -- --ollama         # agent on a local model
//! cargo run -p sample-22-mcp-microservice -- --serve 0.0.0.0:8080   # just the service
//! # register the running service with Claude Code:
//! claude mcp add --transport http orders http://127.0.0.1:8080/mcp --header "Authorization: Bearer dev-token"
//! ```

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use sample_common::{is_live, pick_model, BoxError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sinew::mcp::{McpClient, McpHttpConfig, McpServer};
use sinew::prelude::*;
use sinew::tool;

const TOKEN: &str = "dev-token";

#[derive(Clone, Serialize)]
struct Order {
    id: String,
    customer: String,
    status: String,
    total: f64,
}

/// The service's own state (a database in a real service).
type Db = Arc<Mutex<BTreeMap<String, Order>>>;

/// Look up an order by id.
#[tool]
fn get_order(
    #[state] db: Db,
    /// Order id, e.g. "A-1001".
    id: String,
) -> Result<Order, String> {
    db.lock()
        .unwrap()
        .get(&id)
        .cloned()
        .ok_or(format!("no order {id}"))
}

#[derive(Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum Status {
    Paid,
    Shipped,
    Cancelled,
}

/// Change an order's status.
#[tool]
fn update_order_status(
    #[state] db: Db,
    /// Order id, e.g. "A-1001".
    id: String,
    /// New status.
    status: Status,
) -> Result<Order, String> {
    let mut db = db.lock().unwrap();
    let o = db.get_mut(&id).ok_or(format!("no order {id}"))?;
    o.status = match status {
        Status::Paid => "paid",
        Status::Shipped => "shipped",
        Status::Cancelled => "cancelled",
    }
    .into();
    Ok(o.clone())
}

fn tools(db: Db) -> ToolBox {
    ToolBox::new()
        .with(get_order::tool(db.clone()))
        .with(update_order_status::tool(db))
}

/// REST routes + the MCP endpoint in one axum app.
fn app(db: Db) -> axum::Router {
    let rest_db = db.clone();
    let mcp = McpServer::new("orders", "1.0.0")
        .with_instructions("Tools for the orders service.")
        .with_tools(tools(db))
        .http_router(McpHttpConfig::default().with_bearer_token(TOKEN));
    axum::Router::new()
        .route(
            "/orders",
            axum::routing::get(move || {
                let db = rest_db.clone();
                async move { axum::Json(db.lock().unwrap().values().cloned().collect::<Vec<_>>()) }
            }),
        )
        .merge(mcp)
}

fn seed() -> Db {
    let mut m = BTreeMap::new();
    for (id, customer, status, total) in [
        ("A-1001", "Ana", "paid", 129.90),
        ("A-1002", "Bruno", "paid", 54.00),
    ] {
        m.insert(
            id.to_string(),
            Order {
                id: id.into(),
                customer: customer.into(),
                status: status.into(),
                total,
            },
        );
    }
    Arc::new(Mutex::new(m))
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--serve") {
        let addr = args
            .get(i + 1)
            .map(String::as_str)
            .unwrap_or("127.0.0.1:8080");
        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("orders service on http://{addr}  (REST /orders, MCP /mcp, token {TOKEN})");
        axum::serve(listener, app(seed()))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        return Ok(());
    }

    // 1. Start the microservice in-process on a free port.
    let db = seed();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let base = format!("http://{}", listener.local_addr()?);
    let svc = app(db.clone());
    tokio::spawn(async move { axum::serve(listener, svc).await });
    println!("orders service on {base}");

    // 2. Another service (here: an agent) connects over MCP with the token.
    let client = McpClient::connect_http_with_token(format!("{base}/mcp"), TOKEN).await?;
    for t in client.list_tools().await? {
        println!("  tool {:<20} {}", t.name, t.input_schema);
    }

    // 3. The MCP tools are ordinary Sinew tools for a ReAct agent.
    let model = pick_model(vec![
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "get_order".into(),
                arguments: json!({"id": "A-1002"}),
            }],
        ),
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c2".into(),
                name: "update_order_status".into(),
                arguments: json!({"id": "A-1002", "status": "shipped"}),
            }],
        ),
        Message::assistant("Order A-1002 for Bruno (54.00) is now marked as shipped."),
    ])?;
    let agent = react_agent(model, client.toolbox().await?)?;
    let out = agent
        .invoke(AgentState::new(vec![Message::user(
            "Look up order A-1002 and mark it as shipped. Answer in one sentence.",
        )]))
        .await?;
    println!(
        "\nagent: {}",
        out.messages
            .last()
            .map(|m| m.content.as_str())
            .unwrap_or("")
    );

    // 4. The service's own REST view reflects the change made through MCP.
    let status = db.lock().unwrap()["A-1002"].status.clone();
    println!("REST view of A-1002: status = {status}");
    if !is_live() {
        assert_eq!(status, "shipped");
    }
    client.close().await;
    Ok(())
}
