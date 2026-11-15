//! Agent2Agent (A2A): one agent serves another over HTTP.
//!
//! - **remote agent**: a currency `react_agent` with an `exchange_rate`
//!   tool, served by `A2aServer` with an Agent Card at
//!   `/.well-known/agent-card.json`;
//! - **local agent**: a second `react_agent` whose only tool is an
//!   `A2aAgentTool` built from that card, so it delegates over A2A;
//! - **streaming**: the A2A client streams a reply token by token.
//!
//! ```bash
//! cargo run -p sample-14-a2a
//! cargo run -p sample-14-a2a -- --ollama
//! cargo run -p sample-14-a2a -- --model groq:llama-3.3-70b-versatile
//! # only serve the remote agent (any A2A client can call it):
//! cargo run -p sample-14-a2a -- --serve 127.0.0.1:9100
//! ```

use std::io::Write;

use futures_util::StreamExt;
use sample_common::{pick_model, BoxError};
use serde_json::{json, Value};
use sinew::prelude::*;
use sinew_a2a::{
    A2aAgentTool, A2aClient, A2aServer, AgentCard, AgentSkill, GraphExecutor, StreamResponse,
};

fn exchange_tools() -> ToolBox {
    ToolBox::new().with(FnTool::new(
        "exchange_rate",
        "Exchange rate between two ISO currency codes (units of `to` per one `from`)",
        json!({
            "type": "object",
            "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
            "required": ["from", "to"]
        }),
        |a: Value| {
            let rate = |c: &str| match c.to_uppercase().as_str() {
                "USD" => Some(1.0),
                "EUR" => Some(0.92),
                "BRL" => Some(5.45),
                "CAD" => Some(1.38),
                _ => None,
            };
            let from = a["from"].as_str().unwrap_or_default();
            let to = a["to"].as_str().unwrap_or_default();
            match (rate(from), rate(to)) {
                (Some(f), Some(t)) => Ok(json!({ "from": from, "to": to, "rate": t / f })),
                _ => Ok(json!({ "error": "unknown currency; try USD, EUR, BRL or CAD" })),
            }
        },
    ))
}

fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
    }
}

/// The remote agent's server (card URL filled in when it binds).
fn currency_server() -> Result<A2aServer, BoxError> {
    let model = pick_model(vec![
        Message::assistant_with_tools(
            "",
            vec![
                call("r1", "exchange_rate", json!({"from": "USD", "to": "EUR"})),
                call("r2", "exchange_rate", json!({"from": "USD", "to": "BRL"})),
            ],
        ),
        Message::assistant("250 USD is 230.00 EUR (rate 0.92) and 1362.50 BRL (rate 5.45)."),
        Message::assistant(
            "The Canadian dollar trades at about 1.38 per US dollar, so 100 USD buys roughly 138 CAD.",
        ),
    ])?;
    let graph = react_agent(model, exchange_tools())?;
    let executor = GraphExecutor::agent_with_system(
        graph,
        Some("You convert currencies. Always use the exchange_rate tool; answer in one sentence."),
    );
    let card = AgentCard::new("Currency Agent", "Converts amounts between currencies", "")
        .with_skill(
            AgentSkill::new(
                "convert",
                "Convert currency",
                "Converts an amount from one currency to others",
            )
            .with_tags(["finance", "currency"])
            .with_examples(["How much is 250 USD in EUR?"]),
        );
    Ok(A2aServer::new(card, executor))
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--serve") {
        let addr = args.get(i + 1).map_or("127.0.0.1:9100", String::as_str);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let url = format!("http://{}", listener.local_addr()?);
        println!("A2A agent on {url}/ (card at {url}/.well-known/agent-card.json)");
        currency_server()?.serve(listener).await?;
        return Ok(());
    }

    // 1. Start the remote agent on a free local port.
    let server = currency_server()?.spawn("127.0.0.1:0").await?;
    println!("remote agent listening on {}", server.url());

    // 2. Discover it and wrap it as a tool.
    let tool = A2aAgentTool::discover(&server.base_url()).await?;
    let spec = tool.spec();
    println!("discovered tool `{}`: {}\n", spec.name, spec.description);

    // 3. A local agent that delegates to it.
    let local_model = pick_model(vec![
        Message::assistant_with_tools(
            "",
            vec![call(
                "l1",
                &spec.name,
                json!({"input": "Convert 250 USD to EUR and to BRL"}),
            )],
        ),
        Message::assistant("Your 250 USD is worth 230.00 EUR or 1362.50 BRL."),
    ])?;
    let agent = react_agent(local_model, ToolBox::new().with(tool))?;
    let out = agent
        .invoke(AgentState::new(vec![
            Message::system(
                "You are a travel assistant. Delegate every currency question to the currency agent tool, then answer briefly.",
            ),
            Message::user("I have 250 USD. How much is that in euros and in Brazilian reais?"),
        ]))
        .await?;
    for m in &out.messages[2..] {
        match m.role {
            Role::Assistant if m.has_tool_calls() => m
                .tool_calls
                .iter()
                .for_each(|c| println!("→ A2A call {}({})", c.name, c.arguments)),
            Role::Tool => println!("← {}", m.content),
            _ => println!("\n{}", m.content),
        }
    }

    // 4. Stream straight from the remote agent with the A2A client.
    println!("\n== message/stream ==");
    let client = A2aClient::discover(&server.base_url()).await?;
    let mut events = client
        .stream_text("How many Canadian dollars do I get for 100 USD?")
        .await?;
    while let Some(ev) = events.next().await {
        match ev? {
            StreamResponse::Task(t) => println!("[task {} {:?}]", &t.id[..8], t.status.state),
            StreamResponse::ArtifactUpdate(a) => {
                print!("{}", sinew_a2a::types::parts_text(&a.artifact.parts));
                std::io::stdout().flush()?;
            }
            StreamResponse::StatusUpdate(s) if s.is_final => {
                println!("\n[{:?}]", s.status.state)
            }
            StreamResponse::StatusUpdate(_) => {}
            StreamResponse::Message(m) => println!("{}", m.text()),
        }
    }
    server.shutdown().await?;
    Ok(())
}
