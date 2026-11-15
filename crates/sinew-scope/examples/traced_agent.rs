//! The `react_agent` from the umbrella example, traced to a scope server
//! through `HttpSink`.
//!
//! Self-contained (spawns an in-memory server on an ephemeral port):
//!     cargo run -p sinew-scope --example traced_agent
//! Against a running `sinew-scope serve`:
//!     cargo run -p sinew-scope --example traced_agent -- --url http://127.0.0.1:7878
//! With local Ollama instead of the scripted model:
//!     SINEW_MODEL=qwen3:4b-instruct-2507-q4_K_M \
//!       cargo run -p sinew-scope --example traced_agent -- --ollama --url http://127.0.0.1:7878
//!
//! Flags: `--url <base>`, `--project <name>` (default `examples`),
//! `--token <t>` (or `SINEW_SCOPE_TOKEN`), `--ollama`, `--question <text>`.

use std::sync::Arc;

use serde_json::{json, Value};
use sinew_core::{
    mock::ScriptedModel, providers::OpenAiCompat, ChatModel, FnTool, Message, Tool, ToolBox,
    ToolCall,
};
use sinew_graph::prebuilt::{react_agent, AgentState};
use sinew_scope::{router, Store};
use sinew_trace::{HttpSink, HttpSinkConfig, Tracer};

fn calculator() -> impl Tool {
    FnTool::new(
        "multiply",
        "Multiply two numbers",
        json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
        |args: Value| {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!(a * b))
        },
    )
}

struct Args {
    url: Option<String>,
    project: String,
    token: Option<String>,
    ollama: bool,
    question: String,
}

fn args() -> Args {
    let mut a = Args {
        url: None,
        project: "examples".into(),
        token: std::env::var("SINEW_SCOPE_TOKEN")
            .ok()
            .filter(|t| !t.is_empty()),
        ollama: false,
        question: "What is 37 times 3?".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--url" => a.url = it.next(),
            "--project" => a.project = it.next().unwrap_or(a.project),
            "--token" => a.token = it.next(),
            "--question" => a.question = it.next().unwrap_or(a.question),
            "--ollama" => a.ollama = true,
            other => eprintln!("ignoring unknown argument `{other}`"),
        }
    }
    a
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = args();

    // Use the given server, or spawn one in-process bound to loopback only.
    let (url, local_store) = match &args.url {
        Some(u) => (u.trim_end_matches('/').to_string(), None),
        None => {
            let store = Store::memory().await?;
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let url = format!("http://{}", listener.local_addr()?);
            tokio::spawn(sinew_scope::server::serve_with_shutdown(
                listener,
                router(store.clone(), None),
                std::future::pending(),
            ));
            println!("spawned in-process scope server at {url}");
            (url, Some(store))
        }
    };

    let model: Arc<dyn ChatModel> = if args.ollama {
        Arc::new(OpenAiCompat::ollama(
            std::env::var("SINEW_MODEL").unwrap_or_else(|_| "qwen3:4b-instruct-2507-q4_K_M".into()),
        ))
    } else {
        Arc::new(ScriptedModel::new([
            Message::assistant_with_tools(
                "",
                vec![ToolCall {
                    id: "call_1".into(),
                    name: "multiply".into(),
                    arguments: json!({"a": 37, "b": 3}),
                }],
            ),
            Message::assistant("37 × 3 = 111."),
        ]))
    };

    let mut config = HttpSinkConfig::new(&url).project(&args.project);
    if let Some(t) = &args.token {
        config = config.token(t);
    }
    let sink = Arc::new(HttpSink::new(config));
    let tracer = Tracer::new().with_shared_sink(sink.clone());
    let agent = react_agent(model, ToolBox::new().with(calculator()))?.with_tracer(tracer.clone());

    let out = agent
        .invoke(AgentState::new(vec![
            Message::system("Use the tools for arithmetic. Answer briefly."),
            Message::user(args.question.clone()),
        ]))
        .await?;
    println!(
        "answer: {}",
        out.last().map(|m| m.content.as_str()).unwrap_or("")
    );

    // Deliver the last batch before exiting (flush blocks, so keep it off the
    // async worker threads).
    tokio::task::spawn_blocking(move || tracer.flush()).await?;
    println!(
        "runs sent: {}, dropped: {} → project `{}` on {url}",
        sink.sent(),
        sink.dropped(),
        args.project
    );

    if let Some(store) = local_store {
        let page = store
            .list_traces(&sinew_scope::TraceQuery {
                project: Some(args.project.clone()),
                ..Default::default()
            })
            .await?;
        for t in &page.traces {
            println!(
                "trace {}  {} {}  {}ms  {} runs  tokens {}→{}",
                &t.trace_id[..8],
                t.kind,
                t.name,
                t.latency_ms.unwrap_or(0),
                t.run_count,
                t.input_tokens,
                t.output_tokens
            );
        }
    } else {
        println!("open {url}/#/traces to see it");
    }
    Ok(())
}
