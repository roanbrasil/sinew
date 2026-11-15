//! `sinew-serve`: deploy Sinew graphs and agents behind a low-latency HTTP
//! API, the Rust counterpart of LangGraph Platform / LangServe.
//!
//! - **Assistants**: named graphs of any state type (`Serialize +
//!   DeserializeOwned`), listed with their configuration.
//! - **Threads**: durable conversations backed by `sinew-checkpoint`
//!   ([`SqliteStore`] by default, [`PostgresStore`] with the `postgres`
//!   feature) or [`MemoryStore`]; read, edit and list their state history.
//! - **Runs**: wait for the result, stream it over SSE (`values`, `updates`,
//!   `tokens`, `events`, `custom`), or start it in the background and poll,
//!   join or cancel it. Interrupted threads resume with a new run (optionally
//!   editing the state first). Concurrent runs on one thread follow a
//!   [`MultitaskStrategy`]: reject, enqueue or interrupt. Optional webhooks
//!   on completion.
//! - **OpenAI façade**: `POST /v1/chat/completions` (streaming or not) and
//!   `GET /v1/models` over registered agents and plain chat models, so any
//!   OpenAI client can talk to a Sinew agent.
//! - **Operations**: bearer API keys, CORS, `/ok` `/health` `/ready`,
//!   graceful shutdown, request tracing into `sinew-trace` / Sinew Scope.
//! - **[`ServeClient`]**: typed client for all of the above.
//!
//! ```no_run
//! use std::sync::Arc;
//! use sinew_core::{mock::ScriptedModel, Message, ToolBox};
//! use sinew_graph::prebuilt::react_agent;
//! use sinew_serve::{Assistant, RunCreate, ServeClient, Server};
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let model = Arc::new(ScriptedModel::new([Message::assistant("Hello!")]));
//! let server = Server::builder()
//!     .assistant(Assistant::agent("agent", react_agent(model, ToolBox::new())?))
//!     .build()
//!     .spawn("127.0.0.1:0")
//!     .await?;
//!
//! let client = ServeClient::new(server.base_url());
//! let run = client
//!     .wait(Some("thread-1"), &RunCreate::new("agent", "hi".into()))
//!     .await?;
//! println!("{:?}", run.output);
//! server.shutdown().await?;
//! # Ok(()) }
//! ```
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

pub mod api;
mod assistant;
mod client;
mod openai;
mod runs;
mod server;
mod sse;
pub mod store;

pub use api::{
    AssistantInfo, Command, MultitaskStrategy, OnDisconnect, Run, RunCreate, RunStatus,
    StateUpdate, StreamMode, Thread, ThreadCreate, ThreadQuery, ThreadState, ThreadStatus,
};
pub use assistant::{Assistant, GraphAssistant};
pub use client::{ClientError, RunStream, ServeClient, StreamPart};
pub use server::{
    runtime, CorsConfig, RunningServer, ServeConfig, Server, ServerBuilder, WebhookPolicy,
};
#[cfg(feature = "postgres")]
pub use store::PostgresStore;
#[cfg(feature = "sqlite")]
pub use store::SqliteStore;
pub use store::{MemoryStore, StoreError, ThreadStore};
