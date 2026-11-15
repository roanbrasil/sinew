//! [Agent2Agent (A2A)](https://a2a-protocol.org) protocol for Sinew,
//! version 0.3, JSON-RPC 2.0 over HTTP with Server-Sent Events.
//!
//! - **Server** ([`A2aServer`]): serves an [`AgentExecutor`] with the Agent
//!   Card at `/.well-known/agent-card.json` and the methods `message/send`,
//!   `message/stream`, `tasks/get`, `tasks/cancel` and `tasks/resubscribe`.
//!   Tasks live in a swappable [`TaskStore`].
//! - **Adapters**: [`ChatModelExecutor`] serves any `ChatModel`;
//!   [`GraphExecutor`] serves any compiled graph, e.g. `react_agent`, with
//!   graph interrupts mapped to `input-required`. Tokens stream out as
//!   artifact chunks the moment the model produces them.
//! - **Client** ([`A2aClient`]): fetch the card, send, stream, get, cancel,
//!   resubscribe.
//! - **Delegation** ([`A2aAgentTool`]): a remote agent as a Sinew `Tool`, so
//!   a local agent can hand work to it.
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! use std::sync::Arc;
//! use futures_util::StreamExt;
//! use sinew_core::{mock::ScriptedModel, Message};
//! use sinew_a2a::{A2aClient, A2aServer, AgentCard, ChatModelExecutor};
//!
//! // Serve a model.
//! let model = Arc::new(ScriptedModel::new([Message::assistant("Hello from A2A")]));
//! let card = AgentCard::new("greeter", "Says hello", "");   // url filled in by spawn
//! let server = A2aServer::new(card, ChatModelExecutor::new(model))
//!     .spawn("127.0.0.1:0")
//!     .await?;
//!
//! // Talk to it.
//! let client = A2aClient::discover(&server.base_url()).await?;
//! let mut events = client.stream_text("hi").await?;
//! while let Some(ev) = events.next().await {
//!     if let Some(text) = ev?.text() {
//!         print!("{text}");
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! # Protocol notes
//!
//! - Implements A2A **0.3.0** (`kind` discriminators, `message/send`-style
//!   method names). A2A 1.0 renamed methods and dropped `kind`; it is not
//!   implemented yet.
//! - Push notifications and the authenticated extended card are not
//!   supported; their methods answer with the spec's `-32003` / `-32007`
//!   errors, and the card says `pushNotifications: false`.
//! - Executors run on their own tokio task. A client that disconnects does
//!   not stop the task; `tasks/resubscribe` reattaches and first receives the
//!   current task snapshot (a backfill of everything produced so far).
//! - `message/send` waits until the turn ends (`completed`, `failed`,
//!   `canceled`, `rejected`, `input-required`, `auth-required`) unless
//!   `configuration.blocking` is `false`.
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

pub mod adapters;
pub mod client;
pub mod error;
pub mod executor;
pub mod jsonrpc;
pub mod server;
pub mod store;
pub mod tool;
pub mod types;

pub use adapters::{ChatModelExecutor, GraphExecutor};
pub use client::{fetch_card, A2aClient, A2aStream};
pub use error::{A2aError, Result};
pub use executor::{
    AgentEvent, AgentExecutor, BoxError, EventQueue, RequestContext, TextArtifactWriter,
};
pub use jsonrpc::RpcError;
pub use server::{A2aServer, ServerHandle};
pub use store::{InMemoryTaskStore, TaskStore};
pub use tool::A2aAgentTool;
pub use types::{
    AgentCapabilities, AgentCard, AgentSkill, Artifact, Message, MessageSendParams, Part, Role,
    SendMessageResult, StreamResponse, Task, TaskArtifactUpdateEvent, TaskState, TaskStatus,
    TaskStatusUpdateEvent,
};
