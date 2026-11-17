//! An agent harness on Sinew primitives: the loop, tools and safety rails
//! that turn a chat model into a working agent (in the spirit of Claude
//! Code, LangChain deepagents and the OpenAI Agents SDK).
//!
//! - [`Agent`] / [`AgentBuilder`]: any [`ChatModel`](sinew_core::ChatModel)
//!   or `provider:model` spec, instructions, tools, budgets, structured final
//!   output. Runs on a `sinew-graph` loop (`agent → approval? → tools →
//!   agent`) with a checkpoint after every node, so threads are resumable.
//! - Built-in tools, each opt-in: workspace-confined filesystem
//!   (`read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`),
//!   `shell` (deny/allow lists, timeout, truncation), `http_fetch`,
//!   `write_todos` (plan kept in the thread state), `think`, `load_skill`,
//!   and `task` for subagents with their own context.
//! - [`Hook`]s around model and tool calls; a [`PermissionPolicy`] (allow /
//!   ask / deny with path and command patterns); human approval through a
//!   graph interrupt and [`Agent::resume`].
//! - Context management: token estimates, automatic compaction of older
//!   turns, offloading of large tool outputs to files.
//! - [`Budget`]s (steps, tokens, cost, wall clock) and [`Guardrail`]s (PII
//!   redaction, prompt-injection heuristic).
//! - Project instructions (`AGENTS.md`, `CLAUDE.md`) and skills folders.
//! - MCP servers' tools (feature `mcp`), tracing through `sinew-trace`.
//! - The `sinew-agent` CLI (feature `cli`, on by default).
//!
//! ```no_run
//! # async fn demo() -> sinew_harness::Result<()> {
//! use futures_util::StreamExt;
//! use sinew_harness::{Agent, AgentEvent};
//!
//! let agent = Agent::builder()
//!     .model_spec("ollama:qwen3:4b-instruct-2507-q4_K_M")
//!     .workspace(".")
//!     .filesystem_read_only()
//!     .todos()
//!     .build()?;
//!
//! let mut events = agent.stream("thread-1", "What does this project do?").await?;
//! while let Some(ev) = events.next().await {
//!     match ev {
//!         AgentEvent::Token { text } => print!("{text}"),
//!         AgentEvent::ToolStart { name, .. } => println!("\n[{name}]"),
//!         AgentEvent::Done { result } => println!("\n{} steps", result.steps),
//!         _ => {}
//!     }
//! }
//! # Ok(()) }
//! ```
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

pub mod agent;
pub mod budget;
pub mod context;
pub mod error;
pub mod events;
pub mod glob;
pub mod guardrails;
pub mod hooks;
pub mod instructions;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod permissions;
pub mod state;
pub mod tools;

pub use agent::{Agent, AgentBuilder, DEFAULT_INSTRUCTIONS};
pub use budget::{Budget, Pricing};
pub use context::{CompactionInfo, ContextConfig};
pub use error::{HarnessError, Result};
pub use events::{AgentEvent, AgentOutcome, AgentStream, RunResult};
pub use guardrails::{Guardrail, GuardrailStage, GuardrailVerdict, InjectionDetector, PiiRedactor};
pub use hooks::{Hook, HookContext, ToolDecision};
pub use permissions::{Permission, PermissionPolicy, PermissionRule};
pub use sinew_graph::prebuilt::{LoopAction, LoopGuard};
pub use state::{
    ApprovalDecision, ApprovalRequest, CallDecision, HarnessState, StopReason, Todo, TodoStatus,
    UsageTotals,
};
pub use tools::{HttpOptions, ShellOptions, SubagentSpec, SubagentTools, Workspace};
