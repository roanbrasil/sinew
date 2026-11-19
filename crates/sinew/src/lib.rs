//! Sinew: LLM building blocks (`core`), agent graphs (`graph`) and tracing (`trace`),
//! plus optional retrieval, agent harness, deployment server, protocols and evaluations.
//!
//! | Feature | Adds |
//! |---|---|
//! | `sqlite` / `postgres` | `checkpoint`: durable checkpointers |
//! | `rag` (`pgvector`, `qdrant`, `pdf`) | `rag`: embeddings, loaders, splitters, vector stores, retrievers |
//! | `harness` | `harness`: agent harness with built-in tools, subagents, permissions |
//! | `tools` | `builtin_tools`: web search, web fetch, Wikipedia, calculator, date/time, HTTP requests, e-mail (all `sinew-tools` families) |
//! | `memory` | `memory`: long-term memory across threads (extraction, dedup, recall; `MemoryHook` with `harness`) |
//! | `govern` (`govern-opa`, `govern-cedar`) | `govern`: agent control plane: policies before every tool and model call, agent registry, kill switch, decision events (`GovernHook` with `harness`) |
//! | `serve` | `serve`: deployment server, client and OpenAI-compatible endpoint |
//! | `a2a` | `a2a`: Agent2Agent protocol client and server |
//! | `mcp` / `mcp-http` | `mcp`: Model Context Protocol client and server (stdio; `mcp-http` adds Streamable HTTP for microservices) |
//! | `eval` | `eval`: datasets, evaluators, experiments, prompt hub client |
//! | `http-trace` | `trace::HttpSink` for `sinew-scope serve` |
//! | `otlp` / `otlp-proto` | `trace::OtlpSink` for any OpenTelemetry collector |
//! | `macros` (default) | `#[sinew::tool]` and `tools!`: plain functions become tools; `prompt!` / `chat_prompt!`: templates checked at compile time |
//! | `identity` | signed agent identity and authenticated messages between agents (`core::identity`, `graph::prebuilt::{MessageAuth, PeerAgentTool}`) |
//! | `sqlite-cache`, `schemars` | persistent response cache; schemas from Rust types |
//! | `full` | all of the above except Postgres-backed features and `govern-cedar` |
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

pub use sinew_core as core;
#[cfg(feature = "macros")]
pub use sinew_core::tool;
pub use sinew_core::tools;
#[cfg(feature = "macros")]
pub use sinew_core::{chat_prompt, prompt};
pub use sinew_graph as graph;
pub use sinew_trace as trace;

#[cfg(feature = "a2a")]
pub use sinew_a2a as a2a;
#[cfg(any(feature = "sqlite", feature = "postgres"))]
pub use sinew_checkpoint as checkpoint;
#[cfg(feature = "eval")]
pub use sinew_eval as eval;
#[cfg(feature = "govern")]
pub use sinew_govern as govern;
#[cfg(feature = "harness")]
pub use sinew_harness as harness;
#[cfg(feature = "mcp")]
pub use sinew_mcp as mcp;
#[cfg(feature = "memory")]
pub use sinew_memory as memory;
#[cfg(feature = "rag")]
pub use sinew_rag as rag;
#[cfg(feature = "serve")]
pub use sinew_serve as serve;
#[cfg(feature = "tools")]
pub use sinew_tools as builtin_tools;

pub mod prelude {
    #[cfg(feature = "macros")]
    pub use sinew_core::tool;
    pub use sinew_core::tools;
    #[cfg(feature = "macros")]
    pub use sinew_core::{chat_prompt, prompt};
    pub use sinew_core::{
        providers::{from_spec, Anthropic, Gemini, OpenAiCompat},
        AsyncFnTool, ChatModel, ChatModelExt, ChatPromptTemplate, ChatRequest, ChatResponse,
        ContentPart, FnTool, Message, PromptTemplate, Role, Runnable, RunnableExt, StrOutputParser,
        StreamEvent, StructuredMode, StructuredOutput, Tool, ToolBox, ToolCall, ToolSpec, Usage,
    };
    #[cfg(feature = "macros")]
    pub use sinew_graph::State;
    pub use sinew_graph::{
        prebuilt::{create_supervisor, react_agent, AgentState},
        BoxError, Command, GraphEvent, MemoryCheckpointer, NodeContext, RunOutcome, SendTo,
        StateGraph, StreamMode, END, START,
    };
    pub use sinew_trace::{JsonlSink, MemorySink, Tracer};
}
