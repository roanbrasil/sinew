//! Sinew: LLM building blocks (`core`), agent graphs (`graph`) and tracing (`trace`),
//! plus optional durable checkpoints (`checkpoint`), MCP (`mcp`) and evaluations (`eval`).
//!
//! | Feature | Adds |
//! |---|---|
//! | `sqlite` / `postgres` | `checkpoint`: durable checkpointers |
//! | `mcp` | `mcp`: Model Context Protocol client and server |
//! | `eval` | `eval`: datasets, evaluators, experiments |
//! | `http-trace` | `trace::HttpSink` for `sinew-scope serve` |
//! | `full` | all of the above except Postgres |
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

pub use sinew_core as core;
pub use sinew_graph as graph;
pub use sinew_trace as trace;

#[cfg(any(feature = "sqlite", feature = "postgres"))]
pub use sinew_checkpoint as checkpoint;
#[cfg(feature = "eval")]
pub use sinew_eval as eval;
#[cfg(feature = "mcp")]
pub use sinew_mcp as mcp;

pub mod prelude {
    pub use sinew_core::{
        providers::{from_spec, Anthropic, Gemini, OpenAiCompat},
        ChatModel, ChatRequest, FnTool, Message, PromptTemplate, Role, StreamEvent, StructuredMode,
        StructuredOutput, Tool, ToolBox, ToolCall, ToolSpec,
    };
    pub use sinew_graph::{
        prebuilt::{react_agent, AgentState},
        BoxError, GraphEvent, MemoryCheckpointer, NodeContext, RunOutcome, SendTo, StateGraph, END,
    };
    pub use sinew_trace::{JsonlSink, MemorySink, Tracer};
}
