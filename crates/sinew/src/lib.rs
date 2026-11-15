//! Sinew: LLM building blocks (`core`), agent graphs (`graph`) and tracing (`trace`).

pub use sinew_core as core;
pub use sinew_graph as graph;
pub use sinew_trace as trace;

pub mod prelude {
    pub use sinew_core::{
        providers::{Anthropic, OpenAiCompat},
        ChatModel, ChatRequest, FnTool, Message, PromptTemplate, Role, StreamEvent, StructuredMode,
        StructuredOutput, Tool, ToolBox, ToolCall, ToolSpec,
    };
    pub use sinew_graph::{
        prebuilt::{react_agent, AgentState},
        BoxError, GraphEvent, MemoryCheckpointer, NodeContext, RunOutcome, SendTo, StateGraph, END,
    };
    pub use sinew_trace::{JsonlSink, MemorySink, Tracer};
}
