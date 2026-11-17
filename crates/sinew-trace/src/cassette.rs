//! Turn recorded runs into a [`Cassette`] for record/replay tests.

use serde_json::Value;
use sinew_core::{
    vcr::{Cassette, Interaction, ToolInteraction},
    ChatRequest, ChatResponse, Message, Usage,
};

use crate::{Run, RunKind};

/// Builds a cassette from the runs of one trace (e.g. a production bug):
/// every LLM run becomes a model interaction and every tool run a tool call,
/// in start order.
///
/// Traces store the messages of each model call but not the tool schemas or
/// sampling options, so replay the cassette with
/// [`MatchRequests::Messages`](sinew_core::vcr::MatchRequests::Messages).
///
/// ```
/// use sinew_trace::{cassette_from_runs, MemorySink, Tracer};
/// # let runs = Vec::new();
/// let cassette = cassette_from_runs(&runs);
/// assert!(cassette.interactions.is_empty());
/// ```
pub fn cassette_from_runs(runs: &[Run]) -> Cassette {
    let mut runs: Vec<&Run> = runs.iter().collect();
    runs.sort_by_key(|r| r.start_ms);
    let mut c = Cassette::new();
    for r in runs {
        match r.kind {
            RunKind::Llm => {
                let messages: Vec<Message> = r
                    .inputs
                    .get("messages")
                    .and_then(|m| serde_json::from_value(m.clone()).ok())
                    .unwrap_or_default();
                let model = r
                    .metadata
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or(&r.name)
                    .to_string();
                let response = r
                    .outputs
                    .as_ref()
                    .and_then(|o| serde_json::from_value::<Message>(o.clone()).ok())
                    .map(|message| ChatResponse {
                        message,
                        usage: Usage {
                            input_tokens: r.input_tokens.unwrap_or(0),
                            output_tokens: r.output_tokens.unwrap_or(0),
                            ..Default::default()
                        },
                        model: model.clone(),
                    });
                c.interactions.push(Interaction {
                    model,
                    request: ChatRequest::new(messages),
                    response,
                    stream: None,
                    error: r.error.clone(),
                });
            }
            RunKind::Tool => {
                // Tool runs store the result as the tool message text.
                let result = r.outputs.as_ref().map(|o| match o {
                    Value::String(s) => {
                        serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
                    }
                    v => v.clone(),
                });
                c.tool_calls.push(ToolInteraction {
                    name: r.name.clone(),
                    arguments: r.inputs.clone(),
                    result: if r.error.is_some() { None } else { result },
                    error: r.error.clone(),
                });
            }
            _ => {}
        }
    }
    c
}
