//! Tracing: every graph, node, model call and tool call becomes a [`Run`]
//! with a parent, timings, inputs/outputs and token usage. Runs are pushed
//! to [`TraceSink`]s when they finish (JSONL file, in-memory, or your own).

use std::{
    fmt::Display,
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{
    chat_streaming, ChatModel, ChatRequest, ChatResponse, Message, StreamEvent, ToolBox, ToolCall,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunKind {
    Graph,
    Node,
    Llm,
    Tool,
    Chain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub id: String,
    pub trace_id: String,
    pub parent_id: Option<String>,
    pub name: String,
    pub kind: RunKind,
    pub start_ms: u64,
    pub end_ms: Option<u64>,
    pub inputs: Value,
    pub outputs: Option<Value>,
    pub error: Option<String>,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    #[serde(default)]
    pub metadata: Value,
}

impl Run {
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_ms.map(|e| e.saturating_sub(self.start_ms))
    }
}

/// Lightweight pointer to a run, used to attach children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunRef {
    pub run_id: String,
    pub trace_id: String,
}

pub trait TraceSink: Send + Sync {
    fn record(&self, run: &Run);
}

/// Keeps finished runs in memory (tests, dashboards).
#[derive(Default)]
pub struct MemorySink {
    runs: Mutex<Vec<Run>>,
}

impl MemorySink {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn runs(&self) -> Vec<Run> {
        self.runs.lock().unwrap().clone()
    }
}

impl TraceSink for MemorySink {
    fn record(&self, run: &Run) {
        self.runs.lock().unwrap().push(run.clone());
    }
}

/// Appends one JSON object per finished run. Read it with `sinew-scope`.
pub struct JsonlSink {
    file: Mutex<File>,
}

impl JsonlSink {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            file: Mutex::new(OpenOptions::new().create(true).append(true).open(path)?),
        })
    }
}

impl TraceSink for JsonlSink {
    fn record(&self, run: &Run) {
        if let Ok(line) = serde_json::to_string(run) {
            let mut f = self.file.lock().unwrap();
            // Tracing must never break the application; drop the line on I/O error.
            let _ = writeln!(f, "{line}");
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Clone, Default)]
pub struct Tracer {
    sinks: Vec<Arc<dyn TraceSink>>,
}

impl Tracer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sink(mut self, sink: impl TraceSink + 'static) -> Self {
        self.sinks.push(Arc::new(sink));
        self
    }

    pub fn with_shared_sink(mut self, sink: Arc<dyn TraceSink>) -> Self {
        self.sinks.push(sink);
        self
    }

    pub fn start(
        &self,
        name: impl Into<String>,
        kind: RunKind,
        parent: Option<&RunRef>,
        inputs: Value,
    ) -> RunHandle {
        let id = uuid::Uuid::new_v4().to_string();
        let trace_id = parent
            .map(|p| p.trace_id.clone())
            .unwrap_or_else(|| id.clone());
        RunHandle {
            tracer: self.clone(),
            run: Some(Run {
                id,
                trace_id,
                parent_id: parent.map(|p| p.run_id.clone()),
                name: name.into(),
                kind,
                start_ms: now_ms(),
                end_ms: None,
                inputs,
                outputs: None,
                error: None,
                input_tokens: None,
                output_tokens: None,
                metadata: json!({}),
            }),
        }
    }

    fn emit(&self, run: &Run) {
        for s in &self.sinks {
            s.record(run);
        }
    }
}

/// An open run. Finish it with [`RunHandle::finish`] or [`RunHandle::fail`];
/// a handle dropped without either is recorded as failed.
pub struct RunHandle {
    tracer: Tracer,
    run: Option<Run>,
}

impl RunHandle {
    fn run_mut(&mut self) -> &mut Run {
        self.run.as_mut().expect("run already finished")
    }

    pub fn reference(&self) -> RunRef {
        let r = self.run.as_ref().expect("run already finished");
        RunRef {
            run_id: r.id.clone(),
            trace_id: r.trace_id.clone(),
        }
    }

    pub fn set_usage(&mut self, input_tokens: u32, output_tokens: u32) {
        let r = self.run_mut();
        r.input_tokens = Some(input_tokens);
        r.output_tokens = Some(output_tokens);
    }

    pub fn set_metadata(&mut self, key: &str, value: Value) {
        self.run_mut().metadata[key] = value;
    }

    pub fn finish(mut self, outputs: Value) {
        if let Some(mut r) = self.run.take() {
            r.end_ms = Some(now_ms());
            r.outputs = Some(outputs);
            self.tracer.emit(&r);
        }
    }

    pub fn fail(mut self, error: impl Display) {
        if let Some(mut r) = self.run.take() {
            r.end_ms = Some(now_ms());
            r.error = Some(error.to_string());
            self.tracer.emit(&r);
        }
    }
}

impl Drop for RunHandle {
    fn drop(&mut self) {
        if let Some(mut r) = self.run.take() {
            r.end_ms = Some(now_ms());
            r.error
                .get_or_insert_with(|| "run dropped before finishing".into());
            self.tracer.emit(&r);
        }
    }
}

/// Calls a model, recording an `llm` run when a tracer is present.
pub async fn traced_chat<M: ChatModel + ?Sized>(
    model: &M,
    request: ChatRequest,
    tracer: Option<&Tracer>,
    parent: Option<&RunRef>,
) -> sinew_core::Result<ChatResponse> {
    let Some(tracer) = tracer else {
        return model.chat(request).await;
    };
    let mut run = tracer.start(model.name(), RunKind::Llm, parent, json!({ "messages": request.messages, "tools": request.tools.iter().map(|t| &t.name).collect::<Vec<_>>() }));
    match model.chat(request).await {
        Ok(resp) => {
            run.set_usage(resp.usage.input_tokens, resp.usage.output_tokens);
            run.set_metadata("model", json!(resp.model));
            run.finish(json!(resp.message));
            Ok(resp)
        }
        Err(e) => {
            run.fail(&e);
            Err(e)
        }
    }
}

/// Streams a model call, passing every event to `on_event`, and records an
/// `llm` run (with usage and the number of streamed chunks) when a tracer is
/// present. Returns the accumulated response.
pub async fn traced_chat_stream<M, F>(
    model: &M,
    request: ChatRequest,
    tracer: Option<&Tracer>,
    parent: Option<&RunRef>,
    mut on_event: F,
) -> sinew_core::Result<ChatResponse>
where
    M: ChatModel + ?Sized,
    F: FnMut(&StreamEvent),
{
    let Some(tracer) = tracer else {
        return chat_streaming(model, request, on_event).await;
    };
    let mut run = tracer.start(model.name(), RunKind::Llm, parent, json!({ "messages": request.messages, "tools": request.tools.iter().map(|t| &t.name).collect::<Vec<_>>() }));
    let mut chunks = 0u64;
    let result = chat_streaming(model, request, |ev| {
        if matches!(ev, StreamEvent::TextDelta { .. }) {
            chunks += 1;
        }
        on_event(ev)
    })
    .await;
    match result {
        Ok(resp) => {
            run.set_usage(resp.usage.input_tokens, resp.usage.output_tokens);
            run.set_metadata("model", json!(resp.model));
            run.set_metadata("streamed_chunks", json!(chunks));
            run.finish(json!(resp.message));
            Ok(resp)
        }
        Err(e) => {
            run.fail(&e);
            Err(e)
        }
    }
}

/// Runs a tool call, recording a `tool` run when a tracer is present.
pub async fn traced_tool(
    tools: &ToolBox,
    call: &ToolCall,
    tracer: Option<&Tracer>,
    parent: Option<&RunRef>,
) -> Message {
    let Some(tracer) = tracer else {
        return tools.run(call).await;
    };
    let run = tracer.start(
        call.name.clone(),
        RunKind::Tool,
        parent,
        call.arguments.clone(),
    );
    let msg = tools.run(call).await;
    if msg.content.starts_with("error: ") {
        run.fail(&msg.content["error: ".len()..]);
    } else {
        run.finish(json!(msg.content));
    }
    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    use sinew_core::mock::ScriptedModel;

    #[tokio::test]
    async fn nests_runs_and_records_usage() {
        let sink = Arc::new(MemorySink::new());
        let tracer = Tracer::new().with_shared_sink(sink.clone());
        let root = tracer.start("root", RunKind::Chain, None, json!({}));
        let model = ScriptedModel::new([Message::assistant("hello there")]);
        traced_chat(
            &model,
            ChatRequest::new(vec![Message::user("hi")]),
            Some(&tracer),
            Some(&root.reference()),
        )
        .await
        .unwrap();
        let root_ref = root.reference();
        root.finish(json!("done"));

        let runs = sink.runs();
        assert_eq!(runs.len(), 2);
        let llm = &runs[0];
        assert_eq!(llm.kind, RunKind::Llm);
        assert_eq!(llm.parent_id.as_deref(), Some(root_ref.run_id.as_str()));
        assert_eq!(llm.trace_id, root_ref.trace_id);
        assert_eq!(llm.output_tokens, Some(2));
    }

    #[test]
    fn dropped_run_is_recorded_as_error() {
        let sink = Arc::new(MemorySink::new());
        let tracer = Tracer::new().with_shared_sink(sink.clone());
        drop(tracer.start("x", RunKind::Node, None, json!(null)));
        assert!(sink.runs()[0].error.is_some());
    }
}
