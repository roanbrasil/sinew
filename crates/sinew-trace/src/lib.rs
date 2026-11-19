//! Tracing: every graph, node, model call and tool call becomes a [`Run`]
//! with a parent, timings, inputs/outputs and token usage. Runs are pushed
//! to [`TraceSink`]s when they finish (JSONL file, in-memory, or your own).
//!
//! With the `http` feature, [`HttpSink`] ships finished runs to a
//! `sinew-scope serve` instance in background batches. With the `otlp`
//! feature, [`OtlpSink`] exports them as OpenTelemetry spans (GenAI semantic
//! conventions) to any OTLP/HTTP collector; `otlp-proto` adds the protobuf
//! encoding. See [`otlp`] for the mapping.
//!
//! [`metrics::MetricsSink`] aggregates finished runs into Prometheus counters
//! and histograms (runs, latency, time to first token, tokens, cost, tool
//! calls, errors) and renders the text exposition format.
//!
//! [`Tracer::with_redactor`] masks API keys, tokens, e-mail addresses, card
//! numbers, sensitive fields and custom patterns in every run before any sink
//! sees it; see [`redact`].
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

#[cfg(any(feature = "http", feature = "otlp"))]
mod batch;
mod cassette;
#[cfg(feature = "http")]
mod http;
pub mod metrics;
pub mod otlp;
pub mod redact;
pub use cassette::cassette_from_runs;
#[cfg(feature = "http")]
pub use http::{HttpSink, HttpSinkConfig};
pub use metrics::MetricsSink;
#[cfg(feature = "otlp")]
pub use otlp::{OtlpProtocol, OtlpSink, OtlpSinkConfig};
pub use redact::Redactor;

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
use sinew_core::identity::{current_context, with_context};
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

    /// Pushes out anything buffered. Called by [`Tracer::flush`]; the default
    /// does nothing, which suits synchronous sinks.
    fn flush(&self) {}
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

    fn flush(&self) {
        let _ = self.file.lock().unwrap().flush();
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
    redactor: Option<Arc<Redactor>>,
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

    /// Redacts every finished run's `inputs`, `outputs`, `error` and
    /// `metadata` with `redactor` before it reaches any sink (JSONL, memory,
    /// HTTP, OTLP or your own). Replaces a previously set redactor.
    ///
    /// ```
    /// use sinew_trace::{JsonlSink, Redactor, Tracer};
    ///
    /// # fn main() -> std::io::Result<()> {
    /// # let dir = std::env::temp_dir().join("sinew-redact-doc.jsonl");
    /// let tracer = Tracer::new()
    ///     .with_redactor(Redactor::new().phones(true).field("session_id"))
    ///     .with_sink(JsonlSink::open(&dir)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_redactor(mut self, redactor: Redactor) -> Self {
        self.redactor = Some(Arc::new(redactor));
        self
    }

    /// The redactor set with [`Tracer::with_redactor`], if any.
    pub fn redactor(&self) -> Option<&Redactor> {
        self.redactor.as_deref()
    }

    pub fn start(
        &self,
        name: impl Into<String>,
        kind: RunKind,
        parent: Option<&RunRef>,
        inputs: Value,
    ) -> RunHandle {
        let id = uuid::Uuid::new_v4().to_string();
        let mut metadata = json!({});
        // A root run inside a call context (a request received from another
        // agent, or a tool's own sub-run) joins the context's trace, under
        // the run that made the call, and records who acts for whom.
        let ctx = parent.is_none().then(current_context).flatten();
        let ctx_parent = ctx.as_ref().and_then(|c| {
            Some(RunRef {
                trace_id: c.trace_id.clone()?,
                run_id: c.parent_run_id.clone()?,
            })
        });
        if let Some(c) = &ctx {
            metadata["identity"] = c.to_trace_value();
        }
        let parent = parent.or(ctx_parent.as_ref());
        let trace_id = parent
            .map(|p| p.trace_id.clone())
            .or_else(|| ctx.as_ref().and_then(|c| c.trace_id.clone()))
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
                metadata,
            }),
        }
    }

    /// Flushes every sink. Call it before the process exits so buffered
    /// sinks such as `HttpSink` deliver their last batch.
    pub fn flush(&self) {
        for s in &self.sinks {
            s.flush();
        }
    }

    fn emit(&self, mut run: Run) {
        if self.sinks.is_empty() {
            return;
        }
        if let Some(r) = &self.redactor {
            r.redact_run(&mut run);
        }
        for s in &self.sinks {
            s.record(&run);
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
            self.tracer.emit(r);
        }
    }

    pub fn fail(mut self, error: impl Display) {
        if let Some(mut r) = self.run.take() {
            r.end_ms = Some(now_ms());
            r.error = Some(error.to_string());
            self.tracer.emit(r);
        }
    }
}

impl Drop for RunHandle {
    fn drop(&mut self) {
        if let Some(mut r) = self.run.take() {
            r.end_ms = Some(now_ms());
            r.error
                .get_or_insert_with(|| "run dropped before finishing".into());
            self.tracer.emit(r);
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
            set_cache_usage(&mut run, &resp.usage);
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
/// `llm` run (with usage, the number of streamed chunks and the time to the
/// first token as `metadata.ttft_ms`) when a tracer is present. Returns the
/// accumulated response.
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
    let started = std::time::Instant::now();
    let mut ttft_ms: Option<u64> = None;
    let result = chat_streaming(model, request, |ev| {
        if matches!(ev, StreamEvent::TextDelta { .. }) {
            chunks += 1;
        }
        if ttft_ms.is_none()
            && matches!(
                ev,
                StreamEvent::TextDelta { .. } | StreamEvent::ToolCallDelta { .. }
            )
        {
            ttft_ms = Some(started.elapsed().as_millis() as u64);
        }
        on_event(ev)
    })
    .await;
    match result {
        Ok(resp) => {
            run.set_usage(resp.usage.input_tokens, resp.usage.output_tokens);
            set_cache_usage(&mut run, &resp.usage);
            run.set_metadata("model", json!(resp.model));
            run.set_metadata("streamed_chunks", json!(chunks));
            if let Some(t) = ttft_ms {
                run.set_metadata("ttft_ms", json!(t));
            }
            run.finish(json!(resp.message));
            Ok(resp)
        }
        Err(e) => {
            run.fail(&e);
            Err(e)
        }
    }
}

/// Records prompt-cache token counts the provider reported (read by
/// [`metrics::MetricsSink`] and cost estimates).
fn set_cache_usage(run: &mut RunHandle, usage: &sinew_core::Usage) {
    if usage.cached_input_tokens > 0 {
        run.set_metadata("cached_input_tokens", json!(usage.cached_input_tokens));
    }
    if usage.cache_creation_input_tokens > 0 {
        run.set_metadata(
            "cache_creation_input_tokens",
            json!(usage.cache_creation_input_tokens),
        );
    }
}

/// Records a guardrail event ([`sinew_core::guardrails::GuardEvent`]) as a
/// finished `chain` run named `guardrail:<name>`, with the stage in
/// `inputs`, the whole event in `outputs` and `metadata.guardrail`.
/// Events from a [`Block`](sinew_core::guardrails::GuardAction::Block) or
/// [`Retry`](sinew_core::guardrails::GuardAction::Retry) rule are recorded
/// as failed runs with the reason as the error.
pub fn record_guard_event(
    tracer: &Tracer,
    parent: Option<&RunRef>,
    event: &sinew_core::guardrails::GuardEvent,
) {
    use sinew_core::guardrails::GuardAction;
    let mut run = tracer.start(
        format!("guardrail:{}", event.guard),
        RunKind::Chain,
        parent,
        json!({ "stage": event.stage }),
    );
    run.set_metadata("guardrail", json!(event));
    let blocked = matches!(event.action, GuardAction::Block | GuardAction::Retry { .. })
        || (event.action == GuardAction::Redact && !event.rewrote);
    if blocked {
        run.fail(&event.reason);
    } else {
        run.finish(json!(event));
    }
}

/// A [`GuardObserver`](sinew_core::guardrails::GuardObserver) that records
/// every event with [`record_guard_event`], for
/// [`Guardrails::with_observer`](sinew_core::guardrails::Guardrails::with_observer)
/// or [`Guarded::with_observer`](sinew_core::guardrails::Guarded::with_observer).
///
/// ```
/// use std::sync::Arc;
/// use sinew_core::guardrails::{guard, Guardrails, MaxLength};
/// use sinew_core::{mock::ScriptedModel, ChatModel, ChatModelExt, ChatRequest, Message};
/// use sinew_trace::{guardrail_observer, MemorySink, Tracer};
///
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// let sink = Arc::new(MemorySink::new());
/// let tracer = Tracer::new().with_shared_sink(sink.clone());
/// let model = ScriptedModel::new([Message::assistant("a long answer")])
///     .with_guardrails(Guardrails::new(), [guard(MaxLength::chars(5)).warn()])
///     .with_observer(guardrail_observer(tracer, None));
/// model.chat(ChatRequest::new(vec![Message::user("hi")])).await.unwrap();
/// assert_eq!(sink.runs()[0].name, "guardrail:max_length");
/// # });
/// ```
pub fn guardrail_observer(
    tracer: Tracer,
    parent: Option<RunRef>,
) -> sinew_core::guardrails::GuardObserver {
    Arc::new(move |event| record_guard_event(&tracer, parent.as_ref(), event))
}

/// Runs a tool call, recording a `tool` run when a tracer is present. A
/// tool with a [`source`](sinew_core::Tool::source) (an MCP server) adds it
/// as `tool_source` metadata (`{"server": ..., "version": ...}`).
///
/// Inside a [`CallContext`](sinew_core::identity::CallContext) the run gets
/// `identity` metadata (agent, user, tenant and the delegation chain), and
/// the tool runs with the context pointing at this run, so a remote agent
/// it calls (A2A, MCP) continues the same trace under it.
pub async fn traced_tool(
    tools: &ToolBox,
    call: &ToolCall,
    tracer: Option<&Tracer>,
    parent: Option<&RunRef>,
) -> Message {
    let Some(tracer) = tracer else {
        return tools.run(call).await;
    };
    let mut run = tracer.start(
        call.name.clone(),
        RunKind::Tool,
        parent,
        call.arguments.clone(),
    );
    if let Some(source) = tools.get(&call.name).and_then(|t| t.source()) {
        run.set_metadata("tool_source", json!(source));
    }
    let msg = match current_context() {
        Some(ctx) => {
            run.set_metadata("identity", ctx.to_trace_value());
            let r = run.reference();
            with_context(ctx.run(r.trace_id, r.run_id), tools.run(call)).await
        }
        None => tools.run(call).await,
    };
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

    #[tokio::test]
    async fn sinks_receive_redacted_runs() {
        let memory = Arc::new(MemorySink::new());
        let dir = std::env::temp_dir().join(format!("sinew-redact-{}.jsonl", uuid::Uuid::new_v4()));
        let tracer = Tracer::new()
            .with_redactor(Redactor::new().allow("metadata.owner"))
            .with_shared_sink(memory.clone())
            .with_sink(JsonlSink::open(&dir).unwrap());
        assert!(tracer.redactor().is_some());

        let model = ScriptedModel::new([Message::assistant("write to bob@example.com")]);
        traced_chat(
            &model,
            ChatRequest::new(vec![Message::user(
                "my key is sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz",
            )]),
            Some(&tracer),
            None,
        )
        .await
        .unwrap();
        let mut run = tracer.start(
            "login",
            RunKind::Tool,
            None,
            json!({ "password": "hunter2" }),
        );
        run.set_metadata("owner", json!("ops@example.com"));
        run.fail("bad card 4111 1111 1111 1111");
        tracer.flush();

        let runs = memory.runs();
        let llm = serde_json::to_string(&runs[0]).unwrap();
        assert!(!llm.contains("sk-proj-"), "{llm}");
        assert!(!llm.contains("bob@example.com"), "{llm}");
        assert!(llm.contains("[REDACTED:secret]") && llm.contains("[REDACTED:email]"));
        assert_eq!(runs[1].inputs["password"], "[REDACTED:field]");
        assert_eq!(runs[1].metadata["owner"], "ops@example.com");
        assert_eq!(runs[1].error.as_deref(), Some("bad card [REDACTED:card]"));

        let file = std::fs::read_to_string(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);
        assert!(
            !file.contains("hunter2") && !file.contains("sk-proj-"),
            "{file}"
        );
        assert!(file.contains("[REDACTED:field]"));
    }

    #[test]
    fn tracing_without_redactor_is_unchanged() {
        let sink = Arc::new(MemorySink::new());
        let tracer = Tracer::new().with_shared_sink(sink.clone());
        assert!(tracer.redactor().is_none());
        tracer
            .start("x", RunKind::Tool, None, json!({ "password": "hunter2" }))
            .finish(json!("ana@example.com"));
        let run = &sink.runs()[0];
        assert_eq!(run.inputs["password"], "hunter2");
        assert_eq!(run.outputs, Some(json!("ana@example.com")));
    }

    #[tokio::test]
    async fn tool_runs_carry_the_call_context_and_children_join_the_trace() {
        use sinew_core::identity::{CallContext, Via};
        use sinew_core::FnTool;

        let sink = Arc::new(MemorySink::new());
        let tracer = Tracer::new().with_shared_sink(sink.clone());
        let inner = tracer.clone();
        // A tool that starts its own root run (like a remote agent would).
        let tools = ToolBox::new().with(FnTool::new("lookup", "", json!({}), move |_| {
            inner
                .start("child", RunKind::Graph, None, json!({}))
                .finish(json!("ok"));
            Ok(json!("done"))
        }));
        let call = ToolCall {
            id: "c".into(),
            name: "lookup".into(),
            arguments: json!({}),
        };
        let ctx = CallContext::new("orchestrator")
            .user("ana")
            .delegate_to("billing", Via::A2a);
        let root = tracer.start("root", RunKind::Chain, None, json!({}));
        let root_ref = root.reference();
        with_context(
            ctx,
            traced_tool(&tools, &call, Some(&tracer), Some(&root_ref)),
        )
        .await;
        root.finish(json!(null));

        let runs = sink.runs();
        let child = runs.iter().find(|r| r.name == "child").unwrap();
        let tool = runs.iter().find(|r| r.name == "lookup").unwrap();
        assert_eq!(tool.metadata["identity"]["user"], "ana");
        assert_eq!(tool.metadata["identity"]["chain"][1]["agent"], "billing");
        assert_eq!(child.trace_id, root_ref.trace_id);
        assert_eq!(child.parent_id.as_deref(), Some(tool.id.as_str()));
        assert_eq!(child.metadata["identity"]["agent"], "billing");
        // Without a context nothing changes.
        assert!(runs
            .iter()
            .find(|r| r.name == "root")
            .unwrap()
            .metadata
            .get("identity")
            .is_none());
    }
}
