//! Record and replay model calls and tool results ("VCR cassettes") for
//! deterministic tests without hand-written mocks.
//!
//! Run a test once against a real model in record mode; the requests,
//! responses (or stream events) and tool results are saved to a JSON
//! cassette. Later runs replay the cassette offline and fail loudly if the
//! code sends a different request, so a changed prompt or tool schema is
//! caught instead of silently answered with a stale response.
//!
//! ```
//! # async fn demo() -> sinew_core::Result<()> {
//! use std::sync::Arc;
//! use sinew_core::{vcr::Vcr, ChatModel, ChatRequest, Message, mock::ScriptedModel};
//!
//! let path = std::env::temp_dir().join("sinew-vcr-doc.json");
//! # let _ = std::fs::remove_file(&path);
//! // Record (here against a scripted model; in real tests, a provider).
//! let live = Arc::new(ScriptedModel::new([Message::assistant("Hi there")]));
//! let vcr = Vcr::record(&path);
//! let model = vcr.model(live);
//! model.chat(ChatRequest::new(vec![Message::user("Hello")])).await?;
//! vcr.save()?;
//!
//! // Replay: no model needed, same answer.
//! let vcr = Vcr::replay(&path)?;
//! let model = vcr.model(Arc::new(ScriptedModel::new([])));
//! let r = model.chat(ChatRequest::new(vec![Message::user("Hello")])).await?;
//! assert_eq!(r.message.content, "Hi there");
//! # Ok(()) }
//! ```
//!
//! [`Vcr::from_env`] picks the mode from `SINEW_VCR` (`record`, `replay`,
//! or `auto`: replay when the cassette exists, record otherwise).

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    stream::{events_from_response, ChatStream, StreamAccumulator},
    ChatModel, ChatRequest, ChatResponse, Error, Result, StreamEvent, Tool, ToolBox, ToolSpec,
};

/// Cassette file format version.
pub const CASSETTE_VERSION: u32 = 1;

/// Everything recorded for one test: model interactions and tool calls, in
/// order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Cassette {
    pub version: u32,
    #[serde(default)]
    pub interactions: Vec<Interaction>,
    #[serde(default)]
    pub tool_calls: Vec<ToolInteraction>,
}

impl Cassette {
    pub fn new() -> Self {
        Self {
            version: CASSETTE_VERSION,
            ..Default::default()
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("cannot read cassette {}: {e}", path.display())))?;
        let c: Cassette = serde_json::from_str(&text)?;
        if c.version != CASSETTE_VERSION {
            return Err(Error::Config(format!(
                "cassette {} has version {}, expected {CASSETTE_VERSION}",
                path.display(),
                c.version
            )));
        }
        Ok(c)
    }

    /// Writes pretty JSON (stable key order) so cassettes diff well in review.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)
                .map_err(|e| Error::Config(format!("cannot create {}: {e}", dir.display())))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text + "\n")
            .map_err(|e| Error::Config(format!("cannot write cassette {}: {e}", path.display())))
    }
}

/// One model call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    /// `ChatModel::name` of the recorded model.
    pub model: String,
    pub request: ChatRequest,
    /// Set for `chat` calls (and for streams, folded from the events).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<ChatResponse>,
    /// Set for `chat_stream` calls: the events as received.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<Vec<StreamEvent>>,
    /// Set when the call failed; replay returns `Error::Other` with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// One tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInteraction {
    pub name: String,
    pub arguments: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Record or replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VcrMode {
    Record,
    Replay,
}

/// How replayed requests are compared with recorded ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatchRequests {
    /// The whole request must be equal: messages, tools, options (default).
    #[default]
    Exact,
    /// Only the messages must be equal (tool schemas and options may change).
    Messages,
    /// Replay in order without comparing.
    Sequential,
}

struct Inner {
    mode: VcrMode,
    path: PathBuf,
    matching: MatchRequests,
    cassette: Cassette,
    next_model: usize,
    next_tool: usize,
}

/// A recorder/player shared by the models and tools of one test. Cheap to
/// clone.
#[derive(Clone)]
pub struct Vcr {
    inner: Arc<Mutex<Inner>>,
}

impl Vcr {
    /// Starts an empty cassette that [`save`](Self::save) writes to `path`.
    /// Wrapped models and tools call through and record.
    pub fn record(path: impl Into<PathBuf>) -> Self {
        Self::with(VcrMode::Record, path.into(), Cassette::new())
    }

    /// Loads `path`; wrapped models and tools answer from it and never call
    /// through.
    pub fn replay(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let cassette = Cassette::load(&path)?;
        Ok(Self::with(VcrMode::Replay, path, cassette))
    }

    /// Replays an in-memory cassette, e.g. one exported from a trace with
    /// `sinew_trace::cassette_from_runs` or `GET /api/traces/{id}/cassette`
    /// in sinew-scope.
    pub fn from_cassette(cassette: Cassette) -> Self {
        Self::with(VcrMode::Replay, PathBuf::from("<memory>"), cassette)
    }

    /// Mode from the `SINEW_VCR` environment variable: `record`, `replay`,
    /// or `auto` / unset (replay if the cassette exists, else record).
    pub fn from_env(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        match std::env::var("SINEW_VCR").as_deref() {
            Ok("record") => Ok(Self::record(path)),
            Ok("replay") => Self::replay(path),
            Ok("auto") | Err(_) => {
                if path.exists() {
                    Self::replay(path)
                } else {
                    Ok(Self::record(path))
                }
            }
            Ok(other) => Err(Error::Config(format!(
                "SINEW_VCR must be record, replay or auto, not `{other}`"
            ))),
        }
    }

    fn with(mode: VcrMode, path: PathBuf, cassette: Cassette) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                mode,
                path,
                matching: MatchRequests::default(),
                cassette,
                next_model: 0,
                next_tool: 0,
            })),
        }
    }

    /// How replayed requests are compared (default [`MatchRequests::Exact`]).
    pub fn matching(self, m: MatchRequests) -> Self {
        self.inner.lock().unwrap().matching = m;
        self
    }

    pub fn mode(&self) -> VcrMode {
        self.inner.lock().unwrap().mode
    }

    pub fn is_recording(&self) -> bool {
        self.mode() == VcrMode::Record
    }

    /// A copy of the cassette as recorded (or loaded) so far.
    pub fn cassette(&self) -> Cassette {
        self.inner.lock().unwrap().cassette.clone()
    }

    /// Writes the cassette when recording; a no-op when replaying.
    pub fn save(&self) -> Result<()> {
        let g = self.inner.lock().unwrap();
        match g.mode {
            VcrMode::Record => g.cassette.save(&g.path),
            VcrMode::Replay => Ok(()),
        }
    }

    /// In replay mode, fails unless every recorded interaction was used:
    /// catches code that stopped making calls it used to make.
    pub fn assert_all_played(&self) -> Result<()> {
        let g = self.inner.lock().unwrap();
        if g.mode == VcrMode::Record {
            return Ok(());
        }
        let (m, t) = (g.cassette.interactions.len(), g.cassette.tool_calls.len());
        if g.next_model < m || g.next_tool < t {
            return Err(Error::Other(format!(
                "cassette {}: {} of {m} model calls and {} of {t} tool calls were replayed",
                g.path.display(),
                g.next_model,
                g.next_tool
            )));
        }
        Ok(())
    }

    /// Wraps a model: records its calls, or replays them (then `model` is
    /// never called and can be any placeholder).
    pub fn model(&self, model: Arc<dyn ChatModel>) -> Arc<dyn ChatModel> {
        Arc::new(VcrModel {
            vcr: self.clone(),
            inner: model,
        })
    }

    /// Wraps a tool the same way.
    pub fn tool(&self, tool: Arc<dyn Tool>) -> Arc<dyn Tool> {
        Arc::new(VcrTool {
            vcr: self.clone(),
            inner: tool,
        })
    }

    /// Wraps every tool of a toolbox.
    pub fn toolbox(&self, tools: &ToolBox) -> ToolBox {
        let mut out = ToolBox::new();
        for t in tools.tools() {
            out.register_arc(self.tool(t.clone()));
        }
        out
    }

    fn push(&self, i: Interaction) {
        self.inner.lock().unwrap().cassette.interactions.push(i);
    }

    fn next_interaction(&self, model: &str, req: &ChatRequest) -> Result<Interaction> {
        let mut g = self.inner.lock().unwrap();
        let n = g.next_model;
        let path = g.path.display().to_string();
        let Some(rec) = g.cassette.interactions.get(n).cloned() else {
            return Err(Error::Other(format!(
                "cassette {path}: no recorded model call #{} for `{model}` (re-record with SINEW_VCR=record)",
                n + 1
            )));
        };
        if let Some(diff) = request_diff(g.matching, &rec.request, req) {
            return Err(Error::Other(format!(
                "cassette {path}: model call #{} differs from the recording: {diff} \
                 (re-record with SINEW_VCR=record)",
                n + 1
            )));
        }
        g.next_model += 1;
        Ok(rec)
    }
}

/// Explains the first difference between two requests, or `None` if they
/// match under `m`.
fn request_diff(m: MatchRequests, recorded: &ChatRequest, got: &ChatRequest) -> Option<String> {
    if m == MatchRequests::Sequential {
        return None;
    }
    let (a, b) = (
        serde_json::to_value(&recorded.messages).unwrap_or_default(),
        serde_json::to_value(&got.messages).unwrap_or_default(),
    );
    if a != b {
        let (ra, rb) = (
            a.as_array().cloned().unwrap_or_default(),
            b.as_array().cloned().unwrap_or_default(),
        );
        if ra.len() != rb.len() {
            return Some(format!("recorded {} messages, got {}", ra.len(), rb.len()));
        }
        let i = ra.iter().zip(&rb).position(|(x, y)| x != y).unwrap_or(0);
        return Some(format!(
            "message {i} changed: recorded {} got {}",
            short(&ra[i]),
            short(&rb[i])
        ));
    }
    if m == MatchRequests::Exact {
        let (a, b) = (
            serde_json::to_value(recorded).unwrap_or_default(),
            serde_json::to_value(got).unwrap_or_default(),
        );
        if a != b {
            for key in [
                "tools",
                "temperature",
                "max_tokens",
                "tool_choice",
                "response_format",
            ] {
                if a.get(key) != b.get(key) {
                    return Some(format!("`{key}` changed"));
                }
            }
            return Some("request options changed".into());
        }
    }
    None
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() > 160 {
        format!("{}…", s.chars().take(160).collect::<String>())
    } else {
        s
    }
}

struct VcrModel {
    vcr: Vcr,
    inner: Arc<dyn ChatModel>,
}

#[async_trait]
impl ChatModel for VcrModel {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn capabilities(&self) -> crate::Capabilities {
        self.inner.capabilities()
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        match self.vcr.mode() {
            VcrMode::Record => {
                let res = self.inner.chat(request.clone()).await;
                self.vcr.push(Interaction {
                    model: self.inner.name().to_string(),
                    request,
                    response: res.as_ref().ok().cloned(),
                    stream: None,
                    error: res.as_ref().err().map(ToString::to_string),
                });
                res
            }
            VcrMode::Replay => {
                let rec = self.vcr.next_interaction(self.inner.name(), &request)?;
                if let Some(e) = rec.error {
                    return Err(Error::Other(e));
                }
                match (rec.response, rec.stream) {
                    (Some(r), _) => Ok(r),
                    (None, Some(events)) => {
                        let mut acc = StreamAccumulator::new();
                        events.iter().for_each(|e| acc.push(e));
                        Ok(acc.finish())
                    }
                    (None, None) => {
                        Err(Error::Other("cassette interaction has no response".into()))
                    }
                }
            }
        }
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        match self.vcr.mode() {
            VcrMode::Record => {
                let stream = match self.inner.chat_stream(request.clone()).await {
                    Ok(s) => s,
                    Err(e) => {
                        self.vcr.push(Interaction {
                            model: self.inner.name().to_string(),
                            request,
                            response: None,
                            stream: None,
                            error: Some(e.to_string()),
                        });
                        return Err(e);
                    }
                };
                let state = Recording {
                    vcr: self.vcr.clone(),
                    model: self.inner.name().to_string(),
                    request: Some(request),
                    events: Vec::new(),
                    error: None,
                };
                let s = futures_util::stream::unfold(
                    (stream, state),
                    |(mut stream, mut state)| async move {
                        match stream.next().await {
                            Some(item) => {
                                match &item {
                                    Ok(ev) => state.events.push(ev.clone()),
                                    Err(e) => state.error = Some(e.to_string()),
                                }
                                Some((item, (stream, state)))
                            }
                            None => {
                                state.finish();
                                None
                            }
                        }
                    },
                );
                Ok(Box::pin(s))
            }
            VcrMode::Replay => {
                let rec = self.vcr.next_interaction(self.inner.name(), &request)?;
                if let Some(e) = rec.error.filter(|_| rec.stream.is_none()) {
                    return Err(Error::Other(e));
                }
                let events = match (rec.stream, rec.response) {
                    (Some(events), _) => events,
                    (None, Some(r)) => events_from_response(r),
                    (None, None) => {
                        return Err(Error::Other("cassette interaction has no response".into()))
                    }
                };
                Ok(Box::pin(futures_util::stream::iter(
                    events.into_iter().map(Ok),
                )))
            }
        }
    }
}

/// Collects a recorded stream; written when the stream ends (or is dropped
/// early, so cancelled streams are recorded too).
struct Recording {
    vcr: Vcr,
    model: String,
    request: Option<ChatRequest>,
    events: Vec<StreamEvent>,
    error: Option<String>,
}

impl Recording {
    fn finish(&mut self) {
        let Some(request) = self.request.take() else {
            return;
        };
        let events = std::mem::take(&mut self.events);
        let mut acc = StreamAccumulator::new();
        events.iter().for_each(|e| acc.push(e));
        self.vcr.push(Interaction {
            model: std::mem::take(&mut self.model),
            request,
            response: Some(acc.finish()),
            stream: Some(events),
            error: self.error.take(),
        });
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        self.finish();
    }
}

struct VcrTool {
    vcr: Vcr,
    inner: Arc<dyn Tool>,
}

#[async_trait]
impl Tool for VcrTool {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }

    async fn call(&self, arguments: Value) -> Result<Value> {
        let name = self.inner.spec().name;
        match self.vcr.mode() {
            VcrMode::Record => {
                let res = self.inner.call(arguments.clone()).await;
                self.vcr
                    .inner
                    .lock()
                    .unwrap()
                    .cassette
                    .tool_calls
                    .push(ToolInteraction {
                        name,
                        arguments,
                        result: res.as_ref().ok().cloned(),
                        error: res.as_ref().err().map(|e| match e {
                            Error::Tool { message, .. } => message.clone(),
                            other => other.to_string(),
                        }),
                    });
                res
            }
            VcrMode::Replay => {
                let mut g = self.vcr.inner.lock().unwrap();
                let path = g.path.display().to_string();
                // Tools may run concurrently, so match by name and arguments
                // among the calls not replayed yet, in recorded order.
                let start = g.next_tool;
                let pos = g.cassette.tool_calls[start..]
                    .iter()
                    .position(|t| t.name == name && t.arguments == arguments)
                    .map(|p| p + start);
                let Some(pos) = pos else {
                    return Err(Error::Tool {
                        name,
                        message: format!(
                            "cassette {path}: no recorded call with arguments {} \
                             (re-record with SINEW_VCR=record)",
                            short(&arguments)
                        ),
                    });
                };
                let rec = g.cassette.tool_calls.remove(pos);
                g.cassette.tool_calls.insert(start, rec.clone());
                g.next_tool += 1;
                match (rec.result, rec.error) {
                    (_, Some(e)) => Err(Error::Tool { name, message: e }),
                    (Some(v), None) => Ok(v),
                    (None, None) => Ok(Value::Null),
                }
            }
        }
    }
}
