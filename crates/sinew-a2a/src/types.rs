//! A2A v0.3 data objects: the Agent Card, messages, parts, tasks, artifacts
//! and the streaming update events, with the exact JSON shape of the spec
//! (camelCase fields and `kind` discriminators).
//!
//! Every struct that carries a `kind` in the spec (`Task`, `Message`,
//! `TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`) writes it on
//! serialization and tolerates its absence on input. The unions
//! ([`SendMessageResult`], [`StreamResponse`]) dispatch on `kind`.

use std::collections::BTreeMap;

use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

/// Protocol version this crate implements (written into Agent Cards).
pub const PROTOCOL_VERSION: &str = "0.3.0";

/// Well-known path of the Agent Card (A2A 0.3).
pub const AGENT_CARD_PATH: &str = "/.well-known/agent-card.json";

/// Agent Card path used before A2A 0.3; servers also answer on it.
pub const LEGACY_AGENT_CARD_PATH: &str = "/.well-known/agent.json";

/// Free-form metadata object.
pub type Metadata = Map<String, Value>;

// ---------------------------------------------------------------------------
// Agent Card
// ---------------------------------------------------------------------------

/// Self-describing manifest of an agent, served at [`AGENT_CARD_PATH`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    pub protocol_version: String,
    pub name: String,
    pub description: String,
    /// Endpoint of the preferred transport (the JSON-RPC URL here).
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_transport: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_interfaces: Vec<AgentInterface>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icon_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<AgentProvider>,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documentation_url: Option<String>,
    #[serde(default)]
    pub capabilities: AgentCapabilities,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub security_schemes: BTreeMap<String, SecurityScheme>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub security: Vec<BTreeMap<String, Vec<String>>>,
    #[serde(default)]
    pub default_input_modes: Vec<String>,
    #[serde(default)]
    pub default_output_modes: Vec<String>,
    #[serde(default)]
    pub skills: Vec<AgentSkill>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_authenticated_extended_card: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub signatures: Vec<AgentCardSignature>,
}

impl AgentCard {
    /// A card for a JSON-RPC agent at `url` with streaming enabled, text
    /// input/output modes and no skills yet.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION.into(),
            name: name.into(),
            description: description.into(),
            url: url.into(),
            preferred_transport: Some(TRANSPORT_JSONRPC.into()),
            additional_interfaces: Vec::new(),
            icon_url: None,
            provider: None,
            version: "0.1.0".into(),
            documentation_url: None,
            capabilities: AgentCapabilities {
                streaming: Some(true),
                push_notifications: Some(false),
                state_transition_history: Some(false),
                extensions: Vec::new(),
            },
            security_schemes: BTreeMap::new(),
            security: Vec::new(),
            default_input_modes: vec!["text/plain".into()],
            default_output_modes: vec!["text/plain".into()],
            skills: Vec::new(),
            supports_authenticated_extended_card: None,
            signatures: Vec::new(),
        }
    }

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    pub fn with_skill(mut self, skill: AgentSkill) -> Self {
        self.skills.push(skill);
        self
    }

    pub fn with_provider(
        mut self,
        organization: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        self.provider = Some(AgentProvider {
            organization: organization.into(),
            url: url.into(),
        });
        self
    }

    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.capabilities.streaming = Some(streaming);
        self
    }

    /// Whether the agent declares `capabilities.streaming`.
    pub fn supports_streaming(&self) -> bool {
        self.capabilities.streaming.unwrap_or(false)
    }
}

/// Transport identifier for the JSON-RPC 2.0 binding.
pub const TRANSPORT_JSONRPC: &str = "JSONRPC";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentProvider {
    pub organization: String,
    pub url: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub push_notifications: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_transition_history: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<AgentExtension>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentExtension {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Metadata>,
}

/// Security scheme declared in the card (OpenAPI 3 style).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all_fields = "camelCase")]
pub enum SecurityScheme {
    #[serde(rename = "apiKey")]
    ApiKey {
        #[serde(rename = "in")]
        location: String,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    #[serde(rename = "http")]
    Http {
        scheme: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bearer_format: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    #[serde(rename = "oauth2")]
    OAuth2 {
        /// `OAuthFlows` object, kept as raw JSON.
        flows: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        oauth2_metadata_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    #[serde(rename = "openIdConnect")]
    OpenIdConnect {
        open_id_connect_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    #[serde(rename = "mutualTLS")]
    MutualTls {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

/// One capability of the agent, shown to clients and to delegating models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSkill {
    pub id: String,
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_modes: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_modes: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub security: Option<Vec<BTreeMap<String, Vec<String>>>>,
}

impl AgentSkill {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            tags: Vec::new(),
            examples: Vec::new(),
            input_modes: None,
            output_modes: None,
            security: None,
        }
    }

    pub fn with_tags<I, T>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_examples<I, T>(mut self, examples: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.examples = examples.into_iter().map(Into::into).collect();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentInterface {
    pub url: String,
    pub transport: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentCardSignature {
    pub protected: String,
    pub signature: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<Metadata>,
}

// ---------------------------------------------------------------------------
// Messages and parts
// ---------------------------------------------------------------------------

/// Sender of a [`Message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Agent,
}

/// One turn of communication between a client and an agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename = "message", rename_all = "camelCase")]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
    pub message_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reference_task_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

impl Message {
    /// A message with a fresh random `messageId`.
    pub fn new(role: Role, parts: Vec<Part>) -> Self {
        Self {
            role,
            parts,
            message_id: new_id(),
            task_id: None,
            context_id: None,
            reference_task_ids: Vec::new(),
            extensions: Vec::new(),
            metadata: None,
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::new(Role::User, vec![Part::text(text)])
    }

    pub fn agent_text(text: impl Into<String>) -> Self {
        Self::new(Role::Agent, vec![Part::text(text)])
    }

    /// Continue an existing task (e.g. one waiting in `input-required`).
    pub fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    pub fn with_context_id(mut self, context_id: impl Into<String>) -> Self {
        self.context_id = Some(context_id.into());
        self
    }

    /// Text of all parts joined with newlines (see [`parts_text`]).
    pub fn text(&self) -> String {
        parts_text(&self.parts)
    }
}

/// A piece of content: text, a file (inline bytes or a URI) or JSON data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Part {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Metadata>,
    },
    File {
        file: FileContent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Metadata>,
    },
    Data {
        data: Map<String, Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Metadata>,
    },
}

impl Part {
    pub fn text(text: impl Into<String>) -> Self {
        Part::Text {
            text: text.into(),
            metadata: None,
        }
    }

    /// A data part. Non-object values are wrapped as `{"value": ...}`, since
    /// the spec requires an object.
    pub fn data(value: Value) -> Self {
        let data = match value {
            Value::Object(m) => m,
            other => Map::from_iter([("value".to_string(), other)]),
        };
        Part::Data {
            data,
            metadata: None,
        }
    }

    pub fn file_uri(uri: impl Into<String>, mime_type: Option<String>) -> Self {
        Part::File {
            file: FileContent {
                name: None,
                mime_type,
                bytes: None,
                uri: Some(uri.into()),
            },
            metadata: None,
        }
    }

    /// `bytes_base64` must already be base64-encoded.
    pub fn file_bytes(bytes_base64: impl Into<String>, mime_type: Option<String>) -> Self {
        Part::File {
            file: FileContent {
                name: None,
                mime_type,
                bytes: Some(bytes_base64.into()),
                uri: None,
            },
            metadata: None,
        }
    }

    /// The text of a text part.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Part::Text { text, .. } => Some(text),
            _ => None,
        }
    }
}

/// A file, either inline (`bytes`, base64) or by reference (`uri`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileContent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

/// Human/LLM-readable rendering of parts: text as is, data parts as compact
/// JSON, files as a short `[file name mime uri]` summary (never raw base64).
/// Items are joined with newlines, except consecutive text parts which are
/// concatenated (so streamed text chunks read naturally).
pub fn parts_text(parts: &[Part]) -> String {
    let mut out = String::new();
    let mut last_text = false;
    for p in parts {
        let (piece, is_text) = match p {
            Part::Text { text, .. } => (text.clone(), true),
            Part::Data { data, .. } => (Value::Object(data.clone()).to_string(), false),
            Part::File { file, .. } => {
                let mut s = String::from("[file");
                for v in [&file.name, &file.mime_type, &file.uri]
                    .into_iter()
                    .flatten()
                {
                    s.push(' ');
                    s.push_str(v);
                }
                if let Some(b) = &file.bytes {
                    s.push_str(&format!(" {} bytes base64", b.len()));
                }
                s.push(']');
                (s, false)
            }
        };
        if !out.is_empty() && (!is_text || !last_text) {
            out.push('\n');
        }
        out.push_str(&piece);
        last_text = is_text;
    }
    out
}

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

/// Lifecycle state of a [`Task`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskState {
    Submitted,
    Working,
    InputRequired,
    Completed,
    Canceled,
    Failed,
    Rejected,
    AuthRequired,
    Unknown,
}

impl TaskState {
    /// Completed, canceled, failed or rejected: the task can't continue.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskState::Completed | TaskState::Canceled | TaskState::Failed | TaskState::Rejected
        )
    }

    /// The agent waits for the client (`input-required`, `auth-required`).
    pub fn is_interrupted(self) -> bool {
        matches!(self, TaskState::InputRequired | TaskState::AuthRequired)
    }

    /// Terminal or interrupted: a stream for this turn ends here
    /// (`final: true`).
    pub fn ends_turn(self) -> bool {
        self.is_terminal() || self.is_interrupted()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskStatus {
    pub state: TaskState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    /// ISO 8601 timestamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl TaskStatus {
    /// A status stamped with the current UTC time.
    pub fn new(state: TaskState, message: Option<Message>) -> Self {
        Self {
            state,
            message,
            timestamp: Some(now_rfc3339()),
        }
    }
}

/// A stateful unit of work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename = "task", rename_all = "camelCase")]
pub struct Task {
    pub id: String,
    pub context_id: String,
    pub status: TaskStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

impl Task {
    /// Text of every artifact, in order (see [`parts_text`]).
    pub fn artifacts_text(&self) -> String {
        self.artifacts
            .iter()
            .map(|a| parts_text(&a.parts))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Keeps only the last `n` history messages.
    pub fn truncate_history(&mut self, n: usize) {
        let len = self.history.len();
        if len > n {
            self.history.drain(..len - n);
        }
    }
}

/// An output of a task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Artifact {
    pub artifact_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parts: Vec<Part>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<String>,
}

impl Artifact {
    pub fn new(artifact_id: impl Into<String>, parts: Vec<Part>) -> Self {
        Self {
            artifact_id: artifact_id.into(),
            name: None,
            description: None,
            parts,
            metadata: None,
            extensions: Vec::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Appends parts, merging a text part into a preceding text part so a
    /// streamed artifact stays one growing string instead of thousands of
    /// tiny parts.
    pub fn append_parts(&mut self, parts: Vec<Part>) {
        for p in parts {
            match (self.parts.last_mut(), p) {
                (
                    Some(Part::Text {
                        text,
                        metadata: None,
                    }),
                    Part::Text {
                        text: more,
                        metadata: None,
                    },
                ) => text.push_str(&more),
                (_, p) => self.parts.push(p),
            }
        }
    }
}

/// Streamed when a task's status changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename = "status-update", rename_all = "camelCase")]
pub struct TaskStatusUpdateEvent {
    pub task_id: String,
    pub context_id: String,
    pub status: TaskStatus,
    /// Last event of this stream.
    #[serde(rename = "final", default)]
    pub is_final: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

/// Streamed when an artifact is produced or extended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename = "artifact-update", rename_all = "camelCase")]
pub struct TaskArtifactUpdateEvent {
    pub task_id: String,
    pub context_id: String,
    pub artifact: Artifact,
    /// Append these parts to the artifact with the same id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub append: Option<bool>,
    /// This is the last chunk of the artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_chunk: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

// ---------------------------------------------------------------------------
// Method parameters
// ---------------------------------------------------------------------------

/// Params of `message/send` and `message/stream`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageSendParams {
    pub message: Message,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<MessageSendConfiguration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

impl MessageSendParams {
    pub fn new(message: Message) -> Self {
        Self {
            message,
            configuration: None,
            metadata: None,
        }
    }

    /// `blocking: false`: `message/send` returns as soon as the task exists
    /// instead of waiting for it to finish.
    pub fn non_blocking(mut self) -> Self {
        self.configuration
            .get_or_insert_with(Default::default)
            .blocking = Some(false);
        self
    }

    pub fn with_history_length(mut self, n: usize) -> Self {
        self.configuration
            .get_or_insert_with(Default::default)
            .history_length = Some(n);
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageSendConfiguration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_output_modes: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history_length: Option<usize>,
    /// Push notifications are not supported by the Sinew server; kept as raw
    /// JSON so requests that carry it still parse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub push_notification_config: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocking: Option<bool>,
}

/// Params of `tasks/cancel` and `tasks/resubscribe`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskIdParams {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

/// Params of `tasks/get`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskQueryParams {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

// ---------------------------------------------------------------------------
// Unions
// ---------------------------------------------------------------------------

/// Result of `message/send`: a task, or a direct message reply.
#[derive(Debug, Clone, PartialEq)]
pub enum SendMessageResult {
    Task(Task),
    Message(Message),
}

impl SendMessageResult {
    /// The agent's answer as text: the message text, or the task's artifacts,
    /// or its status message when there are no artifacts.
    pub fn text(&self) -> String {
        match self {
            SendMessageResult::Message(m) => m.text(),
            SendMessageResult::Task(t) => task_text(t),
        }
    }

    pub fn as_task(&self) -> Option<&Task> {
        match self {
            SendMessageResult::Task(t) => Some(t),
            SendMessageResult::Message(_) => None,
        }
    }
}

/// Answer text of a task: its artifacts, else its status message.
pub fn task_text(t: &Task) -> String {
    let text = t.artifacts_text();
    if !text.is_empty() {
        return text;
    }
    t.status
        .message
        .as_ref()
        .map(Message::text)
        .unwrap_or_default()
}

/// One event of `message/stream` or `tasks/resubscribe`.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamResponse {
    Task(Task),
    Message(Message),
    StatusUpdate(TaskStatusUpdateEvent),
    ArtifactUpdate(TaskArtifactUpdateEvent),
}

impl StreamResponse {
    /// Whether the server closes the stream after this event: a direct
    /// message, a `final` status update, or a task snapshot whose state ends
    /// the turn.
    pub fn is_final(&self) -> bool {
        match self {
            StreamResponse::Message(_) => true,
            StreamResponse::StatusUpdate(s) => s.is_final,
            StreamResponse::Task(t) => t.status.state.ends_turn(),
            StreamResponse::ArtifactUpdate(_) => false,
        }
    }

    /// Text carried by this event, if any (artifact chunk text, status
    /// message text, message text).
    pub fn text(&self) -> Option<String> {
        let s = match self {
            StreamResponse::ArtifactUpdate(a) => parts_text(&a.artifact.parts),
            StreamResponse::StatusUpdate(s) => s.status.message.as_ref()?.text(),
            StreamResponse::Message(m) => m.text(),
            StreamResponse::Task(_) => return None,
        };
        Some(s)
    }
}

impl From<Task> for StreamResponse {
    fn from(t: Task) -> Self {
        StreamResponse::Task(t)
    }
}
impl From<Message> for StreamResponse {
    fn from(m: Message) -> Self {
        StreamResponse::Message(m)
    }
}
impl From<TaskStatusUpdateEvent> for StreamResponse {
    fn from(e: TaskStatusUpdateEvent) -> Self {
        StreamResponse::StatusUpdate(e)
    }
}
impl From<TaskArtifactUpdateEvent> for StreamResponse {
    fn from(e: TaskArtifactUpdateEvent) -> Self {
        StreamResponse::ArtifactUpdate(e)
    }
}

fn kind_of(v: &Map<String, Value>) -> Option<&str> {
    v.get("kind").and_then(Value::as_str)
}

impl Serialize for SendMessageResult {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            SendMessageResult::Task(t) => t.serialize(s),
            SendMessageResult::Message(m) => m.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for SendMessageResult {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let map = Map::<String, Value>::deserialize(d)?;
        let v = match kind_of(&map) {
            Some("task") => SendMessageResult::Task(from_map(map)?),
            Some("message") => SendMessageResult::Message(from_map(map)?),
            // Lenient: tasks have `status`, messages have `parts`.
            None if map.contains_key("status") => SendMessageResult::Task(from_map(map)?),
            None if map.contains_key("parts") => SendMessageResult::Message(from_map(map)?),
            other => {
                return Err(D::Error::custom(format!(
                    "expected a task or a message, got kind {other:?}"
                )))
            }
        };
        Ok(v)
    }
}

impl Serialize for StreamResponse {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            StreamResponse::Task(t) => t.serialize(s),
            StreamResponse::Message(m) => m.serialize(s),
            StreamResponse::StatusUpdate(e) => e.serialize(s),
            StreamResponse::ArtifactUpdate(e) => e.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for StreamResponse {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let map = Map::<String, Value>::deserialize(d)?;
        let v = match kind_of(&map) {
            Some("artifact-update") => StreamResponse::ArtifactUpdate(from_map(map)?),
            Some("status-update") => StreamResponse::StatusUpdate(from_map(map)?),
            Some("task") => StreamResponse::Task(from_map(map)?),
            Some("message") => StreamResponse::Message(from_map(map)?),
            None if map.contains_key("artifact") => StreamResponse::ArtifactUpdate(from_map(map)?),
            None if map.contains_key("taskId") && map.contains_key("status") => {
                StreamResponse::StatusUpdate(from_map(map)?)
            }
            None if map.contains_key("status") => StreamResponse::Task(from_map(map)?),
            None if map.contains_key("parts") => StreamResponse::Message(from_map(map)?),
            other => {
                return Err(D::Error::custom(format!(
                    "unknown stream event kind {other:?}"
                )))
            }
        };
        Ok(v)
    }
}

fn from_map<T: serde::de::DeserializeOwned, E: serde::de::Error>(
    map: Map<String, Value>,
) -> Result<T, E> {
    serde_json::from_value(Value::Object(map)).map_err(E::custom)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A fresh random id (UUID v4).
pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Current UTC time as RFC 3339 / ISO 8601 with milliseconds,
/// e.g. `2025-04-02T16:59:25.331Z`.
pub fn now_rfc3339() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60,
        now.subsec_millis()
    )
}
