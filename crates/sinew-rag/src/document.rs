use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Arbitrary JSON metadata attached to a [`Document`].
pub type Metadata = Map<String, Value>;

/// A piece of text plus metadata: what loaders produce, splitters cut,
/// vector stores index and retrievers return.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Document {
    /// Stable id. Stores generate one (a UUID) when it is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The text (LangChain's `page_content`).
    pub content: String,
    #[serde(default)]
    pub metadata: Metadata,
}

impl Document {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: None,
            content: content.into(),
            metadata: Metadata::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn with_metadata_map(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// The `source` metadata field (a path or URL), if it is a string.
    pub fn source(&self) -> Option<&str> {
        self.metadata.get("source").and_then(Value::as_str)
    }

    /// Key used to de-duplicate documents coming from several retrievers:
    /// the id when set, otherwise the content.
    pub(crate) fn dedup_key(&self) -> &str {
        self.id.as_deref().unwrap_or(&self.content)
    }
}

impl From<&str> for Document {
    fn from(s: &str) -> Self {
        Document::new(s)
    }
}

impl From<String> for Document {
    fn from(s: String) -> Self {
        Document::new(s)
    }
}

/// Assigns a fresh UUID to every document without an id and returns all ids.
pub(crate) fn ensure_ids(docs: &mut [Document]) -> Vec<String> {
    docs.iter_mut()
        .map(|d| {
            d.id.get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
                .clone()
        })
        .collect()
}
