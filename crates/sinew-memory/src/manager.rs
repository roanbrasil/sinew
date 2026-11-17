//! [`MemoryManager`]: extraction, consolidation, recall and deletion.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{ChatModel, ChatRequest, Message, Role, StructuredOutput};
use sinew_graph::store::{cosine, PutOptions, SearchQuery, Store, StoreError};
use sinew_rag::Embeddings;

use crate::memory::{
    now_millis, Memory, MemoryError, MemoryKind, RememberReport, Result, ScoredMemory,
};

/// System prompt of the extraction call (first model call of
/// [`MemoryManager::remember`]).
pub const DEFAULT_EXTRACTION_PROMPT: &str = "\
You maintain the long-term memory of an assistant. Read the conversation and \
extract what is worth remembering about the user in future conversations:
- fact: something true about the user or their world (name, job, family, location, projects, tools they use);
- preference: a stable like, dislike or way they want to be helped (language, tone, format, diet);
- episode: an event or decision that may matter later (a trip, a purchase, a deadline, an outcome).
Write each memory as one short, self-contained sentence in the third person (\"The user ...\"), \
in the language of the conversation. Set importance from 0 (trivial) to 1 (essential). \
Only use what the user said or clearly confirmed; ignore the assistant's own suggestions, \
small talk, one-off requests and anything sensitive the user did not ask to be remembered. \
If the user asks to forget something, extract the corrected statement (for example \
\"The user no longer wants to be called Bob\"). Return an empty list when nothing is worth remembering.";

/// System prompt of the consolidation call (second model call of
/// [`MemoryManager::remember`], made only when existing memories are similar
/// to the new facts).
pub const DEFAULT_UPDATE_PROMPT: &str = "\
You keep a user's long-term memory consistent. You receive the existing memories most \
similar to some new facts, each with an id, and the new facts. Return one operation per \
new fact and per existing memory that must change:
- add: the fact is new information not covered by any existing memory; give its content.
- update: an existing memory is about the same subject and the fact refines or corrects it; \
give the existing id and the merged content (one sentence, keeping still-valid details).
- delete: the fact shows an existing memory is no longer true and nothing should replace it, \
or the user asked to forget it; give the existing id.
- none: the fact is already fully covered by an existing memory.
Use only ids from the existing memories. Never copy a fact that duplicates an existing memory as add.";

/// First line of the block produced by [`MemoryManager::context`].
pub const DEFAULT_CONTEXT_HEADER: &str = "\
What you remember about the user from earlier conversations (use it when relevant):";

/// Default namespace prefix: memories of user `u` live under `["memories", u]`.
pub const DEFAULT_NAMESPACE: &str = "memories";

/// The value stored for each memory: the memory plus its embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Record {
    #[serde(flatten)]
    memory: Memory,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding_model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Extracted {
    #[serde(default)]
    memories: Vec<NewFact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NewFact {
    content: String,
    #[serde(default)]
    kind: MemoryKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    importance: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct Operations {
    #[serde(default)]
    operations: Vec<Operation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Action {
    Add,
    Update,
    Delete,
    None,
}

#[derive(Debug, Deserialize)]
struct Operation {
    action: Action,
    #[serde(default)]
    id: Option<Value>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    kind: Option<MemoryKind>,
    #[serde(default)]
    importance: Option<f32>,
}

fn kind_schema() -> Value {
    json!({ "type": "string", "enum": ["fact", "preference", "episode"] })
}

fn extraction_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": { "type": "string" },
                        "kind": kind_schema(),
                        "importance": { "type": "number" }
                    },
                    "required": ["content", "kind"]
                }
            }
        },
        "required": ["memories"]
    })
}

fn update_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": { "type": "string", "enum": ["add", "update", "delete", "none"] },
                        "id": { "type": "string" },
                        "content": { "type": "string" },
                        "kind": kind_schema(),
                        "importance": { "type": "number" }
                    },
                    "required": ["action"]
                }
            }
        },
        "required": ["operations"]
    })
}

/// Long-term memory over any [`Store`]: one namespace per user,
/// model-driven extraction and consolidation, semantic or keyword recall.
///
/// See the [crate docs](crate) for the full flow.
///
/// ```
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// use std::sync::Arc;
/// use sinew_core::mock::ScriptedModel;
/// use sinew_graph::InMemoryStore;
/// use sinew_memory::{MemoryKind, MemoryManager};
///
/// let model = Arc::new(ScriptedModel::new([]));
/// let memory = MemoryManager::new(Arc::new(InMemoryStore::new()), model);
/// memory.add("ana", "The user prefers answers in Portuguese.", MemoryKind::Preference).await?;
/// memory.add("ana", "The user has a cat named Miso.", MemoryKind::Fact).await?;
///
/// let hits = memory.recall("ana", "which language should answers use?", 1).await?;
/// assert_eq!(hits[0].memory.kind, MemoryKind::Preference);
/// assert!(memory.recall("bob", "language", 5).await?.is_empty());
/// # Ok::<_, sinew_memory::MemoryError>(()) }).unwrap();
/// ```
#[derive(Clone)]
pub struct MemoryManager {
    store: Arc<dyn Store>,
    model: Arc<dyn ChatModel>,
    embeddings: Option<Arc<dyn Embeddings>>,
    prefix: Vec<String>,
    extraction_prompt: String,
    update_prompt: String,
    context_header: String,
    candidates: usize,
    min_score: Option<f32>,
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("model", &self.model.name())
            .field(
                "embeddings",
                &self.embeddings.as_ref().map(|e| e.model().to_string()),
            )
            .field("prefix", &self.prefix)
            .field("candidates", &self.candidates)
            .field("min_score", &self.min_score)
            .finish()
    }
}

impl MemoryManager {
    /// Memories go to `store`; `model` extracts and consolidates them in
    /// [`remember`](Self::remember). Without
    /// [`with_embeddings`](Self::with_embeddings), recall ranks by keyword
    /// overlap.
    pub fn new(store: Arc<dyn Store>, model: Arc<dyn ChatModel>) -> Self {
        Self {
            store,
            model,
            embeddings: None,
            prefix: vec![DEFAULT_NAMESPACE.to_string()],
            extraction_prompt: DEFAULT_EXTRACTION_PROMPT.to_string(),
            update_prompt: DEFAULT_UPDATE_PROMPT.to_string(),
            context_header: DEFAULT_CONTEXT_HEADER.to_string(),
            candidates: 5,
            min_score: None,
        }
    }

    /// Ranks by cosine similarity of embeddings. Each memory's vector is
    /// stored next to it with the model name; memories saved without a
    /// vector, or with another model's, are embedded again on first use.
    pub fn with_embeddings(self, embeddings: impl Embeddings + 'static) -> Self {
        self.with_embeddings_arc(Arc::new(embeddings))
    }

    pub fn with_embeddings_arc(mut self, embeddings: Arc<dyn Embeddings>) -> Self {
        self.embeddings = Some(embeddings);
        self
    }

    /// Namespace segments before the user id (default `["memories"]`).
    pub fn with_namespace_prefix<I, T>(mut self, prefix: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.prefix = prefix.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces [`DEFAULT_EXTRACTION_PROMPT`].
    pub fn with_extraction_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.extraction_prompt = prompt.into();
        self
    }

    /// Replaces [`DEFAULT_UPDATE_PROMPT`].
    pub fn with_update_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.update_prompt = prompt.into();
        self
    }

    /// Replaces [`DEFAULT_CONTEXT_HEADER`].
    pub fn with_context_header(mut self, header: impl Into<String>) -> Self {
        self.context_header = header.into();
        self
    }

    /// How many similar existing memories are shown to the model per new
    /// fact during consolidation (default 5).
    pub fn with_candidates(mut self, n: usize) -> Self {
        self.candidates = n.max(1);
        self
    }

    /// Drops recall hits scoring below `score` (cosine similarity with
    /// embeddings, keyword overlap from 0 to 1 without). Keyword recall
    /// always drops memories sharing no word with the query.
    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    pub fn store(&self) -> &Arc<dyn Store> {
        &self.store
    }

    pub fn model(&self) -> &Arc<dyn ChatModel> {
        &self.model
    }

    /// The store namespace of `user_id`: the prefix followed by the id.
    pub fn namespace(&self, user_id: &str) -> Vec<String> {
        let mut ns = self.prefix.clone();
        ns.push(user_id.to_string());
        ns
    }

    /// Saves `content` as a new memory without calling the model (no
    /// deduplication).
    pub async fn add(&self, user_id: &str, content: &str, kind: MemoryKind) -> Result<Memory> {
        let memory = Memory::new(content.trim(), kind);
        self.put(user_id, memory.clone()).await?;
        Ok(memory)
    }

    /// Saves `memory` as is (insert or replace by id), embedding it when
    /// embeddings are configured.
    pub async fn put(&self, user_id: &str, memory: Memory) -> Result<()> {
        let ns = self.namespace(user_id);
        let mut vectors = self.embed(std::slice::from_ref(&memory.content)).await?;
        self.write(&ns, memory, vectors.as_mut().and_then(Vec::pop))
            .await
    }

    pub async fn get(&self, user_id: &str, id: &str) -> Result<Option<Memory>> {
        let ns = self.namespace(user_id);
        let ns_ref: Vec<&str> = ns.iter().map(String::as_str).collect();
        match self.store.get(&ns_ref, id).await? {
            Some(item) => Ok(Some(decode(&item.key, item.value)?.memory)),
            None => Ok(None),
        }
    }

    /// Every memory of `user_id`, most recently updated first.
    pub async fn list(&self, user_id: &str) -> Result<Vec<Memory>> {
        let mut all: Vec<Memory> = self
            .load(&self.namespace(user_id))
            .await?
            .into_iter()
            .map(|r| r.memory)
            .collect();
        all.sort_by_key(|m| std::cmp::Reverse(m.updated_at));
        Ok(all)
    }

    /// Deletes one memory; `false` when it did not exist.
    pub async fn forget(&self, user_id: &str, id: &str) -> Result<bool> {
        let ns = self.namespace(user_id);
        let ns_ref: Vec<&str> = ns.iter().map(String::as_str).collect();
        let existed = self.store.get(&ns_ref, id).await?.is_some();
        if existed {
            self.store.delete(&ns_ref, id).await?;
        }
        Ok(existed)
    }

    /// Deletes every memory of `user_id`; returns how many there were.
    pub async fn forget_all(&self, user_id: &str) -> Result<usize> {
        let ns = self.namespace(user_id);
        let ns_ref: Vec<&str> = ns.iter().map(String::as_str).collect();
        let records = self.load(&ns).await?;
        for r in &records {
            self.store.delete(&ns_ref, &r.memory.id).await?;
        }
        Ok(records.len())
    }

    /// The `k` memories of `user_id` most relevant to `query` (`k == 0`:
    /// all). With embeddings, ranked by cosine similarity; without, by
    /// keyword overlap, and memories sharing no word with the query are
    /// left out. An empty query returns the most recently updated
    /// memories. Ties go to higher importance, then to the newer memory.
    pub async fn recall(&self, user_id: &str, query: &str, k: usize) -> Result<Vec<ScoredMemory>> {
        let ns = self.namespace(user_id);
        let mut records = self.load(&ns).await?;
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let query = query.trim();
        let query_vec = if query.is_empty() {
            None
        } else if let Some(e) = &self.embeddings {
            self.ensure_vectors(&ns, &mut records).await?;
            Some(e.embed_query(query).await?)
        } else {
            None
        };
        let mut hits = self.rank(&records, query, query_vec.as_deref());
        if k > 0 {
            hits.truncate(k);
        }
        Ok(hits)
    }

    /// The recalled memories as a block for a system prompt (the context
    /// header, then one `- content` line per memory), or `None` when
    /// nothing relevant is stored.
    pub async fn context(&self, user_id: &str, query: &str, k: usize) -> Result<Option<String>> {
        let hits = self.recall(user_id, query, k).await?;
        Ok(format_memories(&self.context_header, &hits))
    }

    /// Extracts memories from `messages` and merges them into the user's
    /// memory.
    ///
    /// 1. One structured-output call with the extraction prompt turns the
    ///    conversation (user and assistant text; system and tool messages
    ///    are skipped) into candidate facts, preferences and episodes.
    /// 2. Candidates identical to a stored memory (ignoring case and
    ///    spacing) are dropped.
    /// 3. For each remaining candidate, the most similar stored memories are
    ///    looked up. If there are none, every candidate is added. Otherwise
    ///    a second call with the update prompt decides, per candidate, to
    ///    add it, update an existing memory, delete one, or do nothing.
    ///
    /// `source_thread` is recorded on added and updated memories.
    pub async fn remember(
        &self,
        user_id: &str,
        messages: &[Message],
        source_thread: Option<&str>,
    ) -> Result<RememberReport> {
        let text = transcript(messages);
        if text.trim().is_empty() {
            return Ok(RememberReport::default());
        }
        let extracted: Extracted = StructuredOutput::new("save_memories", extraction_schema())
            .with_description("Memories worth keeping from this conversation.")
            .generate(
                self.model.as_ref(),
                ChatRequest::new(vec![
                    Message::system(self.extraction_prompt.clone()),
                    Message::user(format!("Conversation:\n{text}")),
                ]),
            )
            .await?;

        let ns = self.namespace(user_id);
        let mut records = self.load(&ns).await?;
        let mut seen: HashSet<String> = records.iter().map(|r| norm(&r.memory.content)).collect();
        let facts: Vec<NewFact> = extracted
            .memories
            .into_iter()
            .filter_map(|mut f| {
                f.content = f.content.trim().to_string();
                (!f.content.is_empty() && seen.insert(norm(&f.content))).then_some(f)
            })
            .collect();
        if facts.is_empty() {
            return Ok(RememberReport::default());
        }

        // Vectors computed here are reused when the memories are written.
        let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
        let fact_vecs = if self.embeddings.is_some() {
            self.ensure_vectors(&ns, &mut records).await?;
            let contents: Vec<String> = facts.iter().map(|f| f.content.clone()).collect();
            let vecs = self.embed(&contents).await?.unwrap_or_default();
            for (c, v) in contents.iter().zip(&vecs) {
                cache.insert(c.clone(), v.clone());
            }
            Some(vecs)
        } else {
            None
        };

        let mut candidates: Vec<&Record> = Vec::new();
        for (i, fact) in facts.iter().enumerate() {
            let qv = fact_vecs.as_ref().and_then(|v| v.get(i)).map(Vec::as_slice);
            for hit in self
                .rank(&records, &fact.content, qv)
                .into_iter()
                .filter(|h| h.score > 0.0)
                .take(self.candidates)
            {
                if !candidates.iter().any(|c| c.memory.id == hit.memory.id) {
                    if let Some(r) = records.iter().find(|r| r.memory.id == hit.memory.id) {
                        candidates.push(r);
                    }
                }
            }
        }

        let ops: Vec<Operation> = if candidates.is_empty() {
            facts
                .iter()
                .map(|f| Operation {
                    action: Action::Add,
                    id: None,
                    content: Some(f.content.clone()),
                    kind: Some(f.kind),
                    importance: f.importance,
                })
                .collect()
        } else {
            let existing: Vec<Value> = candidates
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    json!({
                        "id": i.to_string(),
                        "content": r.memory.content,
                        "kind": r.memory.kind,
                    })
                })
                .collect();
            let prompt = format!(
                "Existing memories:\n{}\n\nNew facts:\n{}",
                serde_json::to_string_pretty(&existing).unwrap_or_default(),
                serde_json::to_string_pretty(&facts).unwrap_or_default(),
            );
            let out: Operations = StructuredOutput::new("update_memories", update_schema())
                .with_description("Operations that merge the new facts into the memory.")
                .generate(
                    self.model.as_ref(),
                    ChatRequest::new(vec![
                        Message::system(self.update_prompt.clone()),
                        Message::user(prompt),
                    ]),
                )
                .await?;
            out.operations
        };

        let by_temp: HashMap<String, &Record> = candidates
            .iter()
            .enumerate()
            .map(|(i, r)| (i.to_string(), *r))
            .collect();
        let now = now_millis();
        let mut touched: HashSet<String> = HashSet::new();
        let mut report = RememberReport::default();
        for op in ops {
            let target = op.id.as_ref().and_then(|v| match v {
                Value::String(s) => by_temp.get(s.trim()).copied(),
                Value::Number(n) => by_temp.get(&n.to_string()).copied(),
                _ => None,
            });
            let content = op
                .content
                .as_deref()
                .map(str::trim)
                .filter(|c| !c.is_empty());
            match (op.action, target) {
                (Action::Update, Some(r)) => {
                    let Some(content) = content else { continue };
                    if !touched.insert(r.memory.id.clone()) || content == r.memory.content {
                        continue;
                    }
                    let mut m = r.memory.clone();
                    m.content = content.to_string();
                    m.kind = op.kind.unwrap_or(m.kind);
                    m.importance = op.importance.map(|i| i.clamp(0.0, 1.0)).or(m.importance);
                    m.updated_at = now.max(m.created_at);
                    if let Some(t) = source_thread {
                        m.source_thread = Some(t.to_string());
                    }
                    report.updated.push(m);
                }
                (Action::Delete, Some(r)) => {
                    if touched.insert(r.memory.id.clone()) {
                        report.deleted.push(r.memory.clone());
                    }
                }
                // An update of an unknown id is treated as an addition.
                (Action::Add, _) | (Action::Update, None) => {
                    let Some(content) = content else { continue };
                    let n = norm(content);
                    let duplicate = records.iter().any(|r| norm(&r.memory.content) == n)
                        || report.added.iter().any(|m| norm(&m.content) == n);
                    if duplicate {
                        continue;
                    }
                    let mut m = Memory::new(content, op.kind.unwrap_or_default());
                    m.importance = op.importance.map(|i| i.clamp(0.0, 1.0));
                    m.source_thread = source_thread.map(str::to_string);
                    report.added.push(m);
                }
                _ => {}
            }
        }

        let to_embed: Vec<String> = report
            .added
            .iter()
            .chain(&report.updated)
            .map(|m| m.content.clone())
            .filter(|c| !cache.contains_key(c))
            .collect();
        if let Some(vecs) = self.embed(&to_embed).await? {
            cache.extend(to_embed.into_iter().zip(vecs));
        }
        for m in report.added.iter().chain(&report.updated) {
            let v = cache.get(&m.content).cloned();
            self.write(&ns, m.clone(), v).await?;
        }
        let ns_ref: Vec<&str> = ns.iter().map(String::as_str).collect();
        for m in &report.deleted {
            self.store.delete(&ns_ref, &m.id).await?;
        }
        Ok(report)
    }

    async fn embed(&self, texts: &[String]) -> Result<Option<Vec<Vec<f32>>>> {
        match &self.embeddings {
            Some(e) if !texts.is_empty() => Ok(Some(e.embed_documents(texts).await?)),
            Some(_) => Ok(Some(Vec::new())),
            None => Ok(None),
        }
    }

    async fn write(
        &self,
        ns: &[String],
        memory: Memory,
        embedding: Option<Vec<f32>>,
    ) -> Result<()> {
        let ns_ref: Vec<&str> = ns.iter().map(String::as_str).collect();
        let id = memory.id.clone();
        let record = Record {
            memory,
            embedding_model: embedding
                .as_ref()
                .and(self.embeddings.as_ref())
                .map(|e| e.model().to_string()),
            embedding,
        };
        // With our own vectors, skip the store's index (it would embed the
        // whole record, vector included).
        let options = PutOptions {
            index: self.embeddings.is_some().then_some(false),
            ttl: None,
        };
        self.store
            .put_with(
                &ns_ref,
                &id,
                serde_json::to_value(&record).map_err(StoreError::from)?,
                options,
            )
            .await?;
        Ok(())
    }

    async fn load(&self, ns: &[String]) -> Result<Vec<Record>> {
        let ns_ref: Vec<&str> = ns.iter().map(String::as_str).collect();
        let items = self
            .store
            .search(&ns_ref, SearchQuery::new().limit(0))
            .await?;
        items
            .into_iter()
            .filter(|h| h.item.namespace == ns)
            .map(|h| decode(&h.item.key, h.item.value))
            .collect()
    }

    /// Embeds (and writes back) memories without a vector from the current
    /// embeddings model.
    async fn ensure_vectors(&self, ns: &[String], records: &mut [Record]) -> Result<()> {
        let Some(e) = &self.embeddings else {
            return Ok(());
        };
        let model = e.model().to_string();
        let stale: Vec<usize> = records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.embedding.is_none() || r.embedding_model.as_deref() != Some(&model))
            .map(|(i, _)| i)
            .collect();
        if stale.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = stale
            .iter()
            .map(|&i| records[i].memory.content.clone())
            .collect();
        let vecs = e.embed_documents(&texts).await?;
        for (&i, v) in stale.iter().zip(vecs) {
            records[i].embedding = Some(v.clone());
            records[i].embedding_model = Some(model.clone());
            self.write(ns, records[i].memory.clone(), Some(v)).await?;
        }
        Ok(())
    }

    fn rank(
        &self,
        records: &[Record],
        query: &str,
        query_vec: Option<&[f32]>,
    ) -> Vec<ScoredMemory> {
        let semantic = query_vec.is_some();
        let q_tokens: HashSet<String> = if query.is_empty() || semantic {
            HashSet::new()
        } else {
            sinew_rag::english_tokenize(query).into_iter().collect()
        };
        let mut hits: Vec<ScoredMemory> = records
            .iter()
            .filter_map(|r| {
                let score = if query.is_empty() {
                    0.0
                } else if let Some(q) = query_vec {
                    r.embedding.as_deref().map_or(0.0, |e| cosine(q, e))
                } else {
                    let s = keyword_score(&q_tokens, &r.memory.content);
                    if s <= 0.0 {
                        return None;
                    }
                    s
                };
                if !query.is_empty() && self.min_score.is_some_and(|m| score < m) {
                    return None;
                }
                Some(ScoredMemory {
                    memory: r.memory.clone(),
                    score,
                })
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.memory
                        .importance
                        .unwrap_or(0.0)
                        .partial_cmp(&a.memory.importance.unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| b.memory.updated_at.cmp(&a.memory.updated_at))
        });
        hits
    }
}

fn decode(key: &str, value: Value) -> Result<Record> {
    serde_json::from_value(value).map_err(|source| MemoryError::InvalidRecord {
        key: key.to_string(),
        source,
    })
}

/// Lowercase with collapsed whitespace and no trailing period: the identity
/// used for exact-duplicate checks.
fn norm(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches('.')
        .to_lowercase()
}

/// Shared distinct words over the geometric mean of both word counts (0 to 1).
fn keyword_score(query: &HashSet<String>, content: &str) -> f32 {
    if query.is_empty() {
        return 0.0;
    }
    let words: HashSet<String> = sinew_rag::english_tokenize(content).into_iter().collect();
    if words.is_empty() {
        return 0.0;
    }
    let shared = query.intersection(&words).count() as f32;
    shared / ((query.len() * words.len()) as f32).sqrt()
}

/// Formats recalled memories under `header`, one `- content` line each;
/// `None` when `memories` is empty.
pub fn format_memories(header: &str, memories: &[ScoredMemory]) -> Option<String> {
    if memories.is_empty() {
        return None;
    }
    let mut out = header.to_string();
    for m in memories {
        out.push_str("\n- ");
        out.push_str(&m.memory.content);
    }
    Some(out)
}

/// The conversation as `user: ...` / `assistant: ...` lines. System and tool
/// messages, and assistant messages with no text (tool calls only), are
/// skipped.
pub fn transcript(messages: &[Message]) -> String {
    messages
        .iter()
        .filter_map(|m| {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                _ => return None,
            };
            let text = m.text();
            let text = text.trim();
            (!text.is_empty()).then(|| format!("{role}: {text}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The last turn: from the last user message to the end (all of
/// `messages` when there is no user message).
pub fn last_turn(messages: &[Message]) -> &[Message] {
    match messages.iter().rposition(|m| m.role == Role::User) {
        Some(i) => &messages[i..],
        None => messages,
    }
}

/// Text of the last user message, the default recall query.
pub fn last_user_text(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(Message::text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_and_keyword_score() {
        assert_eq!(norm("  The user  likes tea. "), "the user likes tea");
        let q: HashSet<String> = sinew_rag::english_tokenize("what tea do they like")
            .into_iter()
            .collect();
        assert!(keyword_score(&q, "The user likes green tea") > 0.0);
        assert_eq!(keyword_score(&q, "Lives in Lisbon"), 0.0);
    }

    #[test]
    fn transcript_skips_system_tool_and_empty_messages() {
        let msgs = vec![
            Message::system("sys"),
            Message::user("I am Ana"),
            Message::assistant_with_tools("", vec![]),
            Message::tool_result("c1", "42"),
            Message::assistant("Hi Ana"),
        ];
        assert_eq!(transcript(&msgs), "user: I am Ana\nassistant: Hi Ana");
        assert_eq!(last_turn(&msgs).len(), 4);
        assert_eq!(last_user_text(&msgs).as_deref(), Some("I am Ana"));
    }
}
