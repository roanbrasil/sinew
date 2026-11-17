//! Summarize older turns into one running summary.

use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{message_groups, ContextBudget, ContextStrategy};
use crate::message::{Message, Role};
use crate::{ChatModel, ChatRequest, Result};

/// Instructions given to the summarizing model by default.
pub const DEFAULT_SUMMARY_PROMPT: &str = "You maintain a running summary of a conversation \
between a user and an AI agent, so the agent can continue the work without the original \
messages. You receive the current summary (possibly empty) and new conversation turns. Return \
the updated summary: the user's goals and constraints, decisions made, important facts, numbers \
and identifiers, tool results that still matter, and open tasks. Drop details that no longer \
matter. Plain prose or bullets; no preamble.";

/// First line of the message that holds the summary.
pub const DEFAULT_SUMMARY_HEADER: &str = "Summary of the earlier conversation:";

/// How many recent messages [`Summarize`] keeps verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepRecent {
    /// At most this many non-system messages (whole groups; at least one group).
    Messages(usize),
    /// At most this many tokens (whole groups; at least one group).
    Tokens(usize),
}

const CACHE_ENTRIES: usize = 32;

/// Folds the older part of a conversation into a single summary message
/// produced by a [`ChatModel`], keeping the recent messages verbatim.
///
/// The output is: the system messages of the summarized part, one system
/// message starting with the header ([`DEFAULT_SUMMARY_HEADER`]) and the
/// summary, then the kept recent messages unchanged. The split between the
/// summarized part and the kept part falls between groups (see
/// [`message_groups`]), so tool calls and results stay paired.
///
/// The summary is *running*: it is updated, not recomputed.
///
/// - When the input already contains a summary message (a system message
///   starting with the header, as when the summarized history was stored
///   back), only that summary and the messages after it are sent to the
///   model.
/// - Otherwise the strategy remembers recent summaries keyed by a hash of
///   the messages they cover. Called again with the same conversation plus
///   new messages, as an agent does before every model call, it sends only
///   the previous summary and the new messages.
///
/// Long stretches are summarized in chunks of at most
/// [`max_chunk_tokens`](Self::max_chunk_tokens), each chunk updating the
/// summary, so the summarizer's own request stays small.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use std::sync::Arc;
/// use sinew_core::context::{ContextBudget, ContextStrategy, KeepRecent, Summarize};
/// use sinew_core::mock::ScriptedModel;
/// use sinew_core::{Message, Role};
///
/// let model = Arc::new(ScriptedModel::new([Message::assistant("User wants a trip to Lisbon.")]));
/// let summarize = Summarize::new(model.clone()).keep_recent(KeepRecent::Messages(2));
/// let msgs = vec![
///     Message::user("I want to travel in May."),
///     Message::assistant("Where to?"),
///     Message::user("Lisbon."),
///     Message::assistant("Booking options follow."),
/// ];
/// let out = summarize.apply(msgs, &ContextBudget::new(1_000)).await.unwrap();
/// assert_eq!(out.len(), 3);
/// assert_eq!(out[0].role, Role::System);
/// assert!(out[0].content.ends_with("User wants a trip to Lisbon."));
/// assert_eq!(out[1].content, "Lisbon.");
/// # }
/// ```
pub struct Summarize {
    model: Arc<dyn ChatModel>,
    prompt: String,
    header: String,
    keep_recent: KeepRecent,
    max_chunk_tokens: usize,
    max_tool_result_chars: Option<usize>,
    cache: Mutex<VecDeque<(u64, String)>>,
}

impl std::fmt::Debug for Summarize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Summarize")
            .field("model", &self.model.name())
            .field("header", &self.header)
            .field("keep_recent", &self.keep_recent)
            .field("max_chunk_tokens", &self.max_chunk_tokens)
            .field("max_tool_result_chars", &self.max_tool_result_chars)
            .finish_non_exhaustive()
    }
}

impl Summarize {
    /// Summarizes with `model`, keeping the 10 most recent messages, in
    /// chunks of up to 8 000 tokens, with tool results cut to 2 000
    /// characters in the summarizer's input.
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            model,
            prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
            header: DEFAULT_SUMMARY_HEADER.to_string(),
            keep_recent: KeepRecent::Messages(10),
            max_chunk_tokens: 8_000,
            max_tool_result_chars: Some(2_000),
            cache: Mutex::new(VecDeque::new()),
        }
    }

    /// Replaces the system prompt of the summarizer call.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Replaces the first line of the summary message. It also identifies
    /// an existing summary in the input, so keep it stable.
    pub fn header(mut self, header: impl Into<String>) -> Self {
        self.header = header.into();
        self
    }

    /// How many recent messages stay verbatim (default 10 messages).
    pub fn keep_recent(mut self, keep: KeepRecent) -> Self {
        self.keep_recent = keep;
        self
    }

    /// Maximum transcript tokens per summarizer call (default 8 000).
    pub fn max_chunk_tokens(mut self, n: usize) -> Self {
        self.max_chunk_tokens = n.max(1);
        self
    }

    /// Cuts each tool result in the summarizer's input to this many
    /// characters (default 2 000; `None` sends them whole).
    pub fn max_tool_result_chars(mut self, n: Option<usize>) -> Self {
        self.max_tool_result_chars = n;
        self
    }

    /// Forgets the remembered summaries.
    pub fn clear_cache(&self) {
        self.cache.lock().expect("cache lock").clear();
    }

    fn is_summary(&self, m: &Message) -> bool {
        m.role == Role::System && m.content.starts_with(&self.header)
    }

    fn summary_body<'a>(&self, m: &'a Message) -> &'a str {
        m.content[self.header.len()..].trim()
    }

    fn cached(&self, key: u64) -> Option<String> {
        let cache = self.cache.lock().expect("cache lock");
        cache
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, s)| s.clone())
    }

    fn remember(&self, key: u64, summary: &str) {
        let mut cache = self.cache.lock().expect("cache lock");
        cache.retain(|(k, _)| *k != key);
        if cache.len() >= CACHE_ENTRIES {
            cache.pop_front();
        }
        cache.push_back((key, summary.to_string()));
    }

    async fn update(&self, summary: &str, chunk: &[Message]) -> Result<String> {
        let mut user = String::new();
        user.push_str("Current summary:\n");
        user.push_str(if summary.is_empty() {
            "(none)"
        } else {
            summary
        });
        user.push_str("\n\nNew conversation turns:\n");
        user.push_str(&render_transcript(chunk, self.max_tool_result_chars));
        let request = ChatRequest::new(vec![
            Message::system(self.prompt.clone()),
            Message::user(user),
        ]);
        let resp = self.model.chat(request).await?;
        Ok(resp.message.content.trim().to_string())
    }

    /// Index in `conv` where the kept tail starts.
    fn split(&self, conv: &[Message], budget: &ContextBudget) -> usize {
        let groups = message_groups(conv);
        let mut start = conv.len();
        let (mut msgs, mut tokens) = (0usize, 0usize);
        for (n, g) in groups.iter().rev().enumerate() {
            let t: usize = conv[g.clone()]
                .iter()
                .map(|m| budget.counter.count_message(m))
                .sum();
            let over = match self.keep_recent {
                KeepRecent::Messages(max) => msgs + g.len() > max,
                KeepRecent::Tokens(max) => tokens + t > max,
            };
            if over && n > 0 {
                break;
            }
            msgs += g.len();
            tokens += t;
            start = g.start;
            if over {
                break;
            }
        }
        start
    }
}

#[async_trait]
impl ContextStrategy for Summarize {
    fn name(&self) -> &str {
        "summarize"
    }

    async fn apply(&self, messages: Vec<Message>, budget: &ContextBudget) -> Result<Vec<Message>> {
        let last_summary = messages.iter().rposition(|m| self.is_summary(m));
        let previous = last_summary
            .map(|i| self.summary_body(&messages[i]).to_string())
            .unwrap_or_default();
        let plain: Vec<Message> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .cloned()
            .collect();
        let tail_from = self.split(&plain, budget);
        if tail_from == 0 {
            return Ok(messages);
        }

        // Head: the non-system messages before the split (summarized).
        // System messages in that part move before the summary; earlier
        // summary messages are replaced by the new one. The tail is kept
        // unchanged.
        let mut seen = 0usize;
        let mut systems = Vec::new();
        let mut head = Vec::new();
        let mut tail = Vec::new();
        for m in messages {
            let in_head = seen < tail_from;
            if m.role != Role::System {
                seen += 1;
            }
            if self.is_summary(&m) {
                continue;
            }
            match (in_head, m.role == Role::System) {
                (true, true) => systems.push(m),
                (true, false) => head.push(m),
                (false, _) => tail.push(m),
            }
        }

        // Prefix hashes: key[k] covers `previous` and head[..k].
        let mut hasher = DefaultHasher::new();
        self.header.hash(&mut hasher);
        previous.hash(&mut hasher);
        let mut keys = Vec::with_capacity(head.len() + 1);
        keys.push(hasher.finish());
        for m in &head {
            serde_json::to_string(m)
                .unwrap_or_default()
                .hash(&mut hasher);
            keys.push(hasher.finish());
        }
        let mut start = 0;
        let mut summary = previous;
        for k in (1..=head.len()).rev() {
            if let Some(s) = self.cached(keys[k]) {
                start = k;
                summary = s;
                break;
            }
        }

        // Summarize head[start..] in chunks.
        let mut i = start;
        while i < head.len() {
            let mut j = i;
            let mut tokens = 0;
            while j < head.len() {
                let t = budget.counter.count_message(&head[j]);
                if j > i && tokens + t > self.max_chunk_tokens {
                    break;
                }
                tokens += t;
                j += 1;
            }
            summary = self.update(&summary, &head[i..j]).await?;
            self.remember(keys[j], &summary);
            i = j;
        }

        let mut out = systems;
        out.push(Message::system(format!("{}\n{}", self.header, summary)));
        out.extend(tail);
        Ok(out)
    }
}

/// Renders messages as a plain-text transcript, one `[role] text` line per
/// message and one `[tool call] name(arguments)` line per call. Tool
/// results are labeled with the tool name and cut to `max_tool_result_chars`
/// characters when set.
///
/// ```
/// use serde_json::json;
/// use sinew_core::context::render_transcript;
/// use sinew_core::{Message, ToolCall};
///
/// let call = ToolCall { id: "1".into(), name: "add".into(), arguments: json!({"a": 1}) };
/// let text = render_transcript(
///     &[Message::assistant_with_tools("", vec![call]), Message::tool_result("1", "2")],
///     None,
/// );
/// assert_eq!(text, "[tool call] add({\"a\":1})\n[tool result: add] 2\n");
/// ```
pub fn render_transcript(messages: &[Message], max_tool_result_chars: Option<usize>) -> String {
    let mut names = std::collections::HashMap::new();
    let mut out = String::new();
    for m in messages {
        let text = m.text();
        match m.role {
            Role::Tool => {
                let name = m
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| names.get(id))
                    .map_or("tool", |n: &String| n.as_str());
                let chars = text.chars().count();
                match max_tool_result_chars {
                    Some(max) if chars > max => {
                        let head: String = text.chars().take(max).collect();
                        out.push_str(&format!(
                            "[tool result: {name}] {head} [... {} more characters]\n",
                            chars - max
                        ));
                    }
                    _ => out.push_str(&format!("[tool result: {name}] {text}\n")),
                }
            }
            role => {
                let label = match role {
                    Role::System => "system",
                    Role::User => "user",
                    _ => "assistant",
                };
                if !text.is_empty() {
                    out.push_str(&format!("[{label}] {text}\n"));
                }
                if m.has_media() {
                    out.push_str(&format!("[{label}] (attached media)\n"));
                }
            }
        }
        for c in &m.tool_calls {
            names.insert(c.id.clone(), c.name.clone());
            out.push_str(&format!("[tool call] {}({})\n", c.name, c.arguments));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::testkit::*;
    use crate::context::{check_tool_pairing, EstimateCounter, TokenCounter};
    use crate::mock::ScriptedModel;

    fn replies(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| Message::assistant(format!("summary v{i}")))
            .collect()
    }

    #[tokio::test]
    async fn incremental_updates_send_only_new_messages() {
        let model = Arc::new(ScriptedModel::new(replies(3)));
        let s = Summarize::new(model.clone()).keep_recent(KeepRecent::Messages(2));
        let budget = ContextBudget::new(10_000);
        let mut msgs = vec![Message::system("sys")];
        for i in 0..6 {
            msgs.push(Message::user(format!("u{i}")));
            msgs.push(Message::assistant(format!("a{i}")));
        }
        let out = s.apply(msgs.clone(), &budget).await.unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].content, "sys");
        assert!(out[1].content.ends_with("summary v0"));
        assert_eq!(out[2].content, "u5");

        // Same conversation plus two messages: only the delta is sent.
        msgs.push(Message::user("u6"));
        msgs.push(Message::assistant("a6"));
        let out = s.apply(msgs.clone(), &budget).await.unwrap();
        assert!(out[1].content.ends_with("summary v1"));
        let req = &model.requests()[1];
        let input = &req.messages[1].content;
        assert!(input.contains("Current summary:\nsummary v0"));
        assert!(input.contains("[user] u5"));
        assert!(!input.contains("u4"));

        // Persisted mode: the summarized history is fed back.
        let mut stored = out;
        stored.push(Message::user("u7"));
        stored.push(Message::assistant("a7"));
        let out = s.apply(stored, &budget).await.unwrap();
        assert_eq!(
            out.iter()
                .filter(|m| m.content.starts_with(DEFAULT_SUMMARY_HEADER))
                .count(),
            1
        );
        let input = &model.requests()[2].messages[1].content;
        assert!(input.contains("Current summary:\nsummary v1"));
        assert!(input.contains("[user] u6"));
        assert!(!input.contains("u5"));
        assert_eq!(out.last().unwrap().content, "a7");
    }

    #[tokio::test]
    async fn chunks_long_histories_and_keeps_pairs() {
        let msgs = transcript(3, 500);
        let model = Arc::new(ScriptedModel::new(replies(200)));
        let s = Summarize::new(model.clone())
            .keep_recent(KeepRecent::Tokens(2_000))
            .max_chunk_tokens(4_000);
        let out = s
            .apply(msgs.clone(), &ContextBudget::new(4_000))
            .await
            .unwrap();
        check_tool_pairing(&out).unwrap();
        let calls = model.requests().len();
        assert!(calls > 1, "expected several chunks, got {calls}");
        for req in model.requests() {
            assert!(EstimateCounter.count_text(&req.messages[1].content) < 4_000 + 1_000);
        }
        // A tail plus summary and the system prompt.
        assert_eq!(out[0].content, msgs[0].content);
        assert!(out[1].content.starts_with(DEFAULT_SUMMARY_HEADER));
        assert!(out.len() < 60);
    }

    #[tokio::test]
    async fn nothing_to_summarize_is_a_no_op() {
        let model = Arc::new(ScriptedModel::new(Vec::<Message>::new()));
        let s = Summarize::new(model.clone());
        let msgs = vec![Message::system("s"), Message::user("hi")];
        let out = s
            .apply(msgs.clone(), &ContextBudget::new(10))
            .await
            .unwrap();
        assert_eq!(out, msgs);
        assert!(model.requests().is_empty());
    }
}
