//! Clear or truncate tool results the model has already seen.

use std::collections::HashMap;

use async_trait::async_trait;

use super::{ContextBudget, ContextStrategy};
use crate::message::{Message, Role};
use crate::Result;

/// Start of the placeholder that replaces a cleared tool result.
pub const CLEARED_MARKER: &str = "[tool result cleared to save context";

/// Start of the note appended to a truncated tool result.
pub const TRUNCATED_MARKER: &str = "[tool result truncated to save context";

/// Replaces old tool results with a short placeholder (tool name and
/// size), or truncates them.
///
/// A tool result is *used* once an assistant message follows it: the model
/// has read it and answered. Only used results are touched, oldest first,
/// and the [`keep_last`](Self::keep_last) most recent used results stay
/// intact. Results shorter than [`min_chars`](Self::min_chars) are left
/// alone. With [`until_fits`](Self::until_fits) (the default) clearing stops
/// as soon as the conversation fits the budget.
///
/// The messages themselves stay, so tool calls and results remain paired.
/// Media parts of a cleared or truncated result are removed.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use serde_json::json;
/// use sinew_core::context::{ClearToolResults, ContextBudget, ContextStrategy};
/// use sinew_core::{Message, ToolCall};
///
/// let call = ToolCall { id: "1".into(), name: "read_file".into(), arguments: json!({}) };
/// let msgs = vec![
///     Message::user("summarize the file"),
///     Message::assistant_with_tools("", vec![call]),
///     Message::tool_result("1", "lorem ipsum ".repeat(500)),
///     Message::assistant("It repeats lorem ipsum."),
///     Message::user("thanks"),
/// ];
/// let out = ClearToolResults::new()
///     .keep_last(0)
///     .apply(msgs, &ContextBudget::new(100))
///     .await
///     .unwrap();
/// assert_eq!(
///     out[2].content,
///     "[tool result cleared to save context: `read_file`, 6000 characters]"
/// );
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ClearToolResults {
    keep_last: usize,
    min_chars: usize,
    truncate_to: Option<usize>,
    until_fits: bool,
}

impl Default for ClearToolResults {
    fn default() -> Self {
        Self::new()
    }
}

impl ClearToolResults {
    /// Clears every used tool result but the 2 most recent, oldest first,
    /// until the conversation fits; results under 200 characters are kept.
    pub fn new() -> Self {
        Self {
            keep_last: 2,
            min_chars: 200,
            truncate_to: None,
            until_fits: true,
        }
    }

    /// Used tool results (most recent first) left intact (default 2).
    pub fn keep_last(mut self, n: usize) -> Self {
        self.keep_last = n;
        self
    }

    /// Results with fewer characters than this are left alone (default 200).
    pub fn min_chars(mut self, n: usize) -> Self {
        self.min_chars = n;
        self
    }

    /// Keeps the first `chars` characters of each result plus a note,
    /// instead of replacing the whole result with a placeholder.
    pub fn truncate_to(mut self, chars: usize) -> Self {
        self.truncate_to = Some(chars);
        self
    }

    /// Whether to stop as soon as the conversation fits the budget (default
    /// `true`). With `false`, every eligible result is cleared.
    pub fn until_fits(mut self, yes: bool) -> Self {
        self.until_fits = yes;
        self
    }

    fn rewrite(&self, m: &mut Message, name: &str) {
        let chars = m.content.chars().count();
        m.content = match self.truncate_to {
            None => format!("{CLEARED_MARKER}: `{name}`, {chars} characters]"),
            Some(keep) => {
                let head: String = m.content.chars().take(keep).collect();
                format!("{head}\n{TRUNCATED_MARKER}: `{name}`, {chars} characters in total]")
            }
        };
        m.parts.clear();
    }
}

#[async_trait]
impl ContextStrategy for ClearToolResults {
    fn name(&self) -> &str {
        "clear_tool_results"
    }

    async fn apply(
        &self,
        mut messages: Vec<Message>,
        budget: &ContextBudget,
    ) -> Result<Vec<Message>> {
        // Tool name per call id, and the last assistant message index.
        let mut names: HashMap<String, String> = HashMap::new();
        for m in &messages {
            for c in &m.tool_calls {
                names.insert(c.id.clone(), c.name.clone());
            }
        }
        let Some(last_assistant) = messages.iter().rposition(|m| m.role == Role::Assistant) else {
            return Ok(messages);
        };
        let mut used: Vec<usize> = (0..last_assistant)
            .filter(|&i| messages[i].role == Role::Tool)
            .collect();
        used.truncate(used.len().saturating_sub(self.keep_last));

        let counter = &budget.counter;
        let mut total = if self.until_fits {
            counter.count_messages(&messages)
        } else {
            0
        };
        for i in used {
            if self.until_fits && total <= budget.max_tokens {
                break;
            }
            let m = &messages[i];
            let already =
                m.content.starts_with(CLEARED_MARKER) || m.content.contains(TRUNCATED_MARKER);
            if already || (m.content.chars().count() < self.min_chars && !m.has_media()) {
                continue;
            }
            let name = m
                .tool_call_id
                .as_ref()
                .and_then(|id| names.get(id))
                .map_or("tool", String::as_str)
                .to_string();
            let before = counter.count_message(&messages[i]);
            self.rewrite(&mut messages[i], &name);
            if self.until_fits {
                total = (total + counter.count_message(&messages[i])).saturating_sub(before);
            }
        }
        Ok(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::testkit::*;
    use crate::context::{check_tool_pairing, EstimateCounter, TokenCounter};

    #[tokio::test]
    async fn keeps_unused_and_recent_results() {
        let big = "x".repeat(1_000);
        let msgs = vec![
            Message::user("q"),
            Message::assistant_with_tools("", vec![call("1", "a")]),
            Message::tool_result("1", big.clone()),
            Message::assistant_with_tools("", vec![call("2", "b")]),
            Message::tool_result("2", big.clone()),
            Message::assistant_with_tools("", vec![call("3", "c")]),
            Message::tool_result("3", big.clone()),
        ];
        let out = ClearToolResults::new()
            .keep_last(1)
            .until_fits(false)
            .apply(msgs, &ContextBudget::new(10))
            .await
            .unwrap();
        assert!(out[2].content.starts_with(CLEARED_MARKER));
        assert!(out[2].content.contains("`a`"));
        // Most recent used result kept, unused (current) result kept.
        assert_eq!(out[4].content, big);
        assert_eq!(out[6].content, big);
    }

    #[tokio::test]
    async fn truncation_is_idempotent() {
        let msgs = vec![
            Message::assistant_with_tools("", vec![call("1", "a")]),
            Message::tool_result("1", "y".repeat(1_000)),
            Message::assistant("ok"),
        ];
        let s = ClearToolResults::new()
            .keep_last(0)
            .truncate_to(300)
            .until_fits(false);
        let once = s.apply(msgs, &ContextBudget::new(10)).await.unwrap();
        assert!(once[1].content.starts_with(&"y".repeat(300)));
        assert!(once[1].content.contains("1000 characters in total"));
        let twice = s
            .apply(once.clone(), &ContextBudget::new(10))
            .await
            .unwrap();
        assert_eq!(once, twice);
    }

    #[tokio::test]
    async fn stops_when_it_fits() {
        let msgs = transcript(7, 500);
        let before = EstimateCounter.count_messages(&msgs);
        let budget = ContextBudget::new(before / 2);
        let out = ClearToolResults::new()
            .apply(msgs.clone(), &budget)
            .await
            .unwrap();
        check_tool_pairing(&out).unwrap();
        assert_eq!(out.len(), msgs.len());
        let after = EstimateCounter.count_messages(&out);
        assert!(after <= before / 2, "{after} > {}", before / 2);
        // The newest results are untouched: it stopped early.
        let last_tool = out.iter().rposition(|m| m.role == Role::Tool).unwrap();
        assert!(!out[last_tool].content.starts_with(CLEARED_MARKER));
    }
}
