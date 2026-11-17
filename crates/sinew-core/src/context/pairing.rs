//! Tool-call / tool-result pairing.

use std::collections::HashSet;
use std::ops::Range;

use crate::message::{Message, Role};

/// Splits `messages` into groups that must be kept or dropped together.
///
/// An assistant message with tool calls forms one group with the tool
/// results that directly follow it; every other message is a group of its
/// own (including a tool result that follows no tool call). The ranges cover
/// `messages` in order.
///
/// ```
/// use serde_json::json;
/// use sinew_core::context::message_groups;
/// use sinew_core::{Message, ToolCall};
///
/// let call = ToolCall { id: "1".into(), name: "search".into(), arguments: json!({}) };
/// let msgs = vec![
///     Message::user("find it"),
///     Message::assistant_with_tools("", vec![call]),
///     Message::tool_result("1", "found"),
///     Message::assistant("here it is"),
/// ];
/// assert_eq!(message_groups(&msgs), vec![0..1, 1..3, 3..4]);
/// ```
pub fn message_groups(messages: &[Message]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let mut j = i + 1;
        if messages[i].role == Role::Assistant && messages[i].has_tool_calls() {
            while j < messages.len() && messages[j].role == Role::Tool {
                j += 1;
            }
        }
        out.push(i..j);
        i = j;
    }
    out
}

/// A tool call or tool result that has no partner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingError {
    /// The tool result at `index` does not answer a call of the assistant
    /// message right before its block of tool results.
    OrphanResult {
        index: usize,
        call_id: Option<String>,
    },
    /// The assistant message at `index` has a call with no result after it.
    MissingResult { index: usize, call_id: String },
}

impl std::fmt::Display for PairingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PairingError::OrphanResult { index, call_id } => write!(
                f,
                "tool result at index {index} (call id {}) has no matching tool call",
                call_id.as_deref().unwrap_or("<none>")
            ),
            PairingError::MissingResult { index, call_id } => write!(
                f,
                "tool call {call_id} of the assistant message at index {index} has no result"
            ),
        }
    }
}

impl std::error::Error for PairingError {}

/// Checks that every tool result answers a call of the assistant message
/// before it and every tool call has a result, the shape providers accept.
///
/// ```
/// use serde_json::json;
/// use sinew_core::context::{check_tool_pairing, PairingError};
/// use sinew_core::{Message, ToolCall};
///
/// let call = ToolCall { id: "1".into(), name: "search".into(), arguments: json!({}) };
/// let ok = vec![Message::assistant_with_tools("", vec![call]), Message::tool_result("1", "x")];
/// assert!(check_tool_pairing(&ok).is_ok());
///
/// let orphan = vec![Message::user("hi"), Message::tool_result("1", "x")];
/// assert!(matches!(check_tool_pairing(&orphan), Err(PairingError::OrphanResult { index: 1, .. })));
/// ```
pub fn check_tool_pairing(messages: &[Message]) -> Result<(), PairingError> {
    for g in message_groups(messages) {
        let head = &messages[g.start];
        if head.role == Role::Tool {
            return Err(PairingError::OrphanResult {
                index: g.start,
                call_id: head.tool_call_id.clone(),
            });
        }
        if !head.has_tool_calls() || head.role != Role::Assistant {
            continue;
        }
        let ids: HashSet<&str> = head.tool_calls.iter().map(|c| c.id.as_str()).collect();
        let mut answered = HashSet::new();
        for (i, m) in messages.iter().enumerate().take(g.end).skip(g.start + 1) {
            match m.tool_call_id.as_deref() {
                Some(id) if ids.contains(id) => {
                    answered.insert(id);
                }
                other => {
                    return Err(PairingError::OrphanResult {
                        index: i,
                        call_id: other.map(str::to_string),
                    })
                }
            }
        }
        if let Some(c) = head
            .tool_calls
            .iter()
            .find(|c| !answered.contains(c.id.as_str()))
        {
            return Err(PairingError::MissingResult {
                index: g.start,
                call_id: c.id.clone(),
            });
        }
    }
    Ok(())
}

/// Repairs a history by removal: drops tool results that answer no call
/// and removes tool calls that have no result (an assistant message left
/// with no text, parts or calls is dropped too).
///
/// Strategies in this module never need it on valid input; use it on
/// histories assembled elsewhere. Note that a trailing assistant message
/// whose calls have not run yet loses those calls.
///
/// ```
/// use serde_json::json;
/// use sinew_core::context::{check_tool_pairing, drop_unpaired_tool_messages};
/// use sinew_core::{Message, ToolCall};
///
/// let call = |id: &str| ToolCall { id: id.into(), name: "t".into(), arguments: json!({}) };
/// let broken = vec![
///     Message::tool_result("0", "orphan"),
///     Message::assistant_with_tools("checking", vec![call("1"), call("2")]),
///     Message::tool_result("1", "x"),
/// ];
/// let fixed = drop_unpaired_tool_messages(broken);
/// assert!(check_tool_pairing(&fixed).is_ok());
/// assert_eq!(fixed.len(), 2);
/// assert_eq!(fixed[0].tool_calls.len(), 1);
/// ```
pub fn drop_unpaired_tool_messages(messages: Vec<Message>) -> Vec<Message> {
    let groups = message_groups(&messages);
    let mut slots: Vec<Option<Message>> = messages.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(slots.len());
    for g in groups {
        let mut head = slots[g.start].take().expect("each index is taken once");
        if head.role == Role::Tool {
            continue;
        }
        if head.role != Role::Assistant || !head.has_tool_calls() {
            out.push(head);
            continue;
        }
        let ids: HashSet<String> = head.tool_calls.iter().map(|c| c.id.clone()).collect();
        let mut seen = HashSet::new();
        let mut results = Vec::new();
        for slot in &mut slots[g.start + 1..g.end] {
            let r = slot.take().expect("each index is taken once");
            if let Some(id) = r.tool_call_id.as_deref() {
                if ids.contains(id) && seen.insert(id.to_string()) {
                    results.push(r);
                }
            }
        }
        head.tool_calls.retain(|c| seen.contains(&c.id));
        if head.tool_calls.is_empty() && head.content.is_empty() && head.parts.is_empty() {
            continue;
        }
        out.push(head);
        out.extend(results);
    }
    out
}
