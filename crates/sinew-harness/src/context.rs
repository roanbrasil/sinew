//! Context management: token estimates, history compaction and large
//! tool-output offloading.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sinew_core::{ChatModel, ChatRequest, Message, Role};
use sinew_trace::{traced_chat, RunRef, Tracer};

/// Context-window settings.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextConfig {
    /// Estimated prompt size (system prompt + history) above which older
    /// turns are summarized. Default 24 000 tokens.
    pub max_context_tokens: usize,
    /// Messages kept verbatim at the end of the history when compacting
    /// (default 10). Tool calls and their results are never split.
    pub keep_recent: usize,
    /// Tool outputs longer than this many characters are written to a file
    /// and replaced with a preview and the file path (default 20 000;
    /// `None` disables offloading).
    pub offload_threshold: Option<usize>,
    /// Characters of an offloaded output kept inline (default 2 000).
    pub offload_preview: usize,
    /// Where offloaded outputs go. Default: `<workspace>/.sinew/tool-outputs`,
    /// or a `sinew-harness` folder in the system temp dir without a workspace.
    pub offload_dir: Option<PathBuf>,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 24_000,
            keep_recent: 10,
            offload_threshold: Some(20_000),
            offload_preview: 2_000,
            offload_dir: None,
        }
    }
}

/// Rough token count of a text (about four characters per token).
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Rough token count of one message, including tool calls and overhead.
pub fn estimate_message_tokens(m: &Message) -> usize {
    4 + estimate_tokens(&m.content)
        + m.tool_calls
            .iter()
            .map(|c| 4 + estimate_tokens(&c.name) + estimate_tokens(&c.arguments.to_string()))
            .sum::<usize>()
}

pub fn estimate_messages(messages: &[Message]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

/// Where the kept tail of `messages` starts when compacting, or `None` when
/// there is not enough history to compact. The tail never starts with a tool
/// result, so a tool call and its results stay together.
pub fn compaction_split(messages: &[Message], keep_recent: usize) -> Option<usize> {
    let mut split = messages.len().checked_sub(keep_recent.max(1))?;
    while split > 0 && messages[split].role == Role::Tool {
        split -= 1;
    }
    (split >= 2).then_some(split)
}

/// Result of [`compact`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionInfo {
    pub removed_messages: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
}

/// Header of the message that replaces compacted history.
pub const SUMMARY_HEADER: &str =
    "Summary of the earlier conversation (older turns were compacted):";

fn transcript(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = match m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool result",
        };
        out.push_str(&format!("[{role}] {}\n", m.content));
        for c in &m.tool_calls {
            out.push_str(&format!("[tool call] {}({})\n", c.name, c.arguments));
        }
    }
    out
}

/// Summarizes everything before the split point with `model` and returns
/// the new history: one summary message followed by the kept tail.
pub async fn compact(
    model: &dyn ChatModel,
    messages: &[Message],
    keep_recent: usize,
    tracer: Option<&Tracer>,
    parent: Option<&RunRef>,
) -> sinew_core::Result<Option<(Vec<Message>, CompactionInfo)>> {
    let Some(split) = compaction_split(messages, keep_recent) else {
        return Ok(None);
    };
    let tokens_before = estimate_messages(messages);
    let request = ChatRequest::new(vec![
        Message::system(
            "You compress conversation history for an AI agent that will continue the work. \
             Write a concise but complete summary of the conversation below: the user's goals \
             and constraints, decisions made, important facts and numbers, files read or \
             changed, tool results that still matter, and open tasks. Plain prose or bullets; \
             no preamble.",
        ),
        Message::user(transcript(&messages[..split])),
    ]);
    let resp = traced_chat(model, request, tracer, parent).await?;
    let mut out = vec![Message::system(format!(
        "{SUMMARY_HEADER}\n{}",
        resp.message.content.trim()
    ))];
    out.extend_from_slice(&messages[split..]);
    let info = CompactionInfo {
        removed_messages: split,
        tokens_before,
        tokens_after: estimate_messages(&out),
    };
    Ok(Some((out, info)))
}

/// Makes every assistant tool call have a result: calls left without one
/// (cancelled turn, budget stop, abandoned approval) get a synthetic
/// "not executed" result so the history stays valid for every provider.
pub fn repair_tool_pairs(messages: &mut Vec<Message>) {
    let mut i = 0;
    while i < messages.len() {
        if messages[i].role == Role::Assistant && messages[i].has_tool_calls() {
            let mut j = i + 1;
            let mut answered = Vec::new();
            while j < messages.len() && messages[j].role == Role::Tool {
                if let Some(id) = &messages[j].tool_call_id {
                    answered.push(id.clone());
                }
                j += 1;
            }
            let missing: Vec<Message> = messages[i]
                .tool_calls
                .iter()
                .filter(|c| !answered.contains(&c.id))
                .map(|c| {
                    Message::tool_result(
                        c.id.clone(),
                        "error: not executed (the turn was interrupted)",
                    )
                })
                .collect();
            let n = missing.len();
            messages.splice(j..j, missing);
            i = j + n;
        } else {
            i += 1;
        }
    }
}

/// Writes `output` to `dir/<id>.txt` when it is longer than `threshold`
/// characters and returns a preview plus the path; otherwise returns it
/// unchanged. `display` is how the path is shown to the model.
pub fn offload_output(
    output: String,
    dir: &Path,
    display_dir: &str,
    id: &str,
    threshold: usize,
    preview: usize,
) -> String {
    let chars = output.chars().count();
    if chars <= threshold {
        return output;
    }
    let safe: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let file = format!("{safe}.txt");
    let saved = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(dir.join(&file), &output));
    let head: String = output.chars().take(preview).collect();
    match saved {
        Ok(()) => format!(
            "{head}\n\n[output truncated: {chars} characters in total. The full output was saved to `{display_dir}/{file}`; read it with read_file (offset/limit) or search it with grep.]"
        ),
        Err(e) => format!("{head}\n\n[output truncated: {chars} characters in total; saving the full output failed: {e}]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sinew_core::ToolCall;

    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "t".into(),
            arguments: json!({}),
        }
    }

    #[test]
    fn split_keeps_tool_pairs() {
        let msgs = vec![
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
            Message::assistant_with_tools("", vec![call("1"), call("2")]),
            Message::tool_result("1", "x"),
            Message::tool_result("2", "y"),
            Message::assistant("done"),
        ];
        // keep 2 would start at a tool result; move back to the call.
        assert_eq!(compaction_split(&msgs, 2), Some(3));
        assert_eq!(compaction_split(&msgs, 3), Some(3));
        assert_eq!(compaction_split(&msgs, 6), None);
    }

    #[test]
    fn repairs_dangling_calls() {
        let mut msgs = vec![
            Message::user("a"),
            Message::assistant_with_tools("", vec![call("1"), call("2")]),
            Message::tool_result("1", "x"),
            Message::user("next"),
        ];
        repair_tool_pairs(&mut msgs);
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[3].tool_call_id.as_deref(), Some("2"));
        assert_eq!(msgs[4].content, "next");
        let before = msgs.clone();
        repair_tool_pairs(&mut msgs);
        assert_eq!(msgs, before);
    }

    #[test]
    fn offloads_long_output() {
        let dir = tempfile::tempdir().unwrap();
        let long = "x".repeat(100);
        let out = offload_output(long.clone(), dir.path(), ".sinew/out", "call/1", 50, 10);
        assert!(out.starts_with("xxxxxxxxxx\n"));
        assert!(out.contains(".sinew/out/call_1.txt"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("call_1.txt")).unwrap(),
            long
        );
        assert_eq!(
            offload_output("short".into(), dir.path(), "d", "i", 50, 10),
            "short"
        );
    }
}
