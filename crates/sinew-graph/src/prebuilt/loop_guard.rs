//! Loop and runaway detection for tool-calling agents.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinew_core::{Message, Role, ToolCall};

/// What the agent answers when a [`LoopGuard`] stops the run.
pub const LOOP_STOP_REPLY: &str =
    "I stopped because I was repeating the same actions without making progress.";

/// What to do when a [`LoopGuard`] detects a loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopAction {
    /// Do not run the repeated call; answer it with a tool result telling the
    /// model it is looping, so it can change course (default).
    #[default]
    Nudge,
    /// End the run with [`LOOP_STOP_REPLY`].
    Stop,
    /// Pause with [`NodeContext::interrupt`](crate::NodeContext::interrupt)
    /// and let a human decide. Resume with `true` to run the calls anyway or
    /// `false` to stop.
    Interrupt,
}

/// Detects agents that repeat themselves instead of progressing: the same
/// tool called with the same arguments again and again, or tool calls that
/// keep failing. Checks run per turn (the messages since the last user
/// message), right before tools execute.
///
/// ```
/// use sinew_graph::prebuilt::{LoopAction, LoopGuard, ReactAgent};
/// # fn demo(model: std::sync::Arc<dyn sinew_core::ChatModel>, tools: sinew_core::ToolBox) {
/// let agent = ReactAgent::new(model, tools)
///     .loop_guard(LoopGuard::default().max_repeats(2).on_loop(LoopAction::Stop))
///     .compile();
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopGuard {
    /// Identical calls (same tool, same arguments) allowed per turn before
    /// the next one counts as a loop (default 2: the third is caught).
    pub max_repeats: usize,
    /// Consecutive failing tool results allowed per turn before the run is
    /// treated as stuck (default 3). `0` disables the check.
    pub max_consecutive_errors: usize,
    /// What to do on a repeated call (default [`LoopAction::Nudge`]).
    /// Too many consecutive errors always use [`LoopAction::Stop`] unless
    /// this is [`LoopAction::Interrupt`].
    pub action: LoopAction,
}

impl Default for LoopGuard {
    fn default() -> Self {
        Self {
            max_repeats: 2,
            max_consecutive_errors: 3,
            action: LoopAction::Nudge,
        }
    }
}

impl LoopGuard {
    pub fn max_repeats(mut self, n: usize) -> Self {
        self.max_repeats = n.max(1);
        self
    }

    pub fn max_consecutive_errors(mut self, n: usize) -> Self {
        self.max_consecutive_errors = n;
        self
    }

    pub fn on_loop(mut self, action: LoopAction) -> Self {
        self.action = action;
        self
    }

    /// Inspects the conversation before the tool calls in its last message
    /// run.
    pub fn check(&self, messages: &[Message]) -> LoopCheck {
        let Some((last, earlier)) = messages.split_last() else {
            return LoopCheck::default();
        };
        let turn_start = earlier
            .iter()
            .rposition(|m| m.role == Role::User)
            .map_or(0, |i| i + 1);
        let turn = &earlier[turn_start..];

        let mut seen: HashMap<String, usize> = HashMap::new();
        for m in turn.iter().filter(|m| m.role == Role::Assistant) {
            for c in &m.tool_calls {
                *seen.entry(call_key(c)).or_default() += 1;
            }
        }
        let mut repeated = Vec::new();
        for c in &last.tool_calls {
            let n = seen.entry(call_key(c)).or_default();
            if *n >= self.max_repeats {
                repeated.push(RepeatedCall {
                    id: c.id.clone(),
                    name: c.name.clone(),
                    times: *n,
                });
            }
            *n += 1;
        }

        let consecutive_errors = turn
            .iter()
            .rev()
            .filter(|m| m.role == Role::Tool)
            .take_while(|m| m.content.starts_with("error: "))
            .count();
        let stuck =
            self.max_consecutive_errors > 0 && consecutive_errors >= self.max_consecutive_errors;
        LoopCheck {
            repeated,
            consecutive_errors,
            stuck,
        }
    }

    /// The tool result sent instead of running a repeated call.
    pub fn nudge_message(call: &RepeatedCall) -> String {
        format!(
            "error: loop detected. You already called `{}` with these exact arguments {} time{} \
             in this turn. Do not call it again with the same arguments: use the earlier \
             result, try different arguments or another tool, or give your final answer.",
            call.name,
            call.times,
            if call.times == 1 { "" } else { "s" }
        )
    }
}

/// Result of [`LoopGuard::check`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoopCheck {
    /// Calls in the last message that repeat an earlier call too many times.
    pub repeated: Vec<RepeatedCall>,
    /// Failing tool results at the end of the turn.
    pub consecutive_errors: usize,
    /// `consecutive_errors` reached the limit.
    pub stuck: bool,
}

impl LoopCheck {
    pub fn is_loop(&self) -> bool {
        self.stuck || !self.repeated.is_empty()
    }
}

/// A tool call caught by [`LoopGuard`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepeatedCall {
    pub id: String,
    pub name: String,
    /// How many identical calls this turn already had.
    pub times: usize,
}

/// Tool name plus arguments with object keys sorted, so `{"a":1,"b":2}` and
/// `{"b":2,"a":1}` count as the same call.
fn call_key(c: &ToolCall) -> String {
    fn canon(v: &Value, out: &mut String) {
        match v {
            Value::Object(m) => {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                out.push('{');
                for k in keys {
                    out.push_str(&serde_json::to_string(k).unwrap_or_default());
                    out.push(':');
                    canon(&m[k], out);
                    out.push(',');
                }
                out.push('}');
            }
            Value::Array(a) => {
                out.push('[');
                for x in a {
                    canon(x, out);
                    out.push(',');
                }
                out.push(']');
            }
            other => out.push_str(&other.to_string()),
        }
    }
    let mut s = c.name.clone();
    s.push('\u{1f}');
    canon(&c.arguments, &mut s);
    s
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn call(id: &str, name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: args,
        }
    }

    fn turn(rounds: &[(&str, Value, &str)]) -> Vec<Message> {
        let mut m = vec![Message::user("go")];
        for (i, (name, args, result)) in rounds.iter().enumerate() {
            let id = i.to_string();
            m.push(Message::assistant_with_tools(
                "",
                vec![call(&id, name, args.clone())],
            ));
            m.push(Message::tool_result(id, *result));
        }
        m
    }

    #[test]
    fn third_identical_call_is_a_loop_with_reordered_keys() {
        let g = LoopGuard::default();
        let mut m = turn(&[
            ("search", json!({"q": "x", "n": 1}), "r"),
            ("search", json!({"n": 1, "q": "x"}), "r"),
        ]);
        m.push(Message::assistant_with_tools(
            "",
            vec![call("9", "search", json!({"q": "x", "n": 1}))],
        ));
        let c = g.check(&m);
        assert_eq!(c.repeated.len(), 1);
        assert_eq!(c.repeated[0].times, 2);
        assert!(c.is_loop());
    }

    #[test]
    fn different_arguments_and_previous_turns_do_not_count() {
        let g = LoopGuard::default();
        let mut m = turn(&[
            ("search", json!({"q": "x"}), "r"),
            ("search", json!({"q": "x"}), "r"),
        ]);
        m.push(Message::assistant("done"));
        m.push(Message::user("again"));
        m.push(Message::assistant_with_tools(
            "",
            vec![call("9", "search", json!({"q": "x"}))],
        ));
        assert!(!g.check(&m).is_loop());

        let mut m = turn(&[
            ("search", json!({"q": "a"}), "r"),
            ("search", json!({"q": "b"}), "r"),
        ]);
        m.push(Message::assistant_with_tools(
            "",
            vec![call("9", "search", json!({"q": "c"}))],
        ));
        assert!(!g.check(&m).is_loop());
    }

    #[test]
    fn duplicate_calls_in_one_message_count() {
        let g = LoopGuard::default().max_repeats(1);
        let m = vec![
            Message::user("go"),
            Message::assistant_with_tools(
                "",
                vec![call("1", "f", json!({})), call("2", "f", json!({}))],
            ),
        ];
        let c = g.check(&m);
        assert_eq!(c.repeated.len(), 1);
        assert_eq!(c.repeated[0].id, "2");
    }

    #[test]
    fn consecutive_errors_mark_the_run_stuck() {
        let g = LoopGuard::default();
        let mut m = turn(&[
            ("a", json!({"i": 1}), "error: boom"),
            ("a", json!({"i": 2}), "error: boom"),
            ("a", json!({"i": 3}), "error: boom"),
        ]);
        m.push(Message::assistant_with_tools(
            "",
            vec![call("9", "a", json!({"i": 4}))],
        ));
        let c = g.check(&m);
        assert!(c.stuck && c.repeated.is_empty());
        assert_eq!(c.consecutive_errors, 3);

        let g = LoopGuard::default().max_consecutive_errors(0);
        assert!(!g.check(&m).is_loop());
    }
}
