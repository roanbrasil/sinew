//! Keep the most recent messages.

use async_trait::async_trait;

use super::{message_groups, retain_mask, ContextBudget, ContextStrategy};
use crate::message::{Message, Role};
use crate::Result;

/// Keeps the most recent messages that fit a message count and/or a token
/// limit, and every system message.
///
/// Older messages are dropped a whole group at a time (see
/// [`message_groups`]), so a tool call and its results stay together. The
/// newest group is always kept, even when it alone exceeds the limit. With
/// [`start_on_user`](Self::start_on_user) (on by default) the kept
/// conversation starts at a user message when there is one among the kept
/// groups, since some providers expect the first non-system message to come
/// from the user.
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use sinew_core::context::{ContextBudget, ContextStrategy, KeepLast};
/// use sinew_core::{Message, Role};
///
/// let msgs = vec![
///     Message::system("rules"),
///     Message::user("1"), Message::assistant("a"),
///     Message::user("2"), Message::assistant("b"),
/// ];
/// let out = KeepLast::messages(2).apply(msgs, &ContextBudget::new(1_000)).await.unwrap();
/// let text: Vec<&str> = out.iter().map(|m| m.content.as_str()).collect();
/// assert_eq!(text, ["rules", "2", "b"]);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct KeepLast {
    max_messages: Option<usize>,
    max_tokens: Option<usize>,
    use_budget: bool,
    start_on_user: bool,
}

impl KeepLast {
    /// Keeps at most `n` non-system messages (rounded down to whole groups).
    pub fn messages(n: usize) -> Self {
        Self {
            max_messages: Some(n),
            max_tokens: None,
            use_budget: false,
            start_on_user: true,
        }
    }

    /// Keeps the most recent messages whose tokens, together with the
    /// system messages, are at most `n`.
    pub fn tokens(n: usize) -> Self {
        Self {
            max_messages: None,
            max_tokens: Some(n),
            use_budget: false,
            start_on_user: true,
        }
    }

    /// Keeps as many recent messages as fit the budget it is applied with
    /// (the [`ContextPolicy`](super::ContextPolicy) budget). End a policy
    /// with it to guarantee the budget whenever the system messages and the
    /// newest group fit.
    pub fn fit_budget() -> Self {
        Self {
            max_messages: None,
            max_tokens: None,
            use_budget: true,
            start_on_user: true,
        }
    }

    /// Also limit the number of non-system messages.
    pub fn and_messages(mut self, n: usize) -> Self {
        self.max_messages = Some(n);
        self
    }

    /// Whether to drop leading non-user groups of the kept conversation
    /// (default `true`). Without a user message among the kept groups
    /// nothing extra is dropped.
    pub fn start_on_user(mut self, yes: bool) -> Self {
        self.start_on_user = yes;
        self
    }

    /// The keep-mask for `messages` under this strategy.
    fn mask(&self, messages: &[Message], budget: &ContextBudget) -> Vec<bool> {
        let counter = &budget.counter;
        let mut keep: Vec<bool> = messages.iter().map(|m| m.role == Role::System).collect();
        let system_tokens: usize = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| counter.count_message(m))
            .sum();
        let token_limit = match (self.max_tokens, self.use_budget) {
            (Some(t), true) => Some(t.min(budget.max_tokens)),
            (Some(t), false) => Some(t),
            (None, true) => Some(budget.max_tokens),
            (None, false) => None,
        };
        let token_room = token_limit.map(|t| t.saturating_sub(system_tokens));

        let groups: Vec<_> = message_groups(messages)
            .into_iter()
            .filter(|g| messages[g.start].role != Role::System)
            .collect();
        let (mut used_msgs, mut used_tokens) = (0usize, 0usize);
        let mut kept = Vec::new();
        for g in groups.iter().rev() {
            let n = g.len();
            let t: usize = messages[g.clone()]
                .iter()
                .map(|m| counter.count_message(m))
                .sum();
            let over_msgs = self.max_messages.is_some_and(|max| used_msgs + n > max);
            let over_tokens = token_room.is_some_and(|room| used_tokens + t > room);
            if (over_msgs || over_tokens) && !kept.is_empty() {
                break;
            }
            used_msgs += n;
            used_tokens += t;
            kept.push(g.clone());
            if over_msgs || over_tokens {
                // The newest group alone is over the limit; keep only it.
                break;
            }
        }
        kept.reverse();
        if self.start_on_user {
            if let Some(first_user) = kept
                .iter()
                .position(|g| messages[g.start].role == Role::User)
            {
                kept.drain(..first_user);
            }
        }
        for g in kept {
            for k in &mut keep[g] {
                *k = true;
            }
        }
        keep
    }
}

#[async_trait]
impl ContextStrategy for KeepLast {
    fn name(&self) -> &str {
        "keep_last"
    }

    async fn apply(&self, messages: Vec<Message>, budget: &ContextBudget) -> Result<Vec<Message>> {
        let keep = self.mask(&messages, budget);
        Ok(retain_mask(messages, &keep))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::testkit::*;
    use crate::context::{check_tool_pairing, EstimateCounter, TokenCounter};

    #[tokio::test]
    async fn never_splits_a_tool_group() {
        let msgs = vec![
            Message::user("q"),
            Message::assistant_with_tools("", vec![call("1", "t"), call("2", "t")]),
            Message::tool_result("1", "x"),
            Message::tool_result("2", "y"),
            Message::assistant("done"),
        ];
        let out = KeepLast::messages(3)
            .start_on_user(false)
            .apply(msgs.clone(), &ContextBudget::new(10_000))
            .await
            .unwrap();
        // The tool group (3 messages) plus "done" is 4 > 3, so only "done".
        assert_eq!(out.len(), 1);
        let out = KeepLast::messages(4)
            .start_on_user(false)
            .apply(msgs, &ContextBudget::new(10_000))
            .await
            .unwrap();
        assert_eq!(out.len(), 4);
        check_tool_pairing(&out).unwrap();
    }

    #[tokio::test]
    async fn fits_the_budget_on_long_transcripts() {
        for seed in 0..20 {
            let msgs = transcript(seed, 500);
            check_tool_pairing(&msgs).unwrap();
            let budget = ContextBudget::new(4_000);
            let out = KeepLast::fit_budget().apply(msgs, &budget).await.unwrap();
            check_tool_pairing(&out).unwrap();
            assert!(EstimateCounter.count_messages(&out) <= 4_000);
            assert_eq!(out[0].role, Role::System);
            assert_eq!(out[1].role, Role::User);
        }
    }

    #[tokio::test]
    async fn keeps_newest_group_even_if_over_limit() {
        let msgs = vec![Message::user("old"), Message::user("x ".repeat(500))];
        let out = KeepLast::tokens(10)
            .apply(msgs, &ContextBudget::new(10))
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
    }
}
