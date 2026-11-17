//! Context engineering: fit a conversation into a token budget.
//!
//! Long agent runs grow the message list until it no longer fits the model's
//! context window (or costs more than it should). This module provides
//! strategies that shrink a `Vec<Message>` and a [`ContextPolicy`] that
//! chains them against a token budget:
//!
//! | Strategy | What it does | Needs a model |
//! |---|---|---|
//! | [`KeepLast`] | keeps the last N messages or the last N tokens (system messages always kept) | no |
//! | [`ClearToolResults`] | replaces old tool results with a short placeholder, or truncates them | no |
//! | [`Summarize`] | folds older turns into one running summary message, updated incrementally | yes ([`ChatModel`](crate::ChatModel)) |
//! | [`SemanticSelect`] | keeps the older turns most similar to the latest user message | an embedder ([`TextEmbedder`]) |
//!
//! Every strategy works on *message groups* (see [`message_groups`]): an
//! assistant message with tool calls and the tool results that answer it are
//! kept or dropped together, so the output never contains a tool result
//! without its call, or a call without its results, when the input did not.
//! Providers reject such histories with a 400 error. [`check_tool_pairing`]
//! verifies a history.
//!
//! Token counts come from a [`TokenCounter`]; the default,
//! [`EstimateCounter`], uses [`pricing::estimate_tokens`](crate::pricing::estimate_tokens).
//!
//! ```
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! use sinew_core::context::{ClearToolResults, ContextPolicy, KeepLast};
//! use sinew_core::Message;
//!
//! // Fit into 2 000 tokens: first clear old tool results, then trim.
//! let policy = ContextPolicy::new(2_000)
//!     .then(ClearToolResults::new())
//!     .then(KeepLast::fit_budget());
//!
//! let mut history = vec![Message::system("You are terse.")];
//! for i in 0..200 {
//!     history.push(Message::user(format!("question {i} with some padding text")));
//!     history.push(Message::assistant(format!("answer {i} with some padding text")));
//! }
//! let report = policy.apply_with_report(history, 0).await.unwrap();
//! assert!(report.fits());
//! assert_eq!(report.messages[0].content, "You are terse.");
//! # }
//! ```
//!
//! See the [context engineering guide](https://github.com/roanbrasil/sinew/blob/main/docs/context.md)
//! for when to use each strategy.

mod counter;
mod pairing;
mod policy;
mod semantic;
mod summarize;
mod tool_results;
mod trim;

use async_trait::async_trait;

use crate::{Message, Result};

pub use counter::{EstimateCounter, TokenCounter};
pub use pairing::{check_tool_pairing, drop_unpaired_tool_messages, message_groups, PairingError};
pub use policy::{ContextPolicy, ContextReport};
pub use semantic::{embedder_fn, EmbedderFn, SemanticSelect, TextEmbedder};
pub use summarize::{
    render_transcript, KeepRecent, Summarize, DEFAULT_SUMMARY_HEADER, DEFAULT_SUMMARY_PROMPT,
};
pub use tool_results::{ClearToolResults, CLEARED_MARKER, TRUNCATED_MARKER};
pub use trim::KeepLast;

use std::sync::Arc;

/// The token budget a strategy works against, with the counter that
/// measures it.
#[derive(Clone)]
pub struct ContextBudget {
    /// Maximum tokens the messages may use.
    pub max_tokens: usize,
    /// How tokens are counted.
    pub counter: Arc<dyn TokenCounter>,
}

impl ContextBudget {
    /// A budget of `max_tokens` counted with [`EstimateCounter`].
    ///
    /// ```
    /// use sinew_core::context::ContextBudget;
    /// use sinew_core::Message;
    ///
    /// let budget = ContextBudget::new(100);
    /// assert!(budget.fits(&[Message::user("hello")]));
    /// ```
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            counter: Arc::new(EstimateCounter),
        }
    }

    /// Replaces the token counter.
    pub fn with_counter(mut self, counter: impl TokenCounter + 'static) -> Self {
        self.counter = Arc::new(counter);
        self
    }

    /// Token count of `messages`.
    pub fn count(&self, messages: &[Message]) -> usize {
        self.counter.count_messages(messages)
    }

    /// Whether `messages` fit in the budget.
    pub fn fits(&self, messages: &[Message]) -> bool {
        self.count(messages) <= self.max_tokens
    }
}

impl std::fmt::Debug for ContextBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextBudget")
            .field("max_tokens", &self.max_tokens)
            .finish_non_exhaustive()
    }
}

/// One way to shrink a conversation.
///
/// Implementations must keep tool calls and their results together (work on
/// [`message_groups`]) and must not change the order of the messages they
/// keep. The budget is a target: strategies with their own limits (such as
/// [`KeepLast::messages`]) may ignore it.
///
/// ```
/// use async_trait::async_trait;
/// use sinew_core::context::{ContextBudget, ContextStrategy};
/// use sinew_core::{Message, Result, Role};
///
/// /// Drops every assistant message that has no text and no tool calls.
/// struct DropEmptyReplies;
///
/// #[async_trait]
/// impl ContextStrategy for DropEmptyReplies {
///     fn name(&self) -> &str {
///         "drop_empty_replies"
///     }
///     async fn apply(&self, messages: Vec<Message>, _: &ContextBudget) -> Result<Vec<Message>> {
///         Ok(messages
///             .into_iter()
///             .filter(|m| !(m.role == Role::Assistant && m.content.is_empty() && m.tool_calls.is_empty()))
///             .collect())
///     }
/// }
/// ```
#[async_trait]
pub trait ContextStrategy: Send + Sync {
    /// Short identifier, reported in [`ContextReport::applied`].
    fn name(&self) -> &str;

    /// Returns the shrunk conversation.
    async fn apply(&self, messages: Vec<Message>, budget: &ContextBudget) -> Result<Vec<Message>>;
}

#[async_trait]
impl<T: ContextStrategy + ?Sized> ContextStrategy for Arc<T> {
    fn name(&self) -> &str {
        (**self).name()
    }
    async fn apply(&self, messages: Vec<Message>, budget: &ContextBudget) -> Result<Vec<Message>> {
        (**self).apply(messages, budget).await
    }
}

/// Keeps the messages whose index is `true` in `keep`, in order.
pub(crate) fn retain_mask(messages: Vec<Message>, keep: &[bool]) -> Vec<Message> {
    messages
        .into_iter()
        .zip(keep)
        .filter_map(|(m, k)| k.then_some(m))
        .collect()
}

#[cfg(test)]
pub(crate) mod testkit;
