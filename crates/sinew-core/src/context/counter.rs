//! Token counting for context budgets.

use std::sync::Arc;

use crate::message::{ContentPart, Message};

/// Counts tokens for context budgets.
///
/// Only [`count_text`](Self::count_text) is required; the message-level
/// methods add a per-message overhead, tool-call arguments and a flat
/// allowance per image (800), document (1 500) and audio clip (1 000), the
/// same rules as [`pricing::estimate_request_tokens`](crate::pricing::estimate_request_tokens).
/// Implement it with a real tokenizer for exact counts; any
/// `Fn(&str) -> usize` closure is a counter too.
///
/// ```
/// use sinew_core::context::{EstimateCounter, TokenCounter};
/// use sinew_core::Message;
///
/// let words = |text: &str| text.split_whitespace().count();
/// assert_eq!(words.count_text("one two three"), 3);
/// assert_eq!(words.count_message(&Message::user("one two")), 2 + 4);
/// assert!(EstimateCounter.count_text("hello world") > 0);
/// ```
pub trait TokenCounter: Send + Sync {
    /// Tokens in a piece of text.
    fn count_text(&self, text: &str) -> usize;

    /// Tokens in one message, including a 4-token overhead.
    fn count_message(&self, message: &Message) -> usize {
        let mut n = 4 + self.count_text(&message.content);
        for c in &message.tool_calls {
            n += self.count_text(&c.name) + self.count_text(&c.arguments.to_string());
        }
        for p in &message.parts {
            n += match p {
                ContentPart::Text { text } => self.count_text(text),
                ContentPart::Image { .. } => 800,
                ContentPart::Document { .. } => 1500,
                ContentPart::Audio { .. } => 1000,
            };
        }
        n
    }

    /// Tokens in a list of messages.
    fn count_messages(&self, messages: &[Message]) -> usize {
        messages.iter().map(|m| self.count_message(m)).sum()
    }
}

/// The default counter: [`pricing::estimate_tokens`](crate::pricing::estimate_tokens),
/// about four characters per token. Good for budgeting, not for billing.
#[derive(Debug, Clone, Copy, Default)]
pub struct EstimateCounter;

impl TokenCounter for EstimateCounter {
    fn count_text(&self, text: &str) -> usize {
        crate::pricing::estimate_tokens(text) as usize
    }
}

impl<F> TokenCounter for F
where
    F: Fn(&str) -> usize + Send + Sync,
{
    fn count_text(&self, text: &str) -> usize {
        self(text)
    }
}

impl<T: TokenCounter + ?Sized> TokenCounter for Arc<T> {
    fn count_text(&self, text: &str) -> usize {
        (**self).count_text(text)
    }
    fn count_message(&self, message: &Message) -> usize {
        (**self).count_message(message)
    }
    fn count_messages(&self, messages: &[Message]) -> usize {
        (**self).count_messages(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pricing, ChatRequest};

    #[test]
    fn estimate_counter_matches_request_estimate() {
        let msgs = vec![
            Message::system("be brief"),
            Message::user("what is the capital of France?"),
            Message::user_with_images("this?", [ContentPart::image_url("https://x/a.png")]),
        ];
        let req = pricing::estimate_request_tokens(&ChatRequest::new(msgs.clone()));
        // The request estimate adds 3 tokens of priming.
        assert_eq!(EstimateCounter.count_messages(&msgs) + 3, req as usize);
    }
}
