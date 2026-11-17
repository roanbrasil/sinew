//! Chain strategies against a token budget.

use std::sync::Arc;

use super::{
    ClearToolResults, ContextBudget, ContextStrategy, EstimateCounter, KeepLast, Summarize,
    TokenCounter,
};
use crate::{ChatModel, Message, Result};

/// A token budget and an ordered list of strategies that are applied, one
/// after the other, only while the conversation is over the budget.
///
/// For example "fit into 8k tokens: first clear old tool results, then
/// summarize, then trim":
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use std::sync::Arc;
/// use sinew_core::context::{ClearToolResults, ContextPolicy, KeepLast, Summarize};
/// use sinew_core::mock::ScriptedModel;
/// use sinew_core::Message;
///
/// let summarizer = Arc::new(ScriptedModel::new([Message::assistant("The user asked 60 questions.")]));
/// let policy = ContextPolicy::new(8_000)
///     .then(ClearToolResults::new())
///     .then(Summarize::new(summarizer).max_chunk_tokens(20_000))
///     .then(KeepLast::fit_budget());
///
/// let mut history = Vec::new();
/// for i in 0..60 {
///     history.push(Message::user(format!("question {i}: {}", "context ".repeat(100))));
///     history.push(Message::assistant(format!("answer {i}")));
/// }
/// let report = policy.apply_with_report(history, 0).await.unwrap();
/// assert!(report.fits());
/// assert_eq!(report.applied, ["clear_tool_results", "summarize"]);
/// # }
/// ```
///
/// A conversation that already fits is returned unchanged, without calling
/// any strategy. End the list with [`KeepLast::fit_budget`] to guarantee the
/// budget whenever the system messages and the newest message group fit.
/// Strategies keep tool calls and results paired, so the output is a valid
/// history when the input is.
///
/// A policy is cheap to clone (strategies are shared), and strategies with
/// state (the [`Summarize`] cache) keep it across clones.
#[derive(Clone)]
pub struct ContextPolicy {
    max_tokens: usize,
    counter: Arc<dyn TokenCounter>,
    stages: Vec<Arc<dyn ContextStrategy>>,
}

impl std::fmt::Debug for ContextPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextPolicy")
            .field("max_tokens", &self.max_tokens)
            .field(
                "stages",
                &self.stages.iter().map(|s| s.name()).collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

/// What [`ContextPolicy::apply_with_report`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextReport {
    /// The messages to send.
    pub messages: Vec<Message>,
    /// Tokens before any strategy ran.
    pub tokens_before: usize,
    /// Tokens of [`messages`](Self::messages).
    pub tokens_after: usize,
    /// The budget for the messages (the policy budget minus the reserve).
    pub budget: usize,
    /// Names of the strategies that ran, in order.
    pub applied: Vec<String>,
}

impl ContextReport {
    /// Whether the messages fit the budget.
    pub fn fits(&self) -> bool {
        self.tokens_after <= self.budget
    }

    /// Whether any strategy ran.
    pub fn changed(&self) -> bool {
        !self.applied.is_empty()
    }
}

impl ContextPolicy {
    /// An empty policy with a budget of `max_tokens`, counted with
    /// [`EstimateCounter`].
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            counter: Arc::new(EstimateCounter),
            stages: Vec::new(),
        }
    }

    /// Clears old tool results, then keeps the most recent messages that fit.
    /// No model calls.
    ///
    /// ```
    /// use sinew_core::context::ContextPolicy;
    /// let policy = ContextPolicy::trimming(16_000);
    /// assert_eq!(policy.stage_names(), ["clear_tool_results", "keep_last"]);
    /// ```
    pub fn trimming(max_tokens: usize) -> Self {
        Self::new(max_tokens)
            .then(ClearToolResults::new())
            .then(KeepLast::fit_budget())
    }

    /// Clears old tool results, summarizes older turns with `model`, then
    /// keeps the most recent messages that fit.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use sinew_core::context::ContextPolicy;
    /// use sinew_core::mock::ScriptedModel;
    ///
    /// let policy = ContextPolicy::summarizing(16_000, Arc::new(ScriptedModel::new([])));
    /// assert_eq!(policy.stage_names(), ["clear_tool_results", "summarize", "keep_last"]);
    /// ```
    pub fn summarizing(max_tokens: usize, model: Arc<dyn ChatModel>) -> Self {
        Self::new(max_tokens)
            .then(ClearToolResults::new())
            .then(Summarize::new(model))
            .then(KeepLast::fit_budget())
    }

    /// Replaces the token counter (for example with a real tokenizer).
    pub fn counter(mut self, counter: impl TokenCounter + 'static) -> Self {
        self.counter = Arc::new(counter);
        self
    }

    /// Appends a strategy.
    pub fn then(self, strategy: impl ContextStrategy + 'static) -> Self {
        self.then_arc(Arc::new(strategy))
    }

    /// Appends a shared strategy.
    pub fn then_arc(mut self, strategy: Arc<dyn ContextStrategy>) -> Self {
        self.stages.push(strategy);
        self
    }

    /// The token budget.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// The token counter.
    pub fn token_counter(&self) -> &Arc<dyn TokenCounter> {
        &self.counter
    }

    /// Names of the strategies, in order.
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name()).collect()
    }

    /// Fits `messages` into the budget; see [`apply_with_report`](Self::apply_with_report).
    pub async fn apply(&self, messages: Vec<Message>) -> Result<Vec<Message>> {
        Ok(self.apply_with_report(messages, 0).await?.messages)
    }

    /// Fits `messages` into the budget minus `reserved` tokens (room for
    /// a system prompt or tool schemas added later, or for the answer), and
    /// reports what was done. Strategies run in order until the messages
    /// fit; an error from a strategy is returned as is.
    pub async fn apply_with_report(
        &self,
        messages: Vec<Message>,
        reserved: usize,
    ) -> Result<ContextReport> {
        let budget = ContextBudget {
            max_tokens: self.max_tokens.saturating_sub(reserved),
            counter: self.counter.clone(),
        };
        let tokens_before = budget.count(&messages);
        let mut tokens = tokens_before;
        let mut messages = messages;
        let mut applied = Vec::new();
        for stage in &self.stages {
            if tokens <= budget.max_tokens {
                break;
            }
            messages = stage.apply(messages, &budget).await?;
            tokens = budget.count(&messages);
            applied.push(stage.name().to_string());
        }
        Ok(ContextReport {
            messages,
            tokens_before,
            tokens_after: tokens,
            budget: budget.max_tokens,
            applied,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::testkit::*;
    use crate::context::{check_tool_pairing, embedder_fn, SemanticSelect};
    use crate::mock::ScriptedModel;
    use crate::Role;

    fn summaries(n: usize) -> Arc<ScriptedModel> {
        Arc::new(ScriptedModel::new((0..n).map(|i| {
            Message::assistant(format!("summary {i}: {}", "fact ".repeat(50)))
        })))
    }

    #[tokio::test]
    async fn fitting_input_is_untouched() {
        let msgs = transcript(1, 20);
        let r = ContextPolicy::trimming(1_000_000)
            .apply_with_report(msgs.clone(), 0)
            .await
            .unwrap();
        assert_eq!(r.messages, msgs);
        assert!(!r.changed());
    }

    /// Randomized check over many seeded transcripts, budgets and
    /// pipelines: the result fits and tool pairs hold.
    #[tokio::test]
    async fn randomized_budgets_and_pairing() {
        let mut rng = Rng::new(42);
        for seed in 0..60 {
            let len = 50 + rng.below(600) as usize;
            let msgs = transcript(seed, len);
            check_tool_pairing(&msgs).unwrap();
            let budget = 1_500 + rng.below(12_000) as usize;
            let reserved = rng.below(500) as usize;
            let policy = match seed % 4 {
                0 => ContextPolicy::trimming(budget),
                1 => ContextPolicy::summarizing(budget, summaries(400)),
                2 => ContextPolicy::new(budget)
                    .then(ClearToolResults::new().truncate_to(100 + rng.below(400) as usize))
                    .then(KeepLast::fit_budget().and_messages(5 + rng.below(50) as usize)),
                _ => ContextPolicy::new(budget)
                    .then(SemanticSelect::new(embedder_fn(|t: &[String]| {
                        Ok(t.iter().map(|s| vec![s.len() as f32 % 7.0, 1.0]).collect())
                    })))
                    .then(KeepLast::fit_budget()),
            };
            let r = policy.apply_with_report(msgs, reserved).await.unwrap();
            check_tool_pairing(&r.messages).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            assert!(
                r.fits(),
                "seed {seed}: {} > {} via {:?}",
                r.tokens_after,
                r.budget,
                r.applied
            );
            assert_eq!(r.messages[0].role, Role::System, "seed {seed}");
            assert_eq!(r.tokens_after, EstimateCounter.count_messages(&r.messages));
        }
    }

    #[tokio::test]
    async fn custom_counter_is_used() {
        let msgs = transcript(5, 100);
        let words = |t: &str| t.split_whitespace().count();
        let r = ContextPolicy::trimming(500)
            .counter(words)
            .apply_with_report(msgs, 0)
            .await
            .unwrap();
        assert!(r.fits());
        assert_eq!(r.tokens_after, words.count_messages(&r.messages));
    }
}
