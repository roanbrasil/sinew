# Context engineering

Long agent runs grow the message list until it overflows the model's context window, and every token in
it is paid for on every call. `sinew_core::context` shrinks a `Vec<Message>` to a token budget with
composable strategies, and `ContextPolicy` chains them. The module has no graph dependency; `ReactAgent`
applies a policy before each model call.

```rust
use std::sync::Arc;
use sinew_core::context::{ClearToolResults, ContextPolicy, KeepLast, Summarize};

// "Fit into 8k tokens: first clear old tool results, then summarize, then trim."
let policy = ContextPolicy::new(8_000)
    .then(ClearToolResults::new())
    .then(Summarize::new(summarizer.clone()))
    .then(KeepLast::fit_budget());

let messages = policy.apply(history).await?;
```

## Strategies

| Strategy | Effect | Cost | Use it when |
|---|---|---|---|
| `KeepLast` | keeps the last N messages (`messages(n)`), the last N tokens (`tokens(n)`) or what fits the policy budget (`fit_budget()`); system messages are always kept | none | older turns do not matter, or as the last step of a policy to guarantee the budget |
| `ClearToolResults` | replaces tool results the model has already answered with `[tool result cleared to save context: `name`, N characters]`, or keeps their first N characters (`truncate_to(n)`) | none | tool-heavy agents (file reads, search, HTTP): old outputs are large and rarely needed again, while the calls themselves show what was done |
| `Summarize` | folds older turns into one system message holding a running summary written by a `ChatModel`; the recent messages stay verbatim | one model call per update | the early conversation carries goals, decisions and facts the agent still needs |
| `SemanticSelect` | keeps the recent messages plus the older turns most similar to the latest user message (cosine similarity of embeddings) | one embedding call per new message group | long sessions that move between topics, where the current question refers back to an earlier one |

Pick the cheapest strategy that keeps what the agent needs. A good default for tool-calling agents is
`ContextPolicy::trimming(budget)` (clear tool results, then trim), or `ContextPolicy::summarizing(budget,
model)` (clear tool results, summarize, then trim) when the early turns matter.

### Tool calls and results stay paired

A tool result without the assistant message that requested it, or an assistant tool call without its
result, makes providers reject the request with a 400 error. Every strategy works on message groups
(`message_groups`): an assistant message with tool calls and the tool results that follow it are kept or
dropped together. `ClearToolResults` rewrites the content of a result but keeps the message. As long as the
input is a valid history, the output is one too. `check_tool_pairing` verifies a history, and
`drop_unpaired_tool_messages` repairs one assembled elsewhere by removing the unpaired parts.

### KeepLast

- `KeepLast::messages(n)` counts non-system messages, rounded down to whole groups.
- `KeepLast::tokens(n)` and `fit_budget()` count the system messages too, so the result is at most `n`
  (or the budget) whenever the system messages and the newest group fit.
- The newest group is always kept, even when it alone is over the limit.
- `start_on_user(true)` (the default) drops leading assistant or tool groups of the kept part so the
  conversation starts with a user message, which some providers expect. `and_messages(n)` adds a
  message cap to a token limit.

### ClearToolResults

- A result counts as used once an assistant message follows it. Results of the current step (no
  assistant message after them yet) are never touched.
- `keep_last(n)` (default 2) keeps the `n` most recent used results intact; `min_chars(n)` (default 200)
  leaves short results alone.
- With `until_fits(true)` (the default) it clears oldest first and stops as soon as the conversation fits
  the budget; `until_fits(false)` clears everything eligible.
- It is idempotent: cleared or truncated results are recognized by `CLEARED_MARKER` /
  `TRUNCATED_MARKER` and not rewritten again.

Clearing loses the data. When the agent may need a large output again, save it outside the context first
(the harness offloads large tool outputs to files, see [Harness](harness.md#context-management)) so the
placeholder or the truncated head can point to it.

### Summarize

```rust
use sinew_core::context::{KeepRecent, Summarize};

let summarize = Summarize::new(model)
    .keep_recent(KeepRecent::Tokens(3_000))   // or KeepRecent::Messages(10), the default
    .max_chunk_tokens(8_000)                  // per summarizer call
    .max_tool_result_chars(Some(2_000))       // tool results are cut in the summarizer's input
    .prompt("Summarize for a support agent. Keep order numbers and customer names.");
```

The output is the system messages of the summarized part, one system message starting with
`DEFAULT_SUMMARY_HEADER` (change it with `.header(...)`) followed by the summary, and then the kept recent
messages unchanged. Anthropic and Gemini receive all system messages as the system prompt; OpenAI-style
APIs receive them in place.

The summary is updated, not recomputed:

- If the input contains a summary message (because the shrunk history was stored back, for example with
  `ReactAgent::persist_context(true)`), only that summary and the messages after it go to the model.
- Otherwise the strategy remembers its last summaries keyed by a hash of the messages they cover. An agent
  calls it before every model call with the same history plus new messages, so each call sends only the
  previous summary and the new messages. The cache holds 32 entries; `clear_cache()` empties it.
- A long stretch is summarized in chunks of at most `max_chunk_tokens`, each chunk updating the summary,
  so the summarizer's own request stays small.

The summarizer request is a system prompt and one user message with `Current summary:` and `New conversation
turns:` sections (the transcript format is `render_transcript`). Use a small, cheap model for it. The call
is not traced by `ContextPolicy` itself; wrap the model if you need it in traces.

### SemanticSelect

```rust
use sinew_core::context::SemanticSelect;
use sinew_rag::embeddings::{embeddings_from_spec, ContextEmbedder};

let embeddings = embeddings_from_spec("ollama:bge-m3")?;
let select = SemanticSelect::new(ContextEmbedder(embeddings))
    .keep_recent(6)   // recent messages always kept
    .top_k(4)         // older groups kept by similarity
    .min_score(0.3);
```

The embedder is the `TextEmbedder` trait from `sinew-core`, so the core crate does not depend on
`sinew-rag`. `sinew_rag::embeddings::ContextEmbedder` adapts any `sinew_rag::Embeddings`; `embedder_fn`
wraps a plain function. Each older group is embedded once (as its transcript, cut to 2 000 characters) and
cached by text. Selected groups keep their original order. `SemanticSelect` ignores the budget, so follow
it with `KeepLast::fit_budget()` when the budget must hold.

## ContextPolicy

`ContextPolicy::new(max_tokens)` holds the budget, a token counter and the strategies in order.
`apply_with_report(messages, reserved)`:

1. counts the messages; if they fit `max_tokens - reserved`, returns them unchanged without running any
   strategy;
2. otherwise runs the strategies in order, recounting after each, and stops as soon as the messages fit;
3. returns a `ContextReport` with the messages, `tokens_before`, `tokens_after`, `budget`, the names of the
   strategies that ran (`applied`) and `fits()`.

`reserved` is room for what is added after the policy runs: tool schemas, a system prompt, or the expected
answer length. Strategies only run while the conversation is over budget, so a cheap strategy placed first
(`ClearToolResults`) often avoids the summarizer call altogether. End with `KeepLast::fit_budget()` to
guarantee the budget whenever the system messages and the newest group fit. A policy is cheap to clone;
clones share strategies and their caches.

Custom strategies implement `ContextStrategy` (`name` and an async `apply(messages, &ContextBudget)`); keep
groups intact and do not reorder the messages you keep.

## Token counting

Budgets are measured with a `TokenCounter`. The default, `EstimateCounter`, uses
`pricing::estimate_tokens` (about four characters per token, at least 1.3 tokens per word) plus 4 tokens
per message and flat allowances for images (800), documents (1 500) and audio (1 000), the same rules as
`pricing::estimate_request_tokens`. Estimates are approximate, so leave headroom below the real context
window. For exact counts implement `count_text` with a real tokenizer, or pass any `Fn(&str) -> usize`:

```rust
let policy = ContextPolicy::trimming(8_000).counter(|text: &str| my_tokenizer.count(text));
```

## In a ReAct agent

```rust
use sinew_core::context::ContextPolicy;
use sinew_graph::prebuilt::ReactAgent;

let agent = ReactAgent::new(model.clone(), tools)
    .system_prompt("You are a research assistant.")
    .context_policy(ContextPolicy::summarizing(16_000, cheap_model))
    .compile()?;
```

- The policy runs before every model call, after `pre_model_hook` and `system_context` if set.
- Text from `system_context` (for example recalled [long-term memories](memory.md)) is added before the
  policy runs, so it counts against the budget and is kept like any system message. It is merged into the
  system prompt when the prompt is added, otherwise inserted as its own system message, and it is never
  stored in the state, even with `.persist_context(true)`.
- The system prompt is part of the messages the policy sees (system messages are always kept); it is added
  unless a system message with the same text is already there. The budget is reduced by the estimated size
  of the tool schemas.
- The state is not changed: checkpoints keep the full history and only the request shrinks. With
  `.persist_context(true)` the policy output replaces `AgentState::messages` (without the system prompt),
  so the stored history shrinks and a summary message is carried and updated across turns.
- When the policy changes the messages, the agent emits a `GraphEvent::Custom` with a `context_policy`
  field: `tokens_before`, `tokens_after`, `budget` and `applied`.

## In the harness

`sinew-harness` keeps its own compaction (`ContextConfig`, see [Harness](harness.md#context-management)):
it summarizes everything before the last `keep_recent` messages when the estimate exceeds
`max_context_tokens`, traces the summarizer call and emits a `Compacted` event. Plugging a `ContextPolicy`
into the harness is tracked in [#160](https://github.com/roanbrasil/sinew/issues/160).

## Performance

`cargo bench -p sinew-core --bench context` runs the policies on generated transcripts of 100, 500 and
2 000 messages with many parallel tool calls, against an 8 000-token budget, after checking once that each
result fits and keeps tool pairs. Without model calls, a policy costs on the order of a few milliseconds for
500 messages, dominated by token estimation over large tool results. Randomized tests
(`context::policy::tests::randomized_budgets_and_pairing`) check budgets and pairing over seeded
transcripts of 50 to 650 messages with random budgets and pipelines.
