# Guardrails

Guardrails check text on its way into and out of a model: the user's message, tool results, and the model's
answer. They live in `sinew_core::guardrails` and work with any `ChatModel`, in runnable chains, in
`ReactAgent`, and in the `sinew-harness` agent.

```rust
use sinew_core::guardrails::{guard, InjectionDetector, JsonSchemaGuard, PiiRedactor, SecretRedactor};
use sinew_core::{ChatModelExt, providers::OpenAiCompat};

let model = OpenAiCompat::openai("gpt-4.1-mini")?.with_guardrails(
    // input: new user messages and tool results
    [guard(PiiRedactor::new()), guard(SecretRedactor::new()), guard(InjectionDetector::new())],
    // output: the model's answer
    [guard(SecretRedactor::new()), guard(JsonSchemaGuard::new(schema)).retry(2)],
);
```

## Concepts

- **Stage** (`GuardrailStage`): `Input` (a user message), `ToolOutput` (a tool result) or `Output` (the model's
  answer).
- **Guardrail**: something that looks at a text at a stage and returns a `GuardrailVerdict`: `Pass`,
  `Modify(new_text)` or `Block(reason)`. Synchronous guardrails implement `Guardrail`; guardrails that do I/O (the
  LLM judge) implement `AsyncGuardrail`. Every `Guardrail` is an `AsyncGuardrail`.
- **Rule** (`GuardRule`, built with `guard(g)`): a guardrail plus the action to take when it does not pass, and
  optionally the stages it is limited to (`.at([GuardrailStage::Output])`).
- **Guardrails**: an ordered list of rules plus an optional event observer. `Guardrails::check(stage, text)`
  runs them and returns a `GuardOutcome` (the text after rewrites, the events, and a `GuardDecision`).

Rules run in order. Rewrites chain: each rule sees the previous rule's output, so `PiiRedactor` before
`MaxLength::chars(n).truncate()` truncates the redacted text, and the other way round it may cut an e-mail
address in half before the redactor sees it. The first rule whose action ends the check (block, refuse,
retry) stops it; later rules do not run.

## Built-in guardrails

| Guardrail | Finds | Verdict | Stages |
|---|---|---|---|
| `PiiRedactor` | e-mail addresses, phone numbers (10–15 digits), card numbers (Luhn-checked, first digit 2–6) | `Modify` with `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_CARD]` | all; `.at(...)` to restrict |
| `SecretRedactor` | API keys and tokens with a known shape (`sk-`, `sk-ant-`, `xai-`, `gsk_`, `AIza`, GitHub, AWS `AKIA`/`ASIA`, Slack, JWT, PEM private keys, `Bearer ...`) | `Modify` with `[REDACTED_SECRET]` (`Bearer [REDACTED_SECRET]`) | all |
| `InjectionDetector` | "ignore previous instructions" and variants, requests to reveal the system prompt, "you are now a ...", chat-template tokens | `Modify` with a warning prefix (`INJECTION_WARNING`), or `Block` with `InjectionDetector::blocking()` | `ToolOutput`; `.at(...)` to change |
| `DenyList` | keywords and phrases (case-insensitive, whole words), topics (a label with keywords), regular expressions | `Block("matched denied topic `<label>`")`, or `Modify` with `.redacting("[DENIED]")` | all |
| `JsonSchemaGuard` | text that is not JSON matching a schema (`type`, `required`, `properties`, `items`, `enum`, `additionalProperties: false`) | `Block` naming the first violation | all |
| `MaxLength` | text longer than `n` characters (Unicode scalar values) | `Block`, or `Modify` with `.truncate()` | all |
| `LlmJudge` | whatever a policy prompt describes, judged by any `ChatModel` | `Block(model's reason)` | all |

The PII and secret patterns are the ones trace redaction uses; they are public in
`sinew_core::guardrails::patterns` and re-exported by `sinew_trace::redact`. See [Redaction](redaction.md).

The pattern-based guardrails are heuristics. They catch common shapes and phrasings, not every case:
combine them with least-privilege tools and approvals ([Production](production.md#3-agents-that-stay-inside-the-lines)).

### LLM judge

```rust
use std::sync::Arc;
use sinew_core::guardrails::{GuardRule, LlmJudge};

let judge = LlmJudge::new(
    Arc::new(OpenAiCompat::openai("gpt-4.1-nano")?),
    "Answers must be about cooking. Refuse medical, legal and financial advice.",
)
.named("topic_judge");

// Off-topic questions get a fixed answer without calling `model`;
// off-topic answers are sent back once with the judge's reason.
let judge = Arc::new(judge);
let model = model.with_guardrails(
    [GuardRule::from_arc(judge.clone()).refuse("I can only help with cooking.")],
    [GuardRule::from_arc(judge).retry(1)],
);
```

`LlmJudge` sends the policy as a system message and the text, with what it is (a user message, an answer, a
tool result), as a user message that tells the judge to treat it as data. It asks for structured output with the
schema `{"allow": bool, "reason": string}` (`JudgeVerdict`), through a forced tool call by default
(`.with_mode(StructuredMode::Json)` for JSON mode). `allow: false` becomes `Block(reason)`.

A model error, or a reply that still does not match the schema after one repair (`.with_retries(n)`), is
returned as an error from the check, and the request fails: an unreachable judge does not let text through.
Each check is one model call (two with a repair), so use a small, fast model.

## Actions

| Action | Builder | On `Modify` | On `Block` |
|---|---|---|---|
| `GuardAction::Redact` (default) | `guard(g)` or `.redact()` | the text is replaced | blocked |
| `GuardAction::Block` | `.block()` | blocked | blocked |
| `GuardAction::Refuse(msg)` | `.refuse("...")` | replaced by `msg` | replaced by `msg` |
| `GuardAction::Warn` | `.warn()` | event recorded, text unchanged | event recorded, text unchanged |
| `GuardAction::Retry { max_retries }` | `.retry(n)` | the model is asked again | the model is asked again |

What "blocked", "replaced by `msg`" and "asked again" mean depends on where the guardrails run:

| | `Guarded` input (user message / tool result) | `Guarded` output | `GuardStep` runnable | `ReactAgent::tool_guardrails` |
|---|---|---|---|---|
| Block | `Err(Error::Guardrail)`, model not called | `Err(Error::Guardrail)` | `Err(Error::Guardrail)` | result replaced by `error: tool output blocked by guardrail ...` |
| Refuse | the message is the answer, model not called | the message replaces the answer | the message is the output | the message replaces the result |
| Retry | same as Block | retry, then `Err(Error::Guardrail)` | same as Block | same as Block |

`Error::Guardrail { guard, stage, reason }` names the guardrail, the stage and the reason.

### Retry

On model output, a `Retry` rule appends the rejected answer and a user message to the request:

```text
Your previous answer was rejected by the `json_schema` guardrail: the JSON does not match the schema:
$ is missing required field `city`. Answer again and fix this.
```

and calls the model again. All output rules run again on the new answer. Each rule has its own budget of
`max_retries` extra calls per request; when a rule fires with its budget spent the request fails with
`Error::Guardrail` (reason suffixed with `(still failing after N retries)`). The total number of model calls for
one request is at most 1 + the sum of the retry budgets. Token usage of every call is added to the returned
`ChatResponse::usage`.

### Warnings and events

Every rule that does not pass records a `GuardEvent` (`guard`, `stage`, `action`, `reason`, `rewrote`):

- `Guarded::chat_with_report(request)` returns `(ChatResponse, GuardReport)`; `GuardReport::events` has every
  event and `warnings()` the ones from `Warn` rules; `retries` counts retries.
- `Guarded::on_event(f)` / `Guardrails::on_event(f)` calls `f` as events happen, including for requests that end
  in an error.
- `sinew_trace::guardrail_observer(tracer, parent)` is an observer that records each event as a
  `guardrail:<name>` trace run (`chain` kind, the event in `outputs` and `metadata.guardrail`; blocks and retries
  are recorded as failed runs). `sinew_trace::record_guard_event` records one event.
- `ReactAgent::tool_guardrails` emits each event as a `GraphEvent::Custom` with a `guardrail` field and, when the
  graph is traced, records it with `record_guard_event` under the `tools` node.

`GuardEvent` serializes to JSON (`{"guard": "...", "stage": "tool_output", "action": "warn", ...}`).

## Composition

### Any chat model: `with_guardrails`

`model.with_guardrails(input, output)` (from `ChatModelExt`) returns a `Guarded<M>`, itself a `ChatModel`, so it
stacks with the other wrappers (`with_retry`, `with_fallbacks`, `with_cache`, routers) and works wherever a
model does. Each argument is anything that converts into `Guardrails`: an array or `Vec` of `GuardRule`s, or a
`Guardrails`. Use `Guardrails::new()` for an empty side.

For each request:

1. **Input.** The messages after the last assistant message (the new user messages and tool results) are
   checked: user messages at `Input`, tool results at `ToolOutput`. Older messages were checked when they were
   new and are not checked again. System messages are not checked.
2. **Model.** The inner model is called with the rewritten request.
3. **Output.** If the response has no tool calls, its text is checked at `Output`. Responses with tool calls are
   intermediate steps of an agent loop and pass unchecked; their tool results are checked as input on the next
   call.

Only `Message::content` is checked; multimodal parts and tool-call arguments pass through.

Wrapper order matters as with the other wrappers: `model.with_guardrails(..).with_retry(..)` retries
transient errors of the whole guarded call (a guardrail block is not retryable), while
`model.with_retry(..).with_guardrails(..)` retries only the provider call.

### Streaming

`Guarded::stream_mode` chooses how output guardrails apply to `chat_stream`. Input guardrails always run before
the stream starts.

- `StreamGuardMode::Buffer` (default): the request runs like `chat_with_report` (the inner model is called with
  `chat`, retries included) and the checked response is replayed as stream events. Nothing reaches the caller
  before every output guardrail has passed; there is no incremental output.
- `StreamGuardMode::Final`: the inner model's events are forwarded as they arrive, and the output guardrails run
  on the complete text when the inner stream ends. Text already delivered cannot be taken back, so a rule that
  would block, refuse, retry or rewrite ends the stream with an `Err(Error::Guardrail)` item in place of
  `StreamEvent::Done`; the consumer must discard what it showed. `Warn` rules record their event and the stream
  ends normally. Responses with tool calls are not checked, as with `chat`.

Use `Buffer` when an unchecked answer must never be shown; use `Final` with `Warn` rules, or when the UI can
retract a message.

### Chains: `Guardrails::step`

`guardrails.step(stage)` is a `Runnable` (`GuardStep`) that checks its input:

- `Runnable<String, String>`: returns the rewritten text, the refusal message, or `Error::Guardrail`.
- `Runnable<ChatResponse, ChatResponse>`: the same on the message text, so it can follow a model; responses
  with tool calls pass unchecked.

```rust
use sinew_core::guardrails::{guard, Guardrails, GuardrailStage, SecretRedactor};

let chain = prompt
    .pipe(model)
    .pipe(Guardrails::new().with(guard(SecretRedactor::new())).step(GuardrailStage::Output))
    .pipe(StrOutputParser);
```

A step after a streaming step merges the chunks first, so it buffers. `Retry` rules behave like `Block` in a
step (there is no model to ask again); wrap the model with `with_guardrails` for retries.

### Graph agents: `ReactAgent::tool_guardrails`

```rust
use sinew_core::guardrails::{guard, InjectionDetector, PiiRedactor};

let agent = ReactAgent::new(model, tools)
    .tool_guardrails([guard(PiiRedactor::new()), guard(InjectionDetector::blocking())])
    .compile()?;
```

Every tool result is checked at `ToolOutput` before it is added to the state, so the model never sees the
unchecked text and the checkpointed state holds the checked version. A blocked result becomes
`error: tool output blocked by guardrail `<name>`: <reason>`, which the model can react to. To check the user's
input and the final answer too, give `ReactAgent` a model wrapped with `with_guardrails`; for custom checks on
the model's reply (for example removing tool calls), use `post_model_hook`.

### Harness

The `sinew-harness` agent uses the same `Guardrail` trait and built-in guardrails, re-exported as
`sinew_harness::guardrails` (`PiiRedactor`, `InjectionDetector`, `SecretRedactor`, `DenyList`, `JsonSchemaGuard`,
`MaxLength`, `apply_guardrails`, ...). `AgentBuilder::guardrail(g)` registers a synchronous guardrail; the harness
applies it with the `Redact` action at all three stages (an input block ends the turn without calling the model,
an output block withholds the answer, a tool-output block replaces the output with an error). For per-rule
actions or an `LlmJudge`, wrap the harness model:

```rust
let agent = Agent::builder()
    .model(model.with_guardrails(Guardrails::new(), [guard(judge).refuse("I can't help with that.")]))
    .guardrail(InjectionDetector::new())
    .build()?;
```

See [Harness](harness.md#guardrails).

## Writing a guardrail

```rust
use sinew_core::guardrails::{Guardrail, GuardrailStage, GuardrailVerdict};

struct NoInternalUrls;

impl Guardrail for NoInternalUrls {
    fn name(&self) -> &str {
        "no_internal_urls"
    }
    fn check(&self, _stage: GuardrailStage, text: &str) -> GuardrailVerdict {
        if text.contains(".corp.example.com") {
            GuardrailVerdict::Block("mentions an internal host".into())
        } else {
            GuardrailVerdict::Pass
        }
    }
}
```

For checks that need I/O, implement `AsyncGuardrail` (with `#[async_trait]`) instead; return `Err` only when the
check itself failed, and a `Block` verdict when the text is not allowed. A guardrail that only looks at some
stages returns `Pass` for the others, or leave it stage-agnostic and restrict the rule with `.at(...)`.

## Limits

- Pattern guardrails are heuristics; `InjectionDetector` in particular catches common phrasings, not every
  injection. Treat tool output as untrusted regardless.
- Only text content is checked. Images, documents, audio and tool-call arguments are not.
- In `StreamGuardMode::Final`, text that fails a check has already been delivered when the error arrives.
- `JsonSchemaGuard` supports the common JSON Schema keywords listed above; unknown keywords are ignored.
- An `LlmJudge` adds a model call per check and its own failure modes (the judge model can be wrong, or be
  manipulated by the text it reviews; the prompt tells it to treat the text as data, which reduces but does not
  remove that risk).
