# Model routing

`sinew_core::router::Router` sends each request to one of several named models. It can choose by rules
(prompt size, tools, images, tenant tier), by the label a small classifier model gives the request, or by
trying a cheap model first and escalating to a stronger one when the answer fails a check. It also skips
models that keep failing and prefers the faster of two equivalent models.

`Router` implements `ChatModel`, so it goes anywhere a model does: agents, graphs, `StructuredOutput`,
runnables and the [resilience wrappers](resilience.md).

```rust
use std::sync::Arc;
use sinew_core::{providers::OpenAiCompat, router::{Router, Validator}, *};

let cheap: Arc<dyn ChatModel> = Arc::new(OpenAiCompat::openai("gpt-4.1-mini")?);
let strong: Arc<dyn ChatModel> = Arc::new(OpenAiCompat::openai("gpt-4.1")?);

let router = Router::builder()
    .route("cheap", cheap)
    .route("strong", strong)
    .rule("long-input", |r| r.estimated_tokens() > 8_000, "strong")
    .default_route("cheap")
    .cascade(["cheap", "strong"], Validator::no_refusal())
    .on_route(|d| println!("{} via {:?}", d.route, d.reason))
    .build()?;

let resp = router.chat("Summarize this paragraph: ...".into()).await?;
println!("served by {}", resp.model);   // the serving model's name, as with any provider
```

## How a route is chosen

For each request:

1. **Rules**, in the order they were added. The first rule whose predicate returns `true` names the
   preferred routes.
2. **Classifier**, when no rule matched and one is configured. Its label names the preferred route. If
   the classifier call fails, the default routes are used and the decision says
   `RouteReason::ClassifierError`.
3. **Default routes**, otherwise. Without `default_route`, this is the first registered route.
4. **Filtering.** A preferred route is skipped when it cannot serve the request (see
   [Capability guard](#capability-guard)) or its circuit is open (see [Health](#health)). Among the
   remaining routes, the one with the lowest recent latency wins; ties go to the route listed first.
5. **Fallback.** When every preferred route was skipped, the default routes and then all routes are
   tried the same way, and `RouteDecision::fallback` is `true`.

If no route at all can serve the request, the call fails with `Error::Config` before any model is called.
If capable routes exist but all of their circuits are open, it fails with `Error::Other` ("open circuit")
without calling them.

## Rules

A rule is a name, a predicate over `RouteInput` and one route (`rule`) or several equivalent routes
(`rule_any`, which picks by health and latency):

```rust
Router::builder()
    .route("local", local)
    .route("hosted-a", hosted_a)
    .route("hosted-b", hosted_b)
    .route("strong", strong)
    .rule("enterprise", |r| r.hint("tier") == Some("enterprise"), "strong")
    .rule("long-input", |r| r.estimated_tokens() > 8_000, "strong")
    .rule_any("tools", |r| r.has_tools(), ["hosted-a", "hosted-b"])
    .default_route("local")
    .build()?;
```

`RouteInput` offers:

| Method | Meaning |
|---|---|
| `request()` | the full `ChatRequest` |
| `estimated_tokens()` | prompt size from `pricing::estimate_request_tokens` (no tokenizer) |
| `has_tools()` | the request offers tools |
| `has_images()`, `has_documents()`, `has_audio()` | some message carries that kind of part |
| `hint(key)`, `hints()` | caller-supplied `RouteHints` |

### Hints: tenant tier and other caller facts

`ChatRequest` has no metadata field, so facts that are not part of the request (the tenant's plan, a
user's region, a latency budget) are passed as `RouteHints`:

```rust
use sinew_core::router::RouteHints;

let hints = RouteHints::new().with("tier", "enterprise");
let resp = router.chat_with_hints(request, &hints).await?;
let stream = router.chat_stream_with_hints(request2, &hints).await?;
```

Calls through the plain `ChatModel` methods (`chat`, `chat_stream`) carry no hints, so a rule on a hint
does not match there. Build one router per tier, or call `chat_with_hints` from the code that knows the
tenant.

## Capability guard

Before choosing, the router derives what the request needs: tools (`request.tools` is not empty),
images, documents (PDF) and audio (`ContentPart` kinds in any message). A route is eligible only when its
`Capabilities` include all of them, so a request with an image never reaches a text-only model, even when
a rule names that model. The image request then falls back to a route with vision.

Capabilities come from each model's `capabilities()`. Custom `ChatModel`s report `Capabilities::none()`
by default, which keeps them away from every request with tools or media; register them with
`route_with_capabilities(name, model, caps)` to state what they support. Other capability flags
(`json_schema`, `sampling`, `forced_tool_choice`, ...) do not affect routing.

`Router::capabilities()` is the **union** of the routes' capabilities: the router accepts a request when
at least one route can serve it, and the guard sends it only to such a route. Code that checks
`capabilities()` before sending images therefore sees `vision: true` when any route has vision.

## Classifier

A `Classifier` asks a small model for a label and maps the label to a route. Rules run first; the
classifier is consulted only when no rule matched.

```rust
use sinew_core::router::Classifier;

let router = Router::builder()
    .route("cheap", cheap)
    .route("strong", strong)
    .classifier(
        Classifier::new(small)
            .label("easy", "greetings, lookups, short rewrites", "cheap")
            .label("hard", "multi-step reasoning, math, code review", "strong"),
    )
    .build()?;
```

- The classifier sees the text of the last user message, truncated to 4000 characters
  (`with_max_input_chars`), and a system prompt listing each label with its description
  (`with_instructions` replaces the first line).
- It answers through `StructuredOutput` with `{"label": "..."}`, restricted by the schema to the
  configured labels: a forced tool call when the classifier model reports tool support, otherwise a JSON
  reply. There is no retry; a failure falls back to the default routes.
- Nothing is cached. Every request that reaches the classifier costs one classifier call. Wrap the
  classifier model with `.with_cache(...)` if repeated prompts are common.
- `Classifier::classify(&request)` is public, to test labels on your own data.

A model trained offline (a logistic regression on embeddings, a keyword scorer) does not need the
classifier API: call it inside a rule predicate.

## Cascade

`cascade(ladder, validator)` applies to `chat`. When the chosen route is in the ladder and its answer
fails the validator, or its call returns an error, the request is sent to the next route in the ladder,
and so on up the ladder. The ladder's routes are filtered like any choice: one that cannot serve the
request or has an open circuit is skipped.

- The first route still comes from rules, the classifier or the default. A request that a rule sends
  straight to `strong` is not cascaded down; a request routed to a route outside the ladder is not
  cascaded at all.
- Each escalation increments `Router::escalations()` and emits a `RouteDecision` with
  `RouteReason::Escalation { from }`.
- When no answer passes, the router returns the last successful answer even though it failed the
  validator, or the last error when every step failed.
- Escalating costs the cheap call plus the strong one. A cascade pays off when most requests pass on the
  first step.

Validators:

| Validator | Accepts |
|---|---|
| `Validator::non_empty()` | non-whitespace text or at least one tool call |
| `Validator::no_refusal()` | `non_empty`, and the text does not start like an English refusal ("I can't", "I'm sorry", "As an AI", ...) |
| `Validator::structured(output)` | replies that `StructuredOutput::parse_message` can parse and validate against the schema |
| `Validator::new(\|resp\| ...)` | your own check on the `ChatResponse` |
| `a.and(b)` | both |

With structured output, run `StructuredOutput::generate` on the router and give the cascade a validator
built from the same `StructuredOutput`. The cheap model's invalid JSON then escalates instead of being
sent back for a retry:

```rust
let so = StructuredOutput::new("invoice", schema).with_retries(0);
let router = Router::builder()
    .route("cheap", cheap)
    .route("strong", strong)
    .cascade(["cheap", "strong"], Validator::structured(so.clone()))
    .build()?;
let invoice: Invoice = so.generate(&router, request).await?;
```

## Health

Each route keeps a small health record behind one `Mutex` plus two atomic counters. Nothing runs in the
background; state changes only when a call finishes or a decision is made.

**Circuit breaker.** `CircuitBreaker::new(failure_threshold, cool_down)` (default: 5 failures, 30 s)
applies to every route:

- After `failure_threshold` consecutive counted failures, the circuit opens and the route is skipped for
  `cool_down`.
- After the cool-down the route is half-open. The next call goes through: a success closes the circuit,
  a counted failure opens it again for another `cool_down`. Concurrent requests in the half-open window
  may all try the route.
- Only `Error::is_retryable()` errors count by default (rate limits, timeouts, 5xx, connection errors),
  so a 400 caused by the request does not take a route out of service. Change it with
  `.failure_if(|e| ...)`; disable the breaker with `CircuitBreaker::disabled()`.

**Latency.** Each successful `chat` updates an exponentially weighted moving average of its duration
(`latency_alpha`, default 0.3 for the newest sample). When several routes qualify (`rule_any`,
`default_routes`), the one with the lowest average wins. A route with no successful call yet counts as
zero, so new routes are tried once before the averages decide. Streams update the breaker but not the
latency average, since opening a stream says little about how long it takes.

`Router::stats()` returns, per route: calls served, failures, consecutive failures, circuit state
(`Closed`, `Open { remaining }`, `HalfOpen`) and the latency average.

## Seeing which route served a call

- `ChatResponse::model` is the serving model's name, as reported by its provider, so traces, cost
  (`ChatResponse::cost`) and Sinew Scope show the real model.
- `on_route(|decision| ...)` runs before every routed call (the first call and each escalation) with the
  route name, the model name, the `RouteReason` and the `fallback` flag. Use it to log, count, or add the
  route to your trace's metadata.
- `Router::route(&request, &hints)` returns the decision without calling the model (it does call the
  classifier when one is consulted), which is useful in tests.

## Streaming

`chat_stream` routes once (rules, classifier, default, filtering) and delegates to the chosen route's
`chat_stream`. The cascade does not apply: by the time the answer could be validated, part of it has
already been delivered. Errors before the stream opens count against the route's circuit; errors in the
middle of a stream do not.

## Combining with the other wrappers

The router does not retry. Put retries and timeouts on each route, so a transient failure is retried on
the same model before it counts against that model's circuit, and wrap the router if you want a last
resort:

```rust
let router = Router::builder()
    .route("cheap", Arc::new(cheap.with_timeout(Duration::from_secs(30)).with_retry(RetryPolicy::new(2))))
    .route("strong", Arc::new(strong.with_retry(RetryPolicy::new(2))))
    .default_route("cheap")
    .build()?
    .with_fallbacks([Arc::new(OpenAiCompat::ollama("qwen3:4b")) as Arc<dyn ChatModel>]);
```
