# Resilience, caching, cost and HTTP performance

## Wrappers

`ChatModelExt` (implemented for every `ChatModel`) wraps a model in another
`ChatModel`, so wrappers stack and the result works anywhere a model does:
agents, graphs, structured output and runnables.

```rust
use std::{sync::Arc, time::Duration};
use sinew_core::{providers::OpenAiCompat, *};

let model = OpenAiCompat::openai("gpt-4.1-mini")?
    .with_cache(InMemoryCache::new(1_000))
    .with_timeout(Duration::from_secs(60))
    .with_retry(RetryPolicy::default())
    .with_rate_limit(RateLimiter::per_minute(500).with_tokens_per_minute(200_000))
    .with_fallbacks([Arc::new(OpenAiCompat::ollama("qwen3:4b")) as Arc<dyn ChatModel>]);
```

Wrappers apply from the inside out. In the example:

- Cache hits skip everything else.
- Each attempt has a 60-second timeout.
- Timeouts and transient errors are retried.
- The limiter paces requests.
- The local model takes over only after the primary's retries are exhausted.

### Retry: `with_retry(RetryPolicy)`

- Exponential backoff:
  `min(initial_backoff * multiplier^n, max_backoff)`. The defaults are 3
  retries, 500 ms initial backoff, multiplier 2 and a 30 s cap.
- Equal jitter is on by default: each delay is a random value between half
  and all of the computed backoff, which avoids thundering herds.
- **Retry-After is honoured.** A 429, or any error response with a
  `Retry-After` header, becomes `Error::RateLimited { retry_after, .. }`.
  The policy then waits exactly that long, capped at `max_retry_after`
  (60 s), instead of using its own backoff. Both delta-seconds and HTTP-date
  values are parsed.
- What is retried by default is `Error::is_retryable()`:
  - 408, 409, 429 and 5xx responses
  - `Error::Timeout`
  - connection, request and timeout errors from reqwest
  - mid-stream errors

  Other 4xx responses are not retried. Override with
  `.retry_if(|e| ...)`.
- Streams: only failures before the first event are retried. Once output has
  started, an error is passed through.
- `RetryPolicy::run(|| async { ... })` retries any async operation, and
  `runnable.retry(policy)` retries any runnable.

### Fallbacks: `with_fallbacks([...])`

Tries each fallback in order. By default every error triggers a fallback.
Use `.fallback_if(|e| e.is_retryable())` to fall back only on transient
errors, so that a bad request isn't sent to every provider. `name()` reports
the primary model.

### Rate limiting: `with_rate_limit(RateLimiter)`

A token bucket for requests per minute (`per_minute`, `per_second`,
`with_burst`), optionally combined with tokens per minute
(`with_tokens_per_minute`).

1. Before each request, the limiter waits for one request slot and for the
   estimated prompt tokens (`pricing::estimate_request_tokens`).
2. After the response, it charges or refunds the difference against the real
   `usage.total_tokens()`.

Sustained throughput therefore converges on the configured TPM even when the
estimates are rough. `RateLimiter` clones share their buckets: build one per
provider quota and pass clones to every model that draws from it.

### Timeouts: `with_timeout(Duration)`

Returns `Error::Timeout(d)` when `chat` takes longer than `d`. For
`chat_stream`, the timeout covers opening the stream.
`.with_stream_idle_timeout(d)` also fails a stream when no event arrives for
`d`.

The shared HTTP client sets a 10 s connect timeout but no overall timeout,
because long generations legitimately take minutes. Add `with_timeout` where
you want one.

## Response cache: `with_cache(cache)`

`CachedModel` answers a request it has seen before without calling the
provider.

- The key is a 128-bit FNV-1a hash of the model name and the full serialized
  `ChatRequest` (messages, parts, tools, temperature and so on). It is stable
  across Rust versions, so persistent caches stay valid.
- `chat_stream` hits are replayed as a stream. Misses stream normally and are
  stored when the stream completes successfully.
- Responses are cached regardless of temperature. Use the cache where a
  repeated answer is what you want: tests, evals, idempotent extraction.

Two caches ship with Sinew:

| Cache | Notes |
|---|---|
| `InMemoryCache::new(capacity)` | LRU, optional `.with_ttl(d)`. Clones share storage. |
| `cache::SqliteCache::connect("sqlite://cache.db")` | Feature `sqlite-cache`. Table `sinew_llm_cache`, optional TTL, survives restarts. |

To plug in your own store (Redis, Postgres and so on), implement
`ResponseCache` (`get` and `put`).

## Usage and cost

`Usage` has these fields:

- `input_tokens`: the whole prompt.
- `output_tokens`: includes reasoning tokens.
- `cached_input_tokens`: prompt tokens read from cache. Reported by OpenAI
  (`prompt_tokens_details.cached_tokens`), Anthropic
  (`cache_read_input_tokens`) and Gemini (`cachedContentTokenCount`).
- `cache_creation_input_tokens`: prompt tokens written to cache (Anthropic).
- `reasoning_tokens`: reported by OpenAI
  (`completion_tokens_details.reasoning_tokens`) and Gemini
  (`thoughtsTokenCount`).

For Anthropic, `input_tokens` now includes cache reads and writes, so it
means the same thing for every provider. The new fields are serialized only
when non-zero, and `Usage` supports `+`, `+=` and `Sum`.

```rust
let resp = model.chat(req).await?;
println!("${:.4}", resp.cost().unwrap_or(0.0));            // model reported by the provider
println!("${:.4}", resp.usage.cost("claude-sonnet-5-5").unwrap_or(0.0));
```

`Usage::cost(model)` uses `sinew_core::pricing`:

- Built-in list prices (USD per million tokens) for current OpenAI,
  Anthropic, Gemini and xAI models, including cached-input prices.
  Anthropic cache writes cost 1.25x input.
- Lookup is by longest prefix after stripping a `provider:` or `vendor/`
  prefix, so `gpt-4o-mini-2024-07-18` and `openrouter:openai/gpt-4o-mini`
  both resolve to `gpt-4o-mini`.
- Unknown and local models return `None`.
- Register your own prices, or override the built-in ones, with
  `pricing::set_price("my-model", ModelPrice::new(input, output).with_cached(c))`.

Prices change and long-context or batch tiers are not modelled, so treat the
result as an estimate.

`pricing::estimate_tokens(text)` and `pricing::estimate_request_tokens(req)`
give tokenizer-free estimates (about 4 characters per token, with a floor of
1.3 tokens per word). Use them for budgeting and rate limiting, not billing.

## HTTP performance

- **One shared client.** Every provider uses `providers::shared_client()` by
  default: one `reqwest::Client`, one connection pool and one TLS session
  cache for the whole process. Only the first request to a host pays for the
  TCP and TLS handshakes. Cloning a client is cheap (an `Arc` clone).
- **Tuned settings.**
  - `TCP_NODELAY`, so small streaming writes aren't held back by Nagle's
    algorithm.
  - TCP keepalive of 30 s.
  - Up to 64 idle pooled connections per host, with a 90 s idle timeout.
  - 10 s connect timeout.
  - HTTP/2 via ALPN with adaptive flow-control windows and keepalive pings
    (`http2` feature, on by default; disable default features to drop it).
- **Inject your own client.** Use `with_http_client(client)` on
  `OpenAiCompat`, `Anthropic` and `Gemini`, for proxies, custom roots or
  different timeouts. `providers::tuned_client_builder()` starts from the
  same defaults.
- **No extra copies.** Request bodies are serialized once, straight into
  the request buffer. Responses are parsed from bytes, with no intermediate
  `String`.
- **Linear streaming.** The SSE parser resumes scanning where the previous
  network chunk ended. A long event delivered in many small chunks is
  scanned once, not rescanned on every chunk.

## Azure OpenAI

```rust
let m = OpenAiCompat::azure("https://my-resource.openai.azure.com", "gpt-4o-prod", "2024-10-21", key);
// or AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_API_VERSION:
let m = OpenAiCompat::azure_from_env("gpt-4o-prod")?;
let m = providers::from_spec("azure:gpt-4o-prod")?;
```

Requests go to
`{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...`
with the key in the `api-key` header. The endpoint can also be just the
resource name. Two lower-level builders work for any OpenAI-compatible
gateway: `OpenAiCompat::with_api_key_header`, `with_header` and
`with_query`.
