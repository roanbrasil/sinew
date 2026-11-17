# 0004. One shared, tuned HTTP client for all providers

- Status: Accepted
- Date: 2026-11-16
- Related: [0005](0005-openai-compatible-presets.md), [docs/resilience.md](../resilience.md)

## Context

Time to first token matters for agents and streaming UIs. A new `reqwest::Client` per model
means a new connection pool, so each model pays its own TCP and TLS handshakes. Default client
settings leave Nagle's algorithm on, which can delay small streaming writes, and a global request
timeout does not fit generations that legitimately take minutes.

## Considered options

- Each provider builds its own `reqwest::Client` with defaults.
- One process-wide client, tuned for LLM APIs, shared by every provider, with an override.

## Decision

[crates/sinew-core/src/providers/mod.rs](../../crates/sinew-core/src/providers/mod.rs) defines
`tuned_client_builder()` (keep-alive pool, `TCP_NODELAY`, TCP keepalive, HTTP/2 via ALPN with the
default `http2` feature, 10 s connect timeout, no overall request timeout) and `shared_client()`,
a process-wide client stored in a `OnceLock`. Every provider uses it unless given another client
with `with_http_client`. Request timeouts are set per model with the `with_timeout` wrapper.

## Consequences

- Good: one connection pool and TLS session cache for all models; the second request to a host
  skips the handshakes. Cloning the client is an `Arc` clone.
- Good: HTTP tuning lives in one function.
- Bad: settings are global. Callers who need a proxy, custom TLS roots or different pool limits
  must build their own client and pass it to each provider.
- Bad: there is no default request timeout; a hung server hangs the call unless the model is
  wrapped with `with_timeout`.
