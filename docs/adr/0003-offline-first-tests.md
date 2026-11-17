# 0003. Offline-first tests

- Status: Accepted
- Date: 2026-11-16
- Related: [docs/testing.md](../testing.md), [CONTRIBUTING.md](../../CONTRIBUTING.md)

## Context

Sinew talks to many LLM providers. Tests that call them need keys, network and money, are slow,
and give different answers on each run. Hand-written mocks drift from what providers really send.
Contributors and CI must be able to run `cargo test --workspace --all-features` with no network.

## Considered options

- Live integration tests against hosted providers in CI.
- Hand-written mocks per test.
- Offline tests by default: recorded provider payloads, a scripted model, and record/replay
  cassettes; live tests opt-in.

## Decision

Three offline layers:

- **Recorded payloads.** Each provider's request encoding and response decoding is unit-tested
  against JSON in the shape the real API returns (for example the `decode_response` tests in
  [crates/sinew-core/src/providers/openai.rs](../../crates/sinew-core/src/providers/openai.rs)).
  CONTRIBUTING.md requires offline tests for every provider mapping.
- **`ScriptedModel`** ([crates/sinew-core/src/mock.rs](../../crates/sinew-core/src/mock.rs))
  returns scripted replies, streams them like a provider and records requests. Every sample uses
  it unless given a real model ([samples/common/src/lib.rs](../../samples/common/src/lib.rs)).
- **Record/replay cassettes** (`sinew_core::vcr`,
  [crates/sinew-core/src/vcr.rs](../../crates/sinew-core/src/vcr.rs)) record real model calls,
  stream events and tool results once and replay them offline, failing when the request changes.
  `sinew_trace::cassette_from_runs` turns a recorded trace into a cassette.

Live tests are `#[ignore]` and opt-in (`cargo test -- --ignored`), for example
[crates/sinew-core/tests/live_ollama.rs](../../crates/sinew-core/tests/live_ollama.rs).

## Consequences

- Good: CI is fast, free and deterministic; contributors need no API keys.
- Good: a changed prompt or tool schema breaks a cassette test instead of being answered with a
  stale response.
- Bad: recorded payloads go stale when a provider changes its API; only live tests catch that,
  and they do not run by default.
- Bad: cassettes are files to commit, and to re-record when prompts change on purpose.
