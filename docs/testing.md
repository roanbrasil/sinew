# Testing agents

Agents are hard to test: model output is non-deterministic, calls cost money and need network, and a
hand-written mock drifts from what the provider really returns. Sinew gives you three layers:

| Layer | Use it for | Network |
|---|---|---|
| `ScriptedModel` | Unit tests of graph logic: you write the model's replies | none |
| Record/replay cassettes (`vcr`) | Regression tests of real behavior: record once against a real model, replay forever | only when recording |
| Live tests (`#[ignore]`) | Smoke tests against a local or hosted model | yes |

## `ScriptedModel`

Returns the messages you give it, in order, and records every request so you can assert on prompts:

```rust
use sinew::core::mock::ScriptedModel;

let model = Arc::new(ScriptedModel::new([
    Message::assistant_with_tools("", vec![call("1", "get_order", json!({"id": "A-1"}))]),
    Message::assistant("Order A-1 is paid."),
]));
let agent = react_agent(model.clone(), tools![get_order])?;
agent.invoke(AgentState::new(vec![Message::user("status of A-1?")])).await?;
assert!(model.requests()[1].messages.last().unwrap().content.contains("paid"));
```

Streaming works too: `chat_stream` replays each message word by word, with tool-call deltas, like a provider.

## Record/replay cassettes

`sinew::core::vcr::Vcr` wraps models and tools. In **record** mode they call through and every request,
response (or stream event) and tool result is saved to a JSON cassette. In **replay** mode the cassette answers
and nothing real runs.

```rust
use sinew::core::vcr::Vcr;

#[tokio::test]
async fn order_agent_answers_status() -> sinew::core::Result<()> {
    // SINEW_VCR=record to (re)record; default "auto" replays when the file exists.
    let vcr = Vcr::from_env("tests/cassettes/order_status.json")?;
    let model = vcr.model(from_spec("anthropic:claude-haiku-5-5")?);   // not called during replay
    let tools = vcr.toolbox(&tools![get_order]);                          // not called during replay

    let agent = react_agent(model, tools)?;
    let out = agent.invoke(AgentState::new(vec![Message::user("status of A-1?")])).await?;
    assert!(out.last().unwrap().content.contains("paid"));

    vcr.save()?;                 // writes the cassette when recording
    vcr.assert_all_played()?;    // when replaying, every recorded call was used
    Ok(())
}
```

Commit the cassette next to the test. CI runs fully offline, in milliseconds, with the exact responses the model
gave when you recorded.

### What replay checks

Replay fails loudly instead of answering with a stale response:

- **Request matching.** The n-th model call must match the n-th recorded call. The error says which message
  changed: `model call #2 differs from the recording: message 3 changed: recorded {...} got {...}`.
  Choose how strict with `.matching(..)`:

  | `MatchRequests` | Compares |
  |---|---|
  | `Exact` (default) | messages, tool schemas, temperature, max tokens, tool choice, response format |
  | `Messages` | messages only (change tool descriptions without re-recording) |
  | `Sequential` | nothing; replays in order |

- **Tool calls** are matched by name and arguments among the calls not yet replayed, so tools that run in
  parallel can finish in any order.
- **`assert_all_played`** catches code that stopped making a call it used to make.
- Errors are recorded too: a recorded provider error or tool error is returned again on replay.

### Streams

A streamed call is recorded event by event and replayed with the same events, so streaming UIs and token
handlers are tested exactly. A call recorded with `chat` can be replayed with `chat_stream` and the other way
round.

### Turning a production trace into a test

Every traced run already contains what a cassette needs. With `sinew-scope`:

```bash
curl -s http://127.0.0.1:7878/api/traces/$TRACE_ID/cassette > tests/cassettes/bug_1234.json
```

or in code, from any runs (for example a `MemorySink`):

```rust
let cassette = sinew::trace::cassette_from_runs(&sink.runs());
let vcr = Vcr::from_cassette(cassette).matching(MatchRequests::Messages);
```

Traces store messages but not tool schemas or sampling options, so replay trace cassettes with
`MatchRequests::Messages`. Now the bug that happened in production is a deterministic test: fix the code, keep
the cassette, and the regression cannot come back silently.

### Re-recording

- Re-record when you change prompts, tools or models on purpose: `SINEW_VCR=record cargo test`.
- Review cassette diffs in pull requests; they are pretty-printed JSON with stable key order.
- Cassettes contain prompts and responses. Do not record secrets or personal data you would not commit.

## Live tests

Mark tests that call a real model with `#[ignore]` and run them on demand:

```bash
cargo test -- --ignored        # e.g. against local Ollama
```

The workspace's own live tests use a local Ollama (`SINEW_MODEL`, `SINEW_EMBED_MODEL`), Qdrant
(`SINEW_QDRANT_URL`), Postgres with pgvector (`SINEW_PGVECTOR_URL`, `SINEW_TEST_PG_URL`), and hosted providers listed
in `SINEW_LIVE_SPECS`:

```bash
SINEW_LIVE_SPECS=anthropic:claude-haiku-4-5,ollama:qwen3:4b cargo test -p sinew-core --test live_providers -- --ignored
```

`live_providers` checks, for every spec, a chat with usage, a stream ending in `Done`, and a forced tool call when
the provider supports tools.

### Nightly

`.github/workflows/live.yml` runs every night (and on demand from the Actions tab):

- **local services**: Ollama with a small model, Qdrant and Postgres/pgvector as service containers; runs all live
  tests of core, harness, RAG and the Postgres checkpointer;
- **hosted providers**: runs `live_providers` for each provider whose API key is set as a repository secret
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`); with no
  secrets the job passes and says so in its summary.

A failing scheduled run opens one issue per job ("Nightly live tests failing: ...") and comments on it on later
failures instead of opening duplicates.

## Fuzzing Sinew's parsers

Sinew's own parsers of untrusted input (provider SSE streams, MCP and A2A JSON-RPC, structured-output JSON, the
document loaders, OTLP ingest, the trace redactor and the guardrail patterns) are fuzzed with
[cargo-fuzz](https://rust-fuzz.github.io/book/cargo-fuzz.html). The targets live in `fuzz/`; each one checks that
its parser never panics and holds a property, for example that an SSE body decodes to the same events however the
network splits it into chunks. A nightly workflow runs every target for a few minutes, and the committed seeds and
crash reproducers are replayed as ordinary tests on stable. How to run them and how to add a target is described in
[CONTRIBUTING.md](../CONTRIBUTING.md#fuzzing).
