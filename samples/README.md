# Sinew samples

Small, self-contained programs that show how to use Sinew. Each one is a normal Cargo crate that
depends on `sinew`, the same way your own project would.

Every sample runs **offline by default**, using a scripted model, so you can read the output
without an API key. Add a flag to run it against a real model:

| Flag | Model | Needs |
|---|---|---|
| *(none)* | scripted, deterministic | nothing |
| `--ollama` | local Ollama, default `qwen3:4b-instruct-2507-q4_K_M` | [Ollama](https://ollama.com) running |
| `--openai` | OpenAI, default `gpt-4.1-mini` | `OPENAI_API_KEY` |
| `--anthropic` | Anthropic, default `claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |

Set `SINEW_MODEL` to override the model name.

```bash
cargo run -p sample-03-tools-agent              # offline
cargo run -p sample-03-tools-agent -- --ollama  # real model
```

| # | Sample | What you learn |
|---|---|---|
| 01 | [hello-chat](01-hello-chat/src/main.rs) | Build a `ChatRequest`, call a model, read the answer and token usage |
| 02 | [prompt-chain](02-prompt-chain/src/main.rs) | `PromptTemplate` and chaining two model calls; missing variables are errors |
| 03 | [tools-agent](03-tools-agent/src/main.rs) | Tools as closures (`FnTool`) and as structs (`Tool` trait); the prebuilt `react_agent` |
| 04 | [custom-graph](04-custom-graph/src/main.rs) | `StateGraph` with typed state, a cycle, a router and a recursion limit |
| 05 | [human-in-the-loop](05-human-in-the-loop/src/main.rs) | `interrupt_before`, `get_state`, `update_state` and resuming a thread |
| 06 | [tracing](06-tracing/src/main.rs) | `Tracer` with JSONL, in-memory and custom sinks; tracing your own code |
| 07 | [custom-model](07-custom-model/src/main.rs) | Implementing `ChatModel`: retry and fallback wrappers |
| 08 | [streaming](08-streaming/src/main.rs) | `chat_stream` token by token; `graph.stream()` events from inside an agent |
| 09 | [parallel-agents](09-parallel-agents/src/main.rs) | `add_fan_out` to three reviewers running concurrently, a reducer, and a join node |
| 10 | [structured-output](10-structured-output/src/main.rs) | `StructuredOutput` → typed structs, tool mode and JSON mode, validation with retry |
| 11 | [mcp](11-mcp/src/main.rs) | Serve a `ToolBox` over MCP and use MCP tools from an agent (feature `mcp`) |
| 12 | [durable-agent](12-durable-agent/src/main.rs) | `SqliteCheckpointer`: pause, restart the process, resume the thread (feature `sqlite`) |
| 13 | [scope-and-evals](13-scope-and-evals/src/main.rs) | `HttpSink` traces to the `sinew-scope` UI and an evaluation experiment (features `eval`, `http-trace`) |

`common/` holds the helper that picks the model from the command line.

All samples are built by CI, and each was run both offline and against a local Ollama model before
being committed.
