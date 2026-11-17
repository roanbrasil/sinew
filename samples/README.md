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
| `--gemini` | Google Gemini, default `gemini-2.5-flash` | `GEMINI_API_KEY` |
| `--xai` | xAI Grok, default `grok-4` | `XAI_API_KEY` |
| `--model provider:model` | anything `from_spec` accepts: `lmstudio:qwen3-8b`, `vllm:Qwen/Qwen3-8B`, `groq:llama-3.3-70b-versatile`, `openrouter:vendor/model`, `http://host:8000/v1\|model`, ... | depends on the provider |

Set `SINEW_MODEL` to override the default model name of the shortcut flags.

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
| 11 | [mcp](11-mcp/src/main.rs) | Serve a `ToolBox`, a resource and a prompt over MCP; use MCP tools from an agent, read the resource and render the prompt (feature `mcp`) |
| 12 | [durable-agent](12-durable-agent/src/main.rs) | `SqliteCheckpointer`: pause, restart the process, resume the thread (feature `sqlite`) |
| 13 | [scope-and-evals](13-scope-and-evals/src/main.rs) | `HttpSink` traces to the `sinew-scope` UI and evaluation experiments on the answer and on the agent trajectory (features `eval`, `http-trace`) |
| 14 | [a2a](14-a2a/src/main.rs) | Serve an agent over the Agent2Agent protocol and delegate to it from another agent with `A2aAgentTool`; `message/stream` |
| 15 | [rag](15-rag/src/main.rs) | Load and split `docs/*.md`, embed, hybrid BM25 + vector retrieval, `RagChain` answers with sources, retriever as an agent tool (`--embed provider:model`) |
| 16 | [runnables](16-runnables/src/main.rs) | LCEL-style pipelines: `prompt.pipe(model).pipe(parser)`, `parallel`, branches, retry/fallback wrappers and cost per model |
| 17 | [multi-agent](17-multi-agent/src/main.rs) | `create_supervisor` routing between worker agents with handoff tools |
| 18 | [time-travel](18-time-travel/src/main.rs) | `interrupt()` inside a node, `Command::resume`, state history, replay and fork from a past checkpoint |
| 19 | [harness](19-harness/src/main.rs) | `sinew-harness` agent in a temp workspace: file tools, a subagent, approvals, a written summary |
| 20 | [serve](20-serve/src/main.rs) | Deploy an agent with `sinew-serve`; call it through `ServeClient` and through the OpenAI-compatible endpoint |
| 21 | [prompt-hub-and-feedback](21-prompt-hub-and-feedback/src/main.rs) | Versioned prompts in the hub, feedback and annotation queues, cost and monitoring in `sinew-scope` |
| 22 | [mcp-microservice](22-mcp-microservice/src/main.rs) | A microservice with REST routes plus an MCP endpoint (Streamable HTTP, bearer auth); typed async tools; an agent using them over HTTP (feature `mcp-http`) |

Coming from LangChain or LangGraph? [docs/migrating-from-langchain.md](../docs/migrating-from-langchain.md)
maps Python concepts to Sinew types and shows the common patterns side by side, with pointers to
these samples.

`common/` holds the helper that picks the model from the command line.

All samples are built by CI, and each was run both offline and against a local Ollama model before
being committed.
