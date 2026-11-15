# Sinew

[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

LLM chains, agent graphs and tracing in Rust — the LangChain / LangGraph / LangSmith
trio as one small, typed, async workspace.

| Crate | Plays the role of | What it gives you |
|---|---|---|
| `sinew-core` | LangChain | `Message`, `ChatModel` trait, tools (`Tool`, `FnTool`, `ToolBox`), `PromptTemplate`, providers: OpenAI-compatible (OpenAI, Ollama, vLLM, llama.cpp) and Anthropic |
| `sinew-graph` | LangGraph | `StateGraph<S>` with typed state, conditional edges, cycles with a recursion limit, checkpoints, `interrupt_before` + `update_state` for human-in-the-loop, prebuilt `react_agent` |
| `sinew-trace` | LangSmith SDK | Runs (graph → node → llm/tool) with timings, I/O, token usage; JSONL and in-memory sinks |
| `sinew-scope` | LangSmith UI (terminal for now) | `sinew-scope traces.jsonl [--io]` prints each trace as a tree |
| `sinew` | umbrella | re-exports + `prelude` |

## Quick start

```bash
cargo run --example react_agent                     # offline, scripted model
SINEW_MODEL=qwen3:4b-instruct-2507-q4_K_M \
  cargo run --example react_agent -- --ollama       # local Ollama
ANTHROPIC_API_KEY=... cargo run --example react_agent -- --anthropic
cargo run -p sinew-scope -- traces.jsonl --io       # inspect the trace
```

```text
trace 145b442e  total tokens 388→38
graph graph  711ms
├─ node agent  462ms
│  └─ llm qwen3:4b-instruct-2507-q4_k_m  461ms  tokens 172→26
├─ node tools  0ms
│  └─ tool multiply  0ms
└─ node agent  249ms
   └─ llm qwen3:4b-instruct-2507-q4_k_m  249ms  tokens 216→12
```

## Samples

The [`samples/`](samples/) folder has runnable programs, from a first chat call to tools, custom graphs,
human-in-the-loop, tracing and custom model wrappers. They run offline by default; add `--ollama`, `--openai`
or `--anthropic` to use a real model.

## A graph in a few lines

```rust
use sinew::prelude::*;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Review { draft: String, approved: bool }

let mut g = StateGraph::<Review>::new();
g.add_node("write", |mut s: Review| async move { s.draft.push_str(" v2"); Ok(s) });
g.add_node("publish", |s: Review| async move { Ok(s) });
g.add_conditional_edges("write", |s: &Review| if s.approved { "publish".into() } else { "write".into() }, ["write", "publish"]);
g.add_edge("publish", END);
g.set_entry_point("write");

let graph = g.compile()?                       // validates edges, reachability, missing exits
    .with_checkpointer(MemoryCheckpointer::new())
    .interrupt_before(["publish"]);            // pause for a human

let paused = graph.run("thread-1", Some(state)).await?;   // stops before `publish`
let (mut s, _next) = graph.get_state("thread-1").await?;
s.draft = "edited by a human".into();
graph.update_state("thread-1", s).await?;
let done = graph.run("thread-1", None).await?;            // resumes
```

## Design choices

- **Whole-state nodes (`S -> S`)** instead of partial updates plus reducers. The compiler checks every
  update; reducers can come later for parallel fan-out.
- **`compile()` reports every problem at once**: unknown targets, nodes without an exit, unreachable nodes,
  duplicate names.
- **`invoke` never silently skips an interrupt**; it returns `GraphError::Interrupted`. Use `run` with a
  checkpointer to pause and resume.
- **Tracing never breaks the app**: sink I/O errors are swallowed; a run handle dropped without finishing
  is recorded as failed instead of disappearing.

## Roadmap

1. Streaming (token deltas + per-step events) and parallel fan-out/fan-in with reducers.
2. Persistent checkpointers (SQLite, Postgres).
3. `sinew-scope` server: OTLP/HTTP ingest, storage, web UI, datasets and evaluations.
4. More providers (Gemini, Bedrock) and structured output (JSON Schema → typed structs).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
