<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew-core

[![crates.io](https://img.shields.io/crates/v/sinew-core.svg)](https://crates.io/crates/sinew-core)
[![docs.rs](https://docs.rs/sinew-core/badge.svg)](https://docs.rs/sinew-core)
[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

Messages, chat models, tools, prompt templates, streaming and structured output: the LangChain-style building blocks of Sinew.

Part of [Sinew](https://github.com/roanbrasil/sinew): LLM chains, agent graphs, tracing and evaluation in Rust.

## Providers

| Kind | Providers |
|---|---|
| Native | Anthropic (`Anthropic`), Google Gemini (`Gemini`, AI Studio or Vertex AI) |
| OpenAI-compatible, hosted | OpenAI, xAI Grok, Groq, Mistral, DeepSeek, OpenRouter, Together, Fireworks, Perplexity, Cerebras, NVIDIA NIM, Hugging Face router |
| OpenAI-compatible, local | Ollama, LM Studio, llama.cpp, vLLM, SGLang, LocalAI, Jan, llamafile, TGI, or any URL |

Every provider streams tokens and supports structured output; what else each one accepts (tools, images, PDF, audio, JSON schema, cached tokens) is in the [capability matrix](https://github.com/roanbrasil/sinew/blob/main/docs/providers.md) and at run time in `model.capabilities()`. Pick one in code or from a string:

```rust
use sinew_core::{providers::from_spec, ChatRequest, Message};

let model = from_spec("gemini:gemini-2.5-flash")?;          // GEMINI_API_KEY
// from_spec("xai:grok-4")?                                  // XAI_API_KEY
// from_spec("lmstudio:qwen3-8b")?                           // local, no key
// from_spec("ollama:qwen3:4b")?                             // OLLAMA_HOST or 127.0.0.1:11434
// from_spec("http://10.0.0.5:8000/v1|my-model")?            // any OpenAI-compatible URL
let reply = model.chat(ChatRequest::new(vec![Message::user("Hi")])).await?;
```

## What is inside

- `Message`, `Role`, `ToolCall`, `ChatRequest`, `ChatResponse`, `Usage`
- `ChatModel` trait with `chat` and `chat_stream` (`StreamEvent`, `StreamAccumulator`)
- `Tool`, `ToolBox`, the `#[tool]` attribute (feature `macros`: functions become tools, schema from argument types) and `tools![...]`; `TypedTool`, `AsyncFnTool`, `FnTool` ([tools guide](https://github.com/roanbrasil/sinew/blob/main/docs/tools.md))
- `PromptTemplate`
- `StructuredOutput` (tool or JSON mode, with retries; `schemars` feature derives the schema from a Rust type)
- `mock::ScriptedModel` for offline tests

## Cargo features

| Feature | Adds |
|---|---|
| `schemars` | `StructuredOutput::for_type::<T>()`, `TypedTool` |
| `macros` | `#[tool]` attribute (implies `schemars`) |

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/roanbrasil/sinew/blob/main/LICENSE-APACHE) or [MIT license](https://github.com/roanbrasil/sinew/blob/main/LICENSE-MIT) at your option.
