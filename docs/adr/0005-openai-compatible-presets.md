# 0005. OpenAI-compatible presets instead of one crate per provider

- Status: Accepted
- Date: 2026-11-16
- Related: [0004](0004-shared-http-client.md), [docs/providers.md](../providers.md)

## Context

LangChain ships one package per provider. Most hosted services (xAI, Groq, Mistral, DeepSeek,
OpenRouter, Together, Fireworks, Perplexity, Cerebras, NVIDIA, Hugging Face) and local servers
(Ollama, LM Studio, llama.cpp, vLLM, SGLang, LocalAI, Jan, llamafile, TGI) speak the OpenAI Chat
Completions format. A crate per provider would multiply versions, releases and duplicated code.

## Considered options

- One crate per provider.
- One client per provider inside `sinew-core`.
- Native clients only for APIs that differ; one OpenAI-compatible client plus a table of presets
  for the rest.

## Decision

`sinew-core` has native clients only where the API differs: `Anthropic` and `Gemini`. Everything
else is `OpenAiCompat` ([crates/sinew-core/src/providers/openai.rs](../../crates/sinew-core/src/providers/openai.rs))
configured from a preset table of base URL, API key variable and `Capabilities`
([crates/sinew-core/src/providers/presets.rs](../../crates/sinew-core/src/providers/presets.rs)).
`from_spec("provider:model")` builds any of them. `providers::catalog()` lists them, and
`docs/providers.md` is generated from it by a test
([crates/sinew-core/tests/provider_matrix.rs](../../crates/sinew-core/tests/provider_matrix.rs)).

## Consequences

- Good: adding a compatible provider is a table entry, not a crate; all of them share one
  implementation of streaming, tools and structured output.
- Good: users depend on one crate and switch providers by changing a string.
- Bad: provider-specific features outside the OpenAI format are not available unless they fit
  `OpenAiCompat`'s extension points (`with_header`, `with_query`, `with_api_key_header`) or the
  provider gets a native client.
- Bad: each service is only as good as its OpenAI emulation; preset capabilities are declared
  conservatively and must be kept up to date by hand.
