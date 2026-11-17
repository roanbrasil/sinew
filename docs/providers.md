# Providers

Sinew talks to models through native clients (Anthropic, Gemini, Azure OpenAI) and one OpenAI-compatible
client with presets for hosted APIs and local servers. Build any of them from a string:

```rust
let model = sinew::core::providers::from_spec("gemini:gemini-2.5-flash")?;
// "anthropic:claude-sonnet-5-5", "xai:grok-4", "groq:llama-3.3-70b-versatile",
// "ollama:qwen3:4b", "lmstudio:qwen3-8b", "http://10.0.0.5:8000/v1|my-model"
```

Anything after the first `:` is the model id, so Ollama tags like `qwen3:4b` work.

## Capability matrix

What each provider's API accepts through Sinew. `—` means unsupported or not verified for that endpoint.
Hosted and local OpenAI-compatible servers also depend on the model: a text-only model behind an endpoint that
accepts images still cannot see them. Check at run time with `model.capabilities()` (any `ChatModel`, through
retry, fallback, cache and timeout wrappers) or before building a client with
`providers::capabilities_for("provider", "model")`.

<!-- generated:providers:start -->
| Provider | Spec | Kind | Tools | Stream | Images | PDF | Audio | JSON schema | Cached tokens | Forced tool |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic | `anthropic:<model>` | native | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | ✓ |
| gemini | `gemini:<model>` | native | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| azure | `azure:<model>` | native | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| openai | `openai:<model>` | hosted | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| xai | `xai:<model>` | hosted | ✓ | ✓ | ✓ | — | — | ✓ | ✓ | ✓ |
| groq | `groq:<model>` | hosted | ✓ | ✓ | ✓ | — | — | ✓ | — | ✓ |
| mistral | `mistral:<model>` | hosted | ✓ | ✓ | ✓ | — | — | ✓ | — | ✓ |
| deepseek | `deepseek:<model>` | hosted | ✓ | ✓ | — | — | — | — | — | ✓ |
| openrouter | `openrouter:<model>` | hosted | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| together | `together:<model>` | hosted | ✓ | ✓ | ✓ | — | — | ✓ | — | ✓ |
| fireworks | `fireworks:<model>` | hosted | ✓ | ✓ | ✓ | — | — | ✓ | — | ✓ |
| perplexity | `perplexity:<model>` | hosted | — | ✓ | ✓ | — | — | ✓ | — | — |
| cerebras | `cerebras:<model>` | hosted | ✓ | ✓ | — | — | — | ✓ | — | ✓ |
| nvidia | `nvidia:<model>` | hosted | ✓ | ✓ | ✓ | — | — | ✓ | — | ✓ |
| huggingface | `huggingface:<model>` | hosted | ✓ | ✓ | ✓ | — | — | ✓ | — | — |
| ollama | `ollama:<model>` | local | ✓ | ✓ | ✓ | — | — | ✓ | — | — |
| lmstudio | `lmstudio:<model>` | local | ✓ | ✓ | ✓ | — | — | ✓ | — | — |
| llamacpp | `llamacpp:<model>` | local | ✓ | ✓ | ✓ | — | — | ✓ | — | — |
| vllm | `vllm:<model>` | local | ✓ | ✓ | ✓ | — | — | ✓ | — | ✓ |
| sglang | `sglang:<model>` | local | ✓ | ✓ | ✓ | — | — | ✓ | — | ✓ |
| localai | `localai:<model>` | local | ✓ | ✓ | ✓ | — | — | — | — | — |
| jan | `jan:<model>` | local | ✓ | ✓ | — | — | — | — | — | — |
| llamafile | `llamafile:<model>` | local | — | ✓ | ✓ | — | — | — | — | — |
| tgi | `tgi:<model>` | local | ✓ | ✓ | ✓ | — | — | ✓ | — | — |

| Provider | Aliases | Base URL / default host | Environment variable |
|---|---|---|---|
| anthropic | `claude` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |
| gemini | `google` | `https://generativelanguage.googleapis.com` | `GEMINI_API_KEY` |
| azure | `azure-openai`, `azure_openai` | `$AZURE_OPENAI_ENDPOINT` | `AZURE_OPENAI_API_KEY` |
| openai | — | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| xai | `grok` | `https://api.x.ai/v1` | `XAI_API_KEY` |
| groq | — | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| mistral | — | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` |
| deepseek | — | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| openrouter | — | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| together | — | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` |
| fireworks | — | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` |
| perplexity | — | `https://api.perplexity.ai` | `PERPLEXITY_API_KEY` |
| cerebras | — | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` |
| nvidia | `nim` | `https://integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` |
| huggingface | `hf` | `https://router.huggingface.co/v1` | `HF_TOKEN` |
| ollama | — | `http://127.0.0.1:11434` | `OLLAMA_HOST` |
| lmstudio | `lm-studio`, `lm_studio` | `http://127.0.0.1:1234` | `LMSTUDIO_HOST` |
| llamacpp | `llama.cpp`, `llama-cpp`, `llama_cpp` | `http://127.0.0.1:8080` | `LLAMACPP_HOST` |
| vllm | — | `http://127.0.0.1:8000` | `VLLM_HOST` |
| sglang | — | `http://127.0.0.1:30000` | `SGLANG_HOST` |
| localai | — | `http://127.0.0.1:8080` | `LOCALAI_HOST` |
| jan | — | `http://127.0.0.1:1337` | `JAN_HOST` |
| llamafile | — | `http://127.0.0.1:8080` | `LLAMAFILE_HOST` |
| tgi | — | `http://127.0.0.1:8080` | `TGI_HOST` |
<!-- generated:providers:end -->

Notes:

- **Anthropic** rejects sampling parameters (`temperature`, `top_p`) and forced tool choice on Claude Fable 5.x,
  Mythos 5.1, Opus 5.5 and Sonnet 5.5 (sampling also on Opus 5, 4.8, 4.7 and Haiku 5.5). Sinew drops `temperature`
  and turns a forced tool choice into `auto` plus an instruction for those models, and `capabilities()` reports it
  per model (`sampling`, `forced_tool_choice`). Structured output uses a tool call instead of a JSON-schema mode.
- **JSON schema** means native schema-constrained output for `ResponseFormat::JsonSchema`. Without it,
  `StructuredOutput` falls back to a forced tool call or to prompting with validation and retries.
- **Cached tokens** means the provider reports prompt-cache hits and Sinew maps them to
  `Usage::cached_input_tokens` (and cost).
- **Local servers** read their host from the environment variable in the second table (default shown). Set
  `LOCAL_API_KEY` if your server requires a key.
- A custom endpoint `http(s)://host/v1|model` uses the OpenAI-compatible client with every capability enabled;
  declare the real ones with `OpenAiCompat::with_capabilities`.

The tables between the markers are generated from `providers::catalog()` by a test
(`SINEW_UPDATE_DOCS=1 cargo test -p sinew-core --test provider_matrix`), so they cannot drift from the code.
