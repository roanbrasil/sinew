# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Native Google Gemini provider (`Gemini`) with tools, streaming, structured output and thought-signature round trips.
- OpenAI-compatible presets: xAI Grok, Groq, Mistral, DeepSeek, OpenRouter, Together, Fireworks, Perplexity, Cerebras, NVIDIA NIM, Hugging Face; local LM Studio, llama.cpp, vLLM, SGLang, LocalAI, Jan, llamafile, TGI.
- `providers::from_spec("provider:model")`; samples accept `--model provider:model`.
- Per-crate READMEs, logo, banner.

## [0.1.0]

- Initial crates: `sinew-core`, `sinew-graph`, `sinew-trace`, `sinew-checkpoint`, `sinew-mcp`, `sinew-scope`, `sinew-eval`, `sinew`; samples 01–13.
