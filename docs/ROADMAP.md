# Sinew roadmap

Sinew aims to be the complete, low-latency agent stack in Rust: the LangChain / LangGraph / LangSmith
feature set, plus what those frameworks leave to users in production — durable execution without
repeated side effects, loop detection, deterministic replay tests, compile-time checked prompts,
first-class Kubernetes, and bindings so other languages can use the Rust runtime.

This file is generated from the GitHub issues; the issues are the source of truth.
Priorities: **P0** do next · **P1** this cycle · **P2** planned · **P3** nice to have / blocked.
★ marks features that other agent frameworks lack or do poorly.

## Themes

- [#111](https://github.com/roanbrasil/sinew/issues/111) Developer experience: macros, typed prompts, debugging
- [#112](https://github.com/roanbrasil/sinew/issues/112) Durable and scalable runtime
- [#113](https://github.com/roanbrasil/sinew/issues/113) Models, providers and cost
- [#114](https://github.com/roanbrasil/sinew/issues/114) Protocols: MCP, A2A, AG-UI, OpenAI APIs
- [#115](https://github.com/roanbrasil/sinew/issues/115) Retrieval (RAG)
- [#116](https://github.com/roanbrasil/sinew/issues/116) Observability, evaluation and Studio
- [#117](https://github.com/roanbrasil/sinew/issues/117) Security and governance
- [#118](https://github.com/roanbrasil/sinew/issues/118) Kubernetes and deployment
- [#119](https://github.com/roanbrasil/sinew/issues/119) Bindings and ecosystem reach
- [#120](https://github.com/roanbrasil/sinew/issues/120) Documentation
- [#121](https://github.com/roanbrasil/sinew/issues/121) Quality, CI and releases
- [#122](https://github.com/roanbrasil/sinew/issues/122) Community and adoption
- [#123](https://github.com/roanbrasil/sinew/issues/123) Vision: from framework to platform

## 0.1.1 — Ergonomics and docs, no breaking changes

| Priority | Issue |
|---|---|
| P0 | [#4](https://github.com/roanbrasil/sinew/issues/4) `#[sinew::tool]` attribute macro |
| P1 | [#5](https://github.com/roanbrasil/sinew/issues/5) `#[derive(State)]` for graph state with reducers |
| P1 | [#6](https://github.com/roanbrasil/sinew/issues/6) Examples on every public item in docs.rs |
| P1 | [#7](https://github.com/roanbrasil/sinew/issues/7) GitHub Release for v0.1.0 |
| P1 | [#26](https://github.com/roanbrasil/sinew/issues/26) Loop and runaway detection for agents ★ |
| P1 | [#27](https://github.com/roanbrasil/sinew/issues/27) Record/replay testing (VCR cassettes for models and tools) ★ |
| P1 | [#51](https://github.com/roanbrasil/sinew/issues/51) Architecture guide: crates, layers and data flow |
| P1 | [#52](https://github.com/roanbrasil/sinew/issues/52) Migration guide from LangChain / LangGraph (Python) |
| P1 | [#54](https://github.com/roanbrasil/sinew/issues/54) Provider capability matrix |
| P1 | [#65](https://github.com/roanbrasil/sinew/issues/65) Graph visualization: Mermaid, DOT and PNG export |
| P1 | [#101](https://github.com/roanbrasil/sinew/issues/101) Release automation: release-plz and crates.io trusted publishing |
| P2 | [#56](https://github.com/roanbrasil/sinew/issues/56) Troubleshooting and error reference |
| P2 | [#58](https://github.com/roanbrasil/sinew/issues/58) Contributor guides: add a provider, vector store, loader, tool or checkpointer |
| P2 | [#99](https://github.com/roanbrasil/sinew/issues/99) Code coverage reporting |
| P2 | [#104](https://github.com/roanbrasil/sinew/issues/104) Signed commits and PR-based workflow on main |
| P2 | [#105](https://github.com/roanbrasil/sinew/issues/105) Community: GitHub Discussions, good first issues and contributor ladder |
| P2 | [#106](https://github.com/roanbrasil/sinew/issues/106) Launch announcement: r/rust, This Week in Rust, Hacker News, Medium |
| P3 | [#8](https://github.com/roanbrasil/sinew/issues/8) Social preview image on GitHub |
| P3 | [#59](https://github.com/roanbrasil/sinew/issues/59) Architecture Decision Records (ADRs) |

## 0.2.0 — sqlx 0.9 (MSRV 1.94), durable execution, full MCP, memory, guardrails, cost features

| Priority | Issue |
|---|---|
| P0 | [#9](https://github.com/roanbrasil/sinew/issues/9) Upgrade to sqlx 0.9 (fixes RUSTSEC-2026-0459) |
| P0 | [#25](https://github.com/roanbrasil/sinew/issues/25) Durable execution: side-effect ledger and idempotent tool calls ★ |
| P1 | [#10](https://github.com/roanbrasil/sinew/issues/10) MCP: resources and prompts |
| P1 | [#11](https://github.com/roanbrasil/sinew/issues/11) MCP: OAuth 2.1 for the HTTP server and client |
| P1 | [#13](https://github.com/roanbrasil/sinew/issues/13) Long-term memory |
| P1 | [#14](https://github.com/roanbrasil/sinew/issues/14) Guardrails as composable runnables |
| P1 | [#34](https://github.com/roanbrasil/sinew/issues/34) Model router: pick the model per request by cost, latency or difficulty ★ |
| P1 | [#36](https://github.com/roanbrasil/sinew/issues/36) Automatic prompt caching and batch APIs |
| P1 | [#45](https://github.com/roanbrasil/sinew/issues/45) Secret and PII redaction in traces |
| P1 | [#55](https://github.com/roanbrasil/sinew/issues/55) Production deployment guide and security checklist |
| P1 | [#63](https://github.com/roanbrasil/sinew/issues/63) Compile-time checked prompt templates (`prompt!` macro) ★ |
| P1 | [#68](https://github.com/roanbrasil/sinew/issues/68) Context engineering toolkit: trimming, summarization and compression |
| P1 | [#70](https://github.com/roanbrasil/sinew/issues/70) Built-in tool library |
| P1 | [#79](https://github.com/roanbrasil/sinew/issues/79) RAG evaluation metrics |
| P1 | [#80](https://github.com/roanbrasil/sinew/issues/80) Prometheus metrics endpoint |
| P1 | [#84](https://github.com/roanbrasil/sinew/issues/84) Playground in sinew-scope |
| P1 | [#87](https://github.com/roanbrasil/sinew/issues/87) Agent trajectory evaluation |
| P1 | [#96](https://github.com/roanbrasil/sinew/issues/96) Fuzz parsers: SSE decoders, JSON-RPC, partial JSON, loaders |
| P1 | [#98](https://github.com/roanbrasil/sinew/issues/98) Nightly live-provider test matrix |
| P2 | [#12](https://github.com/roanbrasil/sinew/issues/12) MCP: sampling and elicitation |
| P2 | [#35](https://github.com/roanbrasil/sinew/issues/35) Semantic cache |
| P2 | [#43](https://github.com/roanbrasil/sinew/issues/43) OpenAI Responses API compatibility in sinew-serve |
| P2 | [#53](https://github.com/roanbrasil/sinew/issues/53) Cookbook: 30 task-oriented recipes |
| P2 | [#60](https://github.com/roanbrasil/sinew/issues/60) Project website and landing page |
| P2 | [#66](https://github.com/roanbrasil/sinew/issues/66) Streaming partial structured output |
| P2 | [#67](https://github.com/roanbrasil/sinew/issues/67) Accurate token counting with provider tokenizers |
| P2 | [#75](https://github.com/roanbrasil/sinew/issues/75) Embedded HNSW index for in-process ANN search ★ |
| P2 | [#76](https://github.com/roanbrasil/sinew/issues/76) More document loaders and connectors |
| P2 | [#78](https://github.com/roanbrasil/sinew/issues/78) Rerankers: local cross-encoders, Voyage, Jina, Cohere |
| P2 | [#89](https://github.com/roanbrasil/sinew/issues/89) Checkpoint encryption at rest |
| P2 | [#90](https://github.com/roanbrasil/sinew/issues/90) Web framework integrations: axum, actix-web, Loco |
| P2 | [#95](https://github.com/roanbrasil/sinew/issues/95) Scheduled and background agents (cron) |
| P2 | [#97](https://github.com/roanbrasil/sinew/issues/97) Property-based tests for graph semantics and reducers |
| P2 | [#100](https://github.com/roanbrasil/sinew/issues/100) Benchmark regression tracking in CI |
| P2 | [#102](https://github.com/roanbrasil/sinew/issues/102) Supply chain: SBOM, signed releases, cargo-vet |
| P2 | [#103](https://github.com/roanbrasil/sinew/issues/103) Reduce compile times and dependency footprint |

## 0.3.0 — More providers and stores, eval gate, event-driven agents, enterprise scope

| Priority | Issue |
|---|---|
| P1 | [#15](https://github.com/roanbrasil/sinew/issues/15) Native AWS Bedrock provider |
| P2 | [#16](https://github.com/roanbrasil/sinew/issues/16) Native Cohere provider |
| P2 | [#17](https://github.com/roanbrasil/sinew/issues/17) Claude on Google Vertex AI |
| P2 | [#18](https://github.com/roanbrasil/sinew/issues/18) `sinew-eval run`: evaluation gate for CI |
| P2 | [#19](https://github.com/roanbrasil/sinew/issues/19) Docker image and Helm chart for sinew-serve and sinew-scope |
| P2 | [#20](https://github.com/roanbrasil/sinew/issues/20) Benchmarks against LangChain / LangGraph (Python) |
| P2 | [#28](https://github.com/roanbrasil/sinew/issues/28) Debugger: step through a run and edit state ★ |
| P2 | [#32](https://github.com/roanbrasil/sinew/issues/32) Event-driven agents: Kafka / NATS triggers and AsyncAPI contracts ★ |
| P2 | [#33](https://github.com/roanbrasil/sinew/issues/33) Multi-tenancy: per-tenant quotas, cost budgets and isolation |
| P2 | [#42](https://github.com/roanbrasil/sinew/issues/42) AG-UI protocol and frontend streaming |
| P2 | [#44](https://github.com/roanbrasil/sinew/issues/44) Policy engine for tool permissions (Cedar / OPA) |
| P2 | [#46](https://github.com/roanbrasil/sinew/issues/46) Red-teaming and multi-turn user simulation in sinew-eval |
| P2 | [#48](https://github.com/roanbrasil/sinew/issues/48) GraphRAG: knowledge graph extraction and retrieval |
| P2 | [#49](https://github.com/roanbrasil/sinew/issues/49) SQL and data agent toolkit |
| P2 | [#69](https://github.com/roanbrasil/sinew/issues/69) Prebuilt reasoning agents: plan-and-execute, reflection, tree search |
| P2 | [#71](https://github.com/roanbrasil/sinew/issues/71) Declarative graphs in YAML/JSON |
| P2 | [#72](https://github.com/roanbrasil/sinew/issues/72) Speech, transcription and image generation traits |
| P2 | [#74](https://github.com/roanbrasil/sinew/issues/74) More vector stores: LanceDB, Milvus, Weaviate, Elasticsearch/OpenSearch, Redis, Chroma |
| P2 | [#77](https://github.com/roanbrasil/sinew/issues/77) Layout-aware parsing: tables, OCR and images in documents |
| P2 | [#81](https://github.com/roanbrasil/sinew/issues/81) Export to Langfuse, Phoenix (OpenInference) and LangSmith |
| P2 | [#82](https://github.com/roanbrasil/sinew/issues/82) sinew-scope: Postgres and ClickHouse storage backends |
| P2 | [#83](https://github.com/roanbrasil/sinew/issues/83) sinew-scope: SSO (OIDC), RBAC and audit log |
| P2 | [#86](https://github.com/roanbrasil/sinew/issues/86) Online A/B tests of prompts and models in production |
| P2 | [#88](https://github.com/roanbrasil/sinew/issues/88) Checkpointers and stores: Redis, DynamoDB, MySQL, S3 |
| P2 | [#93](https://github.com/roanbrasil/sinew/issues/93) MCP registry discovery and remote server catalog |
| P2 | [#94](https://github.com/roanbrasil/sinew/issues/94) OpenAI Agents SDK and Anthropic Agent Skills compatibility |

## 0.4.0 — Kubernetes operator and sandbox, distributed execution, Python bindings, Studio

| Priority | Issue |
|---|---|
| P1 | [#29](https://github.com/roanbrasil/sinew/issues/29) Kubernetes operator with Agent / Graph CRDs ★ |
| P1 | [#30](https://github.com/roanbrasil/sinew/issues/30) Kubernetes Agent Sandbox (agents.x-k8s.io) backend for harness tools ★ |
| P1 | [#39](https://github.com/roanbrasil/sinew/issues/39) Python bindings (PyO3): Sinew core from Python ★ |
| P2 | [#31](https://github.com/roanbrasil/sinew/issues/31) Distributed graph execution with a work queue |
| P2 | [#37](https://github.com/roanbrasil/sinew/issues/37) In-process local inference (no Ollama server) ★ |
| P2 | [#38](https://github.com/roanbrasil/sinew/issues/38) Realtime voice agents |
| P2 | [#85](https://github.com/roanbrasil/sinew/issues/85) Sinew Studio: visual graph builder and live run viewer ★ |

## 1.0.0 — Stable API and semver guarantees

| Priority | Issue |
|---|---|
| P1 | [#57](https://github.com/roanbrasil/sinew/issues/57) API stability, semver and MSRV policy |
| P1 | [#107](https://github.com/roanbrasil/sinew/issues/107) 1.0 API review |
| P3 | [#64](https://github.com/roanbrasil/sinew/issues/64) Typestate graph builder: catch dangling edges at compile time ★ |

## Backlog — Unscheduled and long-term vision

| Priority | Issue |
|---|---|
| P2 | [#22](https://github.com/roanbrasil/sinew/issues/22) ServeClient adds ~600µs to time to first token |
| P2 | [#23](https://github.com/roanbrasil/sinew/issues/23) Guide book (mdBook) published to GitHub Pages |
| P3 | [#21](https://github.com/roanbrasil/sinew/issues/21) A2A 1.0 |
| P3 | [#24](https://github.com/roanbrasil/sinew/issues/24) Supervisor sample gives weak answers with small local models |
| P3 | [#40](https://github.com/roanbrasil/sinew/issues/40) Node.js / TypeScript bindings (napi-rs) |
| P3 | [#41](https://github.com/roanbrasil/sinew/issues/41) WASM target: run agents in the browser and on edge workers ★ |
| P3 | [#47](https://github.com/roanbrasil/sinew/issues/47) Export traces as fine-tuning datasets |
| P3 | [#50](https://github.com/roanbrasil/sinew/issues/50) Browser and computer-use tools |
| P3 | [#61](https://github.com/roanbrasil/sinew/issues/61) Portuguese (pt-BR) translation of the guide |
| P3 | [#62](https://github.com/roanbrasil/sinew/issues/62) Video tutorials and talk material |
| P3 | [#73](https://github.com/roanbrasil/sinew/issues/73) Async runtime agnosticism |
| P3 | [#91](https://github.com/roanbrasil/sinew/issues/91) Desktop and UI examples: Tauri, Leptos, Dioxus |
| P3 | [#92](https://github.com/roanbrasil/sinew/issues/92) Game NPC agents with Bevy |
| P3 | [#108](https://github.com/roanbrasil/sinew/issues/108) Vision: Sinew Cloud-free platform (self-hosted control plane) |
| P3 | [#109](https://github.com/roanbrasil/sinew/issues/109) Vision: Sinew Hub for sharing graphs, tools and prompts |
| P3 | [#110](https://github.com/roanbrasil/sinew/issues/110) Vision: agent mesh — discovery, routing and policy across A2A/MCP agents ★ |

## How to help

Pick an issue labeled `good first issue` or any `P0`/`P1`, comment that you are on it, and open a PR. See [CONTRIBUTING.md](https://github.com/roanbrasil/sinew/blob/main/CONTRIBUTING.md).

