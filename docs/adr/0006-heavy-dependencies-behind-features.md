# 0006. Heavy dependencies behind cargo features

- Status: Accepted
- Date: 2026-11-16
- Related: [architecture guide, cargo features](../architecture.md#cargo-features)

## Context

Sinew covers chat models, graphs, databases, HTTP servers, PDF parsing and OpenTelemetry. Most
users need a fraction of that. Compile time, binary size and the dependency audit surface grow
with every crate pulled in.

## Considered options

- Everything always compiled in the umbrella crate.
- Separate crates only, no umbrella.
- An umbrella with a small default and one feature per optional crate or heavy dependency.

## Decision

The umbrella `sinew` always includes `sinew-core`, `sinew-graph` and `sinew-trace`; every other
crate is an optional dependency behind a feature
([crates/sinew/Cargo.toml](../../crates/sinew/Cargo.toml)). Inside crates, heavy or optional
dependencies are features too: `sqlx` backends (`sqlite`, `postgres`, `pgvector`,
`sqlite-cache`), `axum` for the MCP HTTP server (`http-server`), `pdf-extract` (`pdf`),
`schemars` and the proc macro (`schemars`, `macros`), and the network trace sinks (`http`,
`otlp`). `full` enables everything except Postgres-backed features. CONTRIBUTING.md asks that new
heavy dependencies go behind a feature.

## Consequences

- Good: `sinew` with default features builds core, graph, trace and the `#[tool]` macro only.
- Good: users pick a database or transport without paying for the others.
- Bad: more feature combinations exist than CI builds (CI uses `--all-features`), so a
  combination can break unnoticed.
- Bad: users have to find the right feature; a missing one shows up as a missing module or type.
  The feature tables in the README and the architecture guide are the reference.
