# 0007. No `unsafe` code

- Status: Accepted
- Date: 2026-11-16
- Related: [SECURITY.md](../../SECURITY.md)

## Context

Sinew passes model output to tools, file systems, shells and servers. Memory-safety bugs in this
kind of code become security problems. The hot paths (SSE decoding, graph supersteps, vector
search) are fast enough in safe Rust.

## Considered options

- Allow `unsafe` where profiling justifies it, with review.
- Forbid `unsafe` across the workspace.

## Decision

The workspace sets `unsafe_code = "forbid"` under `[workspace.lints.rust]`
([Cargo.toml](../../Cargo.toml)), and every crate opts in with `[lints] workspace = true`.
`forbid` cannot be overridden with `#[allow]` inside a crate.

## Consequences

- Good: there are no `unsafe` blocks to review; memory safety of Sinew's own code rests on the
  compiler.
- Good: users can state this guarantee in their own audits.
- Bad: some optimizations (unchecked indexing, SIMD intrinsics, custom allocators) are not
  available; performance work has to come from data layout and algorithms.
- Bad: dependencies may still use `unsafe`; this decision covers Sinew's own code only.
  `cargo deny` ([deny.toml](../../deny.toml)) checks advisories and licenses of dependencies.
