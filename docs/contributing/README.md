# Contributor guides

Step-by-step guides for the extension points where most contributions land. Each one lists the trait to
implement (with its signature copied from the source), where the files go, the feature flag, the tests to add, the
docs to update and the changelog line.

| Guide | You are adding | Crate |
|---|---|---|
| [Adding a provider](provider.md) | A chat model provider: an OpenAI-compatible preset or a native client | `sinew-core` |
| [Adding a vector store](vector-store.md) | A `VectorStore` backend | `sinew-rag` |
| [Adding a document loader](loader.md) | A `Loader` for a file format or source | `sinew-rag` |
| [Adding a tool](tool.md) | A tool for agents, or a built-in harness tool | `sinew-core`, `sinew-harness` |
| [Adding a checkpointer or store](checkpointer.md) | A durable `Checkpointer` / `CheckpointStore`, or a long-term `Store` | `sinew-checkpoint`, `sinew-graph` |

## Before you start

1. Open an issue with the matching template: **New provider** or **New integration**. It records what the API
   accepts and how it will be tested, before any code is written.
2. Read [CONTRIBUTING.md](../../CONTRIBUTING.md). The rules that matter most for integrations:
   - `cargo test` never touches the network. Request and response mapping is tested against recorded payloads;
     tests that need a server or a key are `#[ignore]` and run on demand.
   - New heavy dependencies go behind a cargo feature, off by default.
   - Public items have doc comments.
3. Run the full check before opening the pull request:

   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   cargo test --workspace --all-features
   RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
   ```

## Every integration pull request

- [ ] Trait implemented, public items documented.
- [ ] Offline tests (recorded payloads or an in-process backend), plus `#[ignore]` live tests where a server is
      needed, with the command to run them in the test file's header.
- [ ] Feature flag in the crate and, when users reach it through the umbrella crate, in `crates/sinew/Cargo.toml`
      (including `full` when it has no system dependency).
- [ ] The relevant guide under `docs/` and the crate table in `README.md` updated.
- [ ] A line under `## [Unreleased]` → `### Added` in `CHANGELOG.md`, with the issue number.

If an error message you hit while building this was hard to understand, add or improve its entry in
[troubleshooting.md](../troubleshooting.md).
