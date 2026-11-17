# Contributing to Sinew

Thanks for helping. Before opening a pull request:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo deny check   # optional locally: cargo install cargo-deny
```

- Keep crates small and dependency-light; new heavy dependencies go behind a cargo feature.
- Provider request/response mapping must be covered by offline unit tests (no network in `cargo test`).
- Public items need doc comments. Examples in docs should compile.
- Add an entry to `CHANGELOG.md` under "Unreleased".
- Hot paths (streaming decoders, graph supersteps, vector search) must not regress; run the benches (`cargo bench -p <crate>`) when touching them.
- One logical change per PR, with a clear description of the behavior change.

By contributing you agree that your work is dual-licensed under MIT and Apache-2.0, like the rest of the project.

## Releases

Add a line under `## [Unreleased]` in `CHANGELOG.md` with every user-visible change. Releases are cut by merging the automated release PR; see [RELEASING.md](RELEASING.md).
