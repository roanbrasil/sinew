# Contributing to Sinew

Thanks for helping. Before opening a pull request:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

- Keep crates small and dependency-light; new heavy dependencies go behind a cargo feature.
- Provider request/response mapping must be covered by offline unit tests (no network in `cargo test`).
- Public items need doc comments. Examples in docs should compile.
- One logical change per PR, with a clear description of the behavior change.

By contributing you agree that your work is dual-licensed under MIT and Apache-2.0, like the rest of the project.
