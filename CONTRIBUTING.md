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
- Read the [architecture guide](docs/architecture.md) and the [architecture decision records](docs/adr/README.md) before changing a design decision; record a new or superseding ADR when you do.

## Pull requests and signed commits

`main` is protected: every change lands through a pull request, the CI checks below must pass, and commits must
carry a verified signature. Squash merges made on GitHub are signed by GitHub, so a contributor's own commits
only need signing when a PR is merged with a merge commit or rebased.

Sign with your SSH key (simplest):

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
```

Then add the same public key on GitHub under Settings → SSH and GPG keys → **New SSH key** with key type
**Signing Key**. GPG works too (`git config --global user.signingkey <KEYID>` and upload the GPG public key).
Check with `git log --show-signature -1`; GitHub shows a "Verified" badge on the commit.

Required checks (`.github/workflows/ci.yml`): `lint` (rustfmt and clippy with and without all features), `test`
(Linux and macOS), `docs` (rustdoc with `-D warnings`), `msrv` (Rust 1.88), `deny` (licenses and advisories) and
`coverage`. [CODEOWNERS](.github/CODEOWNERS) requests a review from the maintainer automatically.

By contributing you agree that your work is dual-licensed under MIT and Apache-2.0, like the rest of the project.

## Fuzzing

Parsers of untrusted input have [cargo-fuzz](https://rust-fuzz.github.io/book/cargo-fuzz.html) targets in `fuzz/`,
a separate workspace that is not part of the main build (`publish = false`). Each target is a function in
`fuzz/src/` that panics when an invariant breaks; `fuzz/fuzz_targets/<name>.rs` hands it to libFuzzer.

| Target | Checks |
|---|---|
| `sse_parser` | chunked, byte-by-byte and one-shot SSE parsing give the same events |
| `stream_anthropic`, `stream_openai`, `stream_gemini` | provider stream decoders: same result however the body is chunked, successful streams end with `Done`, events fold into a response |
| `structured_json` | JSON extraction from replies, schema validation, list parsing, tool-call arguments assembled from partial JSON deltas |
| `json_rpc_mcp` | MCP `handle_message` and the stateless Streamable HTTP endpoint: a valid JSON-RPC response for every request, none for notifications |
| `json_rpc_a2a` | A2A JSON-RPC endpoint: always `200` with a valid JSON-RPC response (or an SSE stream) |
| `loader_html`, `loader_csv`, `loader_json`, `loader_markdown` | document loaders on in-memory input |
| `otlp_json`, `otlp_proto` | OTLP/HTTP ingest (JSON mapping, protobuf decoder and round trip) |
| `redactor`, `guardrails` | trace redaction and guardrail patterns: no panics, idempotent redaction, no secret left behind |
| `tool_args` | argument parsing in the `#[tool]` expansion: a value or `Error::Tool` for any JSON |

Internal parsers are reached through `#[doc(hidden)] pub mod __fuzz` in `sinew-core` and `sinew-rag`; they are not
public API.

```bash
rustup toolchain install nightly --profile minimal
cargo install cargo-fuzz
cargo +nightly fuzz run --debug-assertions sse_parser fuzz/corpus/sse_parser fuzz/seeds/sse_parser -- -max_total_time=90
cargo +nightly fuzz list                                   # every target
cargo test --manifest-path fuzz/Cargo.toml                 # replay seeds and regressions on stable
```

`fuzz/seeds/<target>/` holds a small committed seed corpus; the corpus libFuzzer grows (`fuzz/corpus/`) and crash
files (`fuzz/artifacts/`) are not committed. When a target finds a crash:

1. minimize it: `cargo +nightly fuzz tmin <target> fuzz/artifacts/<target>/crash-...`;
2. fix the bug in the owning crate and add a unit test there that reproduces it;
3. copy the minimized input to `fuzz/regressions/<target>/<short-name>`, so `cargo test` in `fuzz/` replays it on
   stable from then on;
4. add a `### Fixed` line to `CHANGELOG.md`.

`.github/workflows/fuzz.yml` replays seeds and regressions on every pull request that touches `crates/` or `fuzz/`,
and runs every target for five minutes each night (or on demand, with a custom duration, from the Actions tab),
keeping the grown corpus in the Actions cache and uploading crash files as artifacts.

To add a target: write the check in `fuzz/src/`, register it in `TARGETS` (`fuzz/src/lib.rs`), add
`fuzz/fuzz_targets/<name>.rs` and its `[[bin]]` in `fuzz/Cargo.toml`, a few seeds in `fuzz/seeds/<name>/`, and the
name to the matrix in `fuzz.yml`. The replay test fails if a registered target has no seeds or no binary.

## Extension guides

Adding a provider, vector store, document loader, tool, checkpointer or store? Follow the step-by-step guides in
[docs/contributing/](docs/contributing/README.md), and open the matching issue first (**New provider** or
**New integration**).

## Releases

Add a line under `## [Unreleased]` in `CHANGELOG.md` with every user-visible change. Releases are cut by merging the automated release PR; see [RELEASING.md](RELEASING.md).
