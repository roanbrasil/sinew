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
(Linux and macOS), `docs` (rustdoc with `-D warnings`), `msrv` (Rust 1.94), `deny` (licenses and advisories) and
`coverage`. [CODEOWNERS](.github/CODEOWNERS) requests a review from the maintainer automatically.

By contributing you agree that your work is dual-licensed under MIT and Apache-2.0, like the rest of the project.

## Extension guides

Adding a provider, vector store, document loader, tool, checkpointer or store? Follow the step-by-step guides in
[docs/contributing/](docs/contributing/README.md), and open the matching issue first (**New provider** or
**New integration**).

## Releases

Add a line under `## [Unreleased]` in `CHANGELOG.md` with every user-visible change. Releases are cut by merging the automated release PR; see [RELEASING.md](RELEASING.md).
