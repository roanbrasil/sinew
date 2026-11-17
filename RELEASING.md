# Releasing Sinew

Releases are automated with [release-plz](https://release-plz.dev) and crates.io
[trusted publishing](https://crates.io/docs/trusted-publishing): no API token lives on a laptop or in
repository secrets.

## Flow

1. **Every PR** adds its entry under `## [Unreleased]` in `CHANGELOG.md`.
2. **On every push to `main`**, the `release-pr` job opens or updates a pull request named
   `Release vX.Y.Z` that bumps `workspace.package.version` (all crates share one version).
3. **Before merging the release PR**, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` in
   `CHANGELOG.md` on that branch, add a fresh empty `## [Unreleased]` above it, and update the
   compare links at the bottom.
4. **Merging the release PR** runs the `release` job: it publishes every crate whose new version is not on
   crates.io yet (in dependency order), tags `vX.Y.Z`, and creates the GitHub Release with that version's
   CHANGELOG section.

Version numbers follow [Semantic Versioning](https://semver.org/). Before 1.0, a breaking change bumps the
minor version (0.1 → 0.2) and everything else bumps the patch version. If release-plz proposes the wrong bump,
edit the version in the release PR.

## One-time setup (repository owner)

- **Trusted publishing on crates.io.** For each published crate, open crates.io → the crate → Settings →
  Trusted Publishing → Add, with repository `roanbrasil/sinew`, workflow `release.yml`, environment
  `crates-io`. A crate must exist on crates.io before it can be configured, so a brand-new crate (for example
  `sinew-macros`) is published once by hand with `cargo publish -p <crate>` after `cargo login`.
- **GitHub environment.** Settings → Environments → New environment `crates-io` (optionally restrict it to the
  `main` branch and require a reviewer).
- **Release PRs.** Settings → Actions → General → Workflow permissions → allow GitHub Actions to create pull
  requests.
- **Turn it on.** Settings → Secrets and variables → Actions → Variables → `RELEASE_ENABLED` = `true`. Until then
  the release workflow is skipped, so merges to `main` do not fail on missing setup.

## Manual release (fallback)

```bash
cargo login                       # token with publish-update scope
cargo publish --workspace         # publishes in dependency order
git tag -a vX.Y.Z -m "Sinew X.Y.Z" && git push origin vX.Y.Z
gh release create vX.Y.Z --title "Sinew X.Y.Z" --notes-file notes.md
```

New crates are rate limited on crates.io (a burst of 5, then about one every 10 minutes); new versions of
existing crates have a much higher limit.
