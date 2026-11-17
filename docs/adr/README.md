# Architecture Decision Records

An ADR records one design decision: the problem, what was chosen, and what it costs. Read the
relevant ADR before changing one of these areas, and write a new ADR (or supersede an old one)
when you change a decision. The overall layout is described in the
[architecture guide](../architecture.md).

| # | Decision | Status | Date |
|---|---|---|---|
| [0001](0001-typed-whole-state-nodes.md) | Typed whole-state nodes (`S -> S`) with reducers | Accepted | 2026-11-16 |
| [0002](0002-pregel-supersteps.md) | Pregel supersteps with deterministic task-order merges | Accepted | 2026-11-16 |
| [0003](0003-offline-first-tests.md) | Offline-first tests: recorded payloads, `ScriptedModel`, cassettes | Accepted | 2026-11-16 |
| [0004](0004-shared-http-client.md) | One shared, tuned HTTP client for all providers | Accepted | 2026-11-16 |
| [0005](0005-openai-compatible-presets.md) | OpenAI-compatible presets instead of one crate per provider | Accepted | 2026-11-16 |
| [0006](0006-heavy-dependencies-behind-features.md) | Heavy dependencies behind cargo features | Accepted | 2026-11-16 |
| [0007](0007-no-unsafe-code.md) | No `unsafe` code | Accepted | 2026-11-16 |
| [0008](0008-tracing-never-breaks-the-app.md) | Tracing never breaks the application | Accepted | 2026-11-16 |

## Writing a new ADR

1. Copy the template below into `docs/adr/NNNN-short-title.md`, using the next free number.
2. Keep it short: one decision per file, a page or less.
3. Start with `Status: Proposed`; switch to `Accepted` when the pull request that implements or
   confirms it is merged.
4. Do not rewrite an accepted ADR to reverse it. Write a new one, set the old one's status to
   `Superseded by NNNN` with a link, and link back from the new one.
5. Add the ADR to the table above.

Statuses: `Proposed`, `Accepted`, `Deprecated`, `Superseded by NNNN`.

## Template

The format follows [MADR](https://adr.github.io/madr/), trimmed to the sections used here.

```markdown
# NNNN. Short title of the decision

- Status: Proposed
- Date: YYYY-MM-DD
- Related: issues, PRs, other ADRs

## Context

The problem and the forces at play: requirements, constraints, what users expect,
what the code looks like today. Facts, with file paths.

## Considered options

- Option A
- Option B

## Decision

What was chosen and why it wins over the other options.

## Consequences

- Good: what becomes easier.
- Bad: what becomes harder; the trade-offs accepted.
- Follow-ups: work this decision implies.
```
