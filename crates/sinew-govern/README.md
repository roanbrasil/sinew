<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew-govern

[![crates.io](https://img.shields.io/crates/v/sinew-govern.svg)](https://crates.io/crates/sinew-govern)
[![docs.rs](https://docs.rs/sinew-govern/badge.svg)](https://docs.rs/sinew-govern)
[![CI](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml/badge.svg)](https://github.com/roanbrasil/sinew/actions/workflows/ci.yml)

A control plane for Sinew agents: policies decide, before every tool call and model call, which agent may call
which tool with which arguments, for which user and tenant. A registry of agents, a kill switch and an audit event
for every decision come with it.

Part of [Sinew](https://github.com/roanbrasil/sinew): LLM chains, agent graphs, tracing and evaluation in Rust.

## Quick example

```rust
use std::sync::Arc;
use sinew_govern::{AgentProfile, Condition, ControlPlane, Identity, Rule};
use sinew_graph::prebuilt::ReactAgent;

let cp = ControlPlane::builder()
    .agent(AgentProfile::new("support").owner("cx-team").allow_tools(["get_*", "refund"]))
    .policy(Rule::allow("small-refunds").tools(["refund"]).when(Condition::parse("amount <= 100")?))
    .policy(Rule::ask("large-refunds").tools(["refund"]).when(Condition::parse("amount > 100")?))
    .build();

let agent = ReactAgent::new(model, tools)
    .gate(Arc::new(cp.gate(Identity::agent("support").user("ana"))))
    .compile()?;

cp.suspend("support", "incident 42");   // running turns stop at their next step
```

Or from a policy file:

```yaml
version: 1
agents:
  - { id: support, owner: cx-team, allowed_tools: ["get_*", "refund"] }
policies:
  - { id: small-refunds, effect: allow, tools: [refund], when: ["amount <= 100"] }
  - { id: large-refunds, effect: ask, tools: [refund], when: ["amount > 100"] }
```

## What it covers

- `Decision`: `Allow`, `Ask` (human approval), `Deny`, `Redact` (mask argument fields); deny overrides ask
  overrides redact overrides allow; tools with side effects are denied by default.
- Built-in policies: `Rule` (globs on agent, user, tenant, tool, model; argument conditions; side effects; time
  windows) and `RateLimit`; YAML / JSON `PolicyFile`s with strict validation.
- External engines: `CedarPolicy` (feature `cedar`) and `OpaPolicy` (feature `opa`).
- `ControlPlane`: registry (owner, tenant, allowed tools and models, budgets, status), `suspend`, `quarantine`,
  `disable_tool`, a trace run and a `DecisionEvent` for every decision.
- Enforcement: `ReactAgent::gate`, the harness `GovernHook` (feature `harness`), `governed_toolbox`,
  `govern_model`.

## Features

| Feature | Adds |
|---|---|
| `yaml` (default) | YAML policy files |
| `harness` | `GovernHook` for `sinew-harness` agents |
| `cedar` | `CedarPolicy` (the `cedar-policy` crate) |
| `opa` | `OpaPolicy` (Open Policy Agent data API over HTTP) |

Guide: [docs/governance.md](https://github.com/roanbrasil/sinew/blob/main/docs/governance.md). Architecture:
[docs/architecture.md](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

MIT OR Apache-2.0
