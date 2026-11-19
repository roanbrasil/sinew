# Governance: policies, control plane and kill switch

`sinew-govern` (umbrella feature `govern`, re-exported as `sinew::govern`) decides, before every tool call and
model call, whether an agent may make it: which agent, acting for which user and tenant, may call which tool with
which arguments, or which model. Decisions come from policies written in Rust, in a YAML or JSON file, in Cedar or
in an Open Policy Agent server. A `ControlPlane` holds the policies together with a registry of agents and a kill
switch, and records every decision as a trace run and an audit event.

```rust
use std::sync::Arc;
use sinew::govern::{AgentProfile, Condition, ControlPlane, Identity, Rule};
use sinew::graph::prebuilt::ReactAgent;

let cp = ControlPlane::builder()
    .agent(AgentProfile::new("support").owner("cx-team").allow_tools(["get_*", "refund"]))
    .policy(Rule::allow("small-refunds").tools(["refund"]).when(Condition::parse("amount <= 100")?))
    .policy(Rule::ask("large-refunds").tools(["refund"]).when(Condition::parse("amount > 100")?))
    .tracer(tracer.clone())
    .build();

let agent = ReactAgent::new(model, tools)
    .gate(Arc::new(cp.gate(Identity::agent("support").user("ana").tenant("acme"))))
    .compile()?
    .with_checkpointer(checkpointer);   // `Ask` pauses with an interrupt

// Later, from an admin endpoint or an alert handler:
cp.suspend("support", "incident 42");   // running turns stop at their next step
```

## Concepts

- **Request** (`Request`): one tool call (`kind: tool`, `tool`, `arguments`, `side_effects`) or model call
  (`kind: model`, `model`), with the calling `agent`, the `user` and `tenant` it acts for, the `trace_id` and
  `thread_id`, the evaluation time and free-form `metadata`.
- **Decision** (`Decision`): `Allow`, `Ask { reason }` (a human approves first), `Deny { reason }`, or
  `Redact { fields }` (allow, with these argument fields, JSON pointers such as `/card/number`, replaced by
  `[REDACTED]` before the tool runs and in the recorded decision).
- **Policy** (`Policy` trait): returns a decision for a request, or nothing when it does not apply.
- **Verdict** (`Verdict`): the combined result: the final decision, the id of the policy that decided
  (`None` when the default applied), every matching policy, the redacted fields, and `stop` when the agent is
  suspended.

Combination: **deny overrides ask, ask overrides redact, redact overrides allow**. Among decisions of the same
rank, the first policy in order decides. `Redact` fields from every matching policy are merged, and they also
apply to a call that runs after an `Ask` is approved. A policy that fails (an OPA server that is down) counts as
`Deny`: the engine fails closed.

When no policy decides, `Defaults` apply. The default allows tools without side effects and model calls, and
**denies tools with side effects** (`Tool::side_effects`, `#[tool(side_effects)]`), so a tool that charges, sends
or writes runs only when a policy allows it or asks a human. `Defaults::deny_all()` turns the engine into an
allow-list; `Defaults::allow_all()` keeps it out of the way until you add deny rules.

## Built-in policies

`Rule` returns its decision for every request that matches all of its filters. Filters left empty match
everything.

| Filter | Matches |
|---|---|
| `agents`, `users`, `tenants` | glob patterns (`*`, `?`) on the request's ids; a request without a user never matches a `users` filter |
| `tools` | glob patterns on the tool name; the rule then applies to tool calls only |
| `models` | glob patterns on the model (`anthropic:*`, `*gpt-4*`); the rule then applies to model calls only |
| `side_effects(bool)` | tools with (or without) side effects |
| `when(Condition)`, `arg(field, Op)` | conditions on the arguments, all of which must hold |
| `during(TimeWindow)`, `outside(TimeWindow)` | days of the week and a time range at a fixed UTC offset |

```rust
use sinew::govern::{Condition, Op, RateLimit, Rule, TimeWindow};
use std::time::Duration;

Rule::deny("payment-cap").tools(["pay_*"]).arg("amount", Op::Gt(1_000.0))
    .reason("payments over 1000 are made by people");
Rule::ask("after-hours").side_effects(true)
    .outside(TimeWindow::workdays("09:00", "17:00")?.utc_offset("-05:00")?);
Rule::redact("mask-card", ["/card/number"]).tools(["*"]);
Rule::deny("no-gpt-for-acme").tenants(["acme"]).models(["openai:*"]);
Rule::side_effects_need_approval();
RateLimit::new("refund-rate", 5, Duration::from_secs(60)).tools(["refund"]);   // per agent and tool
```

**Conditions** (`Condition::parse`) are `field operator value`. The field is a JSON pointer (`/payee/country`) or a
dotted path (`payee.country`); values are JSON, and a bare word that is not JSON is a string.

| Condition | Meaning |
|---|---|
| `amount <= 100` | number comparisons `<`, `<=`, `>`, `>=` (numbers in strings count) |
| `currency == "USD"`, `status != closed` | JSON equality (`1` equals `1.0`) |
| `currency in ["USD", "CAD"]`, `country not in ["KP"]` | membership |
| `to matches "@example\\.com$"` | regular expression on a string |
| `customer.ssn exists`, `customer.ssn missing` | presence |

A condition on a field the arguments do not have is false (except `missing`), so `Rule::allow(...).when("amount <=
100")` does not allow a call without an amount; the default decides it.

**Rate limits** (`RateLimit`) deny a request once `max` requests passed through the limiter in the sliding window.
Counts are kept in memory per combination of `RateKey`s (default agent and tool; also user and tenant), and use the
request's time. A request the limiter lets through is counted even if another policy denies it. Use
`.model_calls()` to limit model calls. Counts are per process: with several replicas, divide the limit or use an
external engine.

## Policy files

The same policies as data, in YAML (feature `yaml`, on by default) or JSON:

```yaml
version: 1
defaults:
  tools: allow
  side_effect_tools: deny        # allow | ask | deny
  models: allow
agents:
  - id: billing-bot
    owner: payments-team
    tenant: acme
    allowed_tools: ["get_*", "refund"]
    allowed_models: ["anthropic:*"]
    budget: { max_tool_calls: 500 }
disabled_tools:
  - { tool: "shell", reason: "incident 42" }
policies:
  - id: small-refunds
    effect: allow
    tools: [refund]
    when: ["amount <= 100", "currency in [\"USD\", \"CAD\"]"]
  - id: large-refunds
    effect: ask
    reason: refunds over 100 need a human
    tools: [refund]
    when: ["amount > 100"]
  - id: mask-card
    effect: redact
    fields: ["/card/number"]
  - id: business-hours
    effect: deny
    side_effects: true
    outside: { days: [mon, tue, wed, thu, fri], from: "09:00", to: "17:00", utc_offset: "-05:00" }
  - id: refund-rate
    rate_limit: { max: 5, per_seconds: 60, key: [agent, tool] }
    tools: [refund]
```

```rust
use sinew::govern::{ControlPlane, PolicyFile};

let cp = ControlPlane::builder().policy_file(PolicyFile::from_path("policies.yaml")?).build();
// Reload without restarting: replaces policies, defaults and disabled tools, re-registers agents.
cp.apply_file(PolicyFile::from_path("policies.yaml")?);
```

Policy entries also take `agents`, `users`, `tenants`, `models`, `applies_to: tool | model` and `during`. Parsing
is strict: unknown fields, unknown effects, a `reason` on an `allow`, `fields` outside `redact`, malformed
conditions, times and offsets, and duplicate ids are errors that name the entry, for example
`policy file: policies[1] (id "large-refunds"): when[0]: invalid condition ...`, or the line and column for
syntax errors.

## External policy engines

**Cedar** (feature `cedar`, umbrella `govern-cedar`; not part of `full` because the `cedar-policy` crate and its
parser generator add about 70 packages to the lock file). `CedarPolicy` evaluates Cedar policy text with the principal
`Agent::"<id>"` (a member of `Tenant::"<tenant>"`), the action `Action::"CallTool"` or `Action::"CallModel"`, the
resource `Tool::"<name>"` or `Model::"<model>"`, and a context with `arguments`, `side_effects`, and `user`,
`tenant`, `thread_id` when known.

```cedar
permit(principal in Tenant::"acme", action == Action::"CallTool", resource == Tool::"refund")
when { context.arguments.amount <= 100 };

@ask("refunds over 100 need a human")
permit(principal in Tenant::"acme", action == Action::"CallTool", resource == Tool::"refund")
when { context.arguments.amount > 100 };

@reason("wires are disabled")
forbid(principal, action == Action::"CallTool", resource == Tool::"wire");
```

A `permit` annotated `@ask` turns an allow into `Ask`; a `forbid`'s `@reason` becomes the denial reason. When no
Cedar policy matches, `CedarPolicy` returns no decision so the control plane's defaults apply
(`.deny_when_nothing_matches()` restores Cedar's implicit deny). Cedar has no floating-point numbers or `null`:
non-integer numbers in the arguments become strings and `null` fields are dropped. Argument keys starting with `__`
are prefixed with `arg` so a model cannot write Cedar's JSON escapes (`__entity`) into the context. Evaluation
errors without a determining policy are errors (fail closed). Note that `cedar-policy` turns on serde_json's
`preserve_order` feature, which applies to the whole build: JSON objects then keep insertion order instead of
sorting their keys.

**Open Policy Agent** (feature `opa`, umbrella `govern-opa`). `OpaPolicy` posts `{"input": <Request>}` to an OPA
server's data API (`POST /v1/data/<path>`) and maps `result`: `true` / `false`, an object with `decision`
(`allow`, `ask`, `deny`, `redact`), `reason` and `fields`, or `{"allow": bool, "reason": ...}`. An undefined result
is no decision. Network errors, timeouts and non-2xx answers deny. Each call is one HTTP round trip on the request
path: run OPA next to the agent (sidecar) and keep the 2 s default timeout or lower it.

```rust
let cp = ControlPlane::builder()
    .policy(OpaPolicy::new("opa", "http://localhost:8181", "sinew/tools/decision").bearer_token(token))
    .build();
```

Any other engine fits behind the `Policy` trait (`id` and an async `evaluate`).

## The control plane

`ControlPlane` is cheap to clone; clones share the registry, policies and kill switch, so an admin task can act
while agents run. A check goes through these steps, and the first denial ends it:

1. **Registry**: requests from agents that are not registered are denied (`require_registration(false)` to allow
   them). An agent with a `tenant` cannot act for another tenant; requests without a tenant get the agent's.
2. **Kill switch**: a suspended agent is denied with `Verdict::stop` set; a tool matching a disabled pattern is
   denied for every agent; a quarantined agent cannot call tools with side effects.
3. **Allowed tools and models** of the agent's profile, then its **budget** (`AgentBudget`: tool calls and model
   calls the control plane allowed, until `reset_usage`).
4. **Policies**, combined as above, then the **defaults**.

Each step that decides has a policy id (`control-plane/registry`, `control-plane/kill-switch`,
`control-plane/quarantine`, `control-plane/allowed`, `control-plane/budget`, `control-plane/tenant`), so every
decision says what made it.

`AgentProfile` holds the id, owner, description, tenant, allowed tools and models (glob patterns; unset means no
restriction from the registry), budget and status. Per-turn limits (steps, tokens, cost, wall time) stay in the
runtime: the harness `Budget` and `ReactAgent::max_iterations`.

### Kill switch and quarantine

| Call | Effect |
|---|---|
| `suspend(agent, reason)` | every model call and tool call of the agent is denied; running turns end at their next step |
| `quarantine(agent, reason)` | tools with side effects are denied; read-only tools and model calls still go through the policies, so the agent can be observed |
| `reinstate(agent)` | back to active |
| `disable_tool(pattern, reason)` / `enable_tool(pattern)` | calls of matching tools are denied for every agent (`mcp_github_*`) |

"Next step" means the next check an integration makes: before each model call and before each tool call. A tool
call already running is not cancelled.

### Decisions in traces and the audit log

Every check produces a `DecisionEvent` (id, time, the request with redacted fields masked, the verdict):

- recorded as a finished `chain` run named `policy:<tool or model>` under the node's run (ReactAgent) or the
  control plane's own tracer (`ControlPlaneBuilder::tracer`), with the event in `metadata.governance`; denials
  are failed runs with the reason as the error, so they appear as errors in Sinew Scope and OpenTelemetry
  backends. The tracer's `Redactor` applies as for any run;
- sent to every `DecisionSink` (`MemoryDecisionSink` for tests; your own for a SIEM or a database). Operator actions
  (register, suspend, quarantine, reinstate, disable and enable tool) go to `DecisionSink::record_control` as
  `ControlEvent`s, and so do policy changes (`add_policy`, `set_policies`, `apply_file`) as
  `ControlAction::UpdatePolicies`. The [tamper-evident audit log](#tamper-evident-audit-log) plugs in here.

### Tamper-evident audit log

Feature `audit` adds `sinew_govern::audit`: an append-only, hash-chained log of governance decisions, operator
actions, policy changes, approvals and tool calls with side effects, for audits (ISO/IEC 42001, security reviews,
incidents) that need a record nobody can silently edit.

```rust
use std::sync::Arc;
use sinew_govern::audit::{AuditLog, AuditSigner, SqliteAuditStore};

let signer = AuditSigner::from_hex(&std::env::var("SINEW_AUDIT_KEY")?)?; // from a secret manager
let store = Arc::new(SqliteAuditStore::connect("sqlite://audit.db").await?); // feature audit-sqlite
let audit = AuditLog::builder(store).signer(signer).checkpoint_every(1000).build();

let cp = ControlPlane::builder()
    .policy_file(file)
    .sink(Arc::new(audit.clone())) // every decision, operator action and policy change
    .build();

// What the control plane does not see: the human's answer and the call's outcome.
audit.record_approval(&ApprovalRecord { approver: "ana".into(), approved: true, /* ... */ }).await?;
audit.record_tool_call(&ToolCallRecord { tool: "refund".into(), /* ... */ }).await?;
audit.flush().await; // on shutdown: wait for queued events
```

How it works:

- Each `AuditEntry` has a `seq` (0, 1, 2, ... without gaps), `time_ms`, an `AuditKind` (`decision`, `control`,
  `policy_change`, `approval`, `tool_call`, `checkpoint`, `other`), a JSON `payload`, the `prev_hash` of the entry
  before it (64 zeros for the first) and its own `hash`: SHA-256 over the canonical JSON (keys sorted, no
  whitespace) of `{v, seq, time_ms, kind, prev_hash, payload}`.
- Payloads pass through a `sinew_trace::Redactor` before they are hashed and written (`Redactor::new()` by default:
  API keys and tokens, e-mails, card numbers, `password` / `api_key` / `authorization` fields). Set your own with
  `AuditLogBuilder::redactor`. Decision events already carry the policy's `Redact` fields masked.
- A `checkpoint` entry, written every `checkpoint_every(n)` entries or on `AuditLog::checkpoint()`, holds an Ed25519
  signature (`AuditSigner`, ed25519-dalek) over its own `seq`, the hash of the entry before it and its time, plus the
  signing key's id. Someone who can write to the store but has no key can rewrite the chain and recompute every hash,
  but cannot re-sign the checkpoints, so everything before the last checkpoint is protected. Keep the signing key
  outside the database (`sinew-govern keygen` prints a new pair).
- As a `DecisionSink`, the log queues events and a background task appends them (sinks are called on the request
  path). `flush()` waits for the queue; `failures()` and `last_error()` report events that could not be written.

Stores:

| Store | Feature | Notes |
|---|---|---|
| `MemoryAuditStore` | `audit` | tests, short-lived processes |
| `SqliteAuditStore` | `audit-sqlite` | table `sinew_audit_log`; triggers refuse `UPDATE` and `DELETE`; WAL, `synchronous = FULL` |
| `PostgresAuditStore` | `audit-postgres` | same table; a trigger refuses `UPDATE`, `DELETE` and `TRUNCATE`; several writers allowed (a second writer of the same `seq` gets a conflict and the log retries after the new head) |
| your own | `audit` | implement `AuditStore` (`append`, `last`, `entries`); no update or delete |

Grant the application's database role only `INSERT` and `SELECT` on `sinew_audit_log`.

#### Export and verify

`export_jsonl(store, &mut writer)` writes one entry per line, checkpoints included. `verify_store`, `verify_jsonl`
and `AuditLog::verify` recompute every hash and check every link, the sequence and every checkpoint signature, and
return an `AuditReport` (`entries`, `checkpoints`, `verified_checkpoints`, `head`, `unsigned_tail`, `problems`).
`ChainVerifier` does the same one entry at a time. The same checks from the command line (feature `cli`, plus
`audit-sqlite` / `audit-postgres` for the stores):

```sh
cargo install sinew-govern --features cli,audit-sqlite,audit-postgres
sinew-govern export-audit --sqlite sqlite://audit.db --out audit-2026-11.jsonl
sinew-govern verify-audit --jsonl audit-2026-11.jsonl --public-key <hex> --expect-head 41230:<hash>
sinew-govern verify-audit --postgres postgres://auditor@db/app --public-key-file audit-keys.txt --json
```

`verify-audit` exits with 0 when the log is intact, 1 when tampering is detected (each problem is listed) and 2 when
the input cannot be read.

| Tampering | Detected as |
|---|---|
| an entry's content edited | `HashMismatch` |
| an entry edited and its hash recomputed | `BrokenLink` at the next entry |
| an entry deleted (a checkpoint included) | `OutOfSequence` and `BrokenLink` |
| entries reordered (rows or export lines) | `OutOfSequence`, `HashMismatch`, `BrokenLink` |
| the chain rewritten from some entry on, hashes recomputed | `BadSignature` at the next checkpoint |
| the chain rewritten and checkpoints signed with another key | `UnknownKey` |
| entries removed from the end | `HeadMismatch`, against a head recorded elsewhere (`--expect-head`, `VerifyOptions::expect_head`) or the head the running `AuditLog` knows |

Limits: always pass the public key (`--public-key`); without it, checkpoint signatures are not checked and a forged
log with self-signed checkpoints passes. Entries after the last checkpoint (`unsigned_tail`) are protected against
partial edits but not against a full rewrite of that tail, and truncation can only be detected against a head
recorded outside the store: checkpoint often and copy `AuditLog::head()` (or the last line of each export) to another
system. After a key rotation pass every key used over the log's life.

## Enforcement points

| Runtime | How | Ask | Suspended agent |
|---|---|---|---|
| `ReactAgent` | `.gate(Arc::new(cp.gate(identity)))` | interrupt `{"approval_required": {id, name, arguments, reason}}`; resume with `true` or `false` (needs a checkpointer) | the model sees `skipped: <reason>` for remaining calls; the final message is `The run was stopped: <reason>` |
| `sinew-harness` `Agent` | `.hook(cp.hook(identity))` (feature `harness`) | the harness approval flow (`AgentOutcome::ApprovalRequired`, `Agent::resume`) | the turn ends with `StopReason::Guardrail(reason)` |
| plain `ToolBox` | `cp.governed_toolbox(&toolbox, identity)` | fails the call (no approval channel) | fails the call |
| any `ChatModel` | `cp.govern_model(model, identity)` | fails with `Error::Guardrail { guard: "control_plane", .. }` | same |

`ReactAgent::gate` takes any `AgentGate` (in `sinew_graph::prebuilt`), so other policy engines can use the same
hook. Denied tool calls are answered with `error: denied: <reason>` and the run continues; a denied model call ends
the run. Because a node runs again from the start after an interrupt, calls before the interrupt are checked again
(their results come from the tool ledger).

In the harness, the hook's decision is combined with the agent's `PermissionPolicy` and other hooks (the most
restrictive wins). `auto_approve(true)` (the CLI's `--yes`) turns every `Ask` into `Allow`, the control plane's
included; use `Deny` for limits that must hold without a human. The hook learns whether a tool has side effects
from the control plane's tool catalog (`ControlPlaneBuilder::tool(name, true)` or `.toolbox(&toolbox)`); unknown
tools count as having none. Set `.model("provider:model")` on the hook to check model calls too. The kill switch
uses the new `Hook::should_stop`, called before every model call and before the tools of every step.

### Choosing between permissions, guardrails and governance

- **Harness `PermissionPolicy`**: per-agent rules on tool names and path, command or URL arguments, configured
  where the agent is built. Fine for one agent and one developer.
- **Guardrails** (`sinew_core::guardrails`): check text (user input, tool results, answers) for PII, secrets,
  injection, topics.
- **Governance** (`sinew-govern`): who may call what, with which arguments, for which user and tenant, across many
  agents, changed at runtime by an operator, with a record of every decision.

They compose: a harness agent can keep its permission policy and guardrails and add the control plane hook.

## Mapping to agentic security frameworks

The control plane covers the **Control** part of the Behavior / Identity / Control model of the CSA agentic control
plane, and the evidence part through decision events:

| Area | In this crate | Elsewhere / follow-up |
|---|---|---|
| Identity | agent ids in a registry with owner and tenant; user and tenant on every request; cross-tenant requests denied | signed agent cards and delegated user credentials (#175) |
| Control | policies before every tool and model call, approvals, budgets, rate limits, time windows, kill switch, quarantine, tool disable | durable approvals (harness, `ReactAgent` interrupts) |
| Behavior | every decision as a trace run and an event | loop guard, guardrails, trajectory evaluation; baselines and anomaly detection (#176) |
| Evidence | `DecisionEvent`, `ControlEvent` to `DecisionSink`s; hash-chained, signed `AuditLog` with `verify-audit` | compliance export (#181) |

At a high level, against the OWASP Top 10 for Agentic Applications:

| Risk | Controls here |
|---|---|
| Tool misuse and exploitation | allowed tools per agent, argument conditions, default deny for side effects, rate limits, approvals |
| Identity and privilege abuse | registry, tenant isolation, per-user rules; least privilege through allow-lists |
| Agent goal hijack (prompt injection leading to actions) | the injected instruction still has to pass the policies; pair with `InjectionDetector` on tool output |
| Rogue agents | suspend and quarantine at runtime, disabled tools, unregistered agents denied |
| Cascading failures | budgets and rate limits per agent and tool, the kill switch for a whole tool family |
| Human-agent trust exploitation | `Ask` decisions carry the arguments and the reason to the approver; `GovernHook::risk` sets risk levels shown in harness approval requests (see [harness approvals](harness.md#approvals-that-resist-trust-exploitation)); redaction keeps secrets out of approval screens and traces |

These controls reduce the impact of a compromised or confused agent; they do not detect the compromise itself.
Combine them with guardrails, sandboxing and monitoring ([Production](production.md)).

## Limits and follow-ups

- Usage counters, rate limits and the registry live in memory, per process. A shared store for several replicas
  is a follow-up.
- On-behalf-of user tokens for tools that call external APIs are part of agent identity (#175).
- The harness reports every approval answer to `Hook::on_approval` (an `ApprovalRecord`). `GovernHook::audit(log)`
  forwards those answers to a tamper-evident `AuditLog` as `approval` entries; the control plane's `DecisionSink`s
  still see only the `Ask`. Outside the harness, record answers with `AuditLog::record_approval`.
- `ReactAgent` gates take one identity per compiled graph; build one gate per user (it is a cheap clone of the
  control plane) when the user changes per request.

See the [OWASP agentic controls map](security-owasp.md) for an attack-by-attack view: each of the OWASP Top 10 for Agentic Applications risks with the Sinew controls that stop it, tested by `samples/23-owasp-agentic`.
