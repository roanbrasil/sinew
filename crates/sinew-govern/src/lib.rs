//! A control plane for Sinew agents: which agent may call which tool or
//! model, with which arguments, for which user and tenant, decided by
//! policies before every call, with a kill switch and an audit event for
//! every decision.
//!
//! - [`Policy`]: looks at a [`Request`] (agent, user, tenant, tool and
//!   arguments or model, side effects, trace and thread ids) and returns a
//!   [`Decision`]: `Allow`, `Ask` (human approval), `Deny`, or `Redact`
//!   (allow with argument fields masked). Combined as `Deny` > `Ask` >
//!   `Redact` > `Allow` in a [`PolicySet`], with configurable [`Defaults`]
//!   (tools with side effects are denied unless a policy allows them).
//! - Built-in policies: [`Rule`] (glob patterns on agent, user, tenant,
//!   tool and model; argument [`Condition`]s such as `amount <= 100`;
//!   side effects; [`TimeWindow`]s) and [`RateLimit`]. Write them in Rust
//!   or in a YAML / JSON [`PolicyFile`]. External engines: `CedarPolicy`
//!   (feature `cedar`) and `OpaPolicy` (feature `opa`).
//! - [`ControlPlane`]: a registry of agents ([`AgentProfile`]: owner,
//!   tenant, allowed tools and models, [`AgentBudget`], [`AgentStatus`]),
//!   the policies, a kill switch ([`suspend`](ControlPlane::suspend),
//!   [`quarantine`](ControlPlane::quarantine),
//!   [`disable_tool`](ControlPlane::disable_tool)) and a
//!   [`DecisionEvent`] for every decision, recorded as a trace run and sent
//!   to [`DecisionSink`]s.
//! - Enforcement points: [`ControlPlane::gate`] for `ReactAgent`
//!   (checked before every model call and tool call, so a suspended agent
//!   stops at its next step), `ControlPlane::hook` for `sinew-harness`
//!   (feature `harness`; `Ask` uses the harness approval flow),
//!   [`ControlPlane::governed_toolbox`] for a plain `ToolBox` and
//!   [`ControlPlane::govern_model`] for any `ChatModel`.
//!
//! ```
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! use std::sync::Arc;
//! use serde_json::json;
//! use sinew_core::{mock::ScriptedModel, ChatModel, FnTool, Message, ToolBox, ToolCall};
//! use sinew_govern::{AgentProfile, Condition, ControlPlane, Rule};
//! use sinew_graph::prebuilt::{AgentState, ReactAgent};
//!
//! let cp = ControlPlane::builder()
//!     .agent(AgentProfile::new("support").allow_tools(["refund"]))
//!     .policy(Rule::deny("refund-cap").tools(["refund"])
//!         .when(Condition::parse("amount > 100").unwrap())
//!         .reason("refunds over 100 need a supervisor"))
//!     .build();
//!
//! let model: Arc<dyn ChatModel> = Arc::new(ScriptedModel::new([
//!     Message::assistant_with_tools("", vec![ToolCall {
//!         id: "c1".into(), name: "refund".into(), arguments: json!({"amount": 500}),
//!     }]),
//!     Message::assistant("I could not refund that."),
//! ]));
//! let tools = ToolBox::new().with(FnTool::new("refund", "Refund an order", json!({}), |_| Ok(json!("ok"))));
//! let agent = ReactAgent::new(model, tools).gate(Arc::new(cp.gate("support"))).compile().unwrap();
//! let out = agent.invoke(AgentState::new(vec![Message::user("refund 500")])).await.unwrap();
//! assert_eq!(out.messages[2].content, "error: denied: refunds over 100 need a supervisor");
//! # });
//! ```
//!
//! With feature `audit`, [`audit::AuditLog`] is a tamper-evident,
//! hash-chained log of every decision, operator action and policy change,
//! with Ed25519-signed checkpoints, SQLite / Postgres stores, JSONL export
//! and the `sinew-govern verify-audit` command (feature `cli`).
//!
//! See `docs/governance.md` for the concepts, policy examples and how the
//! controls map to the OWASP agentic risks and the CSA control plane.
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

#[cfg(feature = "audit")]
pub mod audit;
#[cfg(feature = "cedar")]
pub mod cedar;
mod condition;
mod control;
mod decision;
mod error;
pub mod file;
#[cfg(feature = "harness")]
mod harness;
mod integrations;
#[cfg(feature = "opa")]
pub mod opa;
mod pattern;
mod policy;
mod registry;
mod request;
mod sink;
mod time;

#[cfg(feature = "audit")]
pub use audit::{AuditLog, AuditSigner};
#[cfg(feature = "cedar")]
pub use cedar::CedarPolicy;
pub use condition::{Condition, Op};
pub use control::{
    ControlPlane, ControlPlaneBuilder, POLICY_ALLOWED, POLICY_BUDGET, POLICY_KILL_SWITCH,
    POLICY_QUARANTINE, POLICY_REGISTRY, POLICY_TENANT,
};
pub use decision::{redact_fields, Decision, PolicyMatch, Verdict, REDACTED};
pub use error::{GovernError, Result};
pub use file::PolicyFile;
#[cfg(feature = "harness")]
pub use harness::GovernHook;
pub use integrations::{ControlPlaneGate, GovernedModel, GovernedTool};
#[cfg(feature = "opa")]
pub use opa::OpaPolicy;
pub use pattern::glob_match;
pub use policy::{Defaults, Effect, Policy, PolicySet, RateKey, RateLimit, Rule};
pub use registry::{AgentBudget, AgentProfile, AgentStatus, AgentUsage};
pub use request::{Identity, Request, RequestKind};
pub use sink::{
    record_decision, ControlAction, ControlEvent, DecisionEvent, DecisionSink, MemoryDecisionSink,
};
pub use time::{TimeWindow, Weekday};
