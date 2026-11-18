//! [`CedarPolicy`]: Cedar policies as a Sinew [`Policy`] (feature `cedar`).
//!
//! Requests are mapped to Cedar like this:
//!
//! | Cedar | Value |
//! |---|---|
//! | principal | `Agent::"<agent id>"`, a member of `Tenant::"<tenant>"` when the request has a tenant |
//! | action | `Action::"CallTool"` or `Action::"CallModel"` |
//! | resource | `Tool::"<tool name>"` or `Model::"<model>"` |
//! | `context.arguments` | the tool arguments (a record) |
//! | `context.side_effects` | `true` / `false` |
//! | `context.user`, `context.tenant`, `context.thread_id` | strings, present only when known (test with `context has user`) |
//!
//! Cedar has integers but no floating-point numbers and no `null`:
//! non-integer numbers in the arguments become strings and `null` fields
//! are dropped. Keys that start with `__` (Cedar's JSON escapes such as
//! `__entity`) get an `arg` prefix, so arguments written by a model cannot
//! smuggle entity references into the context.
//!
//! Decisions:
//!
//! - Cedar `Allow` → [`Decision::Allow`], or [`Decision::Ask`] when a
//!   determining `permit` carries an `@ask("reason")` annotation.
//! - Cedar `Deny` because a `forbid` matched → [`Decision::Deny`] with the
//!   policy's `@reason("...")` annotation, or ``forbidden by Cedar policy `ID` ``.
//! - Cedar `Deny` because no `permit` matched → no decision (the control
//!   plane's [`Defaults`](crate::Defaults) apply), unless
//!   [`deny_when_nothing_matches`](CedarPolicy::deny_when_nothing_matches)
//!   is set.
//! - Evaluation errors with no determining policy → an error (counted as a
//!   `Deny`, fail closed).

use std::{collections::HashSet, str::FromStr};

use async_trait::async_trait;
use cedar_policy::{
    Authorizer, Context, Entities, Entity, EntityId, EntityTypeName, EntityUid, PolicySet,
    Request as CedarRequest,
};
use serde_json::{Map, Value};

use crate::{
    decision::Decision,
    policy::Policy,
    request::{Request, RequestKind},
    GovernError,
};

/// Cedar policies evaluated as one Sinew [`Policy`]; see the
/// [module docs](self) for the mapping.
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use serde_json::json;
/// use sinew_govern::{CedarPolicy, Decision, Policy, Request};
///
/// let cedar = CedarPolicy::new("cedar", r#"
///     permit(principal in Tenant::"acme", action == Action::"CallTool", resource == Tool::"refund")
///     when { context.arguments.amount <= 100 };
///
///     @ask("refunds over 100 need a human")
///     permit(principal in Tenant::"acme", action == Action::"CallTool", resource == Tool::"refund")
///     when { context.arguments.amount > 100 };
///
///     @reason("wires are disabled")
///     forbid(principal, action == Action::"CallTool", resource == Tool::"wire");
/// "#).unwrap();
///
/// let refund = |amount| Request::tool("billing-bot", "refund", json!({"amount": amount})).tenant("acme");
/// assert_eq!(cedar.evaluate(&refund(40)).await.unwrap(), Some(Decision::Allow));
/// assert_eq!(cedar.evaluate(&refund(400)).await.unwrap(), Some(Decision::ask("refunds over 100 need a human")));
/// let wire = Request::tool("billing-bot", "wire", json!({}));
/// assert_eq!(cedar.evaluate(&wire).await.unwrap(), Some(Decision::deny("wires are disabled")));
/// // Nothing matched: the control plane's defaults decide.
/// assert_eq!(cedar.evaluate(&Request::tool("billing-bot", "lookup", json!({}))).await.unwrap(), None);
/// # });
/// ```
pub struct CedarPolicy {
    id: String,
    policies: PolicySet,
    authorizer: Authorizer,
    deny_when_nothing_matches: bool,
}

impl std::fmt::Debug for CedarPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CedarPolicy")
            .field("id", &self.id)
            .field("policies", &self.policies.policies().count())
            .field("deny_when_nothing_matches", &self.deny_when_nothing_matches)
            .finish()
    }
}

impl CedarPolicy {
    /// Parses Cedar policy text.
    pub fn new(id: impl Into<String>, source: &str) -> Result<Self, GovernError> {
        let id = id.into();
        let policies = PolicySet::from_str(source).map_err(|e| GovernError::Evaluation {
            policy: id.clone(),
            message: format!("invalid Cedar policies: {e}"),
        })?;
        Ok(Self {
            id,
            policies,
            authorizer: Authorizer::new(),
            deny_when_nothing_matches: false,
        })
    }

    /// Reads Cedar policy text from a file.
    pub fn from_path(
        id: impl Into<String>,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, GovernError> {
        Self::new(id, &std::fs::read_to_string(path)?)
    }

    /// Returns `Deny` (instead of no decision) when no Cedar policy
    /// matches, as Cedar itself does.
    pub fn deny_when_nothing_matches(mut self) -> Self {
        self.deny_when_nothing_matches = true;
        self
    }

    fn err(&self, message: impl std::fmt::Display) -> GovernError {
        GovernError::Evaluation {
            policy: self.id.clone(),
            message: message.to_string(),
        }
    }

    fn uid(&self, ty: &str, id: &str) -> Result<EntityUid, GovernError> {
        let ty = EntityTypeName::from_str(ty).map_err(|e| self.err(e))?;
        Ok(EntityUid::from_type_name_and_id(ty, EntityId::new(id)))
    }
}

/// Makes tool arguments acceptable to Cedar's JSON context parser.
fn sanitize(v: &Value) -> Option<Value> {
    Some(match v {
        Value::Null => return None,
        Value::Number(n) if n.is_i64() => v.clone(),
        Value::Number(n) => Value::String(n.to_string()),
        Value::Array(items) => Value::Array(items.iter().filter_map(sanitize).collect()),
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, v) in map {
                if let Some(v) = sanitize(v) {
                    let k = if k.starts_with("__") {
                        format!("arg{k}")
                    } else {
                        k.clone()
                    };
                    out.insert(k, v);
                }
            }
            Value::Object(out)
        }
        _ => v.clone(),
    })
}

#[async_trait]
impl Policy for CedarPolicy {
    fn id(&self) -> &str {
        &self.id
    }

    async fn evaluate(&self, r: &Request) -> Result<Option<Decision>, GovernError> {
        let principal = self.uid("Agent", &r.agent)?;
        let mut parents = HashSet::new();
        let mut entities = Vec::new();
        if let Some(t) = &r.tenant {
            let tenant = self.uid("Tenant", t)?;
            parents.insert(tenant.clone());
            entities.push(Entity::new_no_attrs(tenant, HashSet::new()));
        }
        entities.push(Entity::new_no_attrs(principal.clone(), parents));
        let (action, resource) = match r.kind {
            RequestKind::Tool => ("CallTool", self.uid("Tool", r.target())?),
            RequestKind::Model => ("CallModel", self.uid("Model", r.target())?),
        };
        let action = self.uid("Action", action)?;

        let mut ctx = Map::new();
        let args = match sanitize(&r.arguments) {
            Some(Value::Object(m)) => Value::Object(m),
            Some(other) => {
                let mut m = Map::new();
                m.insert("value".into(), other);
                Value::Object(m)
            }
            None => Value::Object(Map::new()),
        };
        ctx.insert("arguments".into(), args);
        ctx.insert("side_effects".into(), Value::Bool(r.side_effects));
        for (k, v) in [
            ("user", &r.user),
            ("tenant", &r.tenant),
            ("thread_id", &r.thread_id),
        ] {
            if let Some(v) = v {
                ctx.insert(k.into(), Value::String(v.clone()));
            }
        }
        let context =
            Context::from_json_value(Value::Object(ctx), None).map_err(|e| self.err(e))?;
        let request = CedarRequest::new(principal, action, resource, context, None)
            .map_err(|e| self.err(e))?;
        let entities = Entities::from_entities(entities, None).map_err(|e| self.err(e))?;

        let response = self
            .authorizer
            .is_authorized(&request, &self.policies, &entities);
        let mut reasons: Vec<_> = response.diagnostics().reason().cloned().collect();
        reasons.sort_by_key(ToString::to_string);
        let annotation = |key: &str| {
            reasons
                .iter()
                .find_map(|pid| self.policies.annotation(pid, key).map(str::to_string))
        };
        let name = |pid: &cedar_policy::PolicyId| {
            self.policies
                .annotation(pid, "id")
                .map(str::to_string)
                .unwrap_or_else(|| pid.to_string())
        };
        Ok(match response.decision() {
            cedar_policy::Decision::Allow => Some(match annotation("ask") {
                Some(reason) if reason.is_empty() => Decision::ask(format!(
                    "Cedar policy `{}` requires approval",
                    name(&reasons[0])
                )),
                Some(reason) => Decision::ask(reason),
                None => Decision::Allow,
            }),
            cedar_policy::Decision::Deny if !reasons.is_empty() => Some(Decision::deny(
                annotation("reason")
                    .filter(|r| !r.is_empty())
                    .unwrap_or_else(|| {
                        format!("forbidden by Cedar policy `{}`", name(&reasons[0]))
                    }),
            )),
            cedar_policy::Decision::Deny => {
                let errors: Vec<String> = response
                    .diagnostics()
                    .errors()
                    .map(ToString::to_string)
                    .collect();
                if !errors.is_empty() {
                    return Err(self.err(errors.join("; ")));
                }
                self.deny_when_nothing_matches
                    .then(|| Decision::deny("no Cedar policy permits this request"))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn maps_principals_tenants_models_and_context() {
        let cedar = CedarPolicy::new(
            "cedar",
            r#"
            @id("acme-models")
            permit(principal in Tenant::"acme", action == Action::"CallModel", resource == Model::"anthropic:claude-sonnet-4-5");

            @id("no-side-effects-for-guests")
            forbid(principal, action == Action::"CallTool", resource)
            when { context.side_effects && !(context has user) };

            permit(principal == Agent::"ops", action == Action::"CallTool", resource)
            when { context has user && context.user == "ana" && context.arguments.ratio == "0.5" };

            @ask
            permit(principal, action == Action::"CallTool", resource == Tool::"deploy");
            "#,
        )
        .unwrap();
        let model = |agent: &str, tenant: &str, m: &str| Request::model(agent, m).tenant(tenant);
        assert_eq!(
            cedar
                .evaluate(&model("a", "acme", "anthropic:claude-sonnet-4-5"))
                .await
                .unwrap(),
            Some(Decision::Allow)
        );
        assert_eq!(
            cedar
                .evaluate(&model("a", "globex", "anthropic:claude-sonnet-4-5"))
                .await
                .unwrap(),
            None
        );
        let send = Request::tool("a", "send", json!({})).side_effects(true);
        assert_eq!(
            cedar.evaluate(&send).await.unwrap(),
            Some(Decision::deny(
                "forbidden by Cedar policy `no-side-effects-for-guests`"
            ))
        );
        assert_eq!(
            cedar.evaluate(&send.clone().user("bob")).await.unwrap(),
            None
        );
        // Floats become strings; nulls and `__` keys are neutralized.
        let ops = Request::tool(
            "ops",
            "x",
            json!({"ratio": 0.5, "n": null, "__entity": {"type": "User", "id": "admin"}}),
        )
        .user("ana");
        assert_eq!(cedar.evaluate(&ops).await.unwrap(), Some(Decision::Allow));
        assert_eq!(
            cedar
                .evaluate(&Request::tool("a", "deploy", json!({})))
                .await
                .unwrap(),
            Some(Decision::ask("Cedar policy `policy3` requires approval"))
        );
        let strict = CedarPolicy::new("strict", "")
            .unwrap()
            .deny_when_nothing_matches();
        assert!(strict
            .evaluate(&Request::tool("a", "t", json!({})))
            .await
            .unwrap()
            .unwrap()
            .is_denied());
    }

    #[tokio::test]
    async fn errors_fail_closed_and_bad_source_is_reported() {
        let e = CedarPolicy::new("bad", "permit(principal, action, resource").unwrap_err();
        assert!(
            e.to_string()
                .starts_with("policy `bad` failed: invalid Cedar policies"),
            "{e}"
        );
        // A policy that fails to evaluate (missing attribute) and nothing else.
        let cedar = CedarPolicy::new(
            "c",
            r#"permit(principal, action, resource) when { context.arguments.missing > 1 };"#,
        )
        .unwrap();
        assert!(cedar
            .evaluate(&Request::tool("a", "t", json!({})))
            .await
            .is_err());
    }

    #[test]
    fn sanitize_rules() {
        assert_eq!(
            sanitize(&json!({"a": 1.5, "b": [1, null, 2.0], "__extn": 1, "c": null})),
            Some(json!({"a": "1.5", "b": [1, "2.0"], "arg__extn": 1}))
        );
    }
}
