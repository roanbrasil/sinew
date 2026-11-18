//! [`Decision`]s, how several are combined into a [`Verdict`], and
//! argument redaction.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The value written over redacted argument fields.
pub const REDACTED: &str = "[REDACTED]";

/// What one policy says about a request.
///
/// When several policies decide, the most restrictive wins:
/// `Deny` > `Ask` > `Redact` > `Allow` (see [`Decision::severity`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Decision {
    /// The call may run.
    Allow,
    /// The call may run after a human approves it.
    Ask { reason: String },
    /// The call must not run.
    Deny { reason: String },
    /// The call may run with these argument fields (JSON pointers such as
    /// `/card/number`) replaced by [`REDACTED`].
    Redact { fields: Vec<String> },
}

impl Decision {
    pub fn allow() -> Self {
        Decision::Allow
    }

    pub fn ask(reason: impl Into<String>) -> Self {
        Decision::Ask {
            reason: reason.into(),
        }
    }

    pub fn deny(reason: impl Into<String>) -> Self {
        Decision::Deny {
            reason: reason.into(),
        }
    }

    pub fn redact<I, S>(fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Decision::Redact {
            fields: fields.into_iter().map(Into::into).collect(),
        }
    }

    /// Rank used to combine decisions: `Allow` 0, `Redact` 1, `Ask` 2,
    /// `Deny` 3.
    pub fn severity(&self) -> u8 {
        match self {
            Decision::Allow => 0,
            Decision::Redact { .. } => 1,
            Decision::Ask { .. } => 2,
            Decision::Deny { .. } => 3,
        }
    }

    /// `Allow` or `Redact`: the call may run without a human.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Decision::Allow | Decision::Redact { .. })
    }

    pub fn is_denied(&self) -> bool {
        matches!(self, Decision::Deny { .. })
    }

    /// The reason of an `Ask` or `Deny`.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Decision::Ask { reason } | Decision::Deny { reason } => Some(reason),
            _ => None,
        }
    }

    /// `allow`, `ask`, `deny` or `redact`.
    pub fn label(&self) -> &'static str {
        match self {
            Decision::Allow => "allow",
            Decision::Ask { .. } => "ask",
            Decision::Deny { .. } => "deny",
            Decision::Redact { .. } => "redact",
        }
    }
}

/// One policy that returned a decision for a request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyMatch {
    pub policy_id: String,
    pub decision: Decision,
}

/// The combined outcome of every policy for one request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    /// The final decision. A `Redact` here carries the union of every
    /// matching policy's fields.
    pub decision: Decision,
    /// Id of the policy (or control-plane check, `control-plane/...`) that
    /// produced [`decision`](Self::decision); `None` when no policy matched
    /// and the default applied.
    pub policy_id: Option<String>,
    /// Argument fields to redact before the call runs, from every matching
    /// `Redact` decision. Also set when the final decision is `Ask`, for the
    /// call that runs after approval.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redact: Vec<String>,
    /// Every policy that returned a decision, in evaluation order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub matched: Vec<PolicyMatch>,
    /// The agent is suspended: a running turn should end now rather than
    /// only skip this call.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stop: bool,
}

impl Verdict {
    /// A verdict from a single decision.
    pub fn new(decision: Decision, policy_id: Option<String>) -> Self {
        let redact = match &decision {
            Decision::Redact { fields } => fields.clone(),
            _ => Vec::new(),
        };
        Self {
            decision,
            policy_id,
            redact,
            matched: Vec::new(),
            stop: false,
        }
    }

    /// Combines `matched` decisions: the most restrictive wins (the first
    /// one of that rank decides `policy_id`); `Redact` fields are merged.
    /// Returns `None` when nothing matched.
    pub fn combine(matched: Vec<PolicyMatch>) -> Option<Self> {
        let winner = matched
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| {
                a.decision
                    .severity()
                    .cmp(&b.decision.severity())
                    .then(ib.cmp(ia))
            })
            .map(|(i, _)| i)?;
        let mut redact: Vec<String> = Vec::new();
        for m in &matched {
            if let Decision::Redact { fields } = &m.decision {
                for f in fields {
                    if !redact.contains(f) {
                        redact.push(f.clone());
                    }
                }
            }
        }
        let decision = match &matched[winner].decision {
            Decision::Redact { .. } => Decision::Redact {
                fields: redact.clone(),
            },
            d => d.clone(),
        };
        Some(Self {
            decision,
            policy_id: Some(matched[winner].policy_id.clone()),
            redact,
            matched,
            stop: false,
        })
    }

    /// The arguments with [`redact`](Self::redact) applied.
    pub fn redacted(&self, arguments: &Value) -> Value {
        let mut out = arguments.clone();
        redact_fields(&mut out, &self.redact);
        out
    }
}

/// Replaces the value at each JSON pointer in `fields` with [`REDACTED`].
/// Pointers that do not resolve are ignored. A field may also be written as
/// a dotted path (`card.number`).
pub fn redact_fields(value: &mut Value, fields: &[String]) {
    for f in fields {
        let pointer = to_pointer(f);
        if let Some(v) = value.pointer_mut(&pointer) {
            *v = Value::String(REDACTED.into());
        }
    }
}

/// `card.number` → `/card/number`; pointers are returned unchanged.
pub(crate) fn to_pointer(path: &str) -> String {
    let path = path.trim();
    if path.is_empty() || path.starts_with('/') {
        return path.to_string();
    }
    let mut out = String::new();
    for seg in path.split('.') {
        out.push('/');
        out.push_str(&seg.replace('~', "~0").replace('/', "~1"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn m(id: &str, d: Decision) -> PolicyMatch {
        PolicyMatch {
            policy_id: id.into(),
            decision: d,
        }
    }

    #[test]
    fn deny_overrides_ask_overrides_redact_overrides_allow() {
        let v = Verdict::combine(vec![
            m("a", Decision::Allow),
            m("r", Decision::redact(["/x"])),
            m("q", Decision::ask("check")),
            m("d", Decision::deny("no")),
            m("d2", Decision::deny("also no")),
        ])
        .unwrap();
        assert_eq!(v.decision, Decision::deny("no"));
        assert_eq!(v.policy_id.as_deref(), Some("d"));
        assert_eq!(v.redact, vec!["/x"]);

        let v = Verdict::combine(vec![m("a", Decision::Allow), m("q", Decision::ask("c"))]);
        assert_eq!(v.unwrap().policy_id.as_deref(), Some("q"));

        let v = Verdict::combine(vec![
            m("r1", Decision::redact(["/a"])),
            m("a", Decision::Allow),
            m("r2", Decision::redact(["/b", "/a"])),
        ])
        .unwrap();
        assert_eq!(v.decision, Decision::redact(["/a", "/b"]));
        assert!(Verdict::combine(vec![]).is_none());
    }

    #[test]
    fn redacts_pointers_and_dotted_paths() {
        let mut v = json!({"card": {"number": "4111", "exp": "12/30"}, "amount": 3});
        redact_fields(&mut v, &["card.number".into(), "/missing".into()]);
        assert_eq!(
            v,
            json!({"card": {"number": REDACTED, "exp": "12/30"}, "amount": 3})
        );
        assert_eq!(to_pointer("a/b.c"), "/a~1b/c");
    }
}
