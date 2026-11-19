//! [`PeerApprovalPolicy`]: actions taken after peer agents spoke need a
//! verified sender with the authority for them, or a human (OWASP ASI07).

use async_trait::async_trait;
use sinew_core::identity::{PeerLedger, PeerRecord, PeerStatus};

use crate::{
    decision::Decision,
    pattern::glob_match,
    policy::{Effect, Policy},
    request::{Request, RequestKind},
    GovernError,
};

/// Gates sensitive actions (payments, permission changes) that a message
/// from another agent may have "approved".
///
/// It reads a [`PeerLedger`] shared with the wrappers that receive peer
/// messages (`PeerAgentTool`, `Supervisor::message_auth`,
/// `Swarm::message_auth`, `A2aAgentTool::with_peer_ledger`). For a tool
/// call matching one of its actions, in a thread where peer messages were
/// received:
///
/// - a peer message failed authentication (forged, replayed, revoked):
///   `Deny`;
/// - a verified peer whose identity has the action's authority (a
///   capability such as `approve:refunds`) spoke: the requirement is met and
///   the policy abstains, so the other policies decide;
/// - otherwise (unsigned peers, or verified peers without that authority):
///   `Ask`, so a human approves (configurable with
///   [`on_unauthorized`](Self::on_unauthorized)).
///
/// Without peer messages in the thread it abstains: the action is the
/// agent's own and the other policies decide as usual.
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use serde_json::json;
/// use sinew_core::identity::{PeerLedger, PeerRecord};
/// use sinew_govern::{Decision, PeerApprovalPolicy, Policy, Request};
///
/// let ledger = PeerLedger::new();
/// let policy = PeerApprovalPolicy::new(ledger.clone()).action("issue_refund", "approve:refunds");
/// let refund = Request::tool("orchestrator", "issue_refund", json!({"amount": 1250}));
///
/// assert_eq!(policy.evaluate(&refund).await.unwrap(), None); // no peer spoke
///
/// // A verified billing agent (no refund authority) "approves" the refund.
/// ledger.record(PeerRecord::verified("billing-agent", "approved", vec!["billing:read".into()], "k1"));
/// assert!(matches!(policy.evaluate(&refund).await.unwrap(), Some(Decision::Ask { .. })));
///
/// // The real payments supervisor approves it.
/// ledger.record(PeerRecord::verified("payments-supervisor", "ok", vec!["approve:refunds".into()], "k2"));
/// assert_eq!(policy.evaluate(&refund).await.unwrap(), None);
/// # });
/// ```
#[derive(Debug, Clone)]
pub struct PeerApprovalPolicy {
    id: String,
    ledger: PeerLedger,
    actions: Vec<(String, String)>,
    unauthorized: Effect,
}

impl PeerApprovalPolicy {
    /// Id `peer-approval`, no actions yet, `Ask` when unauthorized.
    pub fn new(ledger: PeerLedger) -> Self {
        Self {
            id: "peer-approval".into(),
            ledger,
            actions: Vec::new(),
            unauthorized: Effect::Ask,
        }
    }

    /// Replaces the policy id recorded with its decisions.
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Tools matching `tool_pattern` (glob) need a verified peer with
    /// `authority` (a capability in its identity), or a human.
    pub fn action(mut self, tool_pattern: impl Into<String>, authority: impl Into<String>) -> Self {
        self.actions.push((tool_pattern.into(), authority.into()));
        self
    }

    /// What happens when no verified peer has the authority: `Ask`
    /// (default), `Deny` (no human in the loop), or `Allow` (only log).
    pub fn on_unauthorized(mut self, effect: Effect) -> Self {
        self.unauthorized = effect;
        self
    }

    pub fn ledger(&self) -> &PeerLedger {
        &self.ledger
    }
}

fn describe(records: &[PeerRecord]) -> String {
    let mut seen: Vec<String> = Vec::new();
    for r in records {
        let status = match r.status {
            PeerStatus::Verified => "verified, without that authority",
            PeerStatus::Unsigned => "unverified",
            PeerStatus::Rejected => "rejected",
        };
        let d = format!("`{}` ({status})", r.sender);
        if !seen.contains(&d) {
            seen.push(d);
        }
    }
    seen.join(", ")
}

#[async_trait]
impl Policy for PeerApprovalPolicy {
    fn id(&self) -> &str {
        &self.id
    }

    async fn evaluate(&self, req: &Request) -> Result<Option<Decision>, GovernError> {
        if req.kind != RequestKind::Tool {
            return Ok(None);
        }
        let tool = req.target();
        let Some((_, authority)) = self.actions.iter().find(|(p, _)| glob_match(p, tool)) else {
            return Ok(None);
        };
        let records = self.ledger.for_thread(req.thread_id.as_deref());
        if records.is_empty() {
            return Ok(None);
        }
        if let Some(bad) = records.iter().find(|r| r.is_rejected()) {
            return Ok(Some(Decision::deny(format!(
                "`{tool}` follows a message from `{}` that failed authentication ({})",
                bad.sender,
                bad.reason.as_deref().unwrap_or("rejected")
            ))));
        }
        if records.iter().any(|r| r.has_authority(authority)) {
            return Ok(None);
        }
        let reason = format!(
            "`{tool}` follows messages from peer agents and none is a verified sender with \
             authority `{authority}`: {}",
            describe(&records)
        );
        Ok(match self.unauthorized {
            Effect::Ask => Some(Decision::ask(reason)),
            Effect::Deny => Some(Decision::deny(reason)),
            Effect::Allow => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn decides_by_peer_status_and_thread() {
        let ledger = PeerLedger::new();
        let p = PeerApprovalPolicy::new(ledger.clone())
            .id("pa")
            .action("refund*", "approve:refunds");
        assert_eq!(Policy::id(&p), "pa");
        assert!(p.ledger().is_empty());
        let req = Request::tool("o", "refund_order", json!({})).thread("t1");
        let other = Request::tool("o", "lookup", json!({})).thread("t1");
        let model = Request::model("o", "m");

        ledger.record(PeerRecord::unsigned("billing", "x").thread(Some("t2".into())));
        assert_eq!(p.evaluate(&req).await.unwrap(), None, "other thread");
        ledger.record(PeerRecord::unsigned("billing", "x").thread(Some("t1".into())));
        ledger.record(PeerRecord::unsigned("billing", "y").thread(Some("t1".into())));
        let d = p.evaluate(&req).await.unwrap().unwrap();
        assert_eq!(
            d.reason()
                .unwrap()
                .matches("`billing` (unverified)")
                .count(),
            1
        );
        assert_eq!(p.evaluate(&other).await.unwrap(), None);
        assert_eq!(p.evaluate(&model).await.unwrap(), None);

        let deny = p.clone().on_unauthorized(Effect::Deny);
        assert!(deny.evaluate(&req).await.unwrap().unwrap().is_denied());
        let log = p.clone().on_unauthorized(Effect::Allow);
        assert_eq!(log.evaluate(&req).await.unwrap(), None);

        ledger.record(PeerRecord::rejected("boss", "z", "replayed: nonce 1"));
        let d = p.evaluate(&req).await.unwrap().unwrap();
        assert!(d.is_denied());
        assert!(d.reason().unwrap().contains("replayed"));
        assert!(describe(&ledger.records()).contains("`boss` (rejected)"));
        let mut r = PeerRecord::rejected("x", "z", "r");
        r.reason = None;
        ledger.clear();
        ledger.record(r);
        assert!(p
            .evaluate(&req)
            .await
            .unwrap()
            .unwrap()
            .reason()
            .unwrap()
            .contains("(rejected)"));
    }
}
