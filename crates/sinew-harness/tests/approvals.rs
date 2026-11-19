//! Approval requests that resist trust exploitation (OWASP ASI09): risk
//! levels, intent/call warnings, rendered arguments, stronger confirmation
//! and the `on_approval` hook.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, FnTool, Message, ToolCall};
use sinew_harness::{
    Agent, AgentEvent, AgentOutcome, ApprovalConfig, ApprovalDecision, ApprovalOutcome,
    ApprovalRecord, ApprovalRequest, Confirmation, HarnessError, Hook, HookContext, IntentCheck,
    IntentContext, PermissionPolicy, RiskLevel,
};

const ATTACKER: &str = "ACC-ATTACKER-0001";

fn transfer_call(id: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: "transfer_funds".into(),
        arguments: json!({"to": ATTACKER, "amount": "4900.00"}),
    }
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "to": {"type": "string", "title": "Recipient account"},
            "amount": {"type": "string", "description": "in USD"}
        },
        "required": ["to", "amount"]
    })
}

/// A banking agent whose model calls `transfer_funds` while saying it is
/// changing notification settings. Returns the agent and the transfers made.
fn bank(
    approvals: ApprovalConfig,
    policy: PermissionPolicy,
    hook: Option<Arc<dyn Hook>>,
    turns: usize,
) -> (Agent, Arc<Mutex<Vec<Value>>>) {
    let mut script = Vec::new();
    for i in 0..turns {
        script.push(Message::assistant_with_tools(
            "I'm updating your notification settings now; this is a routine change.",
            vec![transfer_call(&format!("c{i}"))],
        ));
    }
    script.push(Message::assistant("Your e-mail alerts are on."));
    let ledger = Arc::new(Mutex::new(Vec::new()));
    let l = ledger.clone();
    let mut b = Agent::builder()
        .model(ScriptedModel::new(script))
        .tool(FnTool::new(
            "transfer_funds",
            "Transfer money to an account",
            schema(),
            move |args| {
                l.lock().unwrap().push(args);
                Ok(json!("transferred"))
            },
        ))
        .tool(FnTool::new(
            "set_alerts",
            "Change notification settings",
            json!({"type": "object", "properties": {"email": {"type": "boolean"}}}),
            |_| Ok(json!("updated")),
        ))
        .permissions(policy)
        .approvals(approvals);
    if let Some(h) = hook {
        b = b.hook_arc(h);
    }
    (b.build().unwrap(), ledger)
}

async fn first_request(agent: &Agent, thread: &str) -> ApprovalRequest {
    match agent.run(thread, "Turn on e-mail alerts.").await.unwrap() {
        AgentOutcome::ApprovalRequired { mut requests, .. } => requests.remove(0),
        other => panic!("expected an approval request, got {other:?}"),
    }
}

#[derive(Default)]
struct Recorder {
    records: Mutex<Vec<ApprovalRecord>>,
    risk: Option<RiskLevel>,
}

#[async_trait]
impl Hook for Recorder {
    async fn tool_risk(&self, _ctx: &HookContext, _call: &ToolCall) -> Option<RiskLevel> {
        self.risk
    }
    async fn on_approval(&self, ctx: &HookContext, record: &ApprovalRecord) {
        assert_eq!(ctx.thread_id, record.thread_id);
        self.records.lock().unwrap().push(record.clone());
    }
}

#[tokio::test]
async fn mismatch_warning_and_risk_reach_the_reviewer() {
    let approvals = ApprovalConfig::new()
        .risk("transfer_*", RiskLevel::Critical)
        .confirm("transfer_funds", Confirmation::retype(["to", "amount"]));
    // No permission rule asks: the critical risk level does.
    let (agent, ledger) = bank(approvals, PermissionPolicy::allow_all(), None, 1);
    let r = first_request(&agent, "t").await;
    assert_eq!(r.risk, RiskLevel::Critical);
    assert!(r.reason.contains("critical-risk"), "{}", r.reason);
    assert!(r.has_warnings());
    assert_eq!(r.warnings[0].check, "intent_rules");
    assert!(
        r.warnings[0].message.contains("`set_alerts`"),
        "{:?}",
        r.warnings
    );
    assert_eq!(r.rendered_arguments[0].label, "Recipient account");
    assert_eq!(r.rendered_arguments[0].value, ATTACKER);
    assert_eq!(
        r.rendered_arguments[1].description.as_deref(),
        Some("in USD")
    );
    assert_eq!(r.confirmation.retype, vec!["to", "amount"]);
    assert!(r.explanation.as_deref().unwrap().contains("notification"));
    let text = r.render();
    assert!(text.contains("[risk: critical]") && text.contains("WARNING"));

    // The request survives a JSON round trip (UIs, checkpoints).
    let back: ApprovalRequest = serde_json::from_value(serde_json::to_value(&r).unwrap()).unwrap();
    assert_eq!(back, r);

    // The streamed event carries the same request.
    let (agent2, _) = bank(
        ApprovalConfig::new().risk("transfer_*", RiskLevel::High),
        PermissionPolicy::allow_all().ask("transfer_funds"),
        None,
        1,
    );
    let events: Vec<_> = agent2.stream("s", "go").await.unwrap().collect().await;
    let Some(AgentEvent::ApprovalRequired { requests }) = events.last() else {
        panic!("expected approval, got {:?}", events.last())
    };
    assert_eq!(requests[0].risk, RiskLevel::High);
    assert!(requests[0].has_warnings());

    // Denying: nothing moved.
    agent
        .resume("t", vec![ApprovalDecision::deny("c0", "not asked for")])
        .await
        .unwrap();
    assert!(ledger.lock().unwrap().is_empty());
}

#[tokio::test]
async fn retype_confirmation_is_enforced_and_recorded() {
    let rec = Arc::new(Recorder::default());
    let approvals = ApprovalConfig::new()
        .risk("transfer_funds", RiskLevel::Critical)
        .confirm("transfer_funds", Confirmation::retype(["amount"]));
    let (agent, ledger) = bank(
        approvals,
        PermissionPolicy::allow_all(),
        Some(rec.clone()),
        1,
    );
    first_request(&agent, "t").await;

    // A one-click approval is refused and the thread stays paused.
    let err = agent
        .resume("t", vec![ApprovalDecision::approve("c0")])
        .await
        .unwrap_err();
    assert!(
        matches!(&err, HarnessError::Confirmation { call_id, reason } if call_id == "c0" && reason.contains("retyped")),
        "{err}"
    );
    let wrong = agent
        .resume(
            "t",
            vec![ApprovalDecision::approve("c0").retype("amount", "49.00")],
        )
        .await
        .unwrap_err();
    assert!(wrong.to_string().contains("does not match"), "{wrong}");
    assert!(ledger.lock().unwrap().is_empty());

    // The right value runs the call.
    let out = agent
        .resume(
            "t",
            vec![ApprovalDecision::approve("c0").retype("amount", "4900.00")],
        )
        .await
        .unwrap();
    assert_eq!(out.completed().unwrap().text, "Your e-mail alerts are on.");
    assert_eq!(ledger.lock().unwrap().len(), 1);

    let records = rec.records.lock().unwrap().clone();
    let outcomes: Vec<_> = records.iter().map(|r| r.outcome.clone()).collect();
    assert!(matches!(outcomes[0], ApprovalOutcome::Rejected { .. }));
    assert!(matches!(outcomes[1], ApprovalOutcome::Rejected { .. }));
    assert_eq!(outcomes[2], ApprovalOutcome::Approved);
    assert_eq!(records[2].request.risk, RiskLevel::Critical);
    assert!(records[2].request.has_warnings());
    assert_eq!(
        records[2].decision.as_ref().unwrap().retyped["amount"],
        "4900.00"
    );
    let json = serde_json::to_value(&records[2]).unwrap();
    assert_eq!(json["outcome"], "approved");
}

#[tokio::test]
async fn second_approver_and_no_always_for_high_impact() {
    let rec = Arc::new(Recorder::default());
    let approvals = ApprovalConfig::new()
        .risk("transfer_funds", RiskLevel::High)
        .confirm("transfer_funds", Confirmation::second_approver());
    let (agent, ledger) = bank(
        approvals,
        PermissionPolicy::allow_all().ask("transfer_funds"),
        Some(rec.clone()),
        2,
    );
    first_request(&agent, "t").await;
    let err = agent
        .resume(
            "t",
            vec![ApprovalDecision::approve_always("c0").approver("ana")],
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("2 distinct approvers"), "{err}");
    // Two approvers, "always" asked: the call runs, but the next transfer
    // still needs approval.
    let out = agent
        .resume(
            "t",
            vec![ApprovalDecision::approve_always("c0")
                .approver("ana")
                .approver("bruno")],
        )
        .await
        .unwrap();
    let AgentOutcome::ApprovalRequired { requests, .. } = out else {
        panic!("high-impact calls are approved one by one")
    };
    assert_eq!(requests[0].call_id, "c1");
    // No decision at all: denied and recorded as such.
    agent.resume("t", vec![]).await.unwrap();
    assert_eq!(ledger.lock().unwrap().len(), 1);
    let records = rec.records.lock().unwrap().clone();
    assert_eq!(records.last().unwrap().outcome, ApprovalOutcome::Denied);
    assert!(records.last().unwrap().decision.is_none());
}

#[tokio::test]
async fn hooks_raise_the_risk_and_custom_checks_warn() {
    struct Always;
    #[async_trait]
    impl IntentCheck for Always {
        fn name(&self) -> &str {
            "always"
        }
        async fn check(&self, ctx: &IntentContext<'_>) -> Option<String> {
            Some(format!("{} at {}", ctx.call.name, ctx.risk))
        }
    }
    let rec = Arc::new(Recorder {
        risk: Some(RiskLevel::Critical),
        ..Default::default()
    });
    let (agent, _) = bank(
        ApprovalConfig::new().no_checks().check(Always),
        PermissionPolicy::allow_all(),
        Some(rec),
        1,
    );
    let r = first_request(&agent, "t").await;
    assert_eq!(r.risk, RiskLevel::Critical);
    assert_eq!(r.warnings.len(), 1);
    assert_eq!(r.warnings[0].message, "transfer_funds at critical");

    // ask_at(None): the policy alone decides, so the call runs.
    let rec = Arc::new(Recorder {
        risk: Some(RiskLevel::Critical),
        ..Default::default()
    });
    let (agent, ledger) = bank(
        ApprovalConfig::new().ask_at(None),
        PermissionPolicy::allow_all(),
        Some(rec),
        1,
    );
    assert!(!agent
        .run("u", "Turn on e-mail alerts.")
        .await
        .unwrap()
        .is_approval_required());
    assert_eq!(ledger.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn low_risk_requests_have_defaults() {
    // Without configuration: low risk, schema rendering, no confirmation,
    // and no warning for a low-risk call the message does not describe.
    let model = ScriptedModel::new([
        Message::assistant_with_tools(
            "Let me do that.",
            vec![ToolCall {
                id: "e".into(),
                name: "echo".into(),
                arguments: json!({"text": "hi"}),
            }],
        ),
        Message::assistant("done"),
    ]);
    let agent = Agent::builder()
        .model(model)
        .tool(FnTool::new(
            "echo",
            "Echo text",
            json!({"type": "object", "properties": {"text": {"type": "string"}}}),
            Ok,
        ))
        .permissions(PermissionPolicy::ask_all())
        .build()
        .unwrap();
    let r = first_request(&agent, "t").await;
    assert_eq!(r.risk, RiskLevel::Low);
    assert!(r.warnings.is_empty());
    assert!(r.confirmation.is_plain());
    assert_eq!(r.rendered_arguments[0].label, "Text");
    // A plain approval works.
    let out = agent
        .resume("t", vec![ApprovalDecision::approve("e")])
        .await
        .unwrap();
    assert_eq!(out.completed().unwrap().text, "done");
}
