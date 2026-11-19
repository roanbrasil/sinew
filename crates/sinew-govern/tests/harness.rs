//! The control plane enforced in `sinew-harness` agents through
//! `GovernHook` (feature `harness`).

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, FnTool, Message, Role, ToolCall};
use sinew_govern::{AgentProfile, ControlPlane, MemoryDecisionSink, Rule, POLICY_KILL_SWITCH};
use sinew_harness::{Agent, AgentOutcome, ApprovalDecision, RiskLevel, StopReason};

fn call(id: &str, name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args,
    }
}

fn tool_results(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .map(|m| m.content.clone())
        .collect()
}

fn control_plane(audit: Arc<MemoryDecisionSink>) -> ControlPlane {
    ControlPlane::builder()
        .agent(AgentProfile::new("mailer"))
        .policy(
            Rule::ask("emails")
                .tools(["send_email"])
                .reason("emails need approval"),
        )
        .policy(
            Rule::deny("no-bulk")
                .tools(["send_email"])
                .when("to_count > 10".parse().unwrap()),
        )
        .policy(Rule::redact("mask-token", ["/token"]))
        .tool("send_email", true)
        .sink(audit)
        .build()
}

fn build_agent(
    model: Arc<ScriptedModel>,
    cp: &ControlPlane,
    sent: Arc<Mutex<Vec<Value>>>,
) -> Agent {
    Agent::builder()
        .model_arc(model)
        .name("mailer")
        .tool(FnTool::new("send_email", "send", json!({}), move |args| {
            sent.lock().unwrap().push(args);
            Ok(json!("sent"))
        }))
        .tool(FnTool::new("lookup", "look up", json!({}), Ok))
        .hook(cp.hook("mailer"))
        .build()
        .unwrap()
}

#[tokio::test]
async fn ask_maps_to_harness_approvals_and_deny_to_permission_errors() {
    let audit = Arc::new(MemoryDecisionSink::new());
    let cp = control_plane(audit.clone());
    let sent = Arc::new(Mutex::new(Vec::new()));
    let script = || {
        Arc::new(ScriptedModel::new([
            Message::assistant_with_tools(
                "",
                vec![
                    call("e1", "send_email", json!({"to_count": 1})),
                    call("e2", "send_email", json!({"to_count": 50})),
                    call("l1", "lookup", json!({"q": "x", "token": "secret"})),
                ],
            ),
            Message::assistant("finished"),
        ]))
    };

    let model = script();
    let agent = build_agent(model.clone(), &cp, sent.clone());
    let AgentOutcome::ApprovalRequired { requests, .. } = agent.run("t", "mail it").await.unwrap()
    else {
        panic!("expected an approval request")
    };
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].call_id, "e1");
    assert_eq!(requests[0].reason, "emails need approval");
    assert!(sent.lock().unwrap().is_empty());

    let result = agent
        .resume("t", vec![ApprovalDecision::approve("e1")])
        .await
        .unwrap()
        .completed()
        .unwrap();
    assert_eq!(result.text, "finished");
    assert_eq!(sent.lock().unwrap().as_slice(), [json!({"to_count": 1})]);
    let results = tool_results(&model.requests()[1].messages);
    assert_eq!(
        results,
        [
            "sent",
            "error: permission denied: denied by policy `no-bulk`",
            // The redacted argument never reached the tool.
            r#"{"q":"x","token":"[REDACTED]"}"#,
        ]
    );
    let decisions = audit.decisions();
    assert_eq!(decisions.len(), 3);
    assert_eq!(decisions[2].request.arguments["token"], "[REDACTED]");
    assert!(decisions
        .iter()
        .all(|d| d.request.thread_id.as_deref() == Some("t")));
}

#[tokio::test]
async fn kill_switch_ends_harness_turns_with_a_stop_reason() {
    let audit = Arc::new(MemoryDecisionSink::new());
    let cp = control_plane(audit.clone());
    let sent = Arc::new(Mutex::new(Vec::new()));

    // Suspended while waiting for approval: the approved call is skipped.
    let model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools("", vec![call("e1", "send_email", json!({"to_count": 1}))]),
        Message::assistant("never reached"),
    ]));
    let agent = build_agent(model.clone(), &cp, sent.clone());
    assert!(agent
        .run("t", "mail it")
        .await
        .unwrap()
        .is_approval_required());
    cp.suspend("mailer", "compromised credentials");
    let result = agent
        .resume("t", vec![ApprovalDecision::approve("e1")])
        .await
        .unwrap()
        .completed()
        .unwrap();
    let reason = "agent `mailer` is suspended: compromised credentials".to_string();
    assert_eq!(result.stop, Some(StopReason::Guardrail(reason.clone())));
    assert!(sent.lock().unwrap().is_empty());
    assert_eq!(model.requests().len(), 1);
    let state = agent.state("t").await.unwrap().unwrap();
    assert_eq!(
        tool_results(&state.messages),
        [format!("skipped: {reason}")]
    );

    // A new turn stops before the model is called.
    let model = Arc::new(ScriptedModel::new([Message::assistant("hello")]));
    let agent2 = build_agent(model.clone(), &cp, sent.clone());
    let result = agent2.ask("hi").await.unwrap();
    assert_eq!(result.stop, Some(StopReason::Guardrail(reason)));
    assert!(model.requests().is_empty());
    let halts: Vec<_> = audit
        .decisions()
        .into_iter()
        .filter(|d| d.verdict.policy_id.as_deref() == Some(POLICY_KILL_SWITCH))
        .collect();
    assert_eq!(halts.len(), 2);
    assert!(halts.iter().all(|d| d.verdict.stop));

    cp.reinstate("mailer");
    assert_eq!(agent2.ask("hi").await.unwrap().text, "hello");
}

#[tokio::test]
async fn model_calls_are_checked_when_the_model_is_named() {
    let cp = ControlPlane::builder()
        .agent(AgentProfile::new("bot").allow_models(["ollama:*"]))
        .build();
    let model = Arc::new(ScriptedModel::new([Message::assistant("hi")]));
    let agent = Agent::builder()
        .model_arc(model.clone())
        .hook(cp.hook("bot").model("openai:gpt-4.1"))
        .build()
        .unwrap();
    let result = agent.ask("hi").await.unwrap();
    assert_eq!(
        result.stop,
        Some(StopReason::Guardrail(
            "model `openai:gpt-4.1` is not in the allowed models of agent `bot`".into()
        ))
    );
    assert!(model.requests().is_empty());
}

#[tokio::test]
async fn hook_risk_levels_reach_harness_approval_requests() {
    let audit = Arc::new(MemoryDecisionSink::new());
    let cp = control_plane(audit);
    let model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "Sending the e-mail.",
            vec![call("e1", "send_email", json!({"to_count": 1}))],
        ),
        Message::assistant("finished"),
    ]));
    let agent = Agent::builder()
        .model_arc(model)
        .name("mailer")
        .tool(FnTool::new("send_email", "send", json!({}), Ok))
        .hook(
            cp.hook("mailer")
                .risk("send_*", RiskLevel::High)
                .risk("send_email", RiskLevel::Critical),
        )
        .build()
        .unwrap();
    let AgentOutcome::ApprovalRequired { requests, .. } = agent.run("t", "mail it").await.unwrap()
    else {
        panic!("expected an approval request")
    };
    assert_eq!(requests[0].risk, RiskLevel::Critical);
    assert_eq!(requests[0].reason, "emails need approval");
}

#[cfg(feature = "audit")]
#[tokio::test]
async fn approval_answers_go_to_the_audit_log() {
    use sinew_govern::audit::{AuditKind, AuditLog, MemoryAuditStore, VerifyOptions};

    let cp = control_plane(Arc::new(MemoryDecisionSink::new()));
    let store = Arc::new(MemoryAuditStore::new());
    let log = AuditLog::new(store.clone());
    let model = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "",
            vec![call(
                "e1",
                "send_email",
                json!({"to_count": 1, "to": "ana@example.com"}),
            )],
        ),
        Message::assistant_with_tools("", vec![call("e2", "send_email", json!({"to_count": 2}))]),
        Message::assistant("finished"),
    ]));
    let agent = Agent::builder()
        .model_arc(model)
        .name("mailer")
        .tool(FnTool::new("send_email", "send", json!({}), |_| {
            Ok(json!("sent"))
        }))
        .hook(cp.hook("mailer").audit(log.clone()))
        .build()
        .unwrap();

    assert!(agent
        .run("t", "mail it")
        .await
        .unwrap()
        .is_approval_required());
    let next = agent
        .resume(
            "t",
            vec![ApprovalDecision::approve("e1")
                .approver("ana")
                .approver("bo")],
        )
        .await
        .unwrap();
    assert!(next.is_approval_required());
    agent
        .resume("t", vec![ApprovalDecision::deny("e2", "wrong list")])
        .await
        .unwrap();

    let entries = store.snapshot();
    assert_eq!(entries.len(), 2);
    assert!(entries.iter().all(|e| e.kind == AuditKind::Approval));
    let (a, d) = (&entries[0].payload, &entries[1].payload);
    assert_eq!(a["agent"], "mailer");
    assert_eq!(a["target"], "send_email");
    assert_eq!(a["approver"], "ana, bo");
    assert_eq!(a["approved"], true);
    assert_eq!(a["details"]["harness"]["thread_id"], "t");
    // The harness record passed through the redactor.
    assert_eq!(
        a["details"]["harness"]["request"]["arguments"]["to"],
        "[REDACTED:email]"
    );
    assert_eq!(d["approved"], false);
    assert_eq!(d["approver"], "unknown");
    assert_eq!(d["reason"], "wrong list");
    assert!(log.verify(&VerifyOptions::new()).await.unwrap().is_ok());
    assert_eq!(log.failures(), 0);
}
