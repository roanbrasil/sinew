//! The control plane and agent identity: registering verified identities,
//! and requests that take their user, tenant and delegation chain from the
//! call context.

use std::sync::Arc;

use serde_json::json;
use sinew_core::{
    identity::{with_context, AgentIdentity, AgentKey, CallContext, TrustStore, Via},
    mock::ScriptedModel,
    ChatModel, FnTool, Message, ToolBox, ToolCall,
};
use sinew_govern::{
    AgentProfile, ControlPlane, Identity, MemoryDecisionSink, Request, Rule, POLICY_ALLOWED,
};
use sinew_graph::prebuilt::{AgentState, ReactAgent};

#[tokio::test]
async fn registers_only_verified_identities() {
    let issuer = AgentKey::from_bytes(&[1; 32]);
    let key = AgentKey::from_bytes(&[2; 32]);
    let signed = AgentIdentity::new("billing", key.public_key())
        .owner("payments")
        .tenant("acme")
        .version("1.2.0")
        .capability("refunds")
        .allow_tools(["get_*"])
        .sign(&issuer)
        .unwrap();
    let cp = ControlPlane::builder().require_registration(true).build();

    assert!(cp.register_verified(&signed, &TrustStore::new()).is_err());
    assert!(cp.agent("billing").is_none());

    let p = cp
        .register_verified(&signed, &TrustStore::new().trust(issuer.public_key()))
        .unwrap();
    assert_eq!(p.tenant.as_deref(), Some("acme"));
    assert_eq!(
        p.description.as_deref(),
        Some("version 1.2.0; capabilities: refunds")
    );
    let ok = cp
        .check(Request::tool("billing", "get_invoice", json!({})))
        .await;
    assert!(ok.decision.is_allowed());
    let denied = cp
        .check(Request::tool("billing", "refund", json!({})))
        .await;
    assert_eq!(denied.policy_id.as_deref(), Some(POLICY_ALLOWED));
    // No tools signed: none allowed.
    let none = AgentProfile::from_identity(&AgentIdentity::new("x", key.public_key()));
    assert!(!none.tool_allowed("anything"));
}

#[tokio::test]
async fn requests_take_user_tenant_and_chain_from_the_context() {
    let sink = Arc::new(MemoryDecisionSink::new());
    let cp = ControlPlane::builder()
        .agent(AgentProfile::new("billing"))
        .policy(Rule::allow("ana-only").users(["ana"]).tools(["refund"]))
        .sink(sink.clone())
        .build();

    let ctx = CallContext::new("orchestrator")
        .user("ana")
        .tenant("acme")
        .trace("t-1")
        .delegate_to("billing", Via::A2a);
    let v = with_context(
        ctx.clone(),
        cp.check(Request::tool("billing", "refund", json!({})).side_effects(true)),
    )
    .await;
    assert!(v.decision.is_allowed(), "{v:?}");
    let ev = sink.decisions().pop().unwrap();
    assert_eq!(ev.request.user.as_deref(), Some("ana"));
    assert_eq!(ev.request.tenant.as_deref(), Some("acme"));
    assert_eq!(ev.request.trace_id.as_deref(), Some("t-1"));
    assert_eq!(
        ev.request.metadata["delegation_chain"],
        json!(["orchestrator", "billing"])
    );

    // A context for another agent is not used.
    let v = with_context(
        ctx.clone(),
        cp.check(Request::tool("orchestrator", "refund", json!({})).side_effects(true)),
    )
    .await;
    assert!(!v.decision.is_allowed());
    // An explicit user wins over the context.
    let v = with_context(
        ctx.clone(),
        cp.check(
            Request::tool("billing", "refund", json!({}))
                .user("mallory")
                .side_effects(true),
        ),
    )
    .await;
    assert!(!v.decision.is_allowed());
    assert_eq!(Identity::from(&ctx).user.as_deref(), Some("ana"));

    // The ReactAgent gate, built without a user, picks the user up per request.
    let model: Arc<dyn ChatModel> = Arc::new(ScriptedModel::new([
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "refund".into(),
                arguments: json!({}),
            }],
        ),
        Message::assistant("done"),
    ]));
    let tools = ToolBox::new().with(FnTool::new("refund", "", json!({}), |_| Ok(json!("ok"))));
    let cp = cp.clone();
    cp.register_tool("refund", true);
    let agent = ReactAgent::new(model, tools)
        .gate(Arc::new(cp.gate("billing")))
        .compile()
        .unwrap();
    let out = with_context(
        ctx,
        agent.invoke(AgentState::new(vec![Message::user("refund")])),
    )
    .await
    .unwrap();
    assert_eq!(out.messages[2].content, "ok");
}
