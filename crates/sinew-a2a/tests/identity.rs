//! Signed agent cards, verified callers, on-behalf-of credentials and the
//! delegation chain across an A2A hop.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_a2a::{
    jsonrpc::error_code, A2aAgentTool, A2aClient, A2aError, A2aServer, AgentCard, AgentExecutor,
    BoxError, EventQueue, GraphExecutor, Part, RequestContext, ServerHandle,
};
use sinew_core::{
    identity::{
        current_context, user_credential, with_context, AgentCredentials, AgentIdentity, AgentKey,
        CallContext, IdentityVerifier, InMemoryTokenExchanger, OnBehalfOf, TrustStore,
        META_IDENTITY,
    },
    mock::ScriptedModel,
    AsyncFnTool, Message as CoreMessage, ToolBox, ToolCall,
};
use sinew_graph::prebuilt::react_agent;
use sinew_trace::{traced_tool, MemorySink, RunKind, Tracer};

struct Keys {
    issuer: AgentKey,
    orchestrator: AgentCredentials,
    billing_key: AgentKey,
    billing: sinew_core::identity::SignedAgentIdentity,
}

fn keys() -> Keys {
    let issuer = AgentKey::from_bytes(&[1; 32]);
    let okey = AgentKey::from_bytes(&[2; 32]);
    let oid = AgentIdentity::new("orchestrator", okey.public_key())
        .owner("platform")
        .allow_tools(["billing"])
        .sign(&issuer)
        .unwrap();
    let billing_key = AgentKey::from_bytes(&[3; 32]);
    let billing = AgentIdentity::new("billing", billing_key.public_key())
        .owner("payments-team")
        .version("2.0.0")
        .allow_tools(["lookup"])
        .sign(&issuer)
        .unwrap();
    Keys {
        orchestrator: AgentCredentials::new(okey, oid).unwrap(),
        issuer,
        billing_key,
        billing,
    }
}

fn trust(k: &Keys) -> TrustStore {
    TrustStore::new().trust(k.issuer.public_key())
}

fn billing_card(k: &Keys) -> AgentCard {
    AgentCard::new("Billing", "Refunds and invoices", "").with_identity(&k.billing)
}

/// Replies with what the server established about the caller.
struct Whoami;

#[async_trait]
impl AgentExecutor for Whoami {
    async fn execute(&self, ctx: RequestContext, events: EventQueue) -> Result<(), BoxError> {
        let here = current_context();
        let reply = json!({
            "caller": ctx.caller.as_ref().map(|c| c.agent().to_string()),
            "agent": here.as_ref().map(|c| c.agent.clone()),
            "user": here.as_ref().and_then(|c| c.user.clone()),
            "chain": here.as_ref().map(|c| c.chain_agents().iter().map(|s| s.to_string()).collect::<Vec<_>>()),
            "verified": here.as_ref().map(|c| c.chain.last().unwrap().verified),
            "credential": user_credential().map(|c| c.bearer().to_string()),
            "subject": here.and_then(|c| c.subject_token).map(|s| s.expose().to_string()),
            "metadata_has_credential": ctx.metadata.as_ref().is_some_and(|m| m.contains_key("sinew.dev/credential")),
        });
        events.reply(vec![Part::text(reply.to_string())]);
        Ok(())
    }
}

async fn spawn(k: &Keys, executor: impl AgentExecutor) -> ServerHandle {
    A2aServer::new(billing_card(k), executor)
        .with_card_signing_key(k.billing_key.clone())
        .with_identity_verifier(IdentityVerifier::new(trust(k)))
        .spawn("127.0.0.1:0")
        .await
        .unwrap()
}

fn code(e: A2aError) -> i64 {
    match e {
        A2aError::Rpc(r) => r.code,
        other => panic!("expected an RPC error, got {other}"),
    }
}

#[tokio::test]
async fn clients_verify_signed_cards() {
    let k = keys();
    let server = spawn(&k, Whoami).await;
    let client = A2aClient::discover_verified(&server.base_url(), &trust(&k))
        .await
        .unwrap();
    let v = client.verified_identity().unwrap();
    assert_eq!(v.identity.id, "billing");
    assert_eq!(v.identity.owner.as_deref(), Some("payments-team"));

    // An issuer we do not trust.
    let other = TrustStore::new().trust(AgentKey::from_bytes(&[9; 32]).public_key());
    let e = A2aClient::discover_verified(&server.base_url(), &other)
        .await
        .err()
        .unwrap();
    assert!(matches!(e, A2aError::Identity(_)), "{e}");

    // An unsigned card.
    let plain = A2aServer::new(AgentCard::new("Plain", "d", ""), Whoami)
        .spawn("127.0.0.1:0")
        .await
        .unwrap();
    assert!(A2aClient::discover_verified(&plain.base_url(), &trust(&k))
        .await
        .is_err());
    assert!(A2aClient::discover(&plain.base_url())
        .await
        .unwrap()
        .verified_identity()
        .is_none());
}

#[tokio::test]
async fn servers_reject_unauthenticated_and_untrusted_callers() {
    let k = keys();
    let server = spawn(&k, Whoami).await;

    let anonymous = A2aClient::discover(&server.base_url()).await.unwrap();
    assert_eq!(
        code(anonymous.send_text("hi").await.unwrap_err()),
        error_code::IDENTITY_REJECTED
    );
    assert_eq!(
        code(anonymous.get_task("t", None).await.unwrap_err()),
        error_code::IDENTITY_REJECTED
    );

    // Signed, but by an identity from an issuer the server does not trust.
    let rogue_issuer = AgentKey::from_bytes(&[8; 32]);
    let rogue_key = AgentKey::from_bytes(&[7; 32]);
    let rogue = AgentIdentity::new("orchestrator", rogue_key.public_key())
        .sign(&rogue_issuer)
        .unwrap();
    let forged = A2aClient::discover(&server.base_url())
        .await
        .unwrap()
        .with_credentials(AgentCredentials::new(rogue_key, rogue).unwrap());
    let e = forged.send_text("hi").await.unwrap_err();
    assert!(e.to_string().contains("unknown key"), "{e}");

    // Meant for another agent.
    let misdirected = A2aClient::discover(&server.base_url())
        .await
        .unwrap()
        .with_credentials(k.orchestrator.clone())
        .with_audience("hr");
    let e = misdirected.send_text("hi").await.unwrap_err();
    assert!(e.to_string().contains("mismatch"), "{e}");

    // A trusted caller gets through, including tasks/get.
    let client = A2aClient::discover(&server.base_url())
        .await
        .unwrap()
        .with_credentials(k.orchestrator.clone());
    let r = client.send_text("hi").await.unwrap();
    let v: Value = serde_json::from_str(&r.text()).unwrap();
    assert_eq!(v["caller"], "orchestrator");
    assert_eq!(v["chain"], json!(["orchestrator", "billing"]));
    assert_eq!(v["verified"], true);
    assert_eq!(
        code(client.get_task("missing", None).await.unwrap_err()),
        error_code::TASK_NOT_FOUND
    );
}

#[tokio::test]
async fn replayed_requests_are_rejected() {
    let k = keys();
    let server = spawn(&k, Whoami).await;
    let proof = k
        .orchestrator
        .prove(
            Some(&CallContext::new("orchestrator").user("ana")),
            "billing",
        )
        .unwrap();
    let body = json!({
        "jsonrpc": "2.0", "id": 1, "method": "message/send",
        "params": {
            "message": {"kind": "message", "role": "user", "messageId": "m1", "parts": [{"kind": "text", "text": "hi"}]},
            "metadata": { META_IDENTITY: proof.to_value() }
        }
    });
    let http = reqwest::Client::new();
    let url = format!("{}/", server.base_url());
    let first: Value = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(first.get("result").is_some(), "{first}");
    let second: Value = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(second["error"]["code"], error_code::IDENTITY_REJECTED);
    assert!(second["error"]["message"]
        .as_str()
        .unwrap()
        .contains("replayed"));
}

#[tokio::test]
async fn user_context_and_credential_travel_with_the_call() {
    let k = keys();
    let server = spawn(&k, Whoami).await;
    let sts = Arc::new(InMemoryTokenExchanger::new());
    sts.register_subject("ana-session", "ana");
    sts.allow("ana", "billing", ["invoices:read"]);

    let client = A2aClient::discover_verified(&server.base_url(), &trust(&k))
        .await
        .unwrap()
        .with_credentials(k.orchestrator.clone());
    let tools = OnBehalfOf::new(sts.clone())
        .grant("billing", "billing", ["invoices:read"])
        .wrap(&ToolBox::new().with(A2aAgentTool::new("billing", "Billing agent", client)));
    let ctx = CallContext::new("orchestrator")
        .user("ana")
        .tenant("acme")
        .subject_token("ana-session");
    let call = ToolCall {
        id: "1".into(),
        name: "billing".into(),
        arguments: json!({"input": "who am I?"}),
    };
    let out = with_context(ctx, tools.call(&call)).await.unwrap();
    let v: Value = serde_json::from_str(out.as_str().unwrap()).unwrap();
    assert_eq!(v["user"], "ana");
    assert_eq!(v["agent"], "billing");
    let token = v["credential"].as_str().unwrap();
    assert!(token.starts_with("obo-"), "{v}");
    // The credential is the subject for billing's own exchanges, and it
    // was taken out of the metadata the executor sees.
    assert_eq!(v["subject"], token);
    assert_eq!(v["metadata_has_credential"], false);
    let info = sts.introspect(token).unwrap();
    assert_eq!(
        (info.user.as_str(), info.audience.as_str()),
        ("ana", "billing")
    );
    assert_eq!(info.actor.as_deref(), Some("orchestrator"));
    assert!(!server.url().is_empty());
}

#[tokio::test]
async fn delegation_chain_is_visible_in_one_trace() {
    let k = keys();
    let sink = Arc::new(MemorySink::new());
    let tracer = Tracer::new().with_shared_sink(sink.clone());

    // The remote billing agent: a ReactAgent whose `lookup` tool uses the
    // user's credential, exchanged again for the invoices API.
    let sts = Arc::new(InMemoryTokenExchanger::new());
    sts.register_subject("ana-session", "ana");
    sts.allow("ana", "billing", ["invoices:read"]);
    sts.allow("ana", "invoices-api", ["invoices:read"]);
    let lookup = AsyncFnTool::new("lookup", "Find invoices", json!({}), |_| async {
        let c = user_credential().expect("on-behalf-of credential");
        Ok(json!(format!("invoice for {}", c.user.unwrap_or_default())))
    });
    let remote_tools = OnBehalfOf::new(sts.clone())
        .grant("lookup", "invoices-api", ["invoices:read"])
        .wrap(&ToolBox::new().with(lookup));
    let remote_model = Arc::new(ScriptedModel::new([
        CoreMessage::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "r1".into(),
                name: "lookup".into(),
                arguments: json!({}),
            }],
        ),
        CoreMessage::assistant("one invoice"),
    ]));
    let graph = react_agent(remote_model, remote_tools)
        .unwrap()
        .with_tracer(tracer.clone());
    let server = spawn(&k, GraphExecutor::agent(graph)).await;

    // The orchestrator side: a traced tool call to the billing agent.
    let client = A2aClient::discover_verified(&server.base_url(), &trust(&k))
        .await
        .unwrap()
        .with_credentials(k.orchestrator.clone());
    let tools = OnBehalfOf::new(sts.clone())
        .grant("billing", "billing", ["invoices:read"])
        .wrap(&ToolBox::new().with(A2aAgentTool::new("billing", "Billing agent", client)));
    let ctx = CallContext::new("orchestrator")
        .user("ana")
        .subject_token("ana-session");
    let call = ToolCall {
        id: "c1".into(),
        name: "billing".into(),
        arguments: json!({"input": "find my invoice"}),
    };
    let root = tracer.start("orchestrator", RunKind::Chain, None, json!({}));
    let root_ref = root.reference();
    let msg = with_context(
        ctx,
        traced_tool(&tools, &call, Some(&tracer), Some(&root_ref)),
    )
    .await;
    root.finish(json!(null));
    assert_eq!(msg.content, "one invoice");

    let runs = sink.runs();
    let a_tool = runs
        .iter()
        .find(|r| r.name == "billing" && r.kind == RunKind::Tool)
        .unwrap();
    let b_graph = runs.iter().find(|r| r.kind == RunKind::Graph).unwrap();
    let b_tool = runs.iter().find(|r| r.name == "lookup").unwrap();
    // One trace: the remote graph hangs under the orchestrator's tool run.
    for r in [a_tool, b_graph, b_tool] {
        assert_eq!(r.trace_id, root_ref.trace_id, "{}", r.name);
    }
    assert_eq!(b_graph.parent_id.as_deref(), Some(a_tool.id.as_str()));
    // The chain of custody on the remote tool run.
    let chain = &b_tool.metadata["identity"]["chain"];
    assert_eq!(chain[0]["agent"], "orchestrator");
    assert_eq!(chain[0]["via"], "origin");
    assert_eq!(chain[1]["agent"], "billing");
    assert_eq!(chain[1]["via"], "a2a");
    assert_eq!(chain[1]["verified"], true);
    assert_eq!(b_tool.metadata["identity"]["user"], "ana");
    assert_eq!(b_tool.outputs, Some(json!("invoice for ana")));
    assert_eq!(a_tool.metadata["identity"]["agent"], "orchestrator");
    // Two exchanges: orchestrator -> billing, billing -> invoices API.
    assert_eq!(sts.issued_count(), 2);
}
