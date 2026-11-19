//! Signed identities on MCP: the server verifies callers and checks their
//! allowed tools, the client verifies the server, the user's context and
//! on-behalf-of credential reach the tool.

use std::sync::Arc;

use serde_json::{json, Value};
use sinew_core::{
    identity::{
        current_context, user_credential, with_context, AgentCredentials, AgentIdentity, AgentKey,
        CallContext, IdentityVerifier, InMemoryTokenExchanger, OnBehalfOf, TrustStore,
    },
    AsyncFnTool, Tool, ToolBox, ToolCall,
};
use sinew_mcp::{protocol::error_code, McpClient, McpError, McpServer};

async fn connect(server: McpServer) -> McpClient {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let (sr, sw) = tokio::io::split(server_io);
    tokio::spawn(server.serve(sr, sw));
    let (cr, cw) = tokio::io::split(client_io);
    McpClient::connect(cr, cw).await.unwrap()
}

/// Reports what the server established about the call.
fn whoami(name: &str) -> impl Tool {
    AsyncFnTool::new(name, "Who is calling", json!({}), |_| async {
        let ctx = current_context();
        Ok(json!({
            "agent": ctx.as_ref().map(|c| c.agent.clone()),
            "user": ctx.as_ref().and_then(|c| c.user.clone()),
            "chain": ctx.as_ref().map(|c| c.chain_agents().iter().map(|s| s.to_string()).collect::<Vec<_>>()),
            "subject": ctx.and_then(|c| c.subject_token).map(|s| s.expose().to_string()),
            "credential": user_credential().map(|c| c.bearer().to_string()),
        }))
    })
}

struct Setup {
    issuer: AgentKey,
    agent: AgentCredentials,
    server_identity: sinew_core::identity::SignedAgentIdentity,
}

fn setup() -> Setup {
    let issuer = AgentKey::from_bytes(&[1; 32]);
    let key = AgentKey::from_bytes(&[2; 32]);
    let identity = AgentIdentity::new("support-bot", key.public_key())
        .allow_tools(["whoami", "lookup_*"])
        .sign(&issuer)
        .unwrap();
    let skey = AgentKey::from_bytes(&[3; 32]);
    let server_identity = AgentIdentity::new("crm-server", skey.public_key())
        .owner("crm-team")
        .sign(&issuer)
        .unwrap();
    Setup {
        agent: AgentCredentials::new(key, identity).unwrap(),
        issuer,
        server_identity,
    }
}

fn server(s: &Setup) -> McpServer {
    McpServer::new("crm", "1.0")
        .with_tool(whoami("whoami"))
        .with_tool(whoami("delete_customer"))
        .with_identity(s.server_identity.clone())
        .with_identity_verifier(IdentityVerifier::new(
            TrustStore::new().trust(s.issuer.public_key()),
        ))
}

fn rpc_code(e: McpError) -> i64 {
    match e {
        McpError::Rpc(r) => r.code,
        other => panic!("expected an RPC error, got {other}"),
    }
}

#[tokio::test]
async fn verified_callers_get_their_context_and_allowed_tools_only() {
    let s = setup();
    let client = connect(server(&s)).await.with_credentials(s.agent.clone());

    let ctx = CallContext::new("support-bot").user("ana").tenant("acme");
    let r = with_context(ctx, client.call_tool("whoami", json!({})))
        .await
        .unwrap();
    let v: Value = serde_json::from_str(&r.content[0].to_model_text()).unwrap();
    assert_eq!(v["agent"], "crm-server");
    assert_eq!(v["user"], "ana");
    assert_eq!(v["chain"], json!(["support-bot", "crm-server"]));

    // Without a context the agent itself is the origin.
    let r = client.call_tool("whoami", json!({})).await.unwrap();
    let v: Value = serde_json::from_str(&r.content[0].to_model_text()).unwrap();
    assert_eq!(v["chain"], json!(["support-bot", "crm-server"]));
    assert_eq!(v["user"], Value::Null);

    // A tool outside the identity's `allowed_tools`.
    let e = client
        .call_tool("delete_customer", json!({}))
        .await
        .unwrap_err();
    assert_eq!(rpc_code(e), error_code::IDENTITY_REJECTED);

    // As an McpTool, a refusal is a tool error the model can see.
    let tools = client.toolbox().await.unwrap();
    let out = tools
        .run(&ToolCall {
            id: "1".into(),
            name: "delete_customer".into(),
            arguments: json!({}),
        })
        .await;
    assert!(out.content.contains("may not call"), "{}", out.content);
}

#[tokio::test]
async fn unauthenticated_and_untrusted_callers_are_rejected() {
    let s = setup();
    let anonymous = connect(server(&s)).await;
    let e = anonymous.call_tool("whoami", json!({})).await.unwrap_err();
    assert_eq!(rpc_code(e), error_code::IDENTITY_REJECTED);

    let rogue_issuer = AgentKey::from_bytes(&[8; 32]);
    let rogue_key = AgentKey::from_bytes(&[9; 32]);
    let rogue = AgentIdentity::new("support-bot", rogue_key.public_key())
        .allow_tools(["*"])
        .sign(&rogue_issuer)
        .unwrap();
    let forged = connect(server(&s))
        .await
        .with_credentials(AgentCredentials::new(rogue_key, rogue).unwrap());
    let e = forged.call_tool("whoami", json!({})).await.unwrap_err();
    assert!(e.to_string().contains("unknown key"), "{e}");

    let misdirected = connect(server(&s))
        .await
        .with_credentials(s.agent.clone())
        .with_audience("billing-server");
    let e = misdirected
        .call_tool("whoami", json!({}))
        .await
        .unwrap_err();
    assert!(e.to_string().contains("mismatch"), "{e}");

    // A context acting as another agent never leaves the client.
    let client = connect(server(&s)).await.with_credentials(s.agent.clone());
    let e = with_context(
        CallContext::new("admin-bot"),
        client.call_tool("whoami", json!({})),
    )
    .await
    .unwrap_err();
    assert!(matches!(e, McpError::Identity(_)), "{e}");
}

#[tokio::test]
async fn clients_verify_the_server_identity() {
    let s = setup();
    let client = connect(server(&s)).await;
    let trust = TrustStore::new().trust(s.issuer.public_key());
    let v = client.verify_server(&trust).unwrap();
    assert_eq!(v.identity.id, "crm-server");
    assert_eq!(v.identity.owner.as_deref(), Some("crm-team"));
    assert!(client.verify_server(&TrustStore::new()).is_err());

    let plain = connect(McpServer::new("plain", "1.0")).await;
    assert!(plain.server_identity().is_none());
    assert!(matches!(
        plain.verify_server(&trust),
        Err(McpError::Identity(_))
    ));
}

#[tokio::test]
async fn on_behalf_of_credentials_reach_mcp_tools() {
    let s = setup();
    let sts = Arc::new(InMemoryTokenExchanger::new());
    sts.register_subject("ana-session", "ana");
    sts.allow("ana", "crm", ["customers:read"]);

    // A server without identity checks still hands the credential over.
    let open = McpServer::new("crm", "1.0").with_tool(whoami("lookup_customer"));
    let client = connect(open).await;
    let tools = OnBehalfOf::new(sts.clone())
        .grant("lookup_*", "crm", ["customers:read"])
        .wrap(&client.toolbox().await.unwrap());
    let ctx = CallContext::new("support-bot")
        .user("ana")
        .subject_token("ana-session");
    let call = ToolCall {
        id: "1".into(),
        name: "lookup_customer".into(),
        arguments: json!({}),
    };
    let out = with_context(ctx.clone(), tools.call(&call)).await.unwrap();
    let v: Value = serde_json::from_str(out.as_str().unwrap()).unwrap();
    let token = v["credential"].as_str().unwrap();
    assert_eq!(sts.introspect(token).unwrap().audience, "crm");
    assert_eq!(v["agent"], Value::Null, "no verifier: no caller context");

    // With a verifier, the credential is also the context's subject token.
    let server = McpServer::new("crm", "1.0")
        .with_tool(whoami("lookup_customer"))
        .with_identity_verifier(IdentityVerifier::new(
            TrustStore::new().trust(s.issuer.public_key()),
        ));
    let client = connect(server).await.with_credentials(s.agent.clone());
    let tools = OnBehalfOf::new(sts.clone())
        .grant("lookup_*", "crm", ["customers:read"])
        .wrap(&ToolBox::new().with(client.tools().await.unwrap().remove(0)));
    let out = with_context(ctx, tools.call(&call)).await.unwrap();
    let v: Value = serde_json::from_str(out.as_str().unwrap()).unwrap();
    assert_eq!(v["subject"], v["credential"]);
    assert_eq!(v["chain"], json!(["support-bot", "crm"]));
}
