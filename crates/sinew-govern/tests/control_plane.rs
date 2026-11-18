//! The control plane enforced in `ReactAgent`, on a `ToolBox` and on a
//! `ChatModel`, offline with `ScriptedModel`.

use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde_json::{json, Value};
use sinew_core::{
    mock::ScriptedModel, AsyncFnTool, ChatModel, ChatRequest, Error, FnTool, Message, Tool,
    ToolBox, ToolCall, ToolSpec,
};
use sinew_govern::{
    AgentBudget, AgentProfile, AgentStatus, ControlAction, ControlPlane, Decision, Identity,
    MemoryDecisionSink, Request, Rule, POLICY_BUDGET, POLICY_KILL_SWITCH, POLICY_QUARANTINE,
    POLICY_REGISTRY, REDACTED,
};
use sinew_graph::{
    prebuilt::{AgentState, ReactAgent, GATE_STOP_PREFIX},
    GraphEvent, MemoryCheckpointer, RunOutcome,
};
use sinew_trace::{MemorySink, RunKind, Tracer};

fn call(id: &str, name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args,
    }
}

fn tools_msg(calls: Vec<ToolCall>) -> Message {
    Message::assistant_with_tools("", calls)
}

/// A tool that records the arguments it ran with.
struct Recorder {
    name: &'static str,
    side_effects: bool,
    seen: Arc<Mutex<Vec<Value>>>,
}

#[async_trait::async_trait]
impl Tool for Recorder {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.into(),
            description: "records its arguments".into(),
            parameters: json!({"type": "object"}),
        }
    }
    fn side_effects(&self) -> bool {
        self.side_effects
    }
    async fn call(&self, arguments: Value) -> sinew_core::Result<Value> {
        self.seen.lock().unwrap().push(arguments);
        Ok(json!("done"))
    }
}

fn recorder(name: &'static str, side_effects: bool) -> (Recorder, Arc<Mutex<Vec<Value>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    (
        Recorder {
            name,
            side_effects,
            seen: seen.clone(),
        },
        seen,
    )
}

#[tokio::test]
async fn kill_switch_stops_a_running_react_agent_at_the_next_step() {
    let cp = ControlPlane::builder()
        .agent(AgentProfile::new("support"))
        .build();
    // The tool flips the kill switch while the turn is running, as an
    // operator would from another task.
    let operator = cp.clone();
    let tools = ToolBox::new().with(AsyncFnTool::new(
        "lookup",
        "look something up",
        json!({"type": "object"}),
        move |_| {
            let cp = operator.clone();
            async move {
                cp.suspend("support", "incident 42");
                Ok(json!("found it"))
            }
        },
    ));
    let model = Arc::new(ScriptedModel::new([
        tools_msg(vec![call("c1", "lookup", json!({}))]),
        tools_msg(vec![call("c2", "lookup", json!({}))]),
        Message::assistant("never reached"),
    ]));
    let agent = ReactAgent::new(model.clone(), tools)
        .gate(Arc::new(cp.gate("support")))
        .compile()
        .unwrap();
    let out = agent
        .invoke(AgentState::new(vec![Message::user("go")]))
        .await
        .unwrap();

    // One model call, one tool call, then the next step was refused.
    assert_eq!(model.requests().len(), 1);
    assert_eq!(out.messages[2].content, "found it");
    assert_eq!(
        out.last().unwrap().content,
        format!("{GATE_STOP_PREFIX}agent `support` is suspended: incident 42")
    );
    assert_eq!(cp.status("support"), Some(AgentStatus::Suspended));

    // Reinstated, the agent runs again.
    cp.reinstate("support");
    let model = Arc::new(ScriptedModel::new([Message::assistant("back")]));
    let agent = ReactAgent::new(model, ToolBox::new())
        .gate(Arc::new(cp.gate("support")))
        .compile()
        .unwrap();
    let out = agent
        .invoke(AgentState::new(vec![Message::user("hi")]))
        .await
        .unwrap();
    assert_eq!(out.last().unwrap().content, "back");
}

#[tokio::test]
async fn suspension_mid_step_skips_the_remaining_tool_calls() {
    let cp = ControlPlane::builder()
        .agent(AgentProfile::new("bot"))
        .build();
    let operator = cp.clone();
    let (second, seen) = recorder("second", false);
    let tools = ToolBox::new()
        .with(FnTool::new("first", "", json!({}), move |_| {
            operator.suspend("bot", "operator request");
            Ok(json!("ok"))
        }))
        .with(second);
    let model = Arc::new(ScriptedModel::new([tools_msg(vec![
        call("a", "first", json!({})),
        call("b", "second", json!({})),
        call("c", "second", json!({})),
    ])]));
    let agent = ReactAgent::new(model, tools)
        .gate(Arc::new(cp.gate("bot")))
        .compile()
        .unwrap();
    let out = agent
        .invoke(AgentState::new(vec![Message::user("go")]))
        .await
        .unwrap();
    let contents: Vec<&str> = out.messages.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(
        contents[2..],
        [
            "ok",
            "skipped: agent `bot` is suspended: operator request",
            "skipped: agent `bot` is suspended: operator request",
            "The run was stopped: agent `bot` is suspended: operator request",
        ]
    );
    assert!(seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn policies_deny_ask_and_redact_in_a_react_agent() {
    let cp = ControlPlane::builder()
        .agent(AgentProfile::new("billing"))
        .policy(Rule::allow("lookups").tools(["get_*"]))
        .policy(
            Rule::ask("refunds")
                .tools(["refund"])
                .reason("refunds need a human"),
        )
        .policy(Rule::deny("no-wires").tools(["wire"]))
        .policy(Rule::redact("mask-card", ["/card/number"]).tools(["get_*"]))
        .build();
    let (get, get_seen) = recorder("get_customer", false);
    let (refund, refund_seen) = recorder("refund", true);
    let (wire, wire_seen) = recorder("wire", true);
    let tools = ToolBox::new().with(get).with(refund).with(wire);
    let script = || {
        Arc::new(ScriptedModel::new([
            tools_msg(vec![
                call(
                    "g",
                    "get_customer",
                    json!({"id": 1, "card": {"number": "4111"}}),
                ),
                call("w", "wire", json!({"amount": 9})),
                call("r", "refund", json!({"amount": 5})),
            ]),
            Message::assistant("done"),
        ]))
    };
    let build = |model: Arc<ScriptedModel>| {
        ReactAgent::new(model, tools.clone())
            .gate(Arc::new(cp.gate(Identity::agent("billing").user("ana"))))
            .compile()
            .unwrap()
            .with_checkpointer(MemoryCheckpointer::new())
    };

    // Approve the refund.
    let agent = build(script());
    let out = agent
        .run("t1", Some(AgentState::new(vec![Message::user("go")])))
        .await
        .unwrap();
    let RunOutcome::Interrupted { interrupts, .. } = out else {
        panic!("expected an approval interrupt")
    };
    assert_eq!(
        interrupts[0].value["approval_required"],
        json!({"id": "r", "name": "refund", "arguments": {"amount": 5}, "reason": "refunds need a human"})
    );
    let RunOutcome::Completed(state) = agent.resume("t1", true).await.unwrap() else {
        panic!("expected completion")
    };
    let results: Vec<&str> = state.messages[2..5]
        .iter()
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(
        results,
        ["done", "error: denied: denied by policy `no-wires`", "done"]
    );
    assert_eq!(
        get_seen.lock().unwrap()[0],
        json!({"id": 1, "card": {"number": REDACTED}})
    );
    assert_eq!(refund_seen.lock().unwrap().len(), 1);
    assert!(wire_seen.lock().unwrap().is_empty());

    // Refuse it.
    let agent = build(script());
    agent
        .run("t2", Some(AgentState::new(vec![Message::user("go")])))
        .await
        .unwrap();
    let RunOutcome::Completed(state) = agent.resume("t2", false).await.unwrap() else {
        panic!("expected completion")
    };
    assert_eq!(
        state.messages[4].content,
        "error: denied: refunds need a human (not approved)"
    );
    assert_eq!(refund_seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn decisions_are_trace_runs_under_the_node_and_sink_events() {
    let sink = Arc::new(MemorySink::new());
    let tracer = Tracer::new().with_shared_sink(sink.clone());
    let audit = Arc::new(MemoryDecisionSink::new());
    let cp = ControlPlane::builder()
        .agent(AgentProfile::new("bot"))
        .policy(Rule::deny("no-delete").tools(["delete"]))
        .sink(audit.clone())
        .build();
    let tools = ToolBox::new()
        .with(FnTool::new("read", "", json!({}), |_| Ok(json!("text"))))
        .with(FnTool::new("delete", "", json!({}), |_| Ok(json!("gone"))));
    let model = Arc::new(ScriptedModel::new([
        tools_msg(vec![
            call("a", "read", json!({"path": "x"})),
            call("b", "delete", json!({"path": "x"})),
        ]),
        Message::assistant("ok"),
    ]));
    let agent = ReactAgent::new(model, tools)
        .gate(Arc::new(cp.gate("bot")))
        .compile()
        .unwrap()
        .with_tracer(tracer);
    let events: Vec<_> = agent
        .stream("t", Some(AgentState::new(vec![Message::user("go")])))
        .collect()
        .await;

    let runs = sink.runs();
    let policy_runs: Vec<_> = runs
        .iter()
        .filter(|r| r.name.starts_with("policy:"))
        .collect();
    let names: Vec<&str> = policy_runs.iter().map(|r| r.name.as_str()).collect();
    // Two model calls and two tool calls.
    assert_eq!(
        names,
        [
            "policy:scripted",
            "policy:read",
            "policy:delete",
            "policy:scripted"
        ]
    );
    let delete = policy_runs[2];
    assert_eq!(delete.kind, RunKind::Chain);
    assert_eq!(
        delete.error.as_deref(),
        Some("denied by policy `no-delete`")
    );
    assert_eq!(
        delete.metadata["governance"]["verdict"]["policy_id"],
        "no-delete"
    );
    let tools_node = runs.iter().find(|r| r.name == "tools").unwrap();
    assert_eq!(delete.parent_id.as_deref(), Some(tools_node.id.as_str()));
    assert_eq!(delete.inputs["thread_id"], "t");
    assert_eq!(delete.trace_id, tools_node.trace_id);
    assert!(policy_runs[1].error.is_none());

    let decisions = audit.decisions();
    assert_eq!(decisions.len(), 4);
    assert_eq!(decisions[2].request.tool.as_deref(), Some("delete"));
    assert!(decisions[2].verdict.decision.is_denied());
    // The gate also reports non-allow verdicts as custom graph events.
    let gate_events = events
        .iter()
        .filter(|e| matches!(e, GraphEvent::Custom { value, .. } if value.get("gate").is_some()))
        .count();
    assert_eq!(gate_events, 1);
}

#[tokio::test]
async fn registry_quarantine_disabled_tools_and_budgets() {
    let audit = Arc::new(MemoryDecisionSink::new());
    let cp = ControlPlane::builder()
        .agent(AgentProfile::new("a").budget(AgentBudget {
            max_tool_calls: Some(2),
            max_model_calls: None,
        }))
        .agent(AgentProfile::new("m").allow_models(["anthropic:*"]))
        .policy(Rule::allow("everything"))
        .sink(audit.clone())
        .build();
    let tool = |agent: &str, name: &str| Request::tool(agent, name, json!({}));

    let v = cp.check(tool("ghost", "read")).await;
    assert_eq!(v.policy_id.as_deref(), Some(POLICY_REGISTRY));
    assert_eq!(
        cp.stop_reason("ghost").as_deref(),
        Some("agent `ghost` is not registered")
    );

    cp.quarantine("a", "odd behavior");
    let v = cp.check(tool("a", "send").side_effects(true)).await;
    assert_eq!(v.policy_id.as_deref(), Some(POLICY_QUARANTINE));
    assert!(!v.stop);
    assert!(cp.check(tool("a", "read")).await.decision.is_allowed());
    cp.reinstate("a");

    cp.disable_tool("mcp_github_*", "token leaked");
    let v = cp.check(tool("a", "mcp_github_push")).await;
    assert_eq!(v.policy_id.as_deref(), Some(POLICY_KILL_SWITCH));
    assert_eq!(
        v.decision,
        Decision::deny("tool `mcp_github_push` is disabled: token leaked")
    );
    assert!(cp.enable_tool("mcp_github_*"));

    // One allowed tool call so far; spend the budget of two.
    assert_eq!(cp.usage("a").tool_calls, 1);
    assert!(cp.check(tool("a", "read")).await.decision.is_allowed());
    let v = cp.check(tool("a", "read")).await;
    assert_eq!(v.policy_id.as_deref(), Some(POLICY_BUDGET));
    cp.reset_usage("a");
    assert!(cp.check(tool("a", "read")).await.decision.is_allowed());

    // Allowed models.
    assert!(cp
        .check(Request::model("m", "anthropic:claude-sonnet-4-5"))
        .await
        .decision
        .is_allowed());
    assert!(cp
        .check(Request::model("m", "openai:gpt-4.1"))
        .await
        .decision
        .is_denied());

    let actions: Vec<ControlAction> = audit.controls().iter().map(|c| c.action).collect();
    assert_eq!(
        actions,
        [
            ControlAction::Quarantine,
            ControlAction::Reinstate,
            ControlAction::DisableTool,
            ControlAction::EnableTool
        ]
    );
}

#[tokio::test]
async fn governed_toolbox_and_model() {
    let cp = ControlPlane::builder()
        .agent(AgentProfile::new("bot").allow_models(["scripted"]))
        .policy(
            Rule::deny("cap")
                .tools(["pay"])
                .when("amount > 10".parse().unwrap()),
        )
        .policy(Rule::allow("pay").tools(["pay"]))
        .policy(Rule::ask("email").tools(["email"]))
        .build();
    let (pay, paid) = recorder("pay", true);
    let (email, _) = recorder("email", true);
    let tb = cp.governed_toolbox(&ToolBox::new().with(pay).with(email), "bot");
    assert!(tb.get("pay").unwrap().side_effects());

    let ok = tb.run(&call("1", "pay", json!({"amount": 5}))).await;
    assert_eq!(ok.content, "done");
    let no = tb.run(&call("2", "pay", json!({"amount": 50}))).await;
    assert_eq!(
        no.content,
        "error: tool `pay` failed: denied: denied by policy `cap` (policy `cap`)"
    );
    let ask = tb.run(&call("3", "email", json!({}))).await;
    assert!(ask.content.contains("needs approval"));
    assert_eq!(paid.lock().unwrap().len(), 1);

    let model = cp.govern_model(ScriptedModel::new([Message::assistant("hi")]), "bot");
    assert_eq!(model.name(), "scripted");
    model
        .chat(ChatRequest::new(vec![Message::user("x")]))
        .await
        .unwrap();
    cp.suspend("bot", "done for today");
    let err = model
        .chat(ChatRequest::new(vec![Message::user("x")]))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, Error::Guardrail { guard, reason, .. } if guard == "control_plane" && reason.contains("done for today")),
        "{err}"
    );
    let other = cp
        .govern_model(ScriptedModel::new([]), "bot")
        .as_model("openai:gpt-4.1");
    cp.reinstate("bot");
    assert!(other
        .chat(ChatRequest::new(vec![Message::user("x")]))
        .await
        .is_err());
}
