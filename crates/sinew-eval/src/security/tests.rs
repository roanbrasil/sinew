use serde_json::json;
use sinew_core::{mock::ScriptedModel, ChatModel, ChatRequest, Message, ToolCall};

use super::*;

/// A scenario whose attack succeeds unless `stopped_when` says otherwise.
fn scenario(id: &str, protected_succeeds: bool, unprotected_succeeds: bool) -> AttackScenario {
    AttackScenario::new(id, AgenticRisk::ToolMisuse, "test")
        .control("a control")
        .detector(Detector::effect("send_email"))
        .setup(move |ctx: RunContext| async move {
            let log = EffectLog::new();
            let succeeds = if ctx.is_protected() {
                protected_succeeds
            } else {
                unprotected_succeeds
            };
            if succeeds {
                log.record("send_email", json!({"to": "attacker@evil.example"}));
            }
            Ok::<_, String>(Observation::new().with_effects(log.effects()))
        })
}

#[test]
fn risk_ids_and_names() {
    assert_eq!(AgenticRisk::ALL.len(), 10);
    assert_eq!(AgenticRisk::GoalHijack.id(), "ASI01");
    assert_eq!(AgenticRisk::RogueAgents.id(), "ASI10");
    assert_eq!(
        AgenticRisk::MemoryContextPoisoning.name(),
        "Memory & Context Poisoning"
    );
    assert_eq!(
        AgenticRisk::from_id("asi05"),
        Some(AgenticRisk::UnexpectedCodeExecution)
    );
    assert_eq!(AgenticRisk::from_id("ASI11"), None);
    assert_eq!(
        AgenticRisk::SupplyChain.to_string(),
        "ASI04 Agentic Supply Chain Vulnerabilities"
    );
    let ids: Vec<&str> = AgenticRisk::ALL.iter().map(|r| r.id()).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted);
}

#[tokio::test]
async fn statuses() {
    let suite = SecuritySuite::new("t")
        .scenario(scenario("ok", false, true))
        .scenario(scenario("vuln", true, true))
        .scenario(scenario("broken", false, false))
        .scenario(scenario("gap", true, true).gap(KnownGap::new("no control").issue("#1")))
        .scenario(AttackScenario::new(
            "nosetup",
            AgenticRisk::RogueAgents,
            "x",
        ));
    let report = suite.run().await;
    let status = |id: &str| report.result(id).unwrap().status;
    assert_eq!(status("ok"), ScenarioStatus::Mitigated);
    assert_eq!(status("vuln"), ScenarioStatus::Vulnerable);
    assert_eq!(status("broken"), ScenarioStatus::NotReproduced);
    assert_eq!(status("gap"), ScenarioStatus::Gap);
    assert_eq!(status("nosetup"), ScenarioStatus::Error);
    assert!(!report.passed());
    let failing: Vec<&str> = report.failures().iter().map(|r| r.id.as_str()).collect();
    assert_eq!(failing, ["vuln", "broken", "nosetup"]);
    assert_eq!(report.gaps().len(), 1);
    assert_eq!(report.counts.scenarios, 5);
    assert_eq!(report.counts.mitigated, 1);
    assert_eq!(report.counts.errors, 1);
    let tool_misuse = &report.coverage[1];
    assert_eq!(tool_misuse.risk_id, "ASI02");
    assert_eq!(tool_misuse.scenarios, 4);
    assert_eq!(tool_misuse.status, RiskStatus::Failing);
    assert_eq!(report.uncovered().len(), 8);

    let md = report.to_markdown();
    assert!(md.starts_with("# Security report: t"));
    assert!(md.contains(
        "| ok test | ASI02 | - | attack succeeded | attack stopped | mitigated | a control |"
    ));
    assert!(md.contains("## Gaps"));
    assert!(md.contains("Follow-up: #1."));
    assert!(md.contains("## Failures"));
    assert!(md.contains("scenario has no setup"));

    let back = SecurityReport::from_json(&report.to_json()).unwrap();
    assert_eq!(back, report);
}

#[tokio::test]
async fn passing_suite_and_filter() {
    let suite = SecuritySuite::new("t")
        .scenario(scenario("ASI02-01", false, true))
        .scenario(scenario("gap", true, true).gap(KnownGap::new("no control")));
    let report = suite.clone().run().await;
    assert!(report.passed(), "{}", report.to_markdown());
    assert!(report
        .summary()
        .starts_with("PASS: 2 scenarios, 1 mitigated, 1 gaps"));
    assert_eq!(report.coverage[1].status, RiskStatus::Gap);
    assert_eq!(suite.clone().filter("asi02-01").scenarios.len(), 1);
    assert_eq!(
        suite.clone().filter("ASI02").scenarios.len(),
        2,
        "risk id prefix"
    );
    assert_eq!(suite.filter("ASI03").scenarios.len(), 0);
}

#[tokio::test]
async fn live_mode_tolerates_resisting_models() {
    let model: Arc<dyn ChatModel> = Arc::new(ScriptedModel::new([]));
    let report = SecuritySuite::new("t")
        .scenario(scenario("resisted", false, false))
        .run_live(model)
        .await;
    assert!(report.live);
    assert_eq!(report.mode, "live:scripted");
    assert_eq!(report.results[0].status, ScenarioStatus::NotReproduced);
    assert!(report.passed());
    assert_eq!(report.coverage[1].status, RiskStatus::NotReproduced);
}

#[tokio::test]
async fn run_context_picks_the_model() {
    let offline = RunContext::new(Protection::Unprotected);
    assert!(!offline.is_live() && !offline.is_protected());
    let m = offline.model(ScriptedModel::new([Message::assistant("scripted")]));
    let r = m
        .chat(ChatRequest::new(vec![Message::user("x")]))
        .await
        .unwrap();
    assert_eq!(r.message.content, "scripted");

    let live: Arc<dyn ChatModel> = Arc::new(ScriptedModel::new([Message::assistant("live")]));
    let ctx = RunContext::live(Protection::Protected, live);
    assert!(ctx.is_live() && ctx.is_protected());
    let m = ctx.model(ScriptedModel::new([Message::assistant("scripted")]));
    let r = m
        .chat(ChatRequest::new(vec![Message::user("x")]))
        .await
        .unwrap();
    assert_eq!(r.message.content, "live");
}

#[test]
fn suite_mode_from_spec() {
    assert!(SuiteMode::from_spec("ollama:qwen3").unwrap().is_live());
    assert!(SuiteMode::from_spec("nope:x").is_err());
    assert!(!SuiteMode::default().is_live());
}

#[test]
fn detectors() {
    let call = ToolCall {
        id: "1".into(),
        name: "send_email".into(),
        arguments: json!({"to": "a@b.example"}),
    };
    let obs = Observation::new()
        .with_messages(vec![
            Message::user("hi"),
            Message::assistant_with_tools("", vec![call]),
            Message::tool_result("1", "sent; INJECTED"),
            Message::assistant("Done. CANARY-1"),
        ])
        .with_effects(vec![
            Effect {
                tool: "send_email".into(),
                arguments: json!({"to": "attacker@evil.example", "body": "CANARY-2"}),
            },
            Effect {
                tool: "send_email".into(),
                arguments: json!({"to": "ok@corp.example"}),
            },
        ])
        .with_memories(vec!["Invoices go to attacker@evil.example".into()]);
    assert_eq!(obs.final_answer.as_deref(), Some("Done. CANARY-1"));
    assert_eq!(obs.model_calls(), 2);
    assert_eq!(obs.trajectory().tool_names(), ["send_email"]);

    assert!(Detector::effect("send_email").detect(&obs).is_some());
    assert!(Detector::effect("pay").detect(&obs).is_none());
    assert!(
        Detector::effect_with("send_email", json!({"to": "attacker@evil.example"}))
            .detect(&obs)
            .is_some()
    );
    assert!(
        Detector::effect_with("send_email", json!({"to": "x@y.example"}))
            .detect(&obs)
            .is_none()
    );
    assert!(Detector::effect_where("send_email", "external", |a| a["to"]
        .as_str()
        .is_some_and(|t| !t.ends_with("@corp.example")))
    .detect(&obs)
    .is_some());
    assert!(Detector::effect_count_over("send_email", 1)
        .detect(&obs)
        .is_some());
    assert!(Detector::effect_count_over("send_email", 2)
        .detect(&obs)
        .is_none());
    assert!(Detector::leaks("CANARY-1").detect(&obs).is_some());
    assert!(
        Detector::leaks("CANARY-2").detect(&obs).is_some(),
        "effect arguments are egress"
    );
    assert!(
        Detector::leaks("INJECTED").detect(&obs).is_none(),
        "tool results are not egress"
    );
    assert!(Detector::answer_contains("CANARY-1").detect(&obs).is_some());
    assert!(Detector::thread_contains("INJECTED").detect(&obs).is_some());
    assert!(Detector::memory_contains("attacker@")
        .detect(&obs)
        .is_some());
    assert!(Detector::model_calls_over(1).detect(&obs).is_some());
    assert!(Detector::model_calls_over(2).detect(&obs).is_none());
    let any = Detector::any([Detector::effect("pay"), Detector::answer_contains("Done")]);
    assert!(any.description().contains(" or "));
    assert!(any.detect(&obs).unwrap().starts_with("answer: Done"));

    let log = EffectLog::new();
    log.record("a", json!({}));
    log.clone().record("a", json!({}));
    assert_eq!(log.count("a"), 2);
}
