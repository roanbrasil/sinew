//! Runs every OWASP agentic scenario twice, offline: without the mapped
//! controls the attack must succeed, with them it must fail (or the
//! scenario must be a declared gap with a follow-up issue). A regression in
//! any Sinew control fails this test in CI.

use sample_23_owasp_agentic::suite;
use sinew::eval::security::{run_scenario, AgenticRisk, RiskStatus, ScenarioStatus, SuiteMode};

async fn check(id: &str, expected: ScenarioStatus) {
    let suite = suite();
    let scenario = suite.get(id).unwrap_or_else(|| panic!("no scenario {id}"));
    let result = run_scenario(scenario, &SuiteMode::Offline).await;
    assert!(
        result.unprotected.attack_succeeded,
        "{id}: the attack must succeed without controls: {:?}",
        result.unprotected
    );
    assert_eq!(result.status, expected, "{id}: {:#?}", result.protected);
    if expected == ScenarioStatus::Gap {
        let issue = result
            .gap
            .as_ref()
            .and_then(|g| g.issue.as_deref())
            .unwrap_or_default();
        assert!(
            issue.starts_with("https://github.com/roanbrasil/sinew/issues/")
                && issue
                    .rsplit('/')
                    .next()
                    .is_some_and(|n| n.parse::<u32>().is_ok()),
            "{id}: a gap links its follow-up issue, got `{issue}`"
        );
    }
}

macro_rules! scenario_tests {
    ($($name:ident: $id:literal => $status:ident,)*) => {$(
        #[tokio::test]
        async fn $name() {
            check($id, ScenarioStatus::$status).await;
        }
    )*};
}

scenario_tests! {
    asi01_01_tool_output_injection: "ASI01-01" => Mitigated,
    asi01_02_rag_document_injection: "ASI01-02" => Mitigated,
    asi01_03_markdown_image_exfiltration: "ASI01-03" => Mitigated,
    asi02_01_destructive_statement: "ASI02-01" => Mitigated,
    asi02_02_ssrf_internal_endpoint: "ASI02-02" => Mitigated,
    asi03_01_confused_deputy: "ASI03-01" => Mitigated,
    asi03_02_credential_disclosure: "ASI03-02" => Mitigated,
    asi04_01_mcp_tool_poisoning: "ASI04-01" => Gap,
    asi05_01_shell_from_readme: "ASI05-01" => Mitigated,
    asi06_01_thread_state_poisoning: "ASI06-01" => Mitigated,
    asi06_02_long_term_memory_poisoning: "ASI06-02" => Gap,
    asi07_01_spoofed_peer_message: "ASI07-01" => Gap,
    asi08_01_unbounded_tool_loop: "ASI08-01" => Mitigated,
    asi08_02_duplicate_side_effect_after_crash: "ASI08-02" => Mitigated,
    asi09_01_misleading_approval: "ASI09-01" => Gap,
    asi10_01_disables_oversight: "ASI10-01" => Mitigated,
}

#[tokio::test]
async fn full_suite_covers_every_risk_and_passes() {
    let report = suite().run().await;
    assert!(report.passed(), "{}", report.to_markdown());
    assert!(
        report.uncovered().is_empty(),
        "uncovered: {:?}",
        report.uncovered()
    );
    assert_eq!(report.coverage.len(), AgenticRisk::ALL.len());
    assert!(report.coverage.iter().all(|c| c.scenarios >= 1));
    assert!(report
        .coverage
        .iter()
        .all(|c| matches!(c.status, RiskStatus::Mitigated | RiskStatus::Gap)));
    assert_eq!(report.counts.scenarios, 16);

    let md = report.to_markdown();
    assert!(md.contains("| ASI01 | Agent Goal Hijack | 3 | mitigated |"));
    let json = report.to_json();
    let back = sinew::eval::security::SecurityReport::from_json(&json).unwrap();
    assert_eq!(back.results.len(), 16);
}
