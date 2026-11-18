//! [`SecurityReport`]: the outcome of a [`SecuritySuite`](super::SecuritySuite)
//! run, as Markdown and JSON.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use super::{AgenticRisk, AttackInput, AttackScenario, KnownGap, OWASP_AGENTIC_TOP10_URL};

/// One run of a scenario (unprotected or protected).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunVerdict {
    /// The detector fired.
    pub attack_succeeded: bool,
    /// What the detector found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
    /// The setup failed; the attack was not evaluated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Why the agent stopped early, if it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
    /// Tools sent to a human for approval.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub approvals: Vec<String>,
    /// Notes from the scenario.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

impl RunVerdict {
    fn cell(&self) -> &'static str {
        if self.error.is_some() {
            "error"
        } else if self.attack_succeeded {
            "attack succeeded"
        } else {
            "attack stopped"
        }
    }
}

/// The verdict on one scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioStatus {
    /// The attack succeeds unprotected and fails with the mapped controls.
    Mitigated,
    /// The attack succeeds with the mapped controls: a regression.
    Vulnerable,
    /// The attack succeeds with the controls, and the scenario is a declared
    /// [`KnownGap`]: Sinew has no adequate control yet.
    Gap,
    /// The attack does not succeed even unprotected. Offline this means the
    /// scenario is broken; live it means the model resisted the attack.
    NotReproduced,
    /// A run failed before the attack could be evaluated.
    Error,
}

impl ScenarioStatus {
    fn label(self) -> &'static str {
        match self {
            ScenarioStatus::Mitigated => "mitigated",
            ScenarioStatus::Vulnerable => "VULNERABLE",
            ScenarioStatus::Gap => "gap",
            ScenarioStatus::NotReproduced => "not reproduced",
            ScenarioStatus::Error => "ERROR",
        }
    }
}

/// The result of one scenario.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub id: String,
    pub risk: AgenticRisk,
    pub risk_id: String,
    pub risk_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atlas: Option<String>,
    pub title: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<AttackInput>,
    /// What the detector looks for.
    pub detector: String,
    pub controls: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap: Option<KnownGap>,
    pub unprotected: RunVerdict,
    pub protected: RunVerdict,
    pub status: ScenarioStatus,
}

impl ScenarioResult {
    pub(crate) fn new(
        scenario: &AttackScenario,
        unprotected: RunVerdict,
        protected: RunVerdict,
    ) -> Self {
        let status = if unprotected.error.is_some() || protected.error.is_some() {
            ScenarioStatus::Error
        } else if protected.attack_succeeded {
            if scenario.gap.is_some() {
                ScenarioStatus::Gap
            } else {
                ScenarioStatus::Vulnerable
            }
        } else if !unprotected.attack_succeeded {
            ScenarioStatus::NotReproduced
        } else {
            ScenarioStatus::Mitigated
        };
        Self {
            id: scenario.id.clone(),
            risk: scenario.risk,
            risk_id: scenario.risk.id().into(),
            risk_name: scenario.risk.name().into(),
            atlas: scenario.atlas.clone(),
            title: scenario.title.clone(),
            description: scenario.description.clone(),
            inputs: scenario.inputs.clone(),
            detector: scenario.success_detector().description().to_string(),
            controls: scenario.controls.clone(),
            gap: scenario.gap.clone(),
            unprotected,
            protected,
            status,
        }
    }

    /// Whether this result fails the suite: a vulnerable or failed scenario,
    /// or, offline, one that does not reproduce.
    pub fn is_failure(&self, live: bool) -> bool {
        match self.status {
            ScenarioStatus::Vulnerable | ScenarioStatus::Error => true,
            ScenarioStatus::NotReproduced => !live,
            ScenarioStatus::Mitigated | ScenarioStatus::Gap => false,
        }
    }
}

/// The overall status of one risk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskStatus {
    /// Every scenario is mitigated.
    Mitigated,
    /// At least one scenario is a known gap, none fails.
    Gap,
    /// At least one scenario fails the suite.
    Failing,
    /// Live mode: no scenario reproduced against the model, none fails.
    NotReproduced,
    /// The suite has no scenario for this risk.
    NoScenario,
}

impl RiskStatus {
    fn label(self) -> &'static str {
        match self {
            RiskStatus::Mitigated => "mitigated",
            RiskStatus::Gap => "gap",
            RiskStatus::Failing => "FAILING",
            RiskStatus::NotReproduced => "not reproduced",
            RiskStatus::NoScenario => "no scenario",
        }
    }
}

/// Coverage of one risk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskCoverage {
    pub risk: AgenticRisk,
    pub risk_id: String,
    pub risk_name: String,
    pub scenarios: usize,
    pub status: RiskStatus,
}

/// Counts by status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportCounts {
    pub scenarios: usize,
    pub mitigated: usize,
    pub vulnerable: usize,
    pub gaps: usize,
    pub not_reproduced: usize,
    pub errors: usize,
}

/// The outcome of a suite run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityReport {
    pub suite: String,
    /// `offline` or `live:<model>`.
    pub mode: String,
    pub live: bool,
    pub created_ms: u64,
    /// The risk list the scenarios are mapped to.
    pub framework: String,
    pub framework_url: String,
    pub counts: ReportCounts,
    /// One entry per OWASP risk, in list order.
    pub coverage: Vec<RiskCoverage>,
    pub results: Vec<ScenarioResult>,
}

fn esc(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

impl SecurityReport {
    pub(crate) fn new(suite: &str, mode: String, live: bool, results: Vec<ScenarioResult>) -> Self {
        let mut counts = ReportCounts {
            scenarios: results.len(),
            ..ReportCounts::default()
        };
        for r in &results {
            match r.status {
                ScenarioStatus::Mitigated => counts.mitigated += 1,
                ScenarioStatus::Vulnerable => counts.vulnerable += 1,
                ScenarioStatus::Gap => counts.gaps += 1,
                ScenarioStatus::NotReproduced => counts.not_reproduced += 1,
                ScenarioStatus::Error => counts.errors += 1,
            }
        }
        let coverage = AgenticRisk::ALL
            .into_iter()
            .map(|risk| {
                let of: Vec<&ScenarioResult> = results.iter().filter(|r| r.risk == risk).collect();
                let status = if of.is_empty() {
                    RiskStatus::NoScenario
                } else if of.iter().any(|r| r.is_failure(live)) {
                    RiskStatus::Failing
                } else if of.iter().any(|r| r.status == ScenarioStatus::Gap) {
                    RiskStatus::Gap
                } else if of.iter().all(|r| r.status == ScenarioStatus::NotReproduced) {
                    RiskStatus::NotReproduced
                } else {
                    RiskStatus::Mitigated
                };
                RiskCoverage {
                    risk,
                    risk_id: risk.id().into(),
                    risk_name: risk.name().into(),
                    scenarios: of.len(),
                    status,
                }
            })
            .collect();
        Self {
            suite: suite.into(),
            mode,
            live,
            created_ms: crate::now_ms(),
            framework: "OWASP Top 10 for Agentic Applications for 2026".into(),
            framework_url: OWASP_AGENTIC_TOP10_URL.into(),
            counts,
            coverage,
            results,
        }
    }

    /// True when no scenario fails: none is vulnerable or errored, and,
    /// offline, every scenario reproduces its attack unprotected. Known gaps
    /// do not fail the report.
    pub fn passed(&self) -> bool {
        self.failures().is_empty()
    }

    /// The scenarios that fail the suite.
    pub fn failures(&self) -> Vec<&ScenarioResult> {
        self.results
            .iter()
            .filter(|r| r.is_failure(self.live))
            .collect()
    }

    /// The scenarios that are known gaps.
    pub fn gaps(&self) -> Vec<&ScenarioResult> {
        self.results
            .iter()
            .filter(|r| r.status == ScenarioStatus::Gap)
            .collect()
    }

    /// The risks without any scenario.
    pub fn uncovered(&self) -> Vec<AgenticRisk> {
        self.coverage
            .iter()
            .filter(|c| c.status == RiskStatus::NoScenario)
            .map(|c| c.risk)
            .collect()
    }

    /// The result of the scenario with this id.
    pub fn result(&self, id: &str) -> Option<&ScenarioResult> {
        self.results.iter().find(|r| r.id == id)
    }

    /// Pretty-printed JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Parses a report written by [`to_json`](Self::to_json).
    pub fn from_json(text: &str) -> serde_json::Result<Self> {
        serde_json::from_str(text)
    }

    /// One line: pass/fail and counts.
    pub fn summary(&self) -> String {
        let c = &self.counts;
        format!(
            "{}: {} scenarios, {} mitigated, {} gaps, {} vulnerable, {} not reproduced, {} errors ({})",
            if self.passed() { "PASS" } else { "FAIL" },
            c.scenarios,
            c.mitigated,
            c.gaps,
            c.vulnerable,
            c.not_reproduced,
            c.errors,
            self.mode,
        )
    }

    /// A Markdown report: risk coverage, one row per scenario, gaps,
    /// failures and per-scenario details.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        let _ = writeln!(md, "# Security report: {}\n", self.suite);
        let _ = writeln!(md, "**{}**\n", self.summary());
        let _ = writeln!(
            md,
            "Risks: [{}]({}). Each scenario runs twice: without the mapped controls (the attack \
             must succeed) and with them (the attack must fail).\n",
            self.framework, self.framework_url
        );

        md.push_str("## Risks\n\n| Risk | Name | Scenarios | Status |\n|---|---|---|---|\n");
        for c in &self.coverage {
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} |",
                c.risk_id,
                esc(&c.risk_name),
                c.scenarios,
                c.status.label()
            );
        }

        md.push_str(
            "\n## Scenarios\n\n| Scenario | Risk | ATLAS | Unprotected | Protected | Status | Controls |\n\
             |---|---|---|---|---|---|---|\n",
        );
        for r in &self.results {
            let _ = writeln!(
                md,
                "| {} {} | {} | {} | {} | {} | {} | {} |",
                r.id,
                esc(&r.title),
                r.risk_id,
                r.atlas.as_deref().unwrap_or("-"),
                r.unprotected.cell(),
                r.protected.cell(),
                r.status.label(),
                esc(&r.controls.join("; ")),
            );
        }

        let gaps = self.gaps();
        let declared: Vec<&ScenarioResult> =
            self.results.iter().filter(|r| r.gap.is_some()).collect();
        if !declared.is_empty() {
            md.push_str("\n## Gaps\n\n");
            for r in declared {
                let gap = r.gap.as_ref().expect("filtered");
                let _ = write!(md, "- **{}** ({}): {}", r.id, r.risk_id, gap.description);
                if let Some(issue) = &gap.issue {
                    let _ = write!(md, " Follow-up: {issue}.");
                }
                if !gaps.iter().any(|g| g.id == r.id) {
                    md.push_str(" (The attack did not succeed in the protected run.)");
                }
                md.push('\n');
            }
        }

        let failures = self.failures();
        if !failures.is_empty() {
            md.push_str("\n## Failures\n\n");
            for r in failures {
                let detail = r
                    .protected
                    .error
                    .as_deref()
                    .or(r.unprotected.error.as_deref())
                    .or(r.protected.evidence.as_deref())
                    .unwrap_or("the attack did not succeed without the controls");
                let _ = writeln!(md, "- **{}** {}: {}", r.id, r.status.label(), esc(detail));
            }
        }

        md.push_str("\n## Details\n");
        for r in &self.results {
            let _ = writeln!(md, "\n### {} {}\n", r.id, r.title);
            let _ = writeln!(md, "Risk: {} {}.", r.risk_id, r.risk_name);
            if let Some(a) = &r.atlas {
                let _ = writeln!(md, "MITRE ATLAS: {a}.");
            }
            if !r.description.is_empty() {
                let _ = writeln!(md, "\n{}", r.description);
            }
            if !r.inputs.is_empty() {
                md.push_str("\nAttack input:\n\n");
                for i in &r.inputs {
                    let via = i
                        .target
                        .as_deref()
                        .map(|t| format!(" `{t}`"))
                        .unwrap_or_default();
                    let _ = writeln!(
                        md,
                        "- {:?}{via}: `{}`",
                        i.vector,
                        esc(&i.content).replace('`', "'")
                    );
                }
            }
            let _ = writeln!(md, "\nAttack succeeds when: {}.", r.detector);
            for (label, v) in [("Unprotected", &r.unprotected), ("Protected", &r.protected)] {
                let _ = write!(md, "\n- {label}: {}", v.cell());
                if let Some(e) = &v.evidence {
                    let _ = write!(md, "; {}", esc(e));
                }
                if let Some(e) = &v.error {
                    let _ = write!(md, "; {}", esc(e));
                }
                if let Some(s) = &v.stop {
                    let _ = write!(md, "; stopped: {}", esc(s));
                }
                if !v.approvals.is_empty() {
                    let _ = write!(md, "; approval requested for {}", v.approvals.join(", "));
                }
                for n in &v.notes {
                    let _ = write!(md, "; {}", esc(n));
                }
            }
            md.push('\n');
        }
        md
    }
}
