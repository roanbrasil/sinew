//! Attack scenarios for the OWASP Top 10 for Agentic Applications (2026),
//! run against Sinew agents with and without the controls mapped to each
//! risk.
//!
//! [`suite`] returns one [`SecuritySuite`] with at least one scenario per
//! risk. Every scenario builds a real Sinew agent (`ReactAgent`, the harness
//! `Agent`, `MemoryAgent`, an MCP client) with tools that record their side
//! effects, and a scripted model that falls for the attack. The suite runs
//! each scenario twice: unprotected (the attack must succeed) and protected
//! (the attack must fail). Scenarios for which Sinew has no adequate control
//! yet are declared gaps and link a follow-up issue.
//!
//! See `docs/security-owasp.md` for the controls map.

pub mod kit;
pub mod scenarios;

use sinew::eval::security::SecuritySuite;

/// Follow-up issue: validate long-term memory writes (ASI06).
pub const ISSUE_MEMORY_WRITES: &str = "https://github.com/roanbrasil/sinew/issues/187";
/// Follow-up issue: authenticate messages between agents (ASI07).
pub const ISSUE_AGENT_MESSAGES: &str = "https://github.com/roanbrasil/sinew/issues/188";

/// Every scenario, in OWASP list order.
pub fn suite() -> SecuritySuite {
    use scenarios::*;
    SecuritySuite::new("sinew-owasp-agentic").scenarios([
        goal_hijack::tool_output_injection(),
        goal_hijack::rag_document_injection(),
        goal_hijack::markdown_image_exfiltration(),
        tool_misuse::destructive_statement(),
        tool_misuse::ssrf_internal_endpoint(),
        privilege::confused_deputy(),
        privilege::credential_disclosure(),
        supply_chain::mcp_tool_poisoning(),
        code_execution::shell_from_readme(),
        memory::thread_state_poisoning(),
        memory::long_term_memory_poisoning(),
        inter_agent::spoofed_peer_message(),
        cascading::unbounded_tool_loop(),
        cascading::duplicate_side_effect_after_crash(),
        trust::misleading_approval(),
        rogue::disables_oversight(),
    ])
}
