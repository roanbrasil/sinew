//! Per-tool permission policy: allow, ask (human approval) or deny.
//!
//! Rules are written like Claude Code permission rules: a tool-name pattern,
//! optionally followed by a subject pattern in parentheses.
//!
//! | Rule | Matches |
//! |---|---|
//! | `read_file` | every `read_file` call |
//! | `mcp_*` | every tool whose name starts with `mcp_` |
//! | `write_file(src/**)` | writes anywhere under `src/` |
//! | `shell(git *)` | shell commands starting with `git ` |
//! | `http_fetch(https://docs.rs/*)` | fetches of docs.rs pages |
//!
//! The subject of a call is its `path` / `file_path` argument (matched as a
//! path glob, see [`crate::glob`]), its `command` argument (wildcard), or its
//! `url` argument (wildcard). Deny rules win over ask rules, which win over
//! allow rules; calls no rule matches get the policy default.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinew_core::ToolCall;

use crate::glob::{glob_match, wildcard_match};

/// What to do with a tool call. Ordered from least to most restrictive, so
/// `a.max(b)` combines two decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    Allow,
    Ask,
    Deny,
}

/// How a subject argument is matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubjectKind {
    Path,
    Command,
    Url,
}

/// The argument a rule's subject pattern is matched against.
fn subject(call: &ToolCall) -> Option<(SubjectKind, String)> {
    let get = |k: &str| call.arguments.get(k).and_then(Value::as_str);
    if let Some(p) = get("path").or_else(|| get("file_path")) {
        return Some((SubjectKind::Path, p.to_string()));
    }
    if let Some(c) = get("command") {
        return Some((SubjectKind::Command, c.trim().to_string()));
    }
    get("url").map(|u| (SubjectKind::Url, u.to_string()))
}

/// One rule: `tool` or `tool(subject)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionRule {
    tool: String,
    subject: Option<String>,
    action: Permission,
}

impl PermissionRule {
    /// Parses `tool` or `tool(subject)`.
    pub fn parse(spec: &str, action: Permission) -> Self {
        let spec = spec.trim();
        if let Some(open) = spec.find('(') {
            if spec.ends_with(')') {
                return Self {
                    tool: spec[..open].trim().to_string(),
                    subject: Some(spec[open + 1..spec.len() - 1].to_string()),
                    action,
                };
            }
        }
        Self {
            tool: spec.to_string(),
            subject: None,
            action,
        }
    }

    pub fn action(&self) -> Permission {
        self.action
    }

    pub fn matches(&self, call: &ToolCall) -> bool {
        if !wildcard_match(&self.tool, &call.name) {
            return false;
        }
        let Some(pattern) = &self.subject else {
            return true;
        };
        match subject(call) {
            Some((SubjectKind::Path, p)) => glob_match(pattern, &p),
            Some((SubjectKind::Command | SubjectKind::Url, s)) => wildcard_match(pattern, &s),
            None => false,
        }
    }
}

/// Ordered set of rules plus a default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionPolicy {
    rules: Vec<PermissionRule>,
    default: Permission,
}

impl Default for PermissionPolicy {
    /// Everything allowed (library default; the CLI is stricter).
    fn default() -> Self {
        Self::allow_all()
    }
}

impl PermissionPolicy {
    pub fn new(default: Permission) -> Self {
        Self {
            rules: Vec::new(),
            default,
        }
    }

    pub fn allow_all() -> Self {
        Self::new(Permission::Allow)
    }

    pub fn ask_all() -> Self {
        Self::new(Permission::Ask)
    }

    /// Allows read-only tools and asks before writes, shell commands and
    /// network access. This is what `sinew-agent` uses without `--yes`.
    pub fn interactive() -> Self {
        Self::allow_all()
            .ask("write_file")
            .ask("edit_file")
            .ask("shell")
            .ask("http_fetch")
    }

    pub fn with_default(mut self, default: Permission) -> Self {
        self.default = default;
        self
    }

    pub fn rule(mut self, spec: &str, action: Permission) -> Self {
        self.rules.push(PermissionRule::parse(spec, action));
        self
    }

    pub fn allow(self, spec: &str) -> Self {
        self.rule(spec, Permission::Allow)
    }

    pub fn ask(self, spec: &str) -> Self {
        self.rule(spec, Permission::Ask)
    }

    pub fn deny(self, spec: &str) -> Self {
        self.rule(spec, Permission::Deny)
    }

    pub fn default_permission(&self) -> Permission {
        self.default
    }

    pub fn rules(&self) -> &[PermissionRule] {
        &self.rules
    }

    /// Decision for one call: deny rules first, then ask, then allow, then
    /// the default.
    pub fn check(&self, call: &ToolCall) -> Permission {
        let hit = |a: Permission| self.rules.iter().any(|r| r.action == a && r.matches(call));
        if hit(Permission::Deny) {
            Permission::Deny
        } else if hit(Permission::Ask) {
            Permission::Ask
        } else if hit(Permission::Allow) {
            Permission::Allow
        } else {
            self.default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: "c".into(),
            name: name.into(),
            arguments: args,
        }
    }

    #[test]
    fn rules_by_tool_and_subject() {
        let p = PermissionPolicy::ask_all()
            .allow("read_file")
            .allow("write_file(notes/**)")
            .allow("shell(git *)")
            .deny("shell(git push*)")
            .deny("write_file(**/.env)");
        assert_eq!(
            p.check(&call("read_file", json!({"path": "x"}))),
            Permission::Allow
        );
        assert_eq!(
            p.check(&call("write_file", json!({"path": "notes/a/b.md"}))),
            Permission::Allow
        );
        assert_eq!(
            p.check(&call("write_file", json!({"path": "src/main.rs"}))),
            Permission::Ask
        );
        assert_eq!(
            p.check(&call("write_file", json!({"path": "app/.env"}))),
            Permission::Deny
        );
        assert_eq!(
            p.check(&call("shell", json!({"command": "git status"}))),
            Permission::Allow
        );
        assert_eq!(
            p.check(&call("shell", json!({"command": "git push origin"}))),
            Permission::Deny
        );
        assert_eq!(
            p.check(&call("shell", json!({"command": "ls"}))),
            Permission::Ask
        );
    }

    #[test]
    fn interactive_policy() {
        let p = PermissionPolicy::interactive();
        assert_eq!(p.check(&call("grep", json!({}))), Permission::Allow);
        assert_eq!(p.check(&call("edit_file", json!({}))), Permission::Ask);
        assert!(Permission::Deny > Permission::Ask && Permission::Ask > Permission::Allow);
    }
}
