//! Glob patterns for agent ids, users, tenants, tools and models.

/// Whether `text` matches `pattern`, where `*` matches any run of
/// characters (including none) and `?` matches one character. Everything
/// else matches itself, so `mcp_*`, `send_?mail` and `anthropic:*` work as
/// expected; `*` alone matches everything.
///
/// ```
/// use sinew_govern::glob_match;
/// assert!(glob_match("mcp_*", "mcp_search"));
/// assert!(glob_match("anthropic:claude-*", "anthropic:claude-sonnet-4-5"));
/// assert!(!glob_match("get_*", "delete_order"));
/// ```
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0, 0);
    // Position of the last `*` and the text index it was tried at.
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some((pi, ti));
            pi += 1;
        } else if let Some((sp, st)) = star {
            pi = sp + 1;
            ti = st + 1;
            star = Some((sp, st + 1));
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|c| *c == '*')
}

/// Whether any pattern matches. An empty list matches everything.
pub(crate) fn any_match(patterns: &[String], text: Option<&str>) -> bool {
    if patterns.is_empty() {
        return true;
    }
    text.is_some_and(|t| patterns.iter().any(|p| glob_match(p, t)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob_match("*", ""));
        assert!(glob_match("*", "anything at all"));
        assert!(glob_match("refund", "refund"));
        assert!(!glob_match("refund", "refunds"));
        assert!(glob_match("re*d", "refund"));
        assert!(glob_match("*_order", "get_order"));
        assert!(glob_match("a*b*c", "aXXbYYbZc"));
        assert!(!glob_match("a*b*c", "aXXbYY"));
        assert!(glob_match("send_?mail", "send_email"));
        assert!(!glob_match("send_?mail", "send_mail"));
        assert!(glob_match("ten*nt-?", "tenant-1"));
        assert!(any_match(&[], None));
        assert!(!any_match(&["a".into()], None));
        assert!(any_match(&["x".into(), "a*".into()], Some("ab")));
    }
}
