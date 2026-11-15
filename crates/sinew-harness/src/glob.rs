//! Glob patterns for paths and wildcard patterns for commands.
//!
//! Both are compiled to [`regex::Regex`]:
//!
//! - **Path globs** ([`glob_match`]): `*` matches within one path segment,
//!   `**` matches across segments (`src/**/*.rs` also matches `src/a.rs`),
//!   `?` matches one character other than `/`, `{a,b}` is an alternative.
//! - **Wildcards** ([`wildcard_match`]): `*` matches anything (spaces and `/`
//!   included), used for tool names and shell commands such as `git *`.

use regex::Regex;

fn escape_char(c: char, out: &mut String) {
    if c.is_ascii_alphanumeric() || c == '_' || c == ' ' {
        out.push(c);
    } else {
        out.push_str(&regex::escape(&c.to_string()));
    }
}

/// Translates a path glob into an anchored regex source.
pub fn glob_to_regex(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::from("^");
    let mut i = 0;
    let mut in_brace = false;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '*' if chars.get(i + 1) == Some(&'*') => {
                if chars.get(i + 2) == Some(&'/') {
                    out.push_str("(?:.*/)?");
                    i += 3;
                } else {
                    out.push_str(".*");
                    i += 2;
                }
                continue;
            }
            '*' => out.push_str("[^/]*"),
            '?' => out.push_str("[^/]"),
            '{' => {
                in_brace = true;
                out.push_str("(?:");
            }
            '}' if in_brace => {
                in_brace = false;
                out.push(')');
            }
            ',' if in_brace => out.push('|'),
            c => escape_char(c, &mut out),
        }
        i += 1;
    }
    if in_brace {
        out.push(')');
    }
    out.push('$');
    out
}

/// Whether `path` matches the path glob `pattern`.
pub fn glob_match(pattern: &str, path: &str) -> bool {
    let path = path.strip_prefix("./").unwrap_or(path);
    let pattern = pattern.strip_prefix("./").unwrap_or(pattern);
    Regex::new(&glob_to_regex(pattern)).is_ok_and(|re| re.is_match(path))
}

/// Whether `text` matches `pattern`, where `*` matches any run of characters
/// and `?` any single character.
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let mut re = String::from("^");
    for c in pattern.chars() {
        match c {
            '*' => re.push_str("(?s:.*)"),
            '?' => re.push('.'),
            c => escape_char(c, &mut re),
        }
    }
    re.push('$');
    Regex::new(&re).is_ok_and(|r| r.is_match(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_globs() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "src/main.rs"));
        assert!(glob_match("**/*.rs", "src/main.rs"));
        assert!(glob_match("**/*.rs", "main.rs"));
        assert!(glob_match("src/**", "src/a/b/c.txt"));
        assert!(glob_match("src/**/mod.rs", "src/mod.rs"));
        assert!(glob_match("src/**/mod.rs", "src/a/b/mod.rs"));
        assert!(glob_match("*.{md,txt}", "notes.txt"));
        assert!(!glob_match("*.{md,txt}", "notes.rs"));
        assert!(glob_match("./docs/?.md", "docs/a.md"));
        assert!(glob_match("a+b.txt", "a+b.txt"));
    }

    #[test]
    fn wildcards() {
        assert!(wildcard_match("git *", "git status --short"));
        assert!(!wildcard_match("git *", "rm -rf /"));
        assert!(wildcard_match("*", "anything / at all"));
        assert!(wildcard_match("mcp_*", "mcp_search"));
        assert!(wildcard_match("rm -rf /", "rm -rf /"));
        assert!(!wildcard_match("ls", "ls -la"));
    }
}
