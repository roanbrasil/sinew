//! Secret and PII redaction for runs, applied by the [`Tracer`](crate::Tracer)
//! before any sink sees a run.
//!
//! A [`Redactor`] rewrites a run's `inputs`, `outputs`, `error` and `metadata`:
//!
//! - **Value rules** scan every JSON string (and integer, for card numbers) and
//!   replace matches with `[REDACTED:<label>]`:
//!   - `secret`: API keys and tokens with a recognizable shape (`sk-…`,
//!     `sk-ant-…`, `xai-…`, `gsk_…`, `AIza…`, `ghp_…` / `github_pat_…`, AWS
//!     `AKIA…` / `ASIA…`, Slack `xox?-…`, `Bearer …`, JWTs, PEM private keys);
//!   - `email`: e-mail addresses;
//!   - `card`: card numbers, kept only when the digits pass the Luhn check
//!     and start with a card-network digit (2–6), so order IDs and
//!     millisecond timestamps are not masked;
//!   - `phone`: phone numbers (off by default; the pattern is broad);
//!   - custom regular expressions with your own label ([`Redactor::pattern`]).
//! - **Field rules** mask the whole value of any object field whose key is in
//!   the list (`password`, `api_key`, `authorization`, `cookie`, …), compared
//!   case-insensitively with `-` and `_` treated as equal. Many tokens have no
//!   recognizable shape, so the key name is the only signal.
//! - **Allow-list** paths ([`Redactor::allow`]) are never touched.
//!
//! ```
//! use serde_json::json;
//! use sinew_trace::{MemorySink, Redactor, RunKind, Tracer};
//! use std::sync::Arc;
//!
//! let sink = Arc::new(MemorySink::new());
//! let tracer = Tracer::new()
//!     .with_redactor(Redactor::new().allow("metadata.model"))
//!     .with_shared_sink(sink.clone());
//!
//! let run = tracer.start(
//!     "login",
//!     RunKind::Tool,
//!     None,
//!     json!({ "user": "ana@example.com", "password": "hunter2" }),
//! );
//! run.finish(json!("ok"));
//!
//! let inputs = &sink.runs()[0].inputs;
//! assert_eq!(inputs["user"], "[REDACTED:email]");
//! assert_eq!(inputs["password"], "[REDACTED:field]");
//! ```
//!
//! Regular expressions are compiled once per [`Redactor`] (on first use) and
//! every string is first checked against one combined [`regex::RegexSet`];
//! strings with no match are not copied.

use std::{
    borrow::Cow,
    fmt,
    sync::{LazyLock, OnceLock},
};

use regex::{Captures, Regex, RegexSet};
use serde_json::Value;

use crate::Run;

/// E-mail address pattern, shared with `sinew-harness`'s `PiiRedactor`.
pub const EMAIL_PATTERN: &str = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}";

/// Card-number candidate pattern (13 to 19 digits, optionally separated by
/// spaces or dashes). Confirm candidates with [`is_card_number`].
pub const CARD_PATTERN: &str = r"\b(?:\d[ -]?){12,18}\d\b";

/// Phone-number candidate pattern. Confirm candidates with [`is_phone_number`].
pub const PHONE_PATTERN: &str =
    r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\b\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{4}\b";

/// Secret patterns with a recognizable shape. For the `Bearer` pattern the
/// first capture group (the scheme word) is kept.
const SECRET_PATTERNS: &[&str] = &[
    // OpenAI (`sk-`, `sk-proj-`), Anthropic (`sk-ant-`) and other `sk-` keys.
    r"\bsk-[A-Za-z0-9_-]{20,}",
    r"\bxai-[A-Za-z0-9]{20,}",
    r"\bgsk_[A-Za-z0-9]{20,}",
    r"\bAIza[0-9A-Za-z_-]{35}",
    r"\bgh[pousr]_[A-Za-z0-9]{36,}",
    r"\bgithub_pat_[A-Za-z0-9_]{22,}",
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    r"\bxox[abposr]-[A-Za-z0-9-]{10,}",
    r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{16,}",
];

/// Field names masked by [`Redactor::new`], in normalized form (lowercase,
/// `_` instead of `-`).
pub const DEFAULT_FIELDS: &[&str] = &[
    "password",
    "passwd",
    "secret",
    "client_secret",
    "api_key",
    "apikey",
    "x_api_key",
    "authorization",
    "proxy_authorization",
    "cookie",
    "set_cookie",
    "access_token",
    "refresh_token",
    "id_token",
    "private_key",
];

static EMAIL: LazyLock<Regex> = LazyLock::new(|| Regex::new(EMAIL_PATTERN).unwrap());
static CARD: LazyLock<Regex> = LazyLock::new(|| Regex::new(CARD_PATTERN).unwrap());
static PHONE: LazyLock<Regex> = LazyLock::new(|| Regex::new(PHONE_PATTERN).unwrap());
static SECRETS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    SECRET_PATTERNS
        .iter()
        .map(|p| Regex::new(p).unwrap())
        .collect()
});

/// Whether `digits` (ASCII digits only) passes the Luhn checksum.
/// Returns `false` for an empty string or any non-digit character.
pub fn luhn_valid(digits: &str) -> bool {
    if digits.is_empty() {
        return false;
    }
    let mut sum = 0;
    let mut double = false;
    for c in digits.chars().rev() {
        let Some(mut d) = c.to_digit(10) else {
            return false;
        };
        if double {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        sum += d;
        double = !double;
    }
    sum % 10 == 0
}

/// Whether a [`CARD_PATTERN`] match is a plausible card number: 13 to 19
/// digits (separators ignored), first digit 2–6 (the major card networks)
/// and a valid Luhn checksum.
pub fn is_card_number(candidate: &str) -> bool {
    let digits: String = candidate.chars().filter(char::is_ascii_digit).collect();
    (13..=19).contains(&digits.len())
        && matches!(digits.as_bytes()[0], b'2'..=b'6')
        && luhn_valid(&digits)
}

/// Whether a [`PHONE_PATTERN`] match is a plausible phone number (10 to 15
/// digits).
pub fn is_phone_number(candidate: &str) -> bool {
    let n = candidate.chars().filter(char::is_ascii_digit).count();
    (10..=15).contains(&n)
}

/// A custom pattern passed to [`Redactor::pattern`] did not compile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidPattern {
    /// The rejected pattern.
    pub pattern: String,
    /// The regex compiler's message.
    pub message: String,
}

impl fmt::Display for InvalidPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid redaction pattern `{}`: {}",
            self.pattern, self.message
        )
    }
}

impl std::error::Error for InvalidPattern {}

#[derive(Debug, Clone)]
enum Action {
    /// Replace the whole match with the placeholder.
    Mask(String),
    /// Keep capture group 1, then a space and the placeholder.
    KeepPrefix(String),
    /// Replace only matches that pass [`is_card_number`].
    Card,
    /// Replace only matches that pass [`is_phone_number`].
    Phone,
}

#[derive(Debug, Clone)]
struct Rule {
    regex: Regex,
    action: Action,
}

impl Rule {
    fn apply<'t>(&self, text: &'t str) -> Cow<'t, str> {
        match &self.action {
            Action::Mask(p) => self.regex.replace_all(text, p.as_str()),
            Action::KeepPrefix(p) => self
                .regex
                .replace_all(text, |c: &Captures<'_>| format!("{} {p}", &c[1])),
            Action::Card => self.regex.replace_all(text, |c: &Captures<'_>| {
                if is_card_number(&c[0]) {
                    placeholder("card")
                } else {
                    c[0].to_string()
                }
            }),
            Action::Phone => self.regex.replace_all(text, |c: &Captures<'_>| {
                if is_phone_number(&c[0]) {
                    placeholder("phone")
                } else {
                    c[0].to_string()
                }
            }),
        }
    }
}

#[derive(Debug, Clone)]
struct Compiled {
    /// `None` when no value rule is enabled.
    set: Option<RegexSet>,
    rules: Vec<Rule>,
}

#[derive(Debug, Clone)]
struct CustomRule {
    label: String,
    regex: Regex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PathPattern {
    Any,
    Key(String),
}

#[derive(Debug, Clone, Copy)]
enum Seg<'a> {
    Key(&'a str),
    Index(usize),
}

fn placeholder(label: &str) -> String {
    format!("[REDACTED:{label}]")
}

fn normalize_field(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c == '-' {
                '_'
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect()
}

/// `key` equals the normalized `field`, ignoring ASCII case and `-`/`_`.
fn field_eq(key: &str, field: &str) -> bool {
    key.len() == field.len()
        && key.bytes().zip(field.bytes()).all(|(k, f)| {
            let k = if k == b'-' {
                b'_'
            } else {
                k.to_ascii_lowercase()
            };
            k == f
        })
}

/// Rewrites secrets and personal data in runs. See the [module docs](self).
///
/// [`Redactor::new`] enables the `secret`, `email` and `card` value rules and
/// the [`DEFAULT_FIELDS`] field rules; [`Redactor::empty`] starts with
/// nothing enabled. Builder methods toggle each rule.
///
/// ```
/// use serde_json::json;
/// use sinew_trace::Redactor;
///
/// let r = Redactor::new()
///     .phones(true)
///     .pattern("ticket", r"\bTCK-\d{6}\b")
///     .unwrap()
///     .field("session_id")
///     .allow("inputs.messages.*.role");
///
/// assert_eq!(
///     r.redact_str("key sk-ant-api03-abcdefghijklmnopqrstuvwxyz, ticket TCK-123456"),
///     "key [REDACTED:secret], ticket [REDACTED:ticket]"
/// );
///
/// let mut v = json!({ "session_id": "abc", "note": "call +1 514 555 0199" });
/// r.redact_value(&mut v);
/// assert_eq!(v, json!({ "session_id": "[REDACTED:field]", "note": "call [REDACTED:phone]" }));
/// ```
#[derive(Debug, Clone)]
pub struct Redactor {
    secrets: bool,
    emails: bool,
    cards: bool,
    phones: bool,
    custom: Vec<CustomRule>,
    fields: Vec<String>,
    allow: Vec<Vec<PathPattern>>,
    compiled: OnceLock<Compiled>,
}

impl Default for Redactor {
    fn default() -> Self {
        Self::new()
    }
}

impl Redactor {
    /// Secrets, e-mails, card numbers and [`DEFAULT_FIELDS`]; phones off.
    pub fn new() -> Self {
        Self {
            secrets: true,
            emails: true,
            cards: true,
            phones: false,
            custom: Vec::new(),
            fields: DEFAULT_FIELDS.iter().map(|f| (*f).to_string()).collect(),
            allow: Vec::new(),
            compiled: OnceLock::new(),
        }
    }

    /// No rules enabled; add the ones you want.
    pub fn empty() -> Self {
        Self {
            secrets: false,
            emails: false,
            cards: false,
            phones: false,
            custom: Vec::new(),
            fields: Vec::new(),
            allow: Vec::new(),
            compiled: OnceLock::new(),
        }
    }

    fn changed(mut self) -> Self {
        self.compiled = OnceLock::new();
        self
    }

    /// Toggles the `secret` rule (API keys, tokens, JWTs, private keys).
    pub fn secrets(mut self, on: bool) -> Self {
        self.secrets = on;
        self.changed()
    }

    /// Toggles the `email` rule.
    pub fn emails(mut self, on: bool) -> Self {
        self.emails = on;
        self.changed()
    }

    /// Toggles the `card` rule (Luhn-checked).
    pub fn cards(mut self, on: bool) -> Self {
        self.cards = on;
        self.changed()
    }

    /// Toggles the `phone` rule. Off by default: the pattern also matches
    /// some long numeric identifiers.
    pub fn phones(mut self, on: bool) -> Self {
        self.phones = on;
        self.changed()
    }

    /// Adds a custom regular expression; matches become `[REDACTED:<label>]`.
    /// Custom rules run after the `secret` rule and before `email`, `card`
    /// and `phone`.
    pub fn pattern(
        mut self,
        label: impl Into<String>,
        pattern: &str,
    ) -> Result<Self, InvalidPattern> {
        let regex = Regex::new(pattern).map_err(|e| InvalidPattern {
            pattern: pattern.to_string(),
            message: e.to_string(),
        })?;
        self.custom.push(CustomRule {
            label: label.into(),
            regex,
        });
        Ok(self.changed())
    }

    /// Masks the whole value of every object field with this key, at any
    /// depth (case-insensitive, `-` and `_` equal). `null` values are kept.
    pub fn field(mut self, name: &str) -> Self {
        let name = normalize_field(name);
        if !self.fields.contains(&name) {
            self.fields.push(name);
        }
        self
    }

    /// Removes every field rule, including [`DEFAULT_FIELDS`].
    pub fn clear_fields(mut self) -> Self {
        self.fields.clear();
        self
    }

    /// Never redacts the value at `path` or anything below it.
    ///
    /// A path is dot-separated and starts at a run field: `inputs`,
    /// `outputs`, `metadata` or `error`. Each further segment is an object
    /// key or an array index; `*` matches any single key or index. For
    /// example `metadata.model`, `outputs.content` or
    /// `inputs.messages.*.role`.
    pub fn allow(mut self, path: &str) -> Self {
        let pattern = path
            .split('.')
            .map(|s| {
                if s == "*" {
                    PathPattern::Any
                } else {
                    PathPattern::Key(s.to_string())
                }
            })
            .collect();
        self.allow.push(pattern);
        self
    }

    fn compiled(&self) -> &Compiled {
        self.compiled.get_or_init(|| {
            let mut rules = Vec::new();
            let mut patterns: Vec<String> = Vec::new();
            if self.secrets {
                for (re, src) in SECRETS.iter().zip(SECRET_PATTERNS) {
                    let action = if src.contains("(bearer)") {
                        Action::KeepPrefix(placeholder("secret"))
                    } else {
                        Action::Mask(placeholder("secret"))
                    };
                    rules.push(Rule {
                        regex: re.clone(),
                        action,
                    });
                    patterns.push((*src).to_string());
                }
            }
            for c in &self.custom {
                rules.push(Rule {
                    regex: c.regex.clone(),
                    action: Action::Mask(placeholder(&c.label)),
                });
                patterns.push(c.regex.as_str().to_string());
            }
            if self.emails {
                rules.push(Rule {
                    regex: EMAIL.clone(),
                    action: Action::Mask(placeholder("email")),
                });
                patterns.push(EMAIL_PATTERN.to_string());
            }
            if self.cards {
                rules.push(Rule {
                    regex: CARD.clone(),
                    action: Action::Card,
                });
                patterns.push(CARD_PATTERN.to_string());
            }
            if self.phones {
                rules.push(Rule {
                    regex: PHONE.clone(),
                    action: Action::Phone,
                });
                patterns.push(PHONE_PATTERN.to_string());
            }
            // Every pattern compiled on its own, so the set only fails on size
            // limits; then each rule is tried on every string.
            let set = if patterns.is_empty() {
                None
            } else {
                RegexSet::new(&patterns).ok()
            };
            Compiled { set, rules }
        })
    }

    /// Applies the value rules to one string. Returns the input unchanged
    /// (borrowed) when nothing matches.
    pub fn redact_str<'t>(&self, text: &'t str) -> Cow<'t, str> {
        let c = self.compiled();
        if c.rules.is_empty() {
            return Cow::Borrowed(text);
        }
        let mut out = Cow::Borrowed(text);
        match &c.set {
            Some(set) => {
                let hits = set.matches(text);
                if !hits.matched_any() {
                    return out;
                }
                for i in hits.iter() {
                    apply_rule(&c.rules[i], &mut out);
                }
            }
            None => {
                for rule in &c.rules {
                    apply_rule(rule, &mut out);
                }
            }
        }
        out
    }

    /// Redacts a JSON value in place. The value is treated as the root of
    /// the path, so allow-list entries match from their second segment on
    /// (`inputs.a.b` matches `a.b` here); use [`Redactor::redact_run`] for
    /// whole runs.
    pub fn redact_value(&self, value: &mut Value) {
        let mut path = Vec::new();
        self.walk(value, &mut path, true);
    }

    /// Redacts a run's `inputs`, `outputs`, `error` and `metadata` in place.
    /// Ids, names, timings and token counts are left as they are.
    pub fn redact_run(&self, run: &mut Run) {
        self.redact_root("inputs", &mut run.inputs);
        if let Some(out) = run.outputs.as_mut() {
            self.redact_root("outputs", out);
        }
        self.redact_root("metadata", &mut run.metadata);
        if let Some(err) = run.error.as_mut() {
            if !self.path_allowed(&[Seg::Key("error")], false) {
                if let Cow::Owned(s) = self.redact_str(err) {
                    *err = s;
                }
            }
        }
    }

    fn redact_root(&self, root: &'static str, value: &mut Value) {
        let mut path = vec![Seg::Key(root)];
        self.walk(value, &mut path, false);
    }

    fn path_allowed(&self, path: &[Seg<'_>], relative: bool) -> bool {
        self.allow.iter().any(|p| {
            if relative {
                !p.is_empty() && path_matches(&p[1..], path)
            } else {
                path_matches(p, path)
            }
        })
    }

    fn is_masked_field(&self, key: &str) -> bool {
        self.fields.iter().any(|f| field_eq(key, f))
    }

    /// `relative`: the path has no run-field root segment (from
    /// [`Redactor::redact_value`]), so allow patterns skip their first segment.
    fn walk<'a>(&self, value: &'a mut Value, path: &mut Vec<Seg<'a>>, relative: bool) {
        if self.path_allowed(path, relative) {
            return;
        }
        match value {
            Value::String(s) => {
                if let Cow::Owned(r) = self.redact_str(s) {
                    *s = r;
                }
            }
            Value::Number(n) => {
                if self.cards {
                    if let Some(u) = n.as_u64() {
                        if u >= 1_000_000_000_000 && is_card_number(&u.to_string()) {
                            *value = Value::String(placeholder("card"));
                        }
                    }
                }
            }
            Value::Array(items) => {
                for (i, item) in items.iter_mut().enumerate() {
                    path.push(Seg::Index(i));
                    self.walk(item, path, relative);
                    path.pop();
                }
            }
            Value::Object(map) => {
                for (k, v) in map.iter_mut() {
                    path.push(Seg::Key(k.as_str()));
                    if !v.is_null() && self.is_masked_field(k) && !self.path_allowed(path, relative)
                    {
                        *v = Value::String(placeholder("field"));
                    } else {
                        self.walk(v, path, relative);
                    }
                    path.pop();
                }
            }
            Value::Null | Value::Bool(_) => {}
        }
    }
}

fn path_matches(pattern: &[PathPattern], path: &[Seg<'_>]) -> bool {
    pattern.len() == path.len()
        && pattern.iter().zip(path).all(|(p, s)| match (p, s) {
            (PathPattern::Any, _) => true,
            (PathPattern::Key(k), Seg::Key(s)) => k == s,
            (PathPattern::Key(k), Seg::Index(i)) => k.parse::<usize>() == Ok(*i),
        })
}

fn apply_rule(rule: &Rule, out: &mut Cow<'_, str>) {
    let replaced = match rule.apply(out) {
        Cow::Owned(s) => Some(s),
        Cow::Borrowed(_) => None,
    };
    if let Some(s) = replaced {
        *out = Cow::Owned(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Instant;

    fn r() -> Redactor {
        Redactor::new()
    }

    #[test]
    fn masks_api_keys_and_tokens() {
        let cases = [
            "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz012345",
            "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz-_0123",
            "xai-AbCdEfGhIjKlMnOpQrStUvWxYz",
            "gsk_AbCdEfGhIjKlMnOpQrStUvWxYz0123",
            "AIzaSyA-1234567890abcdefghijklmnopqrstu",
            "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
            "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
            "AKIAIOSFODNN7EXAMPLE",
            "xoxb-123456789012-abcdefghij",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        ];
        for secret in cases {
            let out = r().redact_str(&format!("token={secret} end")).into_owned();
            assert_eq!(out, "token=[REDACTED:secret] end", "{secret}");
        }
    }

    #[test]
    fn keeps_bearer_scheme_and_masks_private_keys() {
        assert_eq!(
            r().redact_str("Authorization: Bearer abc.def-ghi_jkl~mno+pqr/stu="),
            "Authorization: Bearer [REDACTED:secret]"
        );
        let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\nIBAAK\n-----END RSA PRIVATE KEY-----";
        assert_eq!(
            r().redact_str(&format!("key:\n{pem}\nbye")),
            "key:\n[REDACTED:secret]\nbye"
        );
    }

    #[test]
    fn masks_emails_and_luhn_valid_cards_only() {
        let out = r().redact_str(
            "mail ana.silva@example.com, card 4111 1111 1111 1111, \
             order 1234567890123, ts 1731772800000, date 2026-11-15",
        );
        assert_eq!(
            out,
            "mail [REDACTED:email], card [REDACTED:card], \
             order 1234567890123, ts 1731772800000, date 2026-11-15"
        );
        // Luhn-valid but starts with 1 (not a card network): kept.
        assert!(luhn_valid("1000000000009"));
        assert_eq!(r().redact_str("id 1000000000009"), "id 1000000000009");
        assert!(!luhn_valid(""));
        assert!(!luhn_valid("12a4"));
    }

    #[test]
    fn phones_are_opt_in() {
        let text = "call +1 (514) 555-0199";
        assert_eq!(r().redact_str(text), text);
        assert_eq!(r().phones(true).redact_str(text), "call [REDACTED:phone]");
    }

    #[test]
    fn rules_can_be_turned_off() {
        let text = "ana@example.com sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz 4111111111111111";
        assert_eq!(Redactor::empty().redact_str(text), text);
        assert_eq!(
            r().emails(false).secrets(false).redact_str(text),
            "ana@example.com sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz [REDACTED:card]"
        );
        assert_eq!(
            r().cards(false).redact_str(text),
            "[REDACTED:email] [REDACTED:secret] 4111111111111111"
        );
    }

    #[test]
    fn custom_patterns() {
        let red = Redactor::empty()
            .pattern("customer", r"\bCUST-\d{4}\b")
            .unwrap();
        assert_eq!(
            red.redact_str("for CUST-0042 and CUST-12"),
            "for [REDACTED:customer] and CUST-12"
        );
        let err = Redactor::empty().pattern("bad", "(").unwrap_err();
        assert_eq!(err.pattern, "(");
        assert!(err.to_string().contains("invalid redaction pattern"));
    }

    #[test]
    fn redacts_nested_json_and_field_names() {
        let mut v = json!({
            "messages": [
                { "role": "user", "content": "I am ana@example.com" },
                { "role": "tool", "content": { "headers": { "Authorization": "Basic Zm9vOmJhcg==", "X-Api-Key": 7 } } }
            ],
            "config": { "PASSWORD": { "nested": "value" }, "api_key": null, "card": 4111111111111111u64, "count": 3 }
        });
        r().redact_value(&mut v);
        assert_eq!(
            v,
            json!({
                "messages": [
                    { "role": "user", "content": "I am [REDACTED:email]" },
                    { "role": "tool", "content": { "headers": { "Authorization": "[REDACTED:field]", "X-Api-Key": "[REDACTED:field]" } } }
                ],
                "config": { "PASSWORD": "[REDACTED:field]", "api_key": null, "card": "[REDACTED:card]", "count": 3 }
            })
        );
    }

    #[test]
    fn custom_and_cleared_field_rules() {
        let mut v = json!({ "session-id": "abc", "password": "p" });
        Redactor::empty().field("Session_ID").redact_value(&mut v);
        assert_eq!(
            v,
            json!({ "session-id": "[REDACTED:field]", "password": "p" })
        );

        let mut v = json!({ "password": "p" });
        r().clear_fields().redact_value(&mut v);
        assert_eq!(v, json!({ "password": "p" }));
    }

    fn run_with(inputs: Value, outputs: Value, metadata: Value, error: Option<&str>) -> Run {
        Run {
            id: "1".into(),
            trace_id: "1".into(),
            parent_id: None,
            name: "ana@example.com".into(),
            kind: crate::RunKind::Llm,
            start_ms: 0,
            end_ms: Some(1),
            inputs,
            outputs: Some(outputs),
            error: error.map(str::to_string),
            input_tokens: None,
            output_tokens: None,
            metadata,
        }
    }

    #[test]
    fn allow_list_paths() {
        let red = r()
            .allow("inputs.messages.*.content")
            .allow("metadata.owner")
            .allow("metadata.creds.password")
            .allow("error");
        let mut run = run_with(
            json!({ "messages": [{ "content": "ana@example.com" }], "other": "ana@example.com" }),
            json!({ "content": "bob@example.com" }),
            json!({ "owner": "ops@example.com", "creds": { "password": "kept", "secret": "x" }, "contact": "ops@example.com" }),
            Some("failed for ana@example.com"),
        );
        red.redact_run(&mut run);
        assert_eq!(run.inputs["messages"][0]["content"], "ana@example.com");
        assert_eq!(run.inputs["other"], "[REDACTED:email]");
        assert_eq!(run.outputs.as_ref().unwrap()["content"], "[REDACTED:email]");
        assert_eq!(run.metadata["owner"], "ops@example.com");
        assert_eq!(run.metadata["creds"]["password"], "kept");
        assert_eq!(run.metadata["creds"]["secret"], "[REDACTED:field]");
        assert_eq!(run.metadata["contact"], "[REDACTED:email]");
        assert_eq!(run.error.as_deref(), Some("failed for ana@example.com"));
        // Names are not redacted.
        assert_eq!(run.name, "ana@example.com");

        // Numeric segments match array indices.
        let mut run = run_with(
            json!({ "m": ["a@example.com", "b@example.com"] }),
            json!(null),
            json!({}),
            None,
        );
        r().allow("inputs.m.1").redact_run(&mut run);
        assert_eq!(
            run.inputs,
            json!({ "m": ["[REDACTED:email]", "b@example.com"] })
        );
    }

    #[test]
    fn redacts_error_strings() {
        let mut run = run_with(
            json!({}),
            json!(null),
            json!({}),
            Some("401 for key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz"),
        );
        r().redact_run(&mut run);
        assert_eq!(run.error.as_deref(), Some("401 for key [REDACTED:secret]"));
    }

    #[test]
    fn unmatched_strings_are_not_copied() {
        assert!(matches!(
            r().redact_str("plain text with numbers 12345 and dashes - ok"),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            Redactor::empty().redact_str("a@b.co"),
            Cow::Borrowed(_)
        ));
    }

    /// A small corpus of strings that must never survive redaction.
    #[test]
    fn known_bad_corpus_has_no_leaks() {
        let secrets = [
            "sk-ant-api03-Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1Qq0",
            "sk-proj-Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1Qq0",
            "AKIAZZZZZZZZZZZZZZZZ",
            "ghs_Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1Qq0Pp9Oo8",
            "jane.doe+test@mail.example.org",
            "5500-0000-0000-0004",
            "378282246310005",
        ];
        let red = r();
        for s in secrets {
            for wrapper in ["{s}", "\"{s}\"", "x={s};", "(see {s})", "line1\n{s}\nline3"] {
                let text = wrapper.replace("{s}", s);
                let out = red.redact_str(&text);
                assert!(!out.contains(s), "leaked {s} in {out}");
            }
        }
    }

    #[test]
    fn large_payload_stays_fast() {
        // About 1 MiB of chat history with a few secrets in it.
        let filler = "The quick brown fox jumps over the lazy dog while the model thinks. ";
        let messages: Vec<Value> = (0..2_000)
            .map(|i| {
                let mut content = filler.repeat(7);
                if i % 500 == 0 {
                    content.push_str(
                        " contact ana@example.com key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz",
                    );
                }
                json!({ "role": if i % 2 == 0 { "user" } else { "assistant" }, "content": content })
            })
            .collect();
        let mut run = run_with(
            json!({ "messages": messages }),
            json!({ "content": filler }),
            json!({ "model": "gpt-x" }),
            None,
        );
        let red = r();
        red.redact_str("warm up");
        let started = Instant::now();
        red.redact_run(&mut run);
        let elapsed = started.elapsed();
        let text = serde_json::to_string(&run.inputs).unwrap();
        assert!(!text.contains("ana@example.com"));
        assert!(!text.contains("sk-proj-"));
        assert_eq!(text.matches("[REDACTED:email]").count(), 4);
        // Generous bound so debug builds on slow CI machines pass.
        assert!(elapsed.as_secs() < 5, "took {elapsed:?}");
    }
}
