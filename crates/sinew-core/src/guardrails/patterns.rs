//! Regular-expression patterns and validators shared by the guardrails in
//! this module and by trace redaction (`sinew_trace::redact`).
//!
//! The patterns are plain strings, so crates that compile their own
//! [`regex::Regex`] or `RegexSet` use exactly the same definitions.

/// E-mail address pattern.
pub const EMAIL_PATTERN: &str = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}";

/// Card-number candidate pattern (13 to 19 digits, optionally separated by
/// spaces or dashes). Confirm candidates with [`is_card_number`].
pub const CARD_PATTERN: &str = r"\b(?:\d[ -]?){12,18}\d\b";

/// Phone-number candidate pattern. Confirm candidates with [`is_phone_number`].
pub const PHONE_PATTERN: &str =
    r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\b\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{4}\b";

/// Secret patterns with a recognizable shape: `sk-` keys (OpenAI,
/// Anthropic and others), xAI, Groq, Google `AIza`, GitHub tokens, AWS
/// access key ids, Slack tokens, JWTs, PEM private keys and `Bearer`
/// credentials.
///
/// For the `Bearer` pattern (the last one) the first capture group, the
/// scheme word, is meant to be kept.
pub const SECRET_PATTERNS: &[&str] = &[
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

/// Prompt-injection heuristics used by
/// [`InjectionDetector`](super::InjectionDetector): "ignore previous
/// instructions" and its variants, requests to reveal the system prompt,
/// role reassignment ("you are now a ...") and chat-template control tokens.
pub const INJECTION_PATTERNS: &[&str] = &[
    r"(?i)ignore\s+(?:all\s+|any\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|earlier|preceding)\s+(?:instructions|prompts|messages|rules)",
    r"(?i)disregard\s+(?:all\s+|any\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|earlier)",
    r"(?i)forget\s+(?:all\s+|everything\s+|your\s+)(?:previous\s+)?(?:instructions|rules|you\s+were\s+told)",
    r"(?i)(?:reveal|print|show|output|repeat)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+prompt|hidden\s+instructions)",
    r"(?i)you\s+are\s+now\s+(?:a|an|in|the)\s+",
    r"(?i)new\s+(?:system\s+)?instructions\s*:",
    r"(?i)<\|(?:im_start|im_end|system|endoftext)\|>|\[/?INST\]|<<SYS>>",
];

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
