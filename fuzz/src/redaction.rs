//! Trace redaction and the guardrail patterns on arbitrary strings.

use std::sync::OnceLock;

use serde_json::Value;
use sinew_core::guardrails::{
    patterns, DenyList, Guardrail, GuardrailStage, GuardrailVerdict, InjectionDetector, MaxLength,
    PiiRedactor, SecretRedactor,
};
use sinew_trace::Redactor;

/// `f` applied twice gives the same text as applied once.
fn assert_idempotent(name: &str, text: &str, f: impl Fn(&str) -> String) {
    let once = f(text);
    let twice = f(&once);
    assert_eq!(once, twice, "{name} is not idempotent on {text:?}");
}

/// The trace redactor never panics, is idempotent, and never leaves a
/// recognizable secret behind.
pub fn redactor(data: &[u8]) {
    let text = String::from_utf8_lossy(data);
    static DEFAULT: OnceLock<Redactor> = OnceLock::new();
    static ALL: OnceLock<Redactor> = OnceLock::new();
    let default = DEFAULT.get_or_init(Redactor::new);
    let all = ALL.get_or_init(|| Redactor::new().phones(true));
    for r in [default, all] {
        assert_idempotent("Redactor::redact_str", &text, |t| {
            r.redact_str(t).into_owned()
        });
        let out = r.redact_str(&text);
        assert!(
            !SecretRedactor::contains_secret(&out),
            "secret left after redaction: {out:?}"
        );
    }
    if let Ok(mut v) = serde_json::from_str::<Value>(&text) {
        all.redact_value(&mut v);
        let once = v.clone();
        all.redact_value(&mut v);
        assert_eq!(once, v, "redact_value is not idempotent");
    }
}

/// The built-in guardrails never panic; the redacting ones are idempotent
/// and `MaxLength::truncate` keeps exactly the limit.
pub fn guardrails(data: &[u8]) {
    let Some((&limit, body)) = data.split_first() else {
        return;
    };
    let text = String::from_utf8_lossy(body);

    let pii = PiiRedactor::new();
    assert_idempotent("PiiRedactor", &text, |t| pii.redact(t));
    let secrets = SecretRedactor::new();
    assert_idempotent("SecretRedactor", &text, |t| secrets.redact(t));
    assert!(
        !SecretRedactor::contains_secret(&secrets.redact(&text)),
        "secret left after redaction"
    );

    for stage in [
        GuardrailStage::Input,
        GuardrailStage::Output,
        GuardrailStage::ToolOutput,
    ] {
        let _ = InjectionDetector::new().check(stage, &text);
        let _ = InjectionDetector::blocking().check(stage, &text);
        let _ = pii.check(stage, &text);
    }
    let _ = patterns::luhn_valid(&text);
    let _ = patterns::is_card_number(&text);
    let _ = patterns::is_phone_number(&text);

    static DENY: OnceLock<(DenyList, DenyList)> = OnceLock::new();
    let (deny, redacting) = DENY.get_or_init(|| {
        let deny = DenyList::new()
            .keywords(["falcon", "c++", "a.b"])
            .topic("weapons", ["gun", " "]);
        let redacting = deny.clone().redacting("[DENIED]");
        (deny, redacting)
    });
    let _ = deny.check(GuardrailStage::Output, &text);
    if let GuardrailVerdict::Modify(out) = redacting.check(GuardrailStage::Output, &text) {
        assert!(
            deny.first_match(&out).is_none(),
            "denied text left: {out:?}"
        );
    }

    let max = usize::from(limit);
    match MaxLength::chars(max)
        .truncate()
        .check(GuardrailStage::Output, &text)
    {
        GuardrailVerdict::Pass => assert!(text.chars().count() <= max),
        GuardrailVerdict::Modify(out) => {
            assert_eq!(out.chars().count(), max);
            assert!(text.starts_with(&out));
        }
        GuardrailVerdict::Block(r) => panic!("truncating MaxLength blocked: {r}"),
    }
}
