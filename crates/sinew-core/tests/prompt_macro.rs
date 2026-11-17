//! `prompt!` and `chat_prompt!`: expansion, parity with `PromptTemplate`,
//! runtime interop and compile errors.

use std::collections::HashMap;

use sinew_core::{chat_prompt, prompt, ChatPromptTemplate, PromptTemplate, Role, Runnable, Vars};

#[test]
fn zero_variables() {
    let p = prompt!("No variables here.");
    assert_eq!(p.format(), "No variables here.");
    assert!(p.variables().is_empty());
    assert_eq!(prompt!("").format(), "");
}

#[test]
fn one_variable() {
    assert_eq!(
        prompt!("Hello, {name}!").name("Roan").format(),
        "Hello, Roan!"
    );
}

#[test]
fn many_variables_any_order_and_repeats() {
    let p = prompt!("{a} then {b} then {c}, and {a} again");
    assert_eq!(p.variables(), ["a", "b", "c"]);
    let filled = p.c(3).a("x").b('y');
    assert_eq!(filled.format(), "x then y then 3, and x again");
    // `format` borrows, so the builder renders more than once.
    assert_eq!(filled.format(), filled.format());
}

#[test]
fn setters_replace_values() {
    let p = prompt!("{x}").x("first").x("second");
    assert_eq!(p.format(), "second");
}

#[test]
fn values_are_any_display() {
    struct Celsius(f32);
    impl std::fmt::Display for Celsius {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:.1} °C", self.0)
        }
    }
    let owned = String::from("Montreal");
    let text = prompt!("{city}: {temp}, {n} readings, ok={ok}")
        .city(&owned)
        .temp(Celsius(-3.5))
        .n(12_u64)
        .ok(true)
        .format();
    assert_eq!(text, "Montreal: -3.5 °C, 12 readings, ok=true");
}

#[test]
fn keyword_and_unicode_names() {
    let p = prompt!("{type} / {ação} / { spaced }");
    assert_eq!(p.variables(), ["type", "ação", "spaced"]);
    assert_eq!(p.r#type("t").ação("a").spaced("s").format(), "t / a / s");
}

#[test]
fn escapes() {
    assert_eq!(prompt!("{{literal}}").format(), "{literal}");
    assert_eq!(prompt!("{{{x}}}").x(1).format(), "{1}");
    assert_eq!(
        prompt!("json: {{\"k\": {n}}}").n(1).format(),
        "json: {\"k\": 1}"
    );
    assert_eq!(prompt!(r"raw \n {{ }}").format(), "raw \\n { }");
}

/// Every template here is rendered by both `prompt!` and `PromptTemplate`
/// with the same values; the outputs must be identical.
#[test]
fn parity_with_prompt_template() {
    fn check(template: &str, vars: &[(&str, &str)], rendered: String) {
        let runtime = PromptTemplate::new(template).unwrap();
        let map: HashMap<&str, String> = vars.iter().map(|(k, v)| (*k, v.to_string())).collect();
        assert_eq!(
            runtime.render(&map).unwrap(),
            rendered,
            "template {template:?}"
        );
    }
    check("plain", &[], prompt!("plain").format());
    check("{a}", &[("a", "1")], prompt!("{a}").a("1").format());
    check(
        "Hi {name}, json: {{\"k\": {n}}}",
        &[("name", "Roan"), ("n", "1")],
        prompt!("Hi {name}, json: {{\"k\": {n}}}")
            .name("Roan")
            .n("1")
            .format(),
    );
    check("{{{{", &[], prompt!("{{{{").format());
    check("}}}}x{{", &[], prompt!("}}}}x{{").format());
    check(
        "{ a }{b}{ a}",
        &[("a", "A"), ("b", "B")],
        prompt!("{ a }{b}{ a}").a("A").b("B").format(),
    );
    check(
        "multi\nline {x}\ttab é",
        &[("x", "{not a var}")],
        prompt!("multi\nline {x}\ttab é").x("{not a var}").format(),
    );
    // Variable lists agree too.
    let runtime = PromptTemplate::new("{b} {a} {b} {c}").unwrap();
    assert_eq!(runtime.variables(), prompt!("{b} {a} {b} {c}").variables());
}

/// Templates `prompt!` rejects at compile time (see `tests/ui_prompt`) are
/// rejected by `PromptTemplate::new` at runtime too.
#[test]
fn rejected_templates_agree() {
    for bad in [
        "Summarize {text in French",
        "Return JSON: {\"a\": 1}",
        "Hello {first name}",
        "{}",
        "a } b",
        "Reply with {answer}} only",
    ] {
        assert!(PromptTemplate::new(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn file_form() {
    let text = prompt!(file = "tests/prompts/summarize.txt")
        .text("Rust has no garbage collector.")
        .lang("French")
        .format();
    assert_eq!(
        text,
        "Summarize the text below in French.\n\nRust has no garbage collector.\n\n\
         Answer as JSON: {\"summary\": \"...\"}\n"
    );
    let runtime = PromptTemplate::new(include_str!("prompts/summarize.txt")).unwrap();
    let p: PromptTemplate = prompt!(file = "tests/prompts/summarize.txt").into();
    assert_eq!(p, runtime);
}

#[tokio::test]
async fn converts_to_runnable_prompt_template() {
    let p = prompt!("Tell me about {topic}");
    let template: PromptTemplate = p.clone().into();
    assert_eq!(template.as_str(), "Tell me about {topic}");
    let out = template
        .invoke(sinew_core::runnable::vars([("topic", "Rust")]))
        .await
        .unwrap();
    assert_eq!(out, p.topic("Rust").format());
}

#[test]
fn to_vars_renders_a_runtime_template() {
    // E.g. a newer version pulled from the prompt hub with the same variables.
    let pulled = PromptTemplate::new("[{lang}] {text}").unwrap();
    let vars: Vars = prompt!("{text} in {lang}").text("hi").lang("en").to_vars();
    assert_eq!(pulled.render_vars(&vars).unwrap(), "[en] hi");
}

#[test]
fn chat_prompt_formats_messages() {
    let p = chat_prompt!(
        system: "You are a {persona}. Reply in {lang}.",
        user: "Explain {topic}.",
        assistant: "Sure, {topic}:",
        user: "Shorter, in {lang}.",
    );
    assert_eq!(p.variables(), ["persona", "lang", "topic"]);
    let filled = p.clone().persona("teacher").lang("pt-BR").topic("traits");
    let req = filled.format();
    let roles: Vec<Role> = req.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        [Role::System, Role::User, Role::Assistant, Role::User]
    );
    assert_eq!(req.messages[0].text(), "You are a teacher. Reply in pt-BR.");
    assert_eq!(req.messages[2].text(), "Sure, traits:");
    assert_eq!(filled.to_messages().len(), 4);

    // Same request as the runtime ChatPromptTemplate with the same values.
    let runtime: ChatPromptTemplate = p.into();
    let expected = runtime.format(&filled.to_vars()).unwrap();
    assert_eq!(
        serde_json::to_value(&req.messages).unwrap(),
        serde_json::to_value(&expected.messages).unwrap()
    );
}

#[test]
fn chat_prompt_from_file_and_no_variables() {
    let req = chat_prompt!(system: "Be brief.", user: file = "tests/prompts/summarize.txt")
        .text("x")
        .lang("en")
        .format();
    assert_eq!(req.messages.len(), 2);
    assert!(req.messages[1]
        .text()
        .starts_with("Summarize the text below in en."));
    assert_eq!(chat_prompt!(user: "hi").format().messages[0].text(), "hi");
}

#[test]
fn crate_override() {
    assert_eq!(prompt!("{x}", crate = "sinew_core").x(1).format(), "1");
}

#[test]
fn compile_errors_are_helpful() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui_prompt/*.rs");
}
