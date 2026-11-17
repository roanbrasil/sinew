//! `docs/providers.md` is generated from `providers::catalog()`; this test
//! fails when they drift. Regenerate with `SINEW_UPDATE_DOCS=1 cargo test -p
//! sinew-core --test provider_matrix`.

use sinew_core::{
    providers::{capabilities_for, catalog, provider_names, ProviderKind},
    Capabilities,
};

const START: &str = "<!-- generated:providers:start -->";
const END: &str = "<!-- generated:providers:end -->";

fn mark(b: bool) -> &'static str {
    if b {
        "✓"
    } else {
        "—"
    }
}

fn generate() -> String {
    let mut out = String::new();
    out.push_str("| Provider | Spec | Kind | Tools | Stream | Images | PDF | Audio | JSON schema | Cached tokens | Forced tool |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for p in catalog() {
        let c = p.capabilities;
        let kind = match p.kind {
            ProviderKind::Native => "native",
            ProviderKind::Hosted => "hosted",
            ProviderKind::Local => "local",
        };
        out.push_str(&format!(
            "| {} | `{}:<model>` | {kind} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            p.name,
            p.name,
            mark(c.tools),
            mark(c.streaming),
            mark(c.vision),
            mark(c.pdf),
            mark(c.audio),
            mark(c.json_schema),
            mark(c.cached_tokens),
            mark(c.forced_tool_choice),
        ));
    }
    out.push_str("\n| Provider | Aliases | Base URL / default host | Environment variable |\n|---|---|---|---|\n");
    for p in catalog() {
        let aliases = if p.aliases.is_empty() {
            "—".to_string()
        } else {
            p.aliases
                .iter()
                .map(|a| format!("`{a}`"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        out.push_str(&format!(
            "| {} | {aliases} | `{}` | `{}` |\n",
            p.name, p.url, p.env_var
        ));
    }
    out
}

#[test]
fn providers_doc_is_up_to_date() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/providers.md");
    let doc = std::fs::read_to_string(path).expect("docs/providers.md");
    let (head, rest) = doc.split_once(START).expect("start marker");
    let (_, tail) = rest.split_once(END).expect("end marker");
    let fresh = format!("{head}{START}\n{}{END}{tail}", generate());
    if fresh != doc {
        if std::env::var("SINEW_UPDATE_DOCS").is_ok() {
            std::fs::write(path, fresh).unwrap();
        } else {
            panic!("docs/providers.md is stale: run SINEW_UPDATE_DOCS=1 cargo test -p sinew-core --test provider_matrix");
        }
    }
}

#[test]
fn catalog_matches_from_spec_names_and_capabilities_for() {
    let names: Vec<&str> = catalog().iter().map(|p| p.name).collect();
    assert_eq!(names, provider_names());
    for p in catalog() {
        assert_eq!(
            capabilities_for(p.name, "claude-sonnet-4-5"),
            Some(p.capabilities),
            "{}",
            p.name
        );
        for a in p.aliases {
            assert!(capabilities_for(a, "m").is_some(), "alias {a}");
        }
    }
    assert_eq!(capabilities_for("nope", "m"), None);
    // Anthropic depends on the model.
    let old = capabilities_for("anthropic", "claude-sonnet-4-5").unwrap();
    let new = capabilities_for("anthropic", "claude-sonnet-5-5").unwrap();
    assert!(old.sampling && old.forced_tool_choice);
    assert!(!new.sampling && !new.forced_tool_choice);
    assert_eq!(Capabilities::default(), Capabilities::none());
}

#[test]
fn models_report_their_capabilities_through_wrappers() {
    use sinew_core::{providers::OpenAiCompat, ChatModel, ChatModelExt, RetryPolicy};
    use std::{sync::Arc, time::Duration};

    let m = OpenAiCompat::ollama("qwen3:4b");
    assert_eq!(m.capabilities(), capabilities_for("ollama", "").unwrap());
    let wrapped = m
        .with_retry(RetryPolicy::default())
        .with_timeout(Duration::from_secs(5));
    assert!(wrapped.capabilities().tools && !wrapped.capabilities().pdf);
    let dynm: Arc<dyn ChatModel> = Arc::new(wrapped);
    assert!(dynm.capabilities().json_schema);
}
