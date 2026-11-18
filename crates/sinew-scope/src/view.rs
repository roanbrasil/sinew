//! Terminal tree viewer for JSONL traces (`sinew-scope view`).

use std::{collections::HashMap, fmt::Write as _, fs, path::Path};

use sinew_trace::Run;

/// Reads a JSONL trace file. Malformed lines are skipped with a warning on
/// stderr.
pub fn load_jsonl(path: impl AsRef<Path>) -> std::io::Result<Vec<Run>> {
    let text = fs::read_to_string(path)?;
    let mut runs: Vec<Run> = Vec::new();
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        match serde_json::from_str(line) {
            Ok(r) => runs.push(r),
            Err(e) => eprintln!("warning: skipping line {}: {e}", n + 1),
        }
    }
    Ok(runs)
}

fn fmt_ms(ms: Option<u64>) -> String {
    match ms {
        None => "open".into(),
        Some(ms) if ms >= 1000 => format!("{:.2}s", ms as f64 / 1000.0),
        Some(ms) => format!("{ms}ms"),
    }
}

fn preview(v: &serde_json::Value) -> String {
    let s = match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let s = s.replace('\n', " ");
    if s.chars().count() > 120 {
        format!("{}…", s.chars().take(120).collect::<String>())
    } else {
        s
    }
}

struct Tree<'a> {
    children: &'a HashMap<&'a str, Vec<&'a Run>>,
    io: bool,
}

impl Tree<'_> {
    fn write(&self, out: &mut String, run: &Run, prefix: &str, last: bool, root: bool) {
        let branch = if root {
            ""
        } else if last {
            "└─ "
        } else {
            "├─ "
        };
        // Lowercase the kind only: run names keep their case.
        let kind = format!("{:?}", run.kind).to_lowercase();
        let mut line = format!(
            "{prefix}{branch}{kind} {}  {}",
            run.name,
            fmt_ms(run.duration_ms())
        );
        if let (Some(i), Some(o)) = (run.input_tokens, run.output_tokens) {
            let _ = write!(line, "  tokens {i}→{o}");
        }
        if let Some(e) = &run.error {
            let _ = write!(line, "  ERROR: {e}");
        }
        let _ = writeln!(out, "{line}");
        let child_prefix = if root {
            String::new()
        } else {
            format!("{prefix}{}", if last { "   " } else { "│  " })
        };
        if self.io {
            let _ = writeln!(out, "{child_prefix}   in : {}", preview(&run.inputs));
            if let Some(o) = &run.outputs {
                let _ = writeln!(out, "{child_prefix}   out: {}", preview(o));
            }
        }
        if let Some(kids) = self.children.get(run.id.as_str()) {
            for (i, k) in kids.iter().enumerate() {
                self.write(out, k, &child_prefix, i + 1 == kids.len(), false);
            }
        }
    }
}

/// The text [`print_traces`] prints.
fn render_traces(mut runs: Vec<Run>, trace_prefix: Option<&str>, io: bool) -> String {
    runs.sort_by_key(|r| r.start_ms);
    if let Some(p) = trace_prefix {
        runs.retain(|r| r.trace_id.starts_with(p));
    }
    let mut children: HashMap<&str, Vec<&Run>> = HashMap::new();
    for r in &runs {
        if let Some(p) = &r.parent_id {
            children.entry(p.as_str()).or_default().push(r);
        }
    }
    let roots: Vec<&Run> = runs
        .iter()
        .filter(|r| {
            r.parent_id
                .as_ref()
                .is_none_or(|p| !runs.iter().any(|x| &x.id == p))
        })
        .collect();
    let tree = Tree {
        children: &children,
        io,
    };
    let mut out = String::new();
    for root in &roots {
        // u64: summing many u32 counts must not overflow.
        let tokens =
            runs.iter()
                .filter(|r| r.trace_id == root.trace_id)
                .fold((0u64, 0u64), |a, r| {
                    (
                        a.0 + u64::from(r.input_tokens.unwrap_or(0)),
                        a.1 + u64::from(r.output_tokens.unwrap_or(0)),
                    )
                });
        // First 8 characters (not bytes: ids may be any UTF-8 string).
        let short: String = root.trace_id.chars().take(8).collect();
        let _ = writeln!(out, "trace {short}  total tokens {}→{}", tokens.0, tokens.1);
        tree.write(&mut out, root, "", true, true);
        out.push('\n');
    }
    let _ = writeln!(out, "{} runs, {} traces", runs.len(), roots.len());
    out
}

/// Prints every trace as a tree, optionally only traces whose id starts
/// with `trace_prefix`, and with inputs/outputs when `io` is set.
pub fn print_traces(runs: Vec<Run>, trace_prefix: Option<&str>, io: bool) {
    print!("{}", render_traces(runs, trace_prefix, io));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sinew_trace::RunKind;

    fn run(id: &str, trace: &str, parent: Option<&str>, kind: RunKind, start: u64) -> Run {
        Run {
            id: id.into(),
            trace_id: trace.into(),
            parent_id: parent.map(str::to_string),
            name: id.into(),
            kind,
            start_ms: start,
            end_ms: Some(start + 5),
            inputs: json!({"q": id}),
            outputs: Some(json!("line one\nline two")),
            error: None,
            input_tokens: None,
            output_tokens: None,
            metadata: json!({}),
        }
    }

    fn sample() -> Vec<Run> {
        let mut root = run("root", "trace-aaaaaaaa", None, RunKind::Graph, 0);
        root.name = "ReactAgent".into();
        root.end_ms = Some(2500);
        let mut llm = run("llm", "trace-aaaaaaaa", Some("root"), RunKind::Llm, 1);
        llm.input_tokens = Some(10);
        llm.output_tokens = Some(4);
        let mut tool = run("tool", "trace-aaaaaaaa", Some("root"), RunKind::Tool, 2);
        tool.error = Some("boom".into());
        let mut nested = run("nested", "trace-aaaaaaaa", Some("tool"), RunKind::Chain, 3);
        nested.end_ms = None;
        nested.outputs = None;
        let other = run("other", "zzzz", None, RunKind::Chain, 10);
        // Parent not in the file: shown as a root of its own.
        let dangling = run("dangling", "yyyy", Some("missing"), RunKind::Tool, 20);
        vec![other, nested, tool, llm, root, dangling]
    }

    #[test]
    fn renders_trees_with_durations_tokens_and_errors() {
        // Regression: the whole line was lowercased, run names included.
        let text = render_traces(sample(), None, false);
        let expected = "trace trace-aa  total tokens 10→4\n\
                        graph ReactAgent  2.50s\n\
                        ├─ llm llm  5ms  tokens 10→4\n\
                        └─ tool tool  5ms  ERROR: boom\n   \
                        └─ chain nested  open\n\
                        \n\
                        trace zzzz  total tokens 0→0\n\
                        chain other  5ms\n\
                        \n\
                        trace yyyy  total tokens 0→0\n\
                        tool dangling  5ms\n\
                        \n\
                        6 runs, 3 traces\n";
        assert_eq!(text, expected);
    }

    #[test]
    fn filters_by_trace_prefix_and_shows_io() {
        let text = render_traces(sample(), Some("zz"), true);
        assert_eq!(
            text,
            "trace zzzz  total tokens 0→0\n\
             chain other  5ms\n   \
             in : {\"q\":\"other\"}\n   \
             out: line one line two\n\
             \n\
             1 runs, 1 traces\n"
        );
        let nested = render_traces(sample(), Some("trace-"), true);
        assert!(nested.contains("│     in : {\"q\":\"llm\"}"), "{nested}");
        assert!(nested.contains("      in : {\"q\":\"nested\"}"), "{nested}");
        assert!(render_traces(vec![], None, true).ends_with("0 runs, 0 traces\n"));
    }

    #[test]
    fn non_ascii_trace_ids_and_long_previews() {
        // Regression: the trace id was cut at byte 8, which panicked
        // inside a multi-byte character.
        let mut r = run("r", "aéééééé", None, RunKind::Llm, 0);
        r.inputs = json!("x".repeat(200));
        r.input_tokens = Some(u32::MAX);
        r.output_tokens = Some(1);
        let mut r2 = run("r2", "aéééééé", Some("r"), RunKind::Llm, 1);
        r2.input_tokens = Some(u32::MAX);
        r2.output_tokens = Some(1);
        let text = render_traces(vec![r, r2], None, true);
        assert!(text.starts_with("trace aééééé"), "{text}");
        // u32::MAX * 2 does not overflow.
        assert!(text.contains("total tokens 8589934590→2"), "{text}");
        assert!(text.contains(&format!("in : {}…", "x".repeat(120))));
    }

    #[test]
    fn loads_jsonl_and_skips_bad_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        let runs = sample();
        let mut text = String::new();
        for r in &runs[..2] {
            text.push_str(&serde_json::to_string(r).unwrap());
            text.push('\n');
        }
        text.push_str("\n{not json}\n");
        fs::write(&path, text).unwrap();
        let loaded = load_jsonl(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "other");
        assert!(load_jsonl(dir.path().join("missing.jsonl")).is_err());
        assert_eq!(fmt_ms(Some(999)), "999ms");
        assert_eq!(fmt_ms(Some(1000)), "1.00s");
    }
}
