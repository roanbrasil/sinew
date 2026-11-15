//! `sinew-scope traces.jsonl [--trace <id-prefix>] [--io]`
//!
//! Prints each trace as a tree with duration, tokens and errors.

use std::{collections::HashMap, fs, process::ExitCode};

use sinew_trace::Run;

struct Opts {
    path: String,
    trace: Option<String>,
    io: bool,
}

fn parse_args() -> Result<Opts, String> {
    let mut args = std::env::args().skip(1);
    let (mut path, mut trace, mut io) = (None, None, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--trace" => trace = Some(args.next().ok_or("--trace needs a value")?),
            "--io" => io = true,
            "-h" | "--help" => return Err(String::new()),
            _ if path.is_none() => path = Some(a),
            _ => return Err(format!("unexpected argument `{a}`")),
        }
    }
    Ok(Opts {
        path: path.ok_or("missing trace file")?,
        trace,
        io,
    })
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

fn print_tree(
    run: &Run,
    children: &HashMap<&str, Vec<&Run>>,
    prefix: &str,
    last: bool,
    root: bool,
    io: bool,
) {
    let branch = if root {
        ""
    } else if last {
        "└─ "
    } else {
        "├─ "
    };
    let mut line = format!(
        "{prefix}{branch}{:?} {}  {}",
        run.kind,
        run.name,
        fmt_ms(run.duration_ms())
    )
    .to_lowercase();
    if let (Some(i), Some(o)) = (run.input_tokens, run.output_tokens) {
        line.push_str(&format!("  tokens {i}→{o}"));
    }
    if let Some(e) = &run.error {
        line.push_str(&format!("  ERROR: {e}"));
    }
    println!("{line}");
    let child_prefix = if root {
        String::new()
    } else {
        format!("{prefix}{}", if last { "   " } else { "│  " })
    };
    if io {
        println!("{child_prefix}   in : {}", preview(&run.inputs));
        if let Some(o) = &run.outputs {
            println!("{child_prefix}   out: {}", preview(o));
        }
    }
    if let Some(kids) = children.get(run.id.as_str()) {
        for (i, k) in kids.iter().enumerate() {
            print_tree(k, children, &child_prefix, i + 1 == kids.len(), false, io);
        }
    }
}

fn main() -> ExitCode {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}");
            }
            eprintln!("usage: sinew-scope <traces.jsonl> [--trace <id-prefix>] [--io]");
            return ExitCode::from(2);
        }
    };
    let text = match fs::read_to_string(&opts.path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: cannot read {}: {e}", opts.path);
            return ExitCode::from(1);
        }
    };
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
    runs.sort_by_key(|r| r.start_ms);
    if let Some(p) = &opts.trace {
        runs.retain(|r| r.trace_id.starts_with(p.as_str()));
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
    for root in &roots {
        let tokens: (u32, u32) =
            runs.iter()
                .filter(|r| r.trace_id == root.trace_id)
                .fold((0, 0), |a, r| {
                    (
                        a.0 + r.input_tokens.unwrap_or(0),
                        a.1 + r.output_tokens.unwrap_or(0),
                    )
                });
        println!(
            "trace {}  total tokens {}→{}",
            &root.trace_id[..8.min(root.trace_id.len())],
            tokens.0,
            tokens.1
        );
        print_tree(root, &children, "", true, true, opts.io);
        println!();
    }
    println!("{} runs, {} traces", runs.len(), roots.len());
    ExitCode::SUCCESS
}
