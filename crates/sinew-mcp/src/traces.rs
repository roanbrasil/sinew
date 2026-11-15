//! Read-only tools over a Sinew JSONL trace file (the format written by
//! `sinew_trace::JsonlSink`). Served by the `sinew-mcp` binary so coding
//! assistants can inspect agent runs.
//!
//! The file is re-read on every call, so new traces show up while an app
//! keeps appending to it.

use std::{
    collections::{BTreeMap, HashMap},
    fmt::Write as _,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{Error, Result, Tool, ToolBox, ToolSpec};
use sinew_trace::{Run, RunKind};

/// Loads every parseable run; malformed lines are counted, not fatal.
pub fn load_runs(path: &Path) -> std::io::Result<(Vec<Run>, usize)> {
    let text = std::fs::read_to_string(path)?;
    let mut runs = Vec::new();
    let mut skipped = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match serde_json::from_str::<Run>(line) {
            Ok(r) => runs.push(r),
            Err(_) => skipped += 1,
        }
    }
    runs.sort_by_key(|r| r.start_ms);
    Ok((runs, skipped))
}

/// Aggregate view of one trace.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceSummary {
    pub trace_id: String,
    pub root_name: String,
    pub root_kind: RunKind,
    pub start_ms: u64,
    pub duration_ms: Option<u64>,
    pub runs: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub errors: usize,
    pub first_error: Option<String>,
}

/// One summary per trace, newest first.
pub fn summarize(runs: &[Run]) -> Vec<TraceSummary> {
    let mut by_trace: BTreeMap<&str, Vec<&Run>> = BTreeMap::new();
    for r in runs {
        by_trace.entry(r.trace_id.as_str()).or_default().push(r);
    }
    let mut out: Vec<TraceSummary> = by_trace
        .into_iter()
        .map(|(id, rs)| {
            // The root is the run whose id is the trace id; if it has not
            // finished yet, fall back to the earliest run.
            let root = rs
                .iter()
                .find(|r| r.id == id)
                .or_else(|| rs.iter().min_by_key(|r| r.start_ms))
                .copied()
                .expect("trace has at least one run");
            let start = rs.iter().map(|r| r.start_ms).min().unwrap_or(0);
            let end = rs.iter().filter_map(|r| r.end_ms).max();
            let errors: Vec<&Run> = rs.iter().filter(|r| r.error.is_some()).copied().collect();
            TraceSummary {
                trace_id: id.to_string(),
                root_name: root.name.clone(),
                root_kind: root.kind,
                start_ms: start,
                duration_ms: if root.id == id {
                    root.duration_ms()
                } else {
                    end.map(|e| e.saturating_sub(start))
                },
                runs: rs.len(),
                input_tokens: rs.iter().map(|r| r.input_tokens.unwrap_or(0) as u64).sum(),
                output_tokens: rs.iter().map(|r| r.output_tokens.unwrap_or(0) as u64).sum(),
                errors: errors.len(),
                // The deepest failure is usually the cause; parents repeat it.
                first_error: errors
                    .iter()
                    .min_by_key(|r| r.end_ms.unwrap_or(u64::MAX))
                    .and_then(|r| r.error.clone()),
            }
        })
        .collect();
    out.sort_by_key(|t| std::cmp::Reverse(t.start_ms));
    out
}

fn kind_str(k: RunKind) -> &'static str {
    match k {
        RunKind::Graph => "graph",
        RunKind::Node => "node",
        RunKind::Llm => "llm",
        RunKind::Tool => "tool",
        RunKind::Chain => "chain",
    }
}

fn fmt_ms(ms: Option<u64>) -> String {
    match ms {
        None => "open".into(),
        Some(ms) if ms >= 1000 => format!("{:.2}s", ms as f64 / 1000.0),
        Some(ms) => format!("{ms}ms"),
    }
}

/// UTC timestamp `YYYY-MM-DD HH:MM:SS` from Unix milliseconds.
pub fn fmt_time(ms: u64) -> String {
    let secs = ms / 1000;
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil-from-days (Howard Hinnant), valid for all dates after 1970.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn short(id: &str) -> &str {
    &id[..8.min(id.len())]
}

fn preview(v: &Value, max: usize) -> String {
    let s = match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let s = s.replace('\n', " ");
    if s.chars().count() > max {
        format!("{}…", s.chars().take(max).collect::<String>())
    } else {
        s
    }
}

fn run_line(r: &Run) -> String {
    let mut line = format!(
        "{} {}  {}",
        kind_str(r.kind),
        r.name,
        fmt_ms(r.duration_ms())
    );
    if let (Some(i), Some(o)) = (r.input_tokens, r.output_tokens) {
        let _ = write!(line, "  tokens {i}→{o}");
    }
    if let Some(e) = &r.error {
        let _ = write!(line, "  ERROR: {e}");
    }
    line
}

/// Renders the runs of one trace as an indented tree.
pub fn render_tree(runs: &[&Run], io: bool, io_chars: usize) -> String {
    let mut children: HashMap<&str, Vec<&Run>> = HashMap::new();
    for r in runs {
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
        .copied()
        .collect();
    let mut out = String::new();
    fn walk(
        r: &Run,
        children: &HashMap<&str, Vec<&Run>>,
        prefix: &str,
        branch: &str,
        child_prefix: &str,
        io: Option<usize>,
        out: &mut String,
    ) {
        let _ = writeln!(out, "{prefix}{branch}{}", run_line(r));
        let inner = format!("{prefix}{child_prefix}");
        if let Some(max) = io {
            let _ = writeln!(out, "{inner}   in : {}", preview(&r.inputs, max));
            if let Some(o) = &r.outputs {
                let _ = writeln!(out, "{inner}   out: {}", preview(o, max));
            }
        }
        if let Some(kids) = children.get(r.id.as_str()) {
            for (i, k) in kids.iter().enumerate() {
                let last = i + 1 == kids.len();
                let (b, c) = if last {
                    ("└─ ", "   ")
                } else {
                    ("├─ ", "│  ")
                };
                walk(k, children, &inner, b, c, io, out);
            }
        }
    }
    let io = io.then_some(io_chars);
    for root in roots {
        walk(root, &children, "", "", "", io, &mut out);
    }
    out
}

/// Builds the toolbox served by the `sinew-mcp` binary.
pub fn trace_toolbox(path: impl Into<PathBuf>) -> ToolBox {
    let path: Arc<PathBuf> = Arc::new(path.into());
    ToolBox::new()
        .with(TraceTool::new(TraceOp::List, path.clone()))
        .with(TraceTool::new(TraceOp::Get, path.clone()))
        .with(TraceTool::new(TraceOp::Search, path))
}

#[derive(Clone, Copy)]
enum TraceOp {
    List,
    Get,
    Search,
}

struct TraceTool {
    op: TraceOp,
    path: Arc<PathBuf>,
}

impl TraceTool {
    fn new(op: TraceOp, path: Arc<PathBuf>) -> Self {
        Self { op, path }
    }

    fn name(&self) -> &'static str {
        match self.op {
            TraceOp::List => "list_traces",
            TraceOp::Get => "get_trace",
            TraceOp::Search => "search_runs",
        }
    }

    fn fail(&self, message: impl Into<String>) -> Error {
        Error::Tool {
            name: self.name().into(),
            message: message.into(),
        }
    }

    fn load(&self) -> Result<(Vec<Run>, usize)> {
        load_runs(&self.path)
            .map_err(|e| self.fail(format!("cannot read {}: {e}", self.path.display())))
    }

    fn list(&self, args: &Value) -> Result<String> {
        let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(20) as usize;
        let errors_only = args
            .get("errors_only")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let (runs, skipped) = self.load()?;
        let all = summarize(&runs);
        let shown: Vec<&TraceSummary> = all
            .iter()
            .filter(|t| !errors_only || t.errors > 0)
            .take(limit)
            .collect();
        let mut out = format!(
            "{} traces in {} ({} runs{}); showing {}, newest first\n",
            all.len(),
            self.path.display(),
            runs.len(),
            if skipped > 0 {
                format!(", {skipped} malformed lines skipped")
            } else {
                String::new()
            },
            shown.len()
        );
        for t in shown {
            let _ = write!(
                out,
                "{}  {} UTC  {} {}  {}  {} runs  tokens {}→{}",
                short(&t.trace_id),
                fmt_time(t.start_ms),
                kind_str(t.root_kind),
                t.root_name,
                fmt_ms(t.duration_ms),
                t.runs,
                t.input_tokens,
                t.output_tokens
            );
            if t.errors > 0 {
                let _ = write!(
                    out,
                    "  {} errors: {}",
                    t.errors,
                    preview(&json!(t.first_error.clone().unwrap_or_default()), 120)
                );
            }
            out.push('\n');
        }
        Ok(out)
    }

    fn get(&self, args: &Value) -> Result<String> {
        let prefix = args
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| self.fail("`id` (trace id or prefix) is required"))?;
        let io = args.get("io").and_then(Value::as_bool).unwrap_or(false);
        let io_chars = args.get("io_chars").and_then(Value::as_u64).unwrap_or(300) as usize;
        let (runs, _) = self.load()?;
        let mut ids: Vec<&str> = runs
            .iter()
            .map(|r| r.trace_id.as_str())
            .filter(|t| t.starts_with(prefix))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let id = match ids.as_slice() {
            [] => return Err(self.fail(format!("no trace matches `{prefix}`"))),
            [one] => *one,
            many => {
                return Err(self.fail(format!(
                    "`{prefix}` is ambiguous; it matches {}",
                    many.iter().map(|s| short(s)).collect::<Vec<_>>().join(", ")
                )))
            }
        };
        let trace_runs: Vec<&Run> = runs.iter().filter(|r| r.trace_id == id).collect();
        let summary = summarize(&trace_runs.iter().map(|r| (*r).clone()).collect::<Vec<_>>())
            .pop()
            .expect("trace has runs");
        let mut out = format!(
            "trace {id}\nstarted {} UTC  duration {}  {} runs  tokens {}→{}  errors {}\n\n",
            fmt_time(summary.start_ms),
            fmt_ms(summary.duration_ms),
            summary.runs,
            summary.input_tokens,
            summary.output_tokens,
            summary.errors
        );
        out.push_str(&render_tree(&trace_runs, io, io_chars));
        Ok(out)
    }

    fn search(&self, args: &Value) -> Result<String> {
        let name = args
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_lowercase);
        let kind = match args.get("kind").and_then(Value::as_str) {
            None => None,
            Some(k) => Some(
                serde_json::from_value::<RunKind>(json!(k.to_lowercase())).map_err(|_| {
                    self.fail(format!(
                        "unknown kind `{k}`; use graph, node, llm, tool or chain"
                    ))
                })?,
            ),
        };
        let error = args.get("error");
        let error_text = error.and_then(Value::as_str).map(str::to_lowercase);
        let errors_only = error.and_then(Value::as_bool);
        let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(50) as usize;
        let (runs, _) = self.load()?;
        let matches: Vec<&Run> = runs
            .iter()
            .rev()
            .filter(|r| {
                name.as_ref()
                    .is_none_or(|n| r.name.to_lowercase().contains(n))
            })
            .filter(|r| kind.is_none_or(|k| r.kind == k))
            .filter(|r| errors_only.is_none_or(|want| r.error.is_some() == want))
            .filter(|r| {
                error_text.as_ref().is_none_or(|t| {
                    r.error
                        .as_ref()
                        .is_some_and(|e| e.to_lowercase().contains(t))
                })
            })
            .collect();
        let mut out = format!(
            "{} matching runs; showing {}, newest first\n",
            matches.len(),
            matches.len().min(limit)
        );
        for r in matches.into_iter().take(limit) {
            let _ = writeln!(
                out,
                "trace {}  run {}  {} UTC  {}",
                short(&r.trace_id),
                short(&r.id),
                fmt_time(r.start_ms),
                run_line(r)
            );
        }
        Ok(out)
    }
}

#[async_trait]
impl Tool for TraceTool {
    fn spec(&self) -> ToolSpec {
        let (description, parameters) = match self.op {
            TraceOp::List => (
                "List recent Sinew traces, newest first, with duration, run count, token usage and errors.",
                json!({
                    "type": "object",
                    "properties": {
                        "limit": { "type": "integer", "minimum": 1, "description": "Max traces to show (default 20)" },
                        "errors_only": { "type": "boolean", "description": "Only traces with at least one failed run" }
                    }
                }),
            ),
            TraceOp::Get => (
                "Show one trace as a run tree (graph → node → llm/tool) with timings, tokens and errors.",
                json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Trace id or a unique prefix (8 chars is usually enough)" },
                        "io": { "type": "boolean", "description": "Include truncated inputs and outputs of each run" },
                        "io_chars": { "type": "integer", "minimum": 20, "description": "Truncation length for io (default 300)" }
                    },
                    "required": ["id"]
                }),
            ),
            TraceOp::Search => (
                "Search runs across all traces by name substring, kind and error.",
                json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Case-insensitive substring of the run name (node, model or tool)" },
                        "kind": { "type": "string", "enum": ["graph", "node", "llm", "tool", "chain"] },
                        "error": {
                            "description": "true: failed runs only; false: successful only; a string: error message contains it",
                            "type": ["boolean", "string"]
                        },
                        "limit": { "type": "integer", "minimum": 1, "description": "Max runs to show (default 50)" }
                    }
                }),
            ),
        };
        ToolSpec {
            name: self.name().into(),
            description: description.into(),
            parameters,
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value> {
        let text = match self.op {
            TraceOp::List => self.list(&arguments),
            TraceOp::Get => self.get(&arguments),
            TraceOp::Search => self.search(&arguments),
        }?;
        Ok(Value::String(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(id: &str, trace: &str, parent: Option<&str>, name: &str, kind: RunKind) -> Run {
        Run {
            id: id.into(),
            trace_id: trace.into(),
            parent_id: parent.map(Into::into),
            name: name.into(),
            kind,
            start_ms: 1_700_000_000_000,
            end_ms: Some(1_700_000_000_250),
            inputs: json!({"q": "hi"}),
            outputs: Some(json!("ok")),
            error: None,
            input_tokens: None,
            output_tokens: None,
            metadata: json!({}),
        }
    }

    fn sample() -> Vec<Run> {
        let mut llm = run("l1", "t1", Some("n1"), "qwen", RunKind::Llm);
        llm.input_tokens = Some(10);
        llm.output_tokens = Some(3);
        let mut tool = run("x1", "t1", Some("n1"), "multiply", RunKind::Tool);
        tool.error = Some("bad args".into());
        let mut other = run("t2", "t2", None, "graph", RunKind::Graph);
        other.start_ms += 5_000;
        other.end_ms = Some(other.start_ms + 1500);
        vec![
            llm,
            tool,
            run("n1", "t1", Some("t1"), "agent", RunKind::Node),
            run("t1", "t1", None, "graph", RunKind::Graph),
            other,
        ]
    }

    #[test]
    fn summarizes_traces() {
        let s = summarize(&sample());
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].trace_id, "t2", "newest first");
        let t1 = &s[1];
        assert_eq!((t1.runs, t1.input_tokens, t1.output_tokens), (4, 10, 3));
        assert_eq!(t1.errors, 1);
        assert_eq!(t1.first_error.as_deref(), Some("bad args"));
        assert_eq!(t1.duration_ms, Some(250));
    }

    #[test]
    fn renders_tree() {
        let runs = sample();
        let refs: Vec<&Run> = runs.iter().filter(|r| r.trace_id == "t1").collect();
        let tree = render_tree(&refs, false, 100);
        assert_eq!(
            tree,
            "graph graph  250ms\n└─ node agent  250ms\n   ├─ llm qwen  250ms  tokens 10→3\n   └─ tool multiply  250ms  ERROR: bad args\n"
        );
    }

    #[test]
    fn formats_utc_time() {
        assert_eq!(fmt_time(0), "1970-01-01 00:00:00");
        assert_eq!(fmt_time(1_700_000_000_000), "2023-11-14 22:13:20");
        assert_eq!(fmt_time(951_782_400_000), "2000-02-29 00:00:00");
    }
}
