//! Filesystem tools rooted in a workspace directory.
//!
//! Every path is resolved against the workspace root. Paths that leave it,
//! lexically (`../..`, absolute paths elsewhere) or through symlinks, are
//! rejected.

use std::{
    fs,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use regex::RegexBuilder;
use serde_json::{json, Value};
use sinew_core::{Tool, ToolSpec};

use super::{arg_bool, arg_str, arg_u64, opt_str, tool_err};
use crate::glob::glob_to_regex;

/// Directories never descended into by `glob`, `grep` and `list_dir -r`.
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".venv", "__pycache__"];
/// Files larger than this are skipped by `grep` and refused by `read_file`.
const MAX_FILE_BYTES: u64 = 10 * 1024 * 1024;
const MAX_LINE_CHARS: usize = 2_000;

/// A directory the filesystem tools are confined to. Cheap to clone.
#[derive(Debug, Clone)]
pub struct Workspace {
    root: Arc<PathBuf>,
}

impl Workspace {
    /// The directory must exist; it is canonicalized.
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = root.as_ref().canonicalize()?;
        if !root.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotADirectory,
                format!("{} is not a directory", root.display()),
            ));
        }
        Ok(Self {
            root: Arc::new(root),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolves `path` (relative to the root, or absolute inside it) and
    /// checks that it stays inside the workspace, following symlinks of the
    /// parts that exist.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let path = path.trim();
        let p = Path::new(if path.is_empty() { "." } else { path });
        let joined = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        };
        let mut out = PathBuf::new();
        for c in joined.components() {
            match c {
                Component::Prefix(_) | Component::RootDir => out.push(c.as_os_str()),
                Component::CurDir => {}
                Component::ParentDir => {
                    out.pop();
                }
                Component::Normal(x) => out.push(x),
            }
        }
        let escape = || format!("path `{path}` is outside the workspace");
        if !out.starts_with(self.root.as_path()) {
            return Err(escape());
        }
        // The deepest part that exists must canonicalize inside the root
        // (catches symlinks pointing outside).
        let mut probe = out.clone();
        loop {
            if let Ok(meta) = fs::symlink_metadata(&probe) {
                match probe.canonicalize() {
                    Ok(real) if real.starts_with(self.root.as_path()) => break,
                    Ok(_) => return Err(escape()),
                    Err(_) if meta.file_type().is_symlink() => {
                        return Err(format!("path `{path}` is a dangling symlink"))
                    }
                    Err(e) => return Err(format!("cannot resolve `{path}`: {e}")),
                }
            }
            if !probe.pop() || !probe.starts_with(self.root.as_path()) {
                break;
            }
        }
        Ok(out)
    }

    /// `path` relative to the root, with `/` separators (`.` for the root).
    pub fn display(&self, path: &Path) -> String {
        let rel = path.strip_prefix(self.root.as_path()).unwrap_or(path);
        let s = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        if s.is_empty() {
            ".".into()
        } else {
            s
        }
    }

    /// Every file under `dir` (recursively, skipping [`SKIP_DIRS`]), at most
    /// `limit` entries.
    fn walk(&self, dir: &Path, limit: usize) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = fs::read_dir(&d) else { continue };
            let mut entries: Vec<_> = rd.flatten().collect();
            entries.sort_by_key(|e| e.file_name());
            for e in entries.into_iter().rev() {
                let Ok(ft) = e.file_type() else { continue };
                let name = e.file_name();
                if ft.is_dir() {
                    if !SKIP_DIRS.iter().any(|s| name == *s) {
                        stack.push(e.path());
                    }
                } else if ft.is_file() {
                    out.push(e.path());
                    if out.len() >= limit {
                        return out;
                    }
                }
            }
        }
        out.sort();
        out
    }
}

fn spec(name: &str, description: &str, parameters: Value) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: description.into(),
        parameters,
    }
}

/// `read_file`: numbered lines, optional `offset` (1-based) and `limit`.
pub struct ReadFile(pub Workspace);
/// `write_file`: create or overwrite a file (parent directories created).
pub struct WriteFile(pub Workspace);
/// `edit_file`: exact string replacement.
pub struct EditFile(pub Workspace);
/// `list_dir`: entries of a directory.
pub struct ListDir(pub Workspace);
/// `glob`: files matching a glob pattern.
pub struct GlobTool(pub Workspace);
/// `grep`: regex search over file contents.
pub struct GrepTool(pub Workspace);

#[async_trait]
impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        spec(
            "read_file",
            "Read a text file from the workspace. Returns numbered lines. Use offset (1-based line number) and limit to read a range of a large file.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root"},
                    "offset": {"type": "integer", "description": "First line to read (1-based, default 1)"},
                    "limit": {"type": "integer", "description": "Maximum lines to read (default 2000)"}
                },
                "required": ["path"]
            }),
        )
    }

    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        let name = "read_file";
        let path = self
            .0
            .resolve(arg_str(name, &args, "path")?)
            .map_err(|e| tool_err(name, e))?;
        let meta = fs::metadata(&path).map_err(|e| tool_err(name, e))?;
        if meta.is_dir() {
            return Err(tool_err(name, "path is a directory; use list_dir"));
        }
        if meta.len() > MAX_FILE_BYTES {
            return Err(tool_err(name, "file is too large (over 10 MB)"));
        }
        let bytes = fs::read(&path).map_err(|e| tool_err(name, e))?;
        if bytes.iter().take(8192).any(|b| *b == 0) {
            return Err(tool_err(name, "file looks binary"));
        }
        let text = String::from_utf8_lossy(&bytes);
        let offset = arg_u64(&args, "offset").unwrap_or(1).max(1) as usize;
        let limit = arg_u64(&args, "limit").unwrap_or(2000).max(1) as usize;
        let total = text.lines().count();
        if total == 0 {
            return Ok(json!("(empty file)"));
        }
        if offset > total {
            return Err(tool_err(
                name,
                format!("offset {offset} is past the end of the file ({total} lines)"),
            ));
        }
        let mut out = String::new();
        let mut last = offset;
        for (i, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
            last = i + 1;
            if line.chars().count() > MAX_LINE_CHARS {
                let cut: String = line.chars().take(MAX_LINE_CHARS).collect();
                out.push_str(&format!("{:>6}\t{cut}… [line truncated]\n", i + 1));
            } else {
                out.push_str(&format!("{:>6}\t{line}\n", i + 1));
            }
        }
        if offset > 1 || last < total {
            out.push_str(&format!(
                "(showing lines {offset}-{last} of {total}; use offset/limit to read more)\n"
            ));
        }
        Ok(Value::String(out))
    }
}

#[async_trait]
impl Tool for WriteFile {
    fn spec(&self) -> ToolSpec {
        spec(
            "write_file",
            "Create or overwrite a file in the workspace with the given content. Parent directories are created.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root"},
                    "content": {"type": "string", "description": "Full file content"}
                },
                "required": ["path", "content"]
            }),
        )
    }

    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        let name = "write_file";
        let rel = arg_str(name, &args, "path")?;
        let content = arg_str(name, &args, "content")?;
        let path = self.0.resolve(rel).map_err(|e| tool_err(name, e))?;
        if path.is_dir() {
            return Err(tool_err(name, "path is a directory"));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| tool_err(name, e))?;
        }
        let existed = path.exists();
        fs::write(&path, content).map_err(|e| tool_err(name, e))?;
        Ok(json!(format!(
            "{} {} ({} bytes, {} lines)",
            if existed { "Overwrote" } else { "Created" },
            self.0.display(&path),
            content.len(),
            content.lines().count()
        )))
    }
}

#[async_trait]
impl Tool for EditFile {
    fn spec(&self) -> ToolSpec {
        spec(
            "edit_file",
            "Replace an exact string in a file. old_string must appear exactly once unless replace_all is true. Read the file first and copy old_string exactly, including whitespace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root"},
                    "old_string": {"type": "string", "description": "Exact text to replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"}
                },
                "required": ["path", "old_string", "new_string"]
            }),
        )
    }

    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        let name = "edit_file";
        let path = self
            .0
            .resolve(arg_str(name, &args, "path")?)
            .map_err(|e| tool_err(name, e))?;
        let old = arg_str(name, &args, "old_string")?;
        let new = arg_str(name, &args, "new_string")?;
        let all = arg_bool(&args, "replace_all").unwrap_or(false);
        if old.is_empty() {
            return Err(tool_err(
                name,
                "old_string is empty; use write_file to create files",
            ));
        }
        if old == new {
            return Err(tool_err(name, "old_string and new_string are identical"));
        }
        let text = fs::read_to_string(&path).map_err(|e| tool_err(name, e))?;
        let count = text.matches(old).count();
        if count == 0 {
            return Err(tool_err(name, "old_string not found in the file"));
        }
        if count > 1 && !all {
            return Err(tool_err(
                name,
                format!("old_string appears {count} times; add surrounding context to make it unique or set replace_all"),
            ));
        }
        let updated = if all {
            text.replace(old, new)
        } else {
            text.replacen(old, new, 1)
        };
        fs::write(&path, updated).map_err(|e| tool_err(name, e))?;
        Ok(json!(format!(
            "Edited {} ({count} replacement{})",
            self.0.display(&path),
            if count == 1 { "" } else { "s" }
        )))
    }
}

#[async_trait]
impl Tool for ListDir {
    fn spec(&self) -> ToolSpec {
        spec(
            "list_dir",
            "List a directory in the workspace. Directories end with `/`; files show their size.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory relative to the workspace root (default: the root)"}
                }
            }),
        )
    }

    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        let name = "list_dir";
        let dir = self
            .0
            .resolve(opt_str(&args, "path").unwrap_or("."))
            .map_err(|e| tool_err(name, e))?;
        let mut entries: Vec<(String, bool, u64)> = fs::read_dir(&dir)
            .map_err(|e| tool_err(name, e))?
            .flatten()
            .map(|e| {
                let meta = e.metadata().ok();
                (
                    e.file_name().to_string_lossy().into_owned(),
                    meta.as_ref().is_some_and(|m| m.is_dir()),
                    meta.map_or(0, |m| m.len()),
                )
            })
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        if entries.is_empty() {
            return Ok(json!(format!("{} is empty", self.0.display(&dir))));
        }
        let lines: Vec<String> = entries
            .into_iter()
            .take(1000)
            .map(|(n, is_dir, size)| {
                if is_dir {
                    format!("{n}/")
                } else {
                    format!("{n}  ({size} bytes)")
                }
            })
            .collect();
        Ok(Value::String(lines.join("\n")))
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn spec(&self) -> ToolSpec {
        spec(
            "glob",
            "Find files by glob pattern, e.g. `**/*.rs` or `src/**/mod.rs`. A pattern without `/` matches file names at any depth. Returns paths relative to the workspace.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "path": {"type": "string", "description": "Directory to search (default: the root)"}
                },
                "required": ["pattern"]
            }),
        )
    }

    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        let name = "glob";
        let pattern = arg_str(name, &args, "pattern")?;
        let base = self
            .0
            .resolve(opt_str(&args, "path").unwrap_or("."))
            .map_err(|e| tool_err(name, e))?;
        let re = regex::Regex::new(&glob_to_regex(pattern.trim_start_matches("./")))
            .map_err(|e| tool_err(name, format!("bad pattern: {e}")))?;
        let by_name = !pattern.contains('/');
        let mut hits = Vec::new();
        for f in self.0.walk(&base, 50_000) {
            let rel = f.strip_prefix(&base).unwrap_or(&f);
            let rel_s = rel.to_string_lossy().replace('\\', "/");
            let target = if by_name {
                rel.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            } else {
                rel_s
            };
            if re.is_match(&target) {
                hits.push(self.0.display(&f));
            }
        }
        if hits.is_empty() {
            return Ok(json!("no files matched"));
        }
        let total = hits.len();
        hits.truncate(500);
        let mut out = hits.join("\n");
        if total > 500 {
            out.push_str(&format!("\n({total} matches; showing the first 500)"));
        }
        Ok(Value::String(out))
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn spec(&self) -> ToolSpec {
        spec(
            "grep",
            "Search file contents with a regular expression. Returns `path:line: text` matches. Optionally restrict to files matching a glob.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression (Rust regex syntax)"},
                    "path": {"type": "string", "description": "File or directory to search (default: the root)"},
                    "glob": {"type": "string", "description": "Only search files whose name/path matches this glob, e.g. `*.rs`"},
                    "case_insensitive": {"type": "boolean"},
                    "max_results": {"type": "integer", "description": "Default 200"}
                },
                "required": ["pattern"]
            }),
        )
    }

    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        let name = "grep";
        let pattern = arg_str(name, &args, "pattern")?;
        let re = RegexBuilder::new(pattern)
            .case_insensitive(arg_bool(&args, "case_insensitive").unwrap_or(false))
            .build()
            .map_err(|e| tool_err(name, format!("bad regex: {e}")))?;
        let base = self
            .0
            .resolve(opt_str(&args, "path").unwrap_or("."))
            .map_err(|e| tool_err(name, e))?;
        let filter = match opt_str(&args, "glob") {
            Some(g) => Some((
                regex::Regex::new(&glob_to_regex(g.trim_start_matches("./")))
                    .map_err(|e| tool_err(name, format!("bad glob: {e}")))?,
                !g.contains('/'),
            )),
            None => None,
        };
        let max = arg_u64(&args, "max_results").unwrap_or(200).max(1) as usize;
        let files = if base.is_file() {
            vec![base.clone()]
        } else {
            self.0.walk(&base, 50_000)
        };
        let mut hits = Vec::new();
        let mut truncated = false;
        'files: for f in files {
            if let Some((re, by_name)) = &filter {
                let rel = f.strip_prefix(&base).unwrap_or(&f);
                let target = if *by_name {
                    rel.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default()
                } else {
                    rel.to_string_lossy().replace('\\', "/")
                };
                if !re.is_match(&target) {
                    continue;
                }
            }
            if fs::metadata(&f).map_or(true, |m| m.len() > MAX_FILE_BYTES) {
                continue;
            }
            let Ok(bytes) = fs::read(&f) else { continue };
            if bytes.iter().take(8192).any(|b| *b == 0) {
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);
            for (i, line) in text.lines().enumerate() {
                if re.is_match(line) {
                    if hits.len() >= max {
                        truncated = true;
                        break 'files;
                    }
                    let shown: String = line.trim_end().chars().take(300).collect();
                    hits.push(format!("{}:{}: {shown}", self.0.display(&f), i + 1));
                }
            }
        }
        if hits.is_empty() {
            return Ok(json!("no matches"));
        }
        let mut out = hits.join("\n");
        if truncated {
            out.push_str(&format!("\n(stopped after {max} matches)"));
        }
        Ok(Value::String(out))
    }
}

/// The filesystem tools for a workspace; `read_only` leaves out
/// `write_file` and `edit_file`.
pub fn fs_tools(ws: &Workspace, read_only: bool) -> Vec<Arc<dyn Tool>> {
    let mut v: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFile(ws.clone())),
        Arc::new(ListDir(ws.clone())),
        Arc::new(GlobTool(ws.clone())),
        Arc::new(GrepTool(ws.clone())),
    ];
    if !read_only {
        v.push(Arc::new(WriteFile(ws.clone())));
        v.push(Arc::new(EditFile(ws.clone())));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src/sub")).unwrap();
        fs::write(
            dir.path().join("src/main.rs"),
            "fn main() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("src/sub/lib.rs"),
            "pub fn add() {}\n// TODO: more\n",
        )
        .unwrap();
        fs::write(dir.path().join("README.md"), "# Demo\nTODO list\n").unwrap();
        let ws = Workspace::new(dir.path()).unwrap();
        (dir, ws)
    }

    #[test]
    fn resolve_blocks_escapes() {
        let (_d, ws) = ws();
        assert!(ws.resolve("src/main.rs").is_ok());
        assert!(ws.resolve("./src/../README.md").is_ok());
        assert!(ws.resolve("new/dir/file.txt").is_ok());
        assert!(ws
            .resolve("../outside.txt")
            .unwrap_err()
            .contains("outside"));
        assert!(ws.resolve("src/../../x").is_err());
        assert!(ws.resolve("/etc/passwd").is_err());
        let inside = ws.root().join("src/main.rs");
        assert!(ws.resolve(inside.to_str().unwrap()).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn resolve_blocks_symlink_escape() {
        let (_d, ws) = ws();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), "s").unwrap();
        std::os::unix::fs::symlink(outside.path(), ws.root().join("link")).unwrap();
        assert!(ws.resolve("link/secret.txt").is_err());
        assert!(ws.resolve("link/new.txt").is_err());
    }

    #[tokio::test]
    async fn read_write_edit() {
        let (_d, ws) = ws();
        let out = ReadFile(ws.clone())
            .call(json!({"path": "src/main.rs", "offset": 2, "limit": 1}))
            .await
            .unwrap();
        let s = out.as_str().unwrap();
        assert!(s.starts_with("     2\t    println!"), "{s}");
        assert!(s.contains("lines 2-2 of 3"));

        WriteFile(ws.clone())
            .call(json!({"path": "notes/a.txt", "content": "one\ntwo two\n"}))
            .await
            .unwrap();
        let e = EditFile(ws.clone())
            .call(json!({"path": "notes/a.txt", "old_string": "two", "new_string": "2"}))
            .await
            .unwrap_err();
        assert!(e.to_string().contains("appears 2 times"));
        EditFile(ws.clone())
            .call(json!({"path": "notes/a.txt", "old_string": "two", "new_string": "2", "replace_all": true}))
            .await
            .unwrap();
        assert_eq!(
            fs::read_to_string(ws.root().join("notes/a.txt")).unwrap(),
            "one\n2 2\n"
        );
        assert!(WriteFile(ws.clone())
            .call(json!({"path": "../evil.txt", "content": "x"}))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn list_glob_grep() {
        let (_d, ws) = ws();
        let l = ListDir(ws.clone()).call(json!({})).await.unwrap();
        assert_eq!(l.as_str().unwrap().lines().next(), Some("src/"));
        let g = GlobTool(ws.clone())
            .call(json!({"pattern": "*.rs"}))
            .await
            .unwrap();
        assert_eq!(g.as_str().unwrap(), "src/main.rs\nsrc/sub/lib.rs");
        let g = GlobTool(ws.clone())
            .call(json!({"pattern": "src/*.rs"}))
            .await
            .unwrap();
        assert_eq!(g.as_str().unwrap(), "src/main.rs");
        let r = GrepTool(ws.clone())
            .call(json!({"pattern": "todo", "case_insensitive": true, "glob": "*.rs"}))
            .await
            .unwrap();
        assert_eq!(r.as_str().unwrap(), "src/sub/lib.rs:2: // TODO: more");
    }
}
