use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use globset::{Glob, GlobSet, GlobSetBuilder};

use super::{CsvLoader, HtmlLoader, JsonLoader, Loader, MarkdownLoader, TextLoader};
use crate::{Document, Error, Result};

type FileLoader = Arc<dyn Fn(&Path) -> Result<Vec<Document>> + Send + Sync>;

/// Walks a directory and loads every file that matches the glob patterns,
/// picking a loader from the extension:
///
/// | Extension | Loader |
/// |---|---|
/// | `md`, `markdown`, `mdx` | [`MarkdownLoader`] |
/// | `html`, `htm`, `xhtml` | [`HtmlLoader`] |
/// | `json` | [`JsonLoader`] |
/// | `jsonl`, `ndjson` | [`JsonLoader::json_lines`] |
/// | `csv` | [`CsvLoader`]; `tsv` with a tab delimiter |
/// | `pdf` | `PdfLoader` (feature `pdf`) |
/// | anything else | [`TextLoader`] |
///
/// Override or add extensions with [`with_loader`](Self::with_loader).
/// Patterns are matched against the path relative to the root
/// (`"**/*.md"`, `"docs/*.txt"`). Hidden files and directories are skipped
/// unless [`include_hidden`](Self::include_hidden). Files are visited in
/// sorted order so results are deterministic; each document also gets a
/// `relative_path` metadata field.
#[derive(Clone)]
pub struct DirectoryLoader {
    root: PathBuf,
    include: Vec<String>,
    exclude: Vec<String>,
    recursive: bool,
    hidden: bool,
    skip_errors: bool,
    loaders: HashMap<String, FileLoader>,
}

impl DirectoryLoader {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            include: Vec::new(),
            exclude: Vec::new(),
            recursive: true,
            hidden: false,
            skip_errors: false,
            loaders: HashMap::new(),
        }
    }

    /// Adds an include pattern (default: every file).
    pub fn glob(mut self, pattern: impl Into<String>) -> Self {
        self.include.push(pattern.into());
        self
    }

    /// Adds an exclude pattern, e.g. `"**/target/**"`.
    pub fn exclude(mut self, pattern: impl Into<String>) -> Self {
        self.exclude.push(pattern.into());
        self
    }

    /// Only the top-level directory.
    pub fn non_recursive(mut self) -> Self {
        self.recursive = false;
        self
    }

    pub fn include_hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Skip files that fail to load instead of returning the first error.
    pub fn skip_errors(mut self) -> Self {
        self.skip_errors = true;
        self
    }

    /// Uses `f` for files with extension `ext` (without the dot).
    pub fn with_loader<F>(mut self, ext: &str, f: F) -> Self
    where
        F: Fn(&Path) -> Result<Vec<Document>> + Send + Sync + 'static,
    {
        self.loaders.insert(ext.to_ascii_lowercase(), Arc::new(f));
        self
    }

    /// The files that would be loaded, sorted.
    pub fn paths(&self) -> Result<Vec<PathBuf>> {
        let include = build_set(&self.include)?;
        let exclude = build_set(&self.exclude)?;
        let mut walker = walkdir::WalkDir::new(&self.root)
            .follow_links(true)
            .sort_by_file_name();
        if !self.recursive {
            walker = walker.max_depth(1);
        }
        let hidden = self.hidden;
        let mut out = Vec::new();
        for entry in walker
            .into_iter()
            .filter_entry(|e| hidden || e.depth() == 0 || !is_hidden(e.file_name()))
        {
            let entry = entry.map_err(|e| Error::Load {
                path: e.path().unwrap_or(&self.root).to_path_buf(),
                message: e.to_string(),
            })?;
            if !entry.file_type().is_file() {
                continue;
            }
            let rel = entry
                .path()
                .strip_prefix(&self.root)
                .unwrap_or(entry.path());
            if include.as_ref().is_some_and(|s| !s.is_match(rel)) {
                continue;
            }
            if exclude.as_ref().is_some_and(|s| s.is_match(rel)) {
                continue;
            }
            out.push(entry.into_path());
        }
        Ok(out)
    }

    fn load_file(&self, path: &Path) -> Result<Vec<Document>> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if let Some(f) = self.loaders.get(&ext) {
            return f(path);
        }
        match ext.as_str() {
            "md" | "markdown" | "mdx" => MarkdownLoader::new(path).load(),
            "html" | "htm" | "xhtml" => HtmlLoader::new(path).load(),
            "json" => JsonLoader::new(path).load(),
            "jsonl" | "ndjson" => JsonLoader::json_lines(path).load(),
            "csv" => CsvLoader::new(path).load(),
            "tsv" => CsvLoader::new(path).with_delimiter(b'\t').load(),
            #[cfg(feature = "pdf")]
            "pdf" => super::PdfLoader::new(path).load(),
            _ => TextLoader::new(path).load(),
        }
    }
}

fn is_hidden(name: &std::ffi::OsStr) -> bool {
    name.to_str().is_some_and(|s| s.starts_with('.'))
}

fn build_set(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut b = GlobSetBuilder::new();
    for p in patterns {
        b.add(Glob::new(p).map_err(|e| Error::Invalid(format!("glob `{p}`: {e}")))?);
    }
    b.build()
        .map(Some)
        .map_err(|e| Error::Invalid(e.to_string()))
}

impl Loader for DirectoryLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let mut docs = Vec::new();
        for path in self.paths()? {
            match self.load_file(&path) {
                Ok(found) => {
                    let rel = path
                        .strip_prefix(&self.root)
                        .unwrap_or(&path)
                        .display()
                        .to_string();
                    docs.extend(found.into_iter().map(|mut d| {
                        d.metadata
                            .entry("relative_path")
                            .or_insert_with(|| rel.clone().into());
                        d
                    }));
                }
                Err(_) if self.skip_errors => {}
                Err(e) => return Err(e),
            }
        }
        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_with_globs_and_picks_loaders() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("sub/deep")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("a.md"), "# A\n\ntext").unwrap();
        std::fs::write(root.join("sub/b.html"), "<p>B &amp; b</p>").unwrap();
        std::fs::write(root.join("sub/deep/c.txt"), "C").unwrap();
        std::fs::write(root.join("sub/deep/d.log"), "D").unwrap();
        std::fs::write(root.join(".git/x.md"), "hidden").unwrap();

        let all = DirectoryLoader::new(root).load().unwrap();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0].metadata["title"], "A");

        let some = DirectoryLoader::new(root)
            .glob("**/*.html")
            .glob("**/*.txt")
            .exclude("sub/deep/**")
            .load()
            .unwrap();
        assert_eq!(some.len(), 1);
        assert_eq!(some[0].content, "B & b");
        assert_eq!(some[0].metadata["relative_path"], "sub/b.html");

        let top = DirectoryLoader::new(root).non_recursive().load().unwrap();
        assert_eq!(top.len(), 1);

        let custom = DirectoryLoader::new(root)
            .glob("**/*.log")
            .with_loader("log", |p| {
                Ok(vec![Document::new(format!("log:{}", p.display()))])
            })
            .load()
            .unwrap();
        assert!(custom[0].content.starts_with("log:"));
        assert!(DirectoryLoader::new(root).glob("[").paths().is_err());
    }
}
