//! Document loaders: plain text, Markdown, HTML, JSON / JSONL, CSV, PDF
//! (feature `pdf`) and a directory walker that picks a loader per file.
//!
//! Loaders are synchronous (file I/O is fast and they are usually run once at
//! indexing time); wrap them in `tokio::task::spawn_blocking` inside a busy
//! server. Every document gets a `source` metadata field with its path.

pub(crate) mod csv;
mod directory;
pub(crate) mod html;
pub(crate) mod json;
#[cfg(feature = "pdf")]
mod pdf;

use std::path::{Path, PathBuf};

pub use self::csv::CsvLoader;
pub use directory::DirectoryLoader;
pub use html::{html_to_text, HtmlLoader};
pub use json::JsonLoader;
#[cfg(feature = "pdf")]
pub use pdf::PdfLoader;

use crate::{Document, Error, Metadata, Result};

/// Produces documents.
pub trait Loader: Send + Sync {
    fn load(&self) -> Result<Vec<Document>>;
}

impl<F> Loader for F
where
    F: Fn() -> Result<Vec<Document>> + Send + Sync,
{
    fn load(&self) -> Result<Vec<Document>> {
        self()
    }
}

pub(crate) fn read_to_string(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    Ok(match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    })
}

pub(crate) fn source_metadata(path: &Path) -> Metadata {
    let mut m = Metadata::new();
    m.insert("source".into(), path.display().to_string().into());
    m
}

/// Loads a whole file as one document (invalid UTF-8 is replaced).
#[derive(Debug, Clone)]
pub struct TextLoader {
    path: PathBuf,
}

impl TextLoader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl Loader for TextLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let content = read_to_string(&self.path)?;
        Ok(vec![
            Document::new(content).with_metadata_map(source_metadata(&self.path))
        ])
    }
}

/// Loads a Markdown file as one document.
///
/// A leading YAML-style front matter block (`---` ... `---`) is removed from
/// the content and its flat `key: value` lines become metadata; the first
/// `# heading` is stored as `title` unless the front matter set one.
#[derive(Debug, Clone)]
pub struct MarkdownLoader {
    path: PathBuf,
}

impl MarkdownLoader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl Loader for MarkdownLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let text = read_to_string(&self.path)?;
        let mut metadata = source_metadata(&self.path);
        let content = parse_markdown(&text, &mut metadata);
        Ok(vec![Document::new(content).with_metadata_map(metadata)])
    }
}

pub(crate) fn parse_markdown(text: &str, metadata: &mut Metadata) -> String {
    let mut body = text;
    let trimmed = text.trim_start_matches('\u{feff}');
    if let Some(rest) = trimmed
        .strip_prefix("---\n")
        .or_else(|| trimmed.strip_prefix("---\r\n"))
    {
        if let Some(end) = rest.find("\n---") {
            for line in rest[..end].lines() {
                if let Some((k, v)) = line.split_once(':') {
                    let (k, v) = (k.trim(), v.trim().trim_matches('"').trim_matches('\''));
                    if !k.is_empty() && !k.starts_with(' ') && !v.is_empty() {
                        metadata.insert(k.to_string(), v.to_string().into());
                    }
                }
            }
            let after = &rest[end + 4..];
            body = after.split_once('\n').map(|(_, b)| b).unwrap_or("");
        }
    }
    if !metadata.contains_key("title") {
        if let Some(t) = body
            .lines()
            .find_map(|l| l.strip_prefix("# ").map(str::trim))
        {
            metadata.insert("title".into(), t.to_string().into());
        }
    }
    body.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_front_matter_and_title() {
        let mut m = Metadata::new();
        let body = parse_markdown(
            "---\nauthor: Ana\ntags: \"x\"\n---\n# Intro\n\nHello\n",
            &mut m,
        );
        assert_eq!(body, "# Intro\n\nHello\n");
        assert_eq!(m["author"], "Ana");
        assert_eq!(m["tags"], "x");
        assert_eq!(m["title"], "Intro");
    }

    #[test]
    fn text_loader_reads_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, "hello").unwrap();
        let docs = TextLoader::new(&p).load().unwrap();
        assert_eq!(docs[0].content, "hello");
        assert_eq!(docs[0].source(), Some(p.display().to_string().as_str()));
        assert!(TextLoader::new(dir.path().join("missing")).load().is_err());
    }
}
