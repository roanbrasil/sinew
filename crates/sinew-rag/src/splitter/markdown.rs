use crate::{Document, Metadata};

/// LangChain's `MarkdownHeaderTextSplitter`: splits Markdown on the given
/// headers and records the enclosing headers of every chunk as metadata.
///
/// ```
/// use sinew_rag::MarkdownHeaderTextSplitter;
/// let s = MarkdownHeaderTextSplitter::new([("#", "h1"), ("##", "h2")]);
/// let docs = s.split_text("# Guide\nintro\n## Install\ncargo add x");
/// assert_eq!(docs[1].content, "cargo add x");
/// assert_eq!(docs[1].metadata["h1"], "Guide");
/// assert_eq!(docs[1].metadata["h2"], "Install");
/// ```
///
/// Header lines inside fenced code blocks are ignored. Consecutive lines with
/// the same headers are merged into one chunk (paragraphs joined with `"  \n"`)
/// unless [`each_line`](Self::each_line). Header lines are removed from the
/// content unless [`keep_headers`](Self::keep_headers). Combine with a
/// [`RecursiveCharacterTextSplitter`](super::RecursiveCharacterTextSplitter)
/// to bound chunk size.
#[derive(Debug, Clone)]
pub struct MarkdownHeaderTextSplitter {
    /// (marker, metadata key), longest marker first.
    headers: Vec<(String, String)>,
    strip_headers: bool,
    each_line: bool,
}

struct Line {
    content: String,
    metadata: Metadata,
}

impl MarkdownHeaderTextSplitter {
    /// `headers` pairs a marker (`"#"`, `"##"`, ...) with a metadata key.
    pub fn new<I, A, B>(headers: I) -> Self
    where
        I: IntoIterator<Item = (A, B)>,
        A: Into<String>,
        B: Into<String>,
    {
        let mut headers: Vec<(String, String)> = headers
            .into_iter()
            .map(|(a, b)| (a.into(), b.into()))
            .collect();
        headers.sort_by_key(|h| std::cmp::Reverse(h.0.len()));
        Self {
            headers,
            strip_headers: true,
            each_line: false,
        }
    }

    /// Keep header lines in the chunk content.
    pub fn keep_headers(mut self) -> Self {
        self.strip_headers = false;
        self
    }

    /// One document per paragraph (block of consecutive non-blank lines)
    /// instead of merging a whole section.
    pub fn each_line(mut self) -> Self {
        self.each_line = true;
        self
    }

    pub fn split_text(&self, text: &str) -> Vec<Document> {
        let mut lines_out: Vec<Line> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        let mut current_meta = Metadata::new();
        let mut header_meta = Metadata::new();
        let mut stack: Vec<(usize, String)> = Vec::new();
        let mut fence: Option<&str> = None;

        let flush = |current: &mut Vec<String>, meta: &Metadata, out: &mut Vec<Line>| {
            if !current.is_empty() {
                out.push(Line {
                    content: current.join("\n"),
                    metadata: meta.clone(),
                });
                current.clear();
            }
        };

        for raw in text.lines() {
            let line = raw.trim();
            match fence {
                None if line.starts_with("```") && line.matches("```").count() == 1 => {
                    fence = Some("```")
                }
                None if line.starts_with("~~~") => fence = Some("~~~"),
                Some(f) if line.starts_with(f) => {
                    fence = None;
                    current.push(line.to_string());
                    current_meta = header_meta.clone();
                    continue;
                }
                _ => {}
            }
            if fence.is_some() {
                current.push(line.to_string());
                continue;
            }
            let header = self.headers.iter().find(|(marker, _)| {
                line.starts_with(marker.as_str())
                    && (line.len() == marker.len() || line[marker.len()..].starts_with(' '))
            });
            match header {
                Some((marker, name)) => {
                    let level = marker.matches('#').count();
                    while stack.last().is_some_and(|(l, _)| *l >= level) {
                        let (_, popped) = stack.pop().unwrap();
                        header_meta.remove(&popped);
                    }
                    stack.push((level, name.clone()));
                    header_meta.insert(name.clone(), line[marker.len()..].trim().into());
                    flush(&mut current, &current_meta, &mut lines_out);
                    if !self.strip_headers {
                        current.push(line.to_string());
                    }
                }
                None if !line.is_empty() => current.push(line.to_string()),
                None => flush(&mut current, &current_meta, &mut lines_out),
            }
            current_meta = header_meta.clone();
        }
        flush(&mut current, &current_meta, &mut lines_out);

        let lines = if self.each_line {
            lines_out
        } else {
            self.aggregate(lines_out)
        };
        lines
            .into_iter()
            .map(|l| Document::new(l.content).with_metadata_map(l.metadata))
            .collect()
    }

    fn aggregate(&self, lines: Vec<Line>) -> Vec<Line> {
        let mut out: Vec<Line> = Vec::new();
        for line in lines {
            if let Some(last) = out.last_mut() {
                if last.metadata == line.metadata {
                    last.content.push_str("  \n");
                    last.content.push_str(&line.content);
                    continue;
                }
                // Keep a header line together with its first content.
                let last_is_header = last
                    .content
                    .rsplit('\n')
                    .next()
                    .is_some_and(|l| l.starts_with('#'));
                if !self.strip_headers
                    && last.metadata.len() < line.metadata.len()
                    && last_is_header
                {
                    last.content.push_str("  \n");
                    last.content.push_str(&line.content);
                    last.metadata = line.metadata;
                    continue;
                }
            }
            out.push(line);
        }
        out
    }

    /// Splits every document; chunks keep the parent's metadata (header
    /// fields win on conflict) plus `chunk_index`.
    pub fn split_documents(&self, docs: &[Document]) -> Vec<Document> {
        let mut out = Vec::new();
        for d in docs {
            for (i, chunk) in self.split_text(&d.content).into_iter().enumerate() {
                let mut c = super::chunk_doc(d, i, chunk.content);
                c.metadata.extend(chunk.metadata);
                out.push(c);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splitter() -> MarkdownHeaderTextSplitter {
        MarkdownHeaderTextSplitter::new([
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ])
    }

    #[test]
    fn splits_on_headers_like_langchain() {
        let md = "# Foo\n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly";
        let docs = splitter().split_text(md);
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].content, "Hi this is Jim  \nHi this is Joe");
        assert_eq!(docs[0].metadata["Header 1"], "Foo");
        assert_eq!(docs[0].metadata["Header 2"], "Bar");
        assert_eq!(docs[1].content, "Hi this is Lance");
        assert_eq!(docs[1].metadata["Header 3"], "Boo");
        assert_eq!(docs[2].content, "Hi this is Molly");
        assert_eq!(docs[2].metadata["Header 2"], "Baz");
        assert!(!docs[2].metadata.contains_key("Header 3"));
    }

    #[test]
    fn ignores_headers_in_code_and_can_keep_them() {
        let md = "# A\n```\n# not a header\n```\ntext";
        let docs = splitter().split_text(md);
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("# not a header"));
        let kept = splitter()
            .keep_headers()
            .split_text("# A\nbody\n## B\nmore");
        assert_eq!(kept[0].content, "# A\nbody");
        assert_eq!(kept[1].content, "## B\nmore");
        // A header followed by a blank line stays with its first paragraph.
        let kept = splitter().keep_headers().split_text("# A\n\n## B\n\nmore");
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].content, "# A  \n## B  \nmore");
        assert_eq!(kept[0].metadata["Header 2"], "B");
        let lines = splitter().each_line().split_text("# A\none\ntwo");
        assert_eq!(lines.len(), 1);
    }
}
