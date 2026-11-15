use std::sync::Arc;

use super::{char_len, chunk_doc, Language, LengthFn, TextSplitter};
use crate::Document;

/// Where a separator goes when text is split on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepSeparator {
    /// Dropped; chunks are re-joined with it.
    None,
    /// At the start of the following piece (LangChain's default).
    Start,
    /// At the end of the preceding piece.
    End,
}

/// LangChain's `RecursiveCharacterTextSplitter`.
///
/// Tries the separators in order (default `"\n\n"`, `"\n"`, `" "`, `""`):
/// the text is split on the first one it contains, pieces shorter than
/// `chunk_size` are merged back into chunks of at most `chunk_size` with up
/// to `chunk_overlap` of trailing context repeated, and pieces that are still
/// too long are split again with the remaining separators. Lengths are
/// measured with the length function (characters by default). Separators are
/// literal strings.
#[derive(Clone)]
pub struct RecursiveCharacterTextSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
    keep_separator: KeepSeparator,
    strip_whitespace: bool,
    add_start_index: bool,
    length: LengthFn,
}

impl RecursiveCharacterTextSplitter {
    /// `chunk_overlap` is clamped to `chunk_size`.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        let chunk_size = chunk_size.max(1);
        Self {
            chunk_size,
            chunk_overlap: chunk_overlap.min(chunk_size),
            separators: ["\n\n", "\n", " ", ""].map(String::from).to_vec(),
            keep_separator: KeepSeparator::Start,
            strip_whitespace: true,
            add_start_index: false,
            length: Arc::new(char_len),
        }
    }

    /// Separators tuned for source code or markup in `language`.
    pub fn for_language(language: Language, chunk_size: usize, chunk_overlap: usize) -> Self {
        Self::new(chunk_size, chunk_overlap).with_separators(language.separators().iter().copied())
    }

    pub fn with_separators<I, S>(mut self, separators: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.separators = separators.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_keep_separator(mut self, keep: KeepSeparator) -> Self {
        self.keep_separator = keep;
        self
    }

    /// Trim whitespace around chunks (default true).
    pub fn with_strip_whitespace(mut self, on: bool) -> Self {
        self.strip_whitespace = on;
        self
    }

    /// Record each chunk's byte offset in the parent as `start_index`
    /// metadata in [`split_documents`](TextSplitter::split_documents).
    pub fn with_start_index(mut self, on: bool) -> Self {
        self.add_start_index = on;
        self
    }

    /// Measures length with `f`, e.g. [`approx_token_len`](super::approx_token_len)
    /// or a real tokenizer.
    pub fn with_length_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> usize + Send + Sync + 'static,
    {
        self.length = Arc::new(f);
        self
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }

    fn split_rec(&self, text: &str, separators: &[String], out: &mut Vec<String>) {
        let mut separator = separators.last().map(String::as_str).unwrap_or("");
        let mut rest: &[String] = &[];
        for (i, s) in separators.iter().enumerate() {
            if s.is_empty() {
                separator = "";
                break;
            }
            if text.contains(s.as_str()) {
                separator = s;
                rest = &separators[i + 1..];
                break;
            }
        }
        let splits = split_on(text, separator, self.keep_separator);
        let merge_sep = if self.keep_separator == KeepSeparator::None {
            separator
        } else {
            ""
        };
        let mut good: Vec<&str> = Vec::new();
        for s in splits {
            if (self.length)(s) < self.chunk_size {
                good.push(s);
                continue;
            }
            if !good.is_empty() {
                self.merge(&good, merge_sep, out);
                good.clear();
            }
            if rest.is_empty() {
                out.push(s.to_string());
            } else {
                self.split_rec(s, rest, out);
            }
        }
        if !good.is_empty() {
            self.merge(&good, merge_sep, out);
        }
    }

    fn join(&self, parts: &[&str], sep: &str) -> Option<String> {
        let text = parts.join(sep);
        let text = if self.strip_whitespace {
            let t = text.trim();
            if t.len() == text.len() {
                text
            } else {
                t.to_string()
            }
        } else {
            text
        };
        (!text.is_empty()).then_some(text)
    }

    /// Greedily packs `splits` into chunks, carrying up to `chunk_overlap`
    /// of the previous chunk into the next one.
    fn merge(&self, splits: &[&str], sep: &str, out: &mut Vec<String>) {
        let sep_len = (self.length)(sep);
        let lens: Vec<usize> = splits.iter().map(|s| (self.length)(s)).collect();
        let mut start = 0;
        let mut total = 0usize;
        for i in 0..splits.len() {
            let len = lens[i];
            let joiner = |count: usize| if count > 0 { sep_len } else { 0 };
            if total + len + joiner(i - start) > self.chunk_size && i > start {
                if let Some(doc) = self.join(&splits[start..i], sep) {
                    out.push(doc);
                }
                while total > self.chunk_overlap
                    || (total + len + joiner(i - start) > self.chunk_size && total > 0)
                {
                    let drop = lens[start] + if i - start > 1 { sep_len } else { 0 };
                    total = total.saturating_sub(drop);
                    start += 1;
                }
            }
            total += len + if i + 1 - start > 1 { sep_len } else { 0 };
        }
        if let Some(doc) = self.join(&splits[start..], sep) {
            out.push(doc);
        }
    }
}

/// Splits `text` on a literal separator; empty pieces are dropped.
fn split_on<'a>(text: &'a str, sep: &str, keep: KeepSeparator) -> Vec<&'a str> {
    if sep.is_empty() {
        return text
            .char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect();
    }
    let mut out = Vec::new();
    let mut last = 0;
    for (i, _) in text.match_indices(sep) {
        match keep {
            KeepSeparator::None => {
                out.push(&text[last..i]);
                last = i + sep.len();
            }
            KeepSeparator::Start => {
                out.push(&text[last..i]);
                last = i;
            }
            KeepSeparator::End => {
                out.push(&text[last..i + sep.len()]);
                last = i + sep.len();
            }
        }
    }
    out.push(&text[last..]);
    out.retain(|s| !s.is_empty());
    out
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

impl TextSplitter for RecursiveCharacterTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        self.split_rec(text, &self.separators, &mut out);
        out
    }

    fn split_documents(&self, docs: &[Document]) -> Vec<Document> {
        let mut out = Vec::new();
        for d in docs {
            let mut index = 0usize;
            let mut prev_len = 0usize;
            for (i, chunk) in self.split_text(&d.content).into_iter().enumerate() {
                let start = if self.add_start_index {
                    let from = (index + prev_len).saturating_sub(self.chunk_overlap);
                    let from = floor_char_boundary(&d.content, from);
                    index = d.content[from..]
                        .find(chunk.as_str())
                        .map(|p| p + from)
                        .unwrap_or(from);
                    prev_len = chunk.len();
                    Some(index)
                } else {
                    None
                };
                let mut doc = chunk_doc(d, i, chunk);
                if let Some(s) = start {
                    doc.metadata.insert("start_index".into(), s.into());
                }
                out.push(doc);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splitter::approx_token_len;

    #[test]
    fn matches_langchain_reference_output() {
        // LangChain test_iterative_text_splitter (chunk_size=10, overlap=1).
        let text = "Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.\nThis is a weird text to write, but gotta test the splittingggg some how.\n\nBye!\n\n-H.";
        let s = RecursiveCharacterTextSplitter::new(10, 1);
        let expected = [
            "Hi.",
            "I'm",
            "Harrison.",
            "How? Are?",
            "You?",
            "Okay then",
            "f f f f.",
            "This is a",
            "weird",
            "text to",
            "write,",
            "but gotta",
            "test the",
            "splitting",
            "gggg",
            "some how.",
            "Bye!",
            "-H.",
        ];
        assert_eq!(s.split_text(text), expected);
    }

    #[test]
    fn overlap_repeats_trailing_words() {
        let s = RecursiveCharacterTextSplitter::new(7, 3)
            .with_separators([" "])
            .with_keep_separator(KeepSeparator::None);
        let chunks = s.split_text("foo bar baz 123");
        assert_eq!(chunks, ["foo bar", "bar baz", "baz 123"]);
    }

    #[test]
    fn separator_modes() {
        let s = RecursiveCharacterTextSplitter::new(4, 0).with_separators(["."]);
        assert_eq!(s.split_text("aa.bb.cc"), ["aa", ".bb", ".cc"]);
        let s = s.clone().with_keep_separator(KeepSeparator::End);
        assert_eq!(s.split_text("aa.bb.cc"), ["aa.", "bb.", "cc"]);
        let s = RecursiveCharacterTextSplitter::new(5, 0)
            .with_separators(["."])
            .with_keep_separator(KeepSeparator::None);
        assert_eq!(s.split_text("aa.bb.cc"), ["aa.bb", "cc"]);
    }

    #[test]
    fn multibyte_and_token_length() {
        let s = RecursiveCharacterTextSplitter::new(3, 0);
        assert_eq!(s.split_text("ãéîõü"), ["ãéî", "õü"]);
        let t = RecursiveCharacterTextSplitter::new(3, 0).with_length_fn(approx_token_len);
        assert_eq!(t.split_text("aaaa bbbb cccc"), ["aaaa bbbb", "cccc"]);
    }

    #[test]
    fn documents_get_ids_and_start_index() {
        let s = RecursiveCharacterTextSplitter::new(7, 3)
            .with_separators([" "])
            .with_keep_separator(KeepSeparator::None)
            .with_start_index(true);
        let doc = Document::new("foo bar baz 123")
            .with_id("d")
            .with_metadata("source", "x");
        let chunks = s.split_documents(&[doc]);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[1].id.as_deref(), Some("d:1"));
        assert_eq!(chunks[1].metadata["source"], "x");
        let starts: Vec<u64> = chunks
            .iter()
            .map(|c| c.metadata["start_index"].as_u64().unwrap())
            .collect();
        assert_eq!(starts, [0, 4, 8]);
    }

    #[test]
    fn code_splitting_prefers_item_boundaries() {
        let code = "fn a() {\n    1\n}\n\nfn b() {\n    2\n}\n";
        let s = RecursiveCharacterTextSplitter::for_language(Language::Rust, 20, 0);
        let chunks = s.split_text(code);
        assert_eq!(chunks, ["fn a() {\n    1\n}", "fn b() {\n    2\n}"]);
    }
}
