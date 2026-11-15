//! Text splitters: cut documents into chunks that fit an embedding model and
//! a prompt.
//!
//! - [`RecursiveCharacterTextSplitter`]: LangChain's default splitter with
//!   the same semantics (ordered separators, `chunk_size`, `chunk_overlap`,
//!   separator kept at the start of the next chunk), language presets for
//!   code ([`Language`]) and a pluggable length function
//!   ([`approx_token_len`], or your own tokenizer).
//! - [`MarkdownHeaderTextSplitter`]: one chunk per section, with the
//!   enclosing headers as metadata.

mod language;
mod markdown;
mod recursive;

use std::sync::Arc;

pub use language::Language;
pub use markdown::MarkdownHeaderTextSplitter;
pub use recursive::{KeepSeparator, RecursiveCharacterTextSplitter};

use crate::Document;

/// Measures text in the unit `chunk_size` is expressed in.
pub type LengthFn = Arc<dyn Fn(&str) -> usize + Send + Sync>;

/// Length in Unicode scalar values (what Python's `len` returns), the
/// default.
pub fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Cheap token estimate: one token per four characters, rounded up. Use it
/// with [`RecursiveCharacterTextSplitter::with_length_fn`] when `chunk_size`
/// should mean tokens and an exact tokenizer is not worth the dependency.
pub fn approx_token_len(s: &str) -> usize {
    char_len(s).div_ceil(4)
}

/// Splits text into chunks.
pub trait TextSplitter: Send + Sync {
    fn split_text(&self, text: &str) -> Vec<String>;

    /// Splits every document. Chunks keep the parent's metadata plus a
    /// `chunk_index`; when the parent has an id, chunk ids are
    /// `"{parent_id}:{index}"`.
    fn split_documents(&self, docs: &[Document]) -> Vec<Document> {
        let mut out = Vec::new();
        for d in docs {
            for (i, chunk) in self.split_text(&d.content).into_iter().enumerate() {
                out.push(chunk_doc(d, i, chunk));
            }
        }
        out
    }
}

pub(crate) fn chunk_doc(parent: &Document, i: usize, content: String) -> Document {
    let mut metadata = parent.metadata.clone();
    metadata.insert("chunk_index".into(), i.into());
    Document {
        id: parent.id.as_ref().map(|p| format!("{p}:{i}")),
        content,
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_functions() {
        assert_eq!(char_len("ação"), 4);
        assert_eq!(approx_token_len(""), 0);
        assert_eq!(approx_token_len("abcde"), 2);
    }
}
