//! Small text utilities shared by BM25 and the fake embeddings.

const STOPWORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be",
    "been", "before", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had",
    "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "my", "no",
    "not", "of", "on", "or", "our", "so", "some", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "too", "up", "us", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "why", "will", "with", "would", "you",
    "your",
];

/// Lower-cased alphanumeric words with common English stop words removed
/// and a light suffix stemmer (`-ing`, `-ed`, `-es`, `-s`, `-ly`, final
/// `-e`), so "restarts", "restarted" and "restart" share a token. A good
/// default for English BM25 (see
/// [`Bm25Retriever::from_documents_with_tokenizer`](crate::Bm25Retriever::from_documents_with_tokenizer)).
pub fn english_tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .map(|w| light_stem(&w))
        .collect()
}

fn light_stem(w: &str) -> String {
    let mut s = w;
    for suffix in ["ing", "ed", "es", "s", "ly"] {
        if s.len() > suffix.len() + 2 && s.ends_with(suffix) {
            if suffix == "s" && s.ends_with("ss") {
                break;
            }
            s = &s[..s.len() - suffix.len()];
            break;
        }
    }
    if s.len() > 3 && s.ends_with('e') {
        s = &s[..s.len() - 1];
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems_and_drops_stop_words() {
        assert_eq!(
            english_tokenize("How do I restart the processes? It restarted; Resumes resume!"),
            ["restart", "process", "restart", "resum", "resum"]
        );
        assert_eq!(english_tokenize("process class"), ["process", "class"]);
    }
}
