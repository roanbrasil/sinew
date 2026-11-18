//! Document loaders on in-memory input.

use sinew_rag::{__fuzz, html_to_text, CsvLoader, JsonLoader};

/// HTML to text never panics; the text has collapsed whitespace (no
/// leading or trailing whitespace, no carriage returns, at most one blank
/// line in a row).
pub fn html(data: &[u8]) {
    let html = String::from_utf8_lossy(data);
    let text = html_to_text(&html);
    assert_eq!(text, text.trim(), "untrimmed text");
    assert!(!text.contains('\r'), "CR in text");
    assert!(!text.contains("\n\n\n"), "more than one blank line");
    if let Some(title) = __fuzz::html_title(&html) {
        assert!(!title.is_empty());
    }
}

/// CSV rows become documents numbered from 0, with or without selected
/// columns; ragged rows and bad UTF-8 are errors or documents, never panics.
pub fn csv(data: &[u8]) {
    let Some((&flags, body)) = data.split_first() else {
        return;
    };
    let delimiter = b",;\t|"[usize::from(flags & 3)];
    let mut loader = CsvLoader::new("fuzz.csv").with_delimiter(delimiter);
    if flags & 4 != 0 {
        loader = loader.with_content_columns(["a", "b"]);
    }
    if flags & 8 != 0 {
        loader = loader.with_metadata_columns(["id"]);
    }
    if let Ok(docs) = __fuzz::csv(&loader, body) {
        for (i, d) in docs.iter().enumerate() {
            assert_eq!(d.metadata["row"], i, "row numbers");
            assert_eq!(d.metadata["source"], "fuzz.csv");
        }
    }
}

/// JSON and JSON Lines records become documents numbered from 1.
pub fn json(data: &[u8]) {
    let Some((&flags, body)) = data.split_first() else {
        return;
    };
    let text = String::from_utf8_lossy(body);
    let mut loader = if flags & 1 != 0 {
        JsonLoader::json_lines("fuzz.jsonl")
    } else {
        JsonLoader::new("fuzz.json")
    };
    if flags & 2 != 0 {
        loader = loader.with_records_pointer("/items");
    }
    if flags & 4 != 0 {
        loader = loader.with_content_pointer("/text");
    }
    if flags & 8 != 0 {
        loader = loader.with_metadata_fields(["id", "title"]);
    }
    if let Ok(docs) = __fuzz::json(&loader, &text) {
        for (i, d) in docs.iter().enumerate() {
            assert_eq!(d.metadata["seq_num"], i + 1, "seq numbers");
        }
    }
}

/// Markdown front matter and title extraction never panic; the body is the
/// end of the file.
pub fn markdown(data: &[u8]) {
    let text = String::from_utf8_lossy(data);
    let (body, metadata) = __fuzz::markdown(&text);
    assert!(text.ends_with(&body), "body is not a suffix of the file");
    if let Some(title) = metadata.get("title") {
        assert!(title.is_string());
    }
}
