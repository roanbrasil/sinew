use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{read_to_string, source_metadata, Loader};
use crate::{Document, Error, Result};

/// Loads JSON or JSON Lines, one document per record.
///
/// - `records` (a JSON Pointer, e.g. `/items`) selects the array of records
///   in a JSON file; without it a top-level array is used, or the whole value
///   as a single record. JSONL files are one record per non-empty line.
/// - `content` (a JSON Pointer relative to the record, e.g. `/text`) picks the
///   text; strings are used as-is, other values are serialized. Without it
///   the whole record is the content.
/// - `metadata_fields` copies top-level record fields into metadata.
///
/// Every document gets `source` and `seq_num` (1-based) metadata.
#[derive(Debug, Clone)]
pub struct JsonLoader {
    path: PathBuf,
    lines: bool,
    records: Option<String>,
    content: Option<String>,
    metadata_fields: Vec<String>,
}

impl JsonLoader {
    /// A `.json` file.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lines: false,
            records: None,
            content: None,
            metadata_fields: Vec::new(),
        }
    }

    /// A `.jsonl` / `.ndjson` file.
    pub fn json_lines(path: impl Into<PathBuf>) -> Self {
        Self {
            lines: true,
            ..Self::new(path)
        }
    }

    pub fn with_records_pointer(mut self, pointer: impl Into<String>) -> Self {
        self.records = Some(pointer.into());
        self
    }

    pub fn with_content_pointer(mut self, pointer: impl Into<String>) -> Self {
        self.content = Some(pointer.into());
        self
    }

    pub fn with_metadata_fields<I, S>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.metadata_fields = fields.into_iter().map(Into::into).collect();
        self
    }

    fn record_to_doc(&self, path: &Path, seq: usize, record: &Value) -> Result<Document> {
        let value = match &self.content {
            Some(p) => record.pointer(p).ok_or_else(|| Error::Load {
                path: path.into(),
                message: format!("record {seq} has no `{p}`"),
            })?,
            None => record,
        };
        let content = match value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let mut metadata = source_metadata(path);
        metadata.insert("seq_num".into(), seq.into());
        for f in &self.metadata_fields {
            if let Some(v) = record.get(f) {
                metadata.insert(f.clone(), v.clone());
            }
        }
        Ok(Document::new(content).with_metadata_map(metadata))
    }
}

impl Loader for JsonLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let text = read_to_string(&self.path)?;
        let bad = |e: serde_json::Error| Error::Load {
            path: self.path.clone(),
            message: e.to_string(),
        };
        let records: Vec<Value> = if self.lines {
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str(l).map_err(bad))
                .collect::<Result<_>>()?
        } else {
            let root: Value = serde_json::from_str(&text).map_err(bad)?;
            let selected = match &self.records {
                Some(p) => root.pointer(p).cloned().ok_or_else(|| Error::Load {
                    path: self.path.clone(),
                    message: format!("no `{p}` in document"),
                })?,
                None => root,
            };
            match selected {
                Value::Array(a) => a,
                other => vec![other],
            }
        };
        records
            .iter()
            .enumerate()
            .map(|(i, r)| self.record_to_doc(&self.path, i + 1, r))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_json_and_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let j = dir.path().join("a.json");
        std::fs::write(
            &j,
            r#"{"items":[{"text":"one","lang":"en"},{"text":"two","lang":"pt"}]}"#,
        )
        .unwrap();
        let docs = JsonLoader::new(&j)
            .with_records_pointer("/items")
            .with_content_pointer("/text")
            .with_metadata_fields(["lang"])
            .load()
            .unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[1].content, "two");
        assert_eq!(docs[1].metadata["lang"], "pt");
        assert_eq!(docs[1].metadata["seq_num"], 2);

        let l = dir.path().join("a.jsonl");
        std::fs::write(&l, "{\"q\":\"a\"}\n\n{\"q\":{\"x\":1}}\n").unwrap();
        let docs = JsonLoader::json_lines(&l)
            .with_content_pointer("/q")
            .load()
            .unwrap();
        assert_eq!(docs[0].content, "a");
        assert_eq!(docs[1].content, r#"{"x":1}"#);
        assert!(JsonLoader::json_lines(&l)
            .with_content_pointer("/nope")
            .load()
            .is_err());
    }
}
