//! Datasets: named lists of examples, stored as JSONL.

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::EvalError;

/// One test case: the inputs passed to the target and, optionally, the
/// reference ("expected") output evaluators compare against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Example {
    #[serde(default = "new_id")]
    pub id: String,
    pub inputs: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<Value>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

impl Example {
    pub fn new(inputs: Value) -> Self {
        Self {
            id: new_id(),
            inputs,
            reference: None,
            metadata: Value::Null,
        }
    }

    pub fn with_reference(mut self, reference: Value) -> Self {
        self.reference = Some(reference);
        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub examples: Vec<Example>,
}

impl Dataset {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    /// Adds an example with an optional reference output.
    pub fn push(&mut self, inputs: Value, reference: Option<Value>) -> &mut Self {
        self.examples.push(Example {
            reference,
            ..Example::new(inputs)
        });
        self
    }

    pub fn with_example(mut self, example: Example) -> Self {
        self.examples.push(example);
        self
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Parses JSONL: one example per line (`{"inputs": ..., "reference": ...}`).
    /// Blank lines are skipped; a malformed line is an error with its number.
    pub fn from_jsonl_str(name: impl Into<String>, text: &str) -> Result<Self, EvalError> {
        let mut ds = Self::new(name);
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let ex: Example = serde_json::from_str(line).map_err(|e| EvalError::Parse {
                line: n + 1,
                message: e.to_string(),
            })?;
            ds.examples.push(ex);
        }
        Ok(ds)
    }

    /// Loads a JSONL file; the dataset is named after the file stem.
    pub fn load_jsonl(path: impl AsRef<Path>) -> Result<Self, EvalError> {
        let path = path.as_ref();
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "dataset".into());
        let mut text = String::new();
        for line in BufReader::new(File::open(path)?).lines() {
            text.push_str(&line?);
            text.push('\n');
        }
        Self::from_jsonl_str(name, &text)
    }

    pub fn to_jsonl_string(&self) -> String {
        let mut out = String::new();
        for ex in &self.examples {
            out.push_str(&serde_json::to_string(ex).unwrap_or_default());
            out.push('\n');
        }
        out
    }

    pub fn save_jsonl(&self, path: impl AsRef<Path>) -> Result<(), EvalError> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(self.to_jsonl_string().as_bytes())?;
        w.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn jsonl_round_trip() {
        let mut ds = Dataset::new("math");
        ds.push(json!({"q": "2+2"}), Some(json!("4")))
            .push(json!({"q": "hi"}), None);
        let dir = std::env::temp_dir().join(format!("sinew-eval-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("math.jsonl");
        ds.save_jsonl(&path).unwrap();
        let back = Dataset::load_jsonl(&path).unwrap();
        assert_eq!(back, ds);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_ids_are_generated_and_errors_have_line_numbers() {
        let ds = Dataset::from_jsonl_str("d", "{\"inputs\": 1}\n\n{\"inputs\": 2}\n").unwrap();
        assert_eq!(ds.len(), 2);
        assert_ne!(ds.examples[0].id, ds.examples[1].id);
        let err = Dataset::from_jsonl_str("d", "{\"inputs\": 1}\nnot json\n").unwrap_err();
        assert!(matches!(err, EvalError::Parse { line: 2, .. }));
    }
}
