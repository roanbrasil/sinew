use std::path::PathBuf;

use super::{source_metadata, Loader};
use crate::{Document, Error, Result};

/// Loads a CSV file, one document per row.
///
/// Like LangChain's `CSVLoader`, the content is one `column: value` line per
/// column (or only `content_columns`), and `metadata_columns` are copied into
/// metadata. Every document gets `source` and `row` (0-based) metadata.
#[derive(Debug, Clone)]
pub struct CsvLoader {
    path: PathBuf,
    delimiter: u8,
    content_columns: Vec<String>,
    metadata_columns: Vec<String>,
}

impl CsvLoader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            delimiter: b',',
            content_columns: Vec::new(),
            metadata_columns: Vec::new(),
        }
    }

    pub fn with_delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }

    pub fn with_content_columns<I, S>(mut self, cols: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.content_columns = cols.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_metadata_columns<I, S>(mut self, cols: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.metadata_columns = cols.into_iter().map(Into::into).collect();
        self
    }
}

impl Loader for CsvLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let reader = self
            .reader_builder()
            .from_path(&self.path)
            .map_err(|e| Error::Load {
                path: self.path.clone(),
                message: e.to_string(),
            })?;
        self.documents(reader)
    }
}

impl CsvLoader {
    fn reader_builder(&self) -> csv::ReaderBuilder {
        let mut b = csv::ReaderBuilder::new();
        b.delimiter(self.delimiter).flexible(true);
        b
    }

    /// Parses CSV `bytes` as if they were the loader's file. Used by the fuzz
    /// targets.
    pub(crate) fn documents_from_bytes(&self, bytes: &[u8]) -> Result<Vec<Document>> {
        self.documents(self.reader_builder().from_reader(bytes))
    }

    fn documents<R: std::io::Read>(&self, mut reader: csv::Reader<R>) -> Result<Vec<Document>> {
        let headers = reader.headers()?.clone();
        let find = |name: &str| -> Result<usize> {
            headers
                .iter()
                .position(|h| h == name)
                .ok_or_else(|| Error::Load {
                    path: self.path.clone(),
                    message: format!("no column `{name}`"),
                })
        };
        let content_idx: Vec<usize> = if self.content_columns.is_empty() {
            (0..headers.len()).collect()
        } else {
            self.content_columns
                .iter()
                .map(|c| find(c))
                .collect::<Result<_>>()?
        };
        let meta_idx: Vec<(usize, &String)> = self
            .metadata_columns
            .iter()
            .map(|c| find(c).map(|i| (i, c)))
            .collect::<Result<_>>()?;

        let mut docs = Vec::new();
        for (row, record) in reader.records().enumerate() {
            let record = record?;
            let mut content = String::new();
            for &i in &content_idx {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(headers.get(i).unwrap_or_default().trim());
                content.push_str(": ");
                content.push_str(record.get(i).unwrap_or_default().trim());
            }
            let mut metadata = source_metadata(&self.path);
            metadata.insert("row".into(), row.into());
            for (i, name) in &meta_idx {
                metadata.insert(
                    (*name).clone(),
                    record.get(*i).unwrap_or_default().to_string().into(),
                );
            }
            docs.push(Document::new(content).with_metadata_map(metadata));
        }
        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_rows() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.csv");
        std::fs::write(&p, "name,city,id\nAna,Recife,1\n\"Bo, Jr\",Lyon,2\n").unwrap();
        let docs = CsvLoader::new(&p).load().unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, "name: Ana\ncity: Recife\nid: 1");
        let docs = CsvLoader::new(&p)
            .with_content_columns(["name"])
            .with_metadata_columns(["id"])
            .load()
            .unwrap();
        assert_eq!(docs[1].content, "name: Bo, Jr");
        assert_eq!(docs[1].metadata["id"], "2");
        assert_eq!(docs[1].metadata["row"], 1);
        assert!(CsvLoader::new(&p)
            .with_content_columns(["zzz"])
            .load()
            .is_err());
    }
}
