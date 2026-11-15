use std::path::PathBuf;

use super::{source_metadata, Loader};
use crate::{Document, Error, Result};

/// Extracts text from a PDF with the pure-Rust `pdf-extract` crate
/// (feature `pdf`). One document per page with a 1-based `page` metadata
/// field, or the whole file as one document with [`single`](Self::single).
///
/// Scanned PDFs (images only) yield empty text; run OCR first.
#[derive(Debug, Clone)]
pub struct PdfLoader {
    path: PathBuf,
    per_page: bool,
}

impl PdfLoader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            per_page: true,
        }
    }

    /// Whole document as one [`Document`].
    pub fn single(mut self) -> Self {
        self.per_page = false;
        self
    }
}

impl Loader for PdfLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let bytes = std::fs::read(&self.path).map_err(|e| Error::io(&self.path, e))?;
        let err = |e: pdf_extract::OutputError| Error::Load {
            path: self.path.clone(),
            message: e.to_string(),
        };
        if !self.per_page {
            let text = pdf_extract::extract_text_from_mem(&bytes).map_err(err)?;
            return Ok(vec![
                Document::new(text).with_metadata_map(source_metadata(&self.path))
            ]);
        }
        let pages = pdf_extract::extract_text_from_mem_by_pages(&bytes).map_err(err)?;
        Ok(pages
            .into_iter()
            .enumerate()
            .map(|(i, text)| {
                let mut m = source_metadata(&self.path);
                m.insert("page".into(), (i + 1).into());
                Document::new(text).with_metadata_map(m)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal one-page PDF with correct xref offsets.
    fn tiny_pdf(text: &str) -> Vec<u8> {
        let stream = format!("BT /F1 18 Tf 20 100 Td ({text}) Tj ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R \
             /Resources << /Font << /F1 5 0 R >> >> >>"
                .to_string(),
            format!(
                "<< /Length {} >>\nstream\n{stream}\nendstream",
                stream.len()
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
                .to_string(),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, o) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
        }
        let xref = out.len();
        out.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
        );
        for off in offsets {
            out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    #[test]
    fn extracts_text_per_page() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.pdf");
        std::fs::write(&p, tiny_pdf("Hello PDF")).unwrap();
        let docs = PdfLoader::new(&p).load().unwrap();
        assert_eq!(docs.len(), 1);
        assert!(
            docs[0].content.contains("Hello PDF"),
            "{:?}",
            docs[0].content
        );
        assert_eq!(docs[0].metadata["page"], 1);
        let single = PdfLoader::new(&p).single().load().unwrap();
        assert!(single[0].content.contains("Hello PDF"));
        std::fs::write(&p, b"not a pdf").unwrap();
        assert!(PdfLoader::new(&p).load().is_err());
    }
}
