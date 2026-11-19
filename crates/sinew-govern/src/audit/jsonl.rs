//! JSONL export and verification: one [`AuditEntry`] per line, in `seq`
//! order, checkpoints included.

use std::io::{BufRead, Write};

use super::{AuditEntry, AuditError, AuditReport, AuditStore, ChainVerifier, VerifyOptions};

/// Writes every entry of `store` to `out`, one JSON object per line, and
/// returns how many were written.
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use std::sync::Arc;
/// use serde_json::json;
/// use sinew_govern::audit::{export_jsonl, verify_jsonl, AuditKind, AuditLog, MemoryAuditStore, VerifyOptions};
///
/// let store = Arc::new(MemoryAuditStore::new());
/// let audit = AuditLog::new(store.clone());
/// audit.append(AuditKind::Other, json!({"n": 1})).await.unwrap();
/// audit.append(AuditKind::Other, json!({"n": 2})).await.unwrap();
///
/// let mut file = Vec::new();
/// assert_eq!(export_jsonl(store.as_ref(), &mut file).await.unwrap(), 2);
/// let report = verify_jsonl(file.as_slice(), &VerifyOptions::new()).unwrap();
/// assert!(report.is_ok());
///
/// // Swap the two lines: detected.
/// let text = String::from_utf8(file).unwrap();
/// let lines: Vec<&str> = text.lines().collect();
/// let swapped = format!("{}\n{}\n", lines[1], lines[0]);
/// assert!(!verify_jsonl(swapped.as_bytes(), &VerifyOptions::new()).unwrap().is_ok());
/// # });
/// ```
pub async fn export_jsonl<W: Write + ?Sized>(
    store: &dyn AuditStore,
    out: &mut W,
) -> Result<u64, AuditError> {
    const PAGE: usize = 1000;
    let mut from = 0;
    let mut n = 0;
    loop {
        let page = store.entries(from, PAGE).await?;
        for e in &page {
            serde_json::to_writer(&mut *out, e).map_err(|e| AuditError::Invalid(e.to_string()))?;
            out.write_all(b"\n")?;
            n += 1;
        }
        match page.last() {
            Some(last) if page.len() == PAGE => from = last.seq.saturating_add(1),
            _ => break,
        }
    }
    out.flush()?;
    Ok(n)
}

/// Parses one line of an export. `line_no` (from 1) is used in errors.
pub fn parse_jsonl_line(line: &str, line_no: usize) -> Result<AuditEntry, AuditError> {
    serde_json::from_str(line).map_err(|e| AuditError::Invalid(format!("line {line_no}: {e}")))
}

/// Reads an export back. Blank lines are skipped.
pub fn read_jsonl<R: BufRead>(input: R) -> Result<Vec<AuditEntry>, AuditError> {
    let mut out = Vec::new();
    for (i, line) in input.lines().enumerate() {
        let line = line?;
        if !line.trim().is_empty() {
            out.push(parse_jsonl_line(&line, i + 1)?);
        }
    }
    Ok(out)
}

/// Verifies an export while reading it, in file order (so reordered lines
/// are detected). Fails only when the input cannot be read or parsed; a
/// tampered log is a report with problems.
pub fn verify_jsonl<R: BufRead>(
    input: R,
    options: &VerifyOptions,
) -> Result<AuditReport, AuditError> {
    let mut v = ChainVerifier::new(options.clone());
    for (i, line) in input.lines().enumerate() {
        let line = line?;
        if !line.trim().is_empty() {
            v.push(&parse_jsonl_line(&line, i + 1)?);
        }
    }
    Ok(v.finish())
}
