//! Project instruction files (`AGENTS.md`, `CLAUDE.md`).

use std::path::{Path, PathBuf};

/// File names looked up in a project directory, in order.
pub const INSTRUCTION_FILES: &[&str] = &[
    "AGENTS.md",
    "CLAUDE.md",
    ".claude/CLAUDE.md",
    ".sinew/AGENTS.md",
];

/// Instruction files that exist in `dir`.
pub fn find_instruction_files(dir: &Path) -> Vec<PathBuf> {
    INSTRUCTION_FILES
        .iter()
        .map(|f| dir.join(f))
        .filter(|p| p.is_file())
        .collect()
}

/// Reads instruction files into one system-prompt section, or `None` if
/// none is readable. Each file is capped at `max_chars`.
pub fn instructions_prompt(
    files: &[PathBuf],
    base: Option<&Path>,
    max_chars: usize,
) -> Option<String> {
    let mut out = String::new();
    for f in files {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        let shown = base
            .and_then(|b| f.strip_prefix(b).ok())
            .unwrap_or(f)
            .display()
            .to_string();
        let body: String = text.chars().take(max_chars).collect();
        out.push_str(&format!("## {shown}\n{body}\n"));
        if text.chars().count() > max_chars {
            out.push_str("[file truncated]\n");
        }
        out.push('\n');
    }
    if out.is_empty() {
        None
    } else {
        Some(format!(
            "# Project instructions\nThe project provides these instructions. Follow them.\n\n{}",
            out.trim_end()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_and_reads_instruction_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "Use tabs.\n").unwrap();
        std::fs::create_dir_all(dir.path().join(".claude")).unwrap();
        std::fs::write(dir.path().join(".claude/CLAUDE.md"), "Be brief.").unwrap();
        let files = find_instruction_files(dir.path());
        assert_eq!(files.len(), 2);
        let p = instructions_prompt(&files, Some(dir.path()), 1000).unwrap();
        assert!(p.contains("## AGENTS.md\nUse tabs."));
        assert!(p.contains("## .claude/CLAUDE.md\nBe brief."));
        assert!(instructions_prompt(&[], None, 10).is_none());
    }
}
