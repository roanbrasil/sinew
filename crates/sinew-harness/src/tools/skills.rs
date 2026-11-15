//! Skills: folders with a `SKILL.md` whose frontmatter names and describes
//! them. Their names and descriptions go into the system prompt; the body is
//! loaded on demand with the `load_skill` tool, so unused skills cost only a
//! line of context.
//!
//! ```text
//! skills/
//! └── release-notes/
//!     ├── SKILL.md        ---\nname: release-notes\ndescription: Write release notes from git log\n---\n...
//!     └── template.md
//! ```

use std::{
    fs,
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{Tool, ToolSpec};

use super::{arg_str, tool_err};

/// One skill found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// The skill's directory.
    pub dir: PathBuf,
    /// `SKILL.md` without its frontmatter.
    pub body: String,
}

/// Splits `---\nkey: value\n---\nbody` into (fields, body).
fn frontmatter(text: &str) -> (Vec<(String, String)>, String) {
    let text = text.trim_start_matches('\u{feff}');
    let Some(rest) = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    else {
        return (Vec::new(), text.to_string());
    };
    let Some(end) = rest.find("\n---") else {
        return (Vec::new(), text.to_string());
    };
    let fields = rest[..end]
        .lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(':')?;
            let v = v.trim().trim_matches('"').trim_matches('\'');
            Some((k.trim().to_string(), v.to_string()))
        })
        .collect();
    let body = rest[end + 4..]
        .trim_start_matches(['-'])
        .trim_start()
        .to_string();
    (fields, body)
}

/// Reads every `*/SKILL.md` directly under `dir` (and `dir/SKILL.md`
/// itself). Folders without a valid frontmatter use the folder name and the
/// first non-heading line as description.
pub fn load_skills(dir: impl AsRef<Path>) -> std::io::Result<Vec<Skill>> {
    let dir = dir.as_ref();
    let mut candidates = vec![dir.to_path_buf()];
    let mut subdirs: Vec<PathBuf> = fs::read_dir(dir)?
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.path())
        .collect();
    subdirs.sort();
    candidates.extend(subdirs);
    let mut out = Vec::new();
    for d in candidates {
        let file = d.join("SKILL.md");
        let Ok(text) = fs::read_to_string(&file) else {
            continue;
        };
        let (fields, body) = frontmatter(&text);
        let get = |k: &str| {
            fields
                .iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.clone())
        };
        let name = get("name").unwrap_or_else(|| {
            d.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "skill".into())
        });
        let description = get("description").unwrap_or_else(|| {
            body.lines()
                .map(str::trim)
                .find(|l| !l.is_empty() && !l.starts_with('#'))
                .unwrap_or("")
                .to_string()
        });
        out.push(Skill {
            name,
            description,
            dir: d,
            body,
        });
    }
    Ok(out)
}

/// System-prompt section listing the skills.
pub fn skills_prompt(skills: &[Skill]) -> String {
    let mut s = String::from(
        "# Skills\nSpecialized instructions are available as skills. When a task matches one, call `load_skill` with its name before starting, then follow it.\n",
    );
    for k in skills {
        s.push_str(&format!("- {}: {}\n", k.name, k.description));
    }
    s
}

/// `load_skill`: returns a skill's instructions and lists its other files.
pub struct LoadSkillTool {
    skills: Vec<Skill>,
}

impl LoadSkillTool {
    pub fn new(skills: Vec<Skill>) -> Self {
        Self { skills }
    }
}

#[async_trait]
impl Tool for LoadSkillTool {
    fn spec(&self) -> ToolSpec {
        let names: Vec<&str> = self.skills.iter().map(|s| s.name.as_str()).collect();
        ToolSpec {
            name: "load_skill".into(),
            description: "Load the full instructions of a skill listed in the system prompt."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": names, "description": "Skill name"}
                },
                "required": ["name"]
            }),
        }
    }

    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        let name = arg_str("load_skill", &args, "name")?;
        let skill = self
            .skills
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| tool_err("load_skill", format!("unknown skill `{name}`")))?;
        let mut out = format!("# Skill: {}\n\n{}\n", skill.name, skill.body.trim());
        let mut files: Vec<String> = fs::read_dir(&skill.dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| n != "SKILL.md")
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        if !files.is_empty() {
            out.push_str(&format!(
                "\nOther files in this skill ({}): {}\n",
                skill.dir.display(),
                files.join(", ")
            ));
        }
        Ok(Value::String(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn loads_skills_from_folders() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("release-notes");
        fs::create_dir_all(&a).unwrap();
        fs::write(
            a.join("SKILL.md"),
            "---\nname: release-notes\ndescription: \"Write release notes\"\n---\n# Steps\n1. Read the log.\n",
        )
        .unwrap();
        fs::write(a.join("template.md"), "x").unwrap();
        let b = dir.path().join("plain");
        fs::create_dir_all(&b).unwrap();
        fs::write(b.join("SKILL.md"), "# Plain\nDoes plain things.\n").unwrap();

        let skills = load_skills(dir.path()).unwrap();
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].name, "plain");
        assert_eq!(skills[0].description, "Does plain things.");
        assert_eq!(skills[1].description, "Write release notes");
        assert!(skills_prompt(&skills).contains("- release-notes: Write release notes"));

        let tool = LoadSkillTool::new(skills);
        let out = tool.call(json!({"name": "release-notes"})).await.unwrap();
        let s = out.as_str().unwrap();
        assert!(s.contains("1. Read the log.") && s.contains("template.md"));
        assert!(!s.contains("description:"));
        assert!(tool.call(json!({"name": "nope"})).await.is_err());
    }
}
