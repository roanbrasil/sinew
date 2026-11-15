//! Prompt hub storage: prompts, immutable versions (content-addressed
//! commits) and movable tags.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::PromptTemplate;
use sinew_eval::hub::{PromptMessage, PromptVersion};
use sqlx::{sqlite::SqliteRow, Row};

use super::{from_json, now_ms, to_json, Result, Store, StoreError};

/// Body of `POST /api/prompts` (same shape as `sinew_eval::PromptPush`).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct PromptPushInput {
    pub name: String,
    pub template: String,
    pub messages: Option<Vec<PromptMessage>>,
    pub description: Option<String>,
    pub model_config: Value,
    pub metadata: Value,
    pub message: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptSummary {
    pub name: String,
    pub description: String,
    pub created_ms: i64,
    pub updated_ms: i64,
    pub version_count: i64,
    pub latest_commit: Option<String>,
    pub latest_version: Option<i64>,
    /// tag → commit
    pub tags: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptDetail {
    pub name: String,
    pub description: String,
    pub created_ms: i64,
    pub updated_ms: i64,
    pub tags: BTreeMap<String, String>,
    /// Newest first.
    pub versions: Vec<PromptVersion>,
}

/// One line of a template diff: `op` is `eq`, `add` or `del`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffLine {
    pub op: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptDiff {
    pub from: PromptVersion,
    pub to: PromptVersion,
    pub template: Vec<DiffLine>,
    pub variables_added: Vec<String>,
    pub variables_removed: Vec<String>,
    /// `[{key, from, to}]` for every top-level model config key that changed.
    pub model_config: Vec<Value>,
}

/// Names and tags: letters, digits, `-`, `_`, `.`; 1–100 chars.
pub fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 100
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        && s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
}

/// Line diff (LCS). Fine for prompt-sized texts.
pub fn line_diff(a: &str, b: &str) -> Vec<DiffLine> {
    let a: Vec<&str> = a.lines().collect();
    let b: Vec<&str> = b.lines().collect();
    let (n, m) = (a.len(), b.len());
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    let line = |op: &str, t: &str| DiffLine {
        op: op.into(),
        text: t.into(),
    };
    while i < n && j < m {
        if a[i] == b[j] {
            out.push(line("eq", a[i]));
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            out.push(line("del", a[i]));
            i += 1;
        } else {
            out.push(line("add", b[j]));
            j += 1;
        }
    }
    out.extend(a[i..].iter().map(|t| line("del", t)));
    out.extend(b[j..].iter().map(|t| line("add", t)));
    out
}

/// SHA-256 (FIPS 180-4), used for content-addressed prompt commits.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, b) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

fn variables(input: &PromptPushInput) -> std::result::Result<Vec<String>, String> {
    let mut vars: Vec<String> = Vec::new();
    let mut add = |t: &str| -> std::result::Result<(), String> {
        let p = PromptTemplate::new(t.to_string()).map_err(|e| e.to_string())?;
        for v in p.variables() {
            if !vars.contains(&v) {
                vars.push(v);
            }
        }
        Ok(())
    };
    match &input.messages {
        Some(list) if !list.is_empty() => {
            for m in list {
                if !matches!(m.role.as_str(), "system" | "user" | "assistant") {
                    return Err(format!("unknown message role `{}`", m.role));
                }
                add(&m.content)?;
            }
        }
        _ => add(&input.template)?,
    }
    Ok(vars)
}

fn row_to_version(row: &SqliteRow) -> Result<PromptVersion> {
    let vars: Vec<String> =
        serde_json::from_str(&row.try_get::<String, _>("input_variables")?).unwrap_or_default();
    Ok(PromptVersion {
        name: row.try_get("name")?,
        commit: row.try_get("commit_hash")?,
        version: row.try_get::<i64, _>("version")?.max(0) as u32,
        template: row.try_get("template")?,
        messages: row
            .try_get::<Option<String>, _>("messages")?
            .and_then(|m| serde_json::from_str(&m).ok()),
        input_variables: vars,
        model_config: from_json(&row.try_get::<String, _>("model_config")?),
        metadata: from_json(&row.try_get::<String, _>("metadata")?),
        message: row.try_get("message")?,
        parent_commit: row.try_get("parent_commit")?,
        created_ms: row.try_get::<i64, _>("created_ms")?.max(0) as u64,
        tags: Vec::new(),
    })
}

impl Store {
    async fn prompt_tags(&self, name: &str) -> Result<BTreeMap<String, String>> {
        let rows =
            sqlx::query("SELECT tag, commit_hash FROM prompt_tags WHERE name = ? ORDER BY tag")
                .bind(name)
                .fetch_all(self.pool())
                .await?;
        rows.iter()
            .map(|r| Ok((r.try_get("tag")?, r.try_get("commit_hash")?)))
            .collect()
    }

    fn attach_tags(v: &mut PromptVersion, tags: &BTreeMap<String, String>) {
        v.tags = tags
            .iter()
            .filter(|(_, c)| **c == v.commit)
            .map(|(t, _)| t.clone())
            .collect();
    }

    /// Pushes a prompt version. Returns the version and whether it is new
    /// (pushing content equal to the latest version stores nothing, but the
    /// requested tags are still applied).
    pub async fn push_prompt(&self, input: &PromptPushInput) -> Result<(PromptVersion, bool)> {
        if !valid_name(&input.name) {
            return Err(StoreError::Invalid(
                "prompt name must be 1-100 letters, digits, `-`, `_` or `.`".into(),
            ));
        }
        if input.template.trim().is_empty() && input.messages.as_ref().is_none_or(|m| m.is_empty())
        {
            return Err(StoreError::Invalid("template is required".into()));
        }
        for t in &input.tags {
            if !valid_name(t) || t == "latest" {
                return Err(StoreError::Invalid(format!("invalid tag `{t}`")));
            }
        }
        let vars = variables(input).map_err(|e| StoreError::Invalid(format!("template: {e}")))?;
        let messages = input.messages.as_ref().filter(|m| !m.is_empty());
        let template = match messages {
            Some(list) if input.template.trim().is_empty() => list
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
            _ => input.template.clone(),
        };
        let now = now_ms();
        let mut tx = self.pool().begin().await?;
        let latest = sqlx::query(
            "SELECT * FROM prompt_versions WHERE name = ? ORDER BY version DESC LIMIT 1",
        )
        .bind(&input.name)
        .fetch_optional(&mut *tx)
        .await?
        .as_ref()
        .map(row_to_version)
        .transpose()?;
        let model_config = if input.model_config.is_null() {
            json!({})
        } else {
            input.model_config.clone()
        };
        let same = latest.as_ref().is_some_and(|l| {
            l.template == template
                && l.messages.as_ref() == messages
                && l.model_config == model_config
        });
        let (version, created) = if let (true, Some(l)) = (same, latest.clone()) {
            (l, false)
        } else {
            let parent = latest.as_ref().map(|l| l.commit.clone());
            let content = json!({
                "name": input.name,
                "parent": parent,
                "template": template,
                "messages": messages,
                "model_config": model_config,
            });
            let commit = sha256_hex(to_json(&content).as_bytes())[..12].to_string();
            let number = latest.as_ref().map(|l| l.version + 1).unwrap_or(1);
            sqlx::query(
                "INSERT INTO prompts (name, description, created_ms, updated_ms) VALUES (?, ?, ?, ?) \
                 ON CONFLICT(name) DO UPDATE SET updated_ms = excluded.updated_ms, \
                 description = CASE WHEN ? IS NULL THEN prompts.description ELSE excluded.description END",
            )
            .bind(&input.name)
            .bind(input.description.clone().unwrap_or_default())
            .bind(now)
            .bind(now)
            .bind(&input.description)
            .execute(&mut *tx)
            .await?;
            let metadata = if input.metadata.is_null() {
                json!({})
            } else {
                input.metadata.clone()
            };
            sqlx::query(
                "INSERT INTO prompt_versions (name, commit_hash, version, template, messages, \
                 input_variables, model_config, metadata, message, parent_commit, created_ms) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&input.name)
            .bind(&commit)
            .bind(i64::from(number))
            .bind(&template)
            .bind(messages.map(|m| serde_json::to_string(m).unwrap_or_default()))
            .bind(serde_json::to_string(&vars).unwrap_or_else(|_| "[]".into()))
            .bind(to_json(&model_config))
            .bind(to_json(&metadata))
            .bind(&input.message)
            .bind(&parent)
            .bind(now)
            .execute(&mut *tx)
            .await?;
            (
                PromptVersion {
                    name: input.name.clone(),
                    commit,
                    version: number,
                    template,
                    messages: messages.cloned(),
                    input_variables: vars,
                    model_config,
                    metadata,
                    message: input.message.clone(),
                    parent_commit: parent,
                    created_ms: now.max(0) as u64,
                    tags: Vec::new(),
                },
                true,
            )
        };
        for t in &input.tags {
            sqlx::query(
                "INSERT INTO prompt_tags (name, tag, commit_hash, updated_ms) VALUES (?, ?, ?, ?) \
                 ON CONFLICT(name, tag) DO UPDATE SET commit_hash = excluded.commit_hash, \
                 updated_ms = excluded.updated_ms",
            )
            .bind(&input.name)
            .bind(t)
            .bind(&version.commit)
            .bind(now)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        let mut version = version;
        let tags = self.prompt_tags(&input.name).await?;
        Self::attach_tags(&mut version, &tags);
        Ok((version, created))
    }

    pub async fn prompts(&self) -> Result<Vec<PromptSummary>> {
        let rows = sqlx::query(
            "SELECT p.*, (SELECT COUNT(*) FROM prompt_versions v WHERE v.name = p.name) AS n, \
             (SELECT commit_hash FROM prompt_versions v WHERE v.name = p.name ORDER BY version DESC LIMIT 1) AS latest_commit, \
             (SELECT MAX(version) FROM prompt_versions v WHERE v.name = p.name) AS latest_version \
             FROM prompts p ORDER BY p.updated_ms DESC",
        )
        .fetch_all(self.pool())
        .await?;
        let mut out = Vec::with_capacity(rows.len());
        for r in &rows {
            let name: String = r.try_get("name")?;
            out.push(PromptSummary {
                tags: self.prompt_tags(&name).await?,
                name,
                description: r.try_get("description")?,
                created_ms: r.try_get("created_ms")?,
                updated_ms: r.try_get("updated_ms")?,
                version_count: r.try_get("n")?,
                latest_commit: r.try_get("latest_commit")?,
                latest_version: r.try_get("latest_version")?,
            });
        }
        Ok(out)
    }

    pub async fn prompt(&self, name: &str) -> Result<Option<PromptDetail>> {
        let Some(row) = sqlx::query("SELECT * FROM prompts WHERE name = ?")
            .bind(name)
            .fetch_optional(self.pool())
            .await?
        else {
            return Ok(None);
        };
        let tags = self.prompt_tags(name).await?;
        let mut versions =
            sqlx::query("SELECT * FROM prompt_versions WHERE name = ? ORDER BY version DESC")
                .bind(name)
                .fetch_all(self.pool())
                .await?
                .iter()
                .map(row_to_version)
                .collect::<Result<Vec<_>>>()?;
        for v in &mut versions {
            Self::attach_tags(v, &tags);
        }
        Ok(Some(PromptDetail {
            name: name.to_string(),
            description: row.try_get("description")?,
            created_ms: row.try_get("created_ms")?,
            updated_ms: row.try_get("updated_ms")?,
            tags,
            versions,
        }))
    }

    /// Resolves `latest`, a tag, `vN`/`N`, or a commit prefix (4+ hex digits).
    pub async fn resolve_prompt(
        &self,
        name: &str,
        reference: &str,
    ) -> Result<Option<PromptVersion>> {
        let r = reference.trim();
        let pool = self.pool();
        let row = if r.is_empty() || r == "latest" {
            sqlx::query(
                "SELECT * FROM prompt_versions WHERE name = ? ORDER BY version DESC LIMIT 1",
            )
            .bind(name)
            .fetch_optional(pool)
            .await?
        } else if let Some(commit) = sqlx::query_scalar::<_, String>(
            "SELECT commit_hash FROM prompt_tags WHERE name = ? AND tag = ?",
        )
        .bind(name)
        .bind(r)
        .fetch_optional(pool)
        .await?
        {
            sqlx::query("SELECT * FROM prompt_versions WHERE name = ? AND commit_hash = ?")
                .bind(name)
                .bind(commit)
                .fetch_optional(pool)
                .await?
        } else if let Some(n) = r
            .strip_prefix('v')
            .unwrap_or(r)
            .parse::<i64>()
            .ok()
            .filter(|_| r.len() < 8 || r.starts_with('v'))
        {
            sqlx::query("SELECT * FROM prompt_versions WHERE name = ? AND version = ?")
                .bind(name)
                .bind(n)
                .fetch_optional(pool)
                .await?
        } else if r.len() >= 4 && r.chars().all(|c| c.is_ascii_hexdigit()) {
            let rows = sqlx::query(
                "SELECT * FROM prompt_versions WHERE name = ? AND commit_hash LIKE ? LIMIT 2",
            )
            .bind(name)
            .bind(format!("{}%", r.to_ascii_lowercase()))
            .fetch_all(pool)
            .await?;
            if rows.len() > 1 {
                return Err(StoreError::Invalid(format!(
                    "ambiguous commit prefix `{r}`"
                )));
            }
            rows.into_iter().next()
        } else {
            None
        };
        let Some(row) = row else {
            return Ok(None);
        };
        let mut v = row_to_version(&row)?;
        let tags = self.prompt_tags(name).await?;
        Self::attach_tags(&mut v, &tags);
        Ok(Some(v))
    }

    /// Points `tag` at the version `reference` resolves to.
    pub async fn set_prompt_tag(
        &self,
        name: &str,
        tag: &str,
        reference: &str,
    ) -> Result<Option<PromptVersion>> {
        if !valid_name(tag) || tag == "latest" {
            return Err(StoreError::Invalid(format!("invalid tag `{tag}`")));
        }
        let Some(mut v) = self.resolve_prompt(name, reference).await? else {
            return Ok(None);
        };
        sqlx::query(
            "INSERT INTO prompt_tags (name, tag, commit_hash, updated_ms) VALUES (?, ?, ?, ?) \
             ON CONFLICT(name, tag) DO UPDATE SET commit_hash = excluded.commit_hash, \
             updated_ms = excluded.updated_ms",
        )
        .bind(name)
        .bind(tag)
        .bind(&v.commit)
        .bind(now_ms())
        .execute(self.pool())
        .await?;
        let tags = self.prompt_tags(name).await?;
        Self::attach_tags(&mut v, &tags);
        Ok(Some(v))
    }

    pub async fn delete_prompt_tag(&self, name: &str, tag: &str) -> Result<bool> {
        let r = sqlx::query("DELETE FROM prompt_tags WHERE name = ? AND tag = ?")
            .bind(name)
            .bind(tag)
            .execute(self.pool())
            .await?;
        Ok(r.rows_affected() > 0)
    }

    /// Deletes a prompt with all versions and tags.
    pub async fn delete_prompt(&self, name: &str) -> Result<bool> {
        let mut tx = self.pool().begin().await?;
        for table in ["prompt_tags", "prompt_versions"] {
            sqlx::query(&format!("DELETE FROM {table} WHERE name = ?"))
                .bind(name)
                .execute(&mut *tx)
                .await?;
        }
        let r = sqlx::query("DELETE FROM prompts WHERE name = ?")
            .bind(name)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(r.rows_affected() > 0)
    }

    pub async fn diff_prompt(
        &self,
        name: &str,
        from: &str,
        to: &str,
    ) -> Result<Option<PromptDiff>> {
        let (Some(a), Some(b)) = (
            self.resolve_prompt(name, from).await?,
            self.resolve_prompt(name, to).await?,
        ) else {
            return Ok(None);
        };
        let render = |v: &PromptVersion| match &v.messages {
            Some(list) => list
                .iter()
                .map(|m| format!("[{}]\n{}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n"),
            None => v.template.clone(),
        };
        let mut keys: Vec<String> = Vec::new();
        for v in [&a.model_config, &b.model_config] {
            if let Value::Object(o) = v {
                for k in o.keys() {
                    if !keys.contains(k) {
                        keys.push(k.clone());
                    }
                }
            }
        }
        keys.sort();
        let model_config = keys
            .into_iter()
            .filter_map(|k| {
                let (x, y) = (a.model_config.get(&k), b.model_config.get(&k));
                (x != y).then(|| json!({ "key": k, "from": x, "to": y }))
            })
            .collect();
        Ok(Some(PromptDiff {
            template: line_diff(&render(&a), &render(&b)),
            variables_added: b
                .input_variables
                .iter()
                .filter(|v| !a.input_variables.contains(v))
                .cloned()
                .collect(),
            variables_removed: a
                .input_variables
                .iter()
                .filter(|v| !b.input_variables.contains(v))
                .cloned()
                .collect(),
            model_config,
            from: a,
            to: b,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let long = vec![b'a'; 1000];
        assert_eq!(sha256_hex(&long).len(), 64);
    }

    #[test]
    fn diffs_lines() {
        let d = line_diff("a\nb\nc", "a\nx\nc\nd");
        let ops: Vec<&str> = d.iter().map(|l| l.op.as_str()).collect();
        assert_eq!(ops, vec!["eq", "del", "add", "eq", "add"]);
        assert!(valid_name("qa-bot.v2"));
        assert!(!valid_name("-x"));
        assert!(!valid_name("a/b"));
    }
}
