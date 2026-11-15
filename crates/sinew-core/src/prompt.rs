use std::{borrow::Cow, collections::HashMap};

use crate::{Error, Result};

/// `{name}` placeholders; `{{` and `}}` are literal braces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplate {
    template: String,
}

enum Piece<'a> {
    Text(&'a str),
    Brace(char),
    Var(&'a str),
}

impl PromptTemplate {
    pub fn new(template: impl Into<String>) -> Result<Self> {
        let t = Self {
            template: template.into(),
        };
        t.pieces()?;
        Ok(t)
    }

    fn pieces(&self) -> Result<Vec<Piece<'_>>> {
        let s = self.template.as_str();
        let bytes = s.as_bytes();
        let (mut out, mut i, mut start) = (Vec::new(), 0, 0);
        while i < bytes.len() {
            match bytes[i] {
                b'{' if bytes.get(i + 1) == Some(&b'{') => {
                    out.push(Piece::Text(&s[start..i]));
                    out.push(Piece::Brace('{'));
                    i += 2;
                    start = i;
                }
                b'}' if bytes.get(i + 1) == Some(&b'}') => {
                    out.push(Piece::Text(&s[start..i]));
                    out.push(Piece::Brace('}'));
                    i += 2;
                    start = i;
                }
                b'{' => {
                    let end = s[i + 1..]
                        .find('}')
                        .ok_or_else(|| Error::Template(format!("unclosed `{{` at byte {i}")))?;
                    let name = s[i + 1..i + 1 + end].trim();
                    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        return Err(Error::Template(format!("invalid variable name `{name}`")));
                    }
                    out.push(Piece::Text(&s[start..i]));
                    out.push(Piece::Var(name));
                    i += end + 2;
                    start = i;
                }
                b'}' => return Err(Error::Template(format!("unmatched `}}` at byte {i}"))),
                _ => i += 1,
            }
        }
        out.push(Piece::Text(&s[start..]));
        Ok(out)
    }

    /// Variable names in order of first appearance.
    pub fn variables(&self) -> Vec<String> {
        let mut seen = Vec::new();
        for p in self.pieces().unwrap_or_default() {
            if let Piece::Var(v) = p {
                if !seen.iter().any(|s: &String| s == v) {
                    seen.push(v.to_string());
                }
            }
        }
        seen
    }

    pub fn render(&self, vars: &HashMap<&str, String>) -> Result<String> {
        self.render_lookup(|v| vars.get(v).map(|s| Cow::Borrowed(s.as_str())))
    }

    /// Renders with owned keys (the [`Vars`](crate::Vars) map runnables use).
    pub fn render_vars(&self, vars: &HashMap<String, String>) -> Result<String> {
        self.render_lookup(|v| vars.get(v).map(|s| Cow::Borrowed(s.as_str())))
    }

    /// Renders from a JSON object: strings are inserted as-is, other values
    /// as JSON text, `null` as an empty string.
    pub fn render_json(&self, vars: &serde_json::Value) -> Result<String> {
        self.render_lookup(|v| {
            vars.get(v).map(|x| match x {
                serde_json::Value::String(s) => Cow::Borrowed(s.as_str()),
                serde_json::Value::Null => Cow::Borrowed(""),
                other => Cow::Owned(other.to_string()),
            })
        })
    }

    /// Renders with a custom variable lookup.
    pub fn render_lookup<'a>(
        &self,
        lookup: impl Fn(&str) -> Option<Cow<'a, str>>,
    ) -> Result<String> {
        let mut out = String::with_capacity(self.template.len());
        for p in self.pieces()? {
            match p {
                Piece::Text(t) => out.push_str(t),
                Piece::Brace(c) => out.push(c),
                Piece::Var(v) => {
                    out.push_str(&lookup(v).ok_or_else(|| Error::MissingVariable(v.to_string()))?)
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_and_escapes() {
        let t = PromptTemplate::new("Hi {name}, json: {{\"k\": {n}}}").unwrap();
        assert_eq!(t.variables(), vec!["name", "n"]);
        let vars = HashMap::from([("name", "Roan".to_string()), ("n", "1".to_string())]);
        assert_eq!(t.render(&vars).unwrap(), "Hi Roan, json: {\"k\": 1}");
    }

    #[test]
    fn missing_variable_errors() {
        let t = PromptTemplate::new("{a}{b}").unwrap();
        let vars = HashMap::from([("a", "x".to_string())]);
        assert!(matches!(t.render(&vars), Err(Error::MissingVariable(v)) if v == "b"));
    }

    #[test]
    fn rejects_malformed() {
        assert!(PromptTemplate::new("oops {name").is_err());
        assert!(PromptTemplate::new("oops }").is_err());
        assert!(PromptTemplate::new("{bad name}").is_err());
    }
}
