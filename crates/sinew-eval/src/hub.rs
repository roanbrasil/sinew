//! [`PromptHub`]: versioned prompts stored in `sinew-scope`.
//!
//! Every push of a changed template creates an immutable version identified
//! by a commit hash (12 hex digits) and a sequential number. Tags such as
//! `prod` or `staging` point at a commit and can be moved. References:
//!
//! | Reference | Resolves to |
//! |---|---|
//! | `name` or `name@latest` | the newest version |
//! | `name@prod` | the commit the `prod` tag points at |
//! | `name@v3` or `name@3` | version number 3 |
//! | `name@1a2b3c4d` | the version whose commit starts with these hex digits |
//!
//! Pulls are cached in memory: commit references forever (they never change),
//! everything else for the cache TTL (default 60 s). When the server cannot
//! be reached, a stale cached copy is returned instead of an error.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_core::{Message, PromptTemplate};

use crate::{client::encode, EvalError, ScopeClient};

/// One chat message of a chat prompt; `content` is a template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: String,
}

/// An immutable prompt version as stored by the hub.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptVersion {
    pub name: String,
    /// Content hash (12 hex digits).
    pub commit: String,
    /// 1-based sequence number.
    pub version: u32,
    /// The template (`{variable}` placeholders, `{{`/`}}` for literal braces).
    pub template: String,
    /// Chat form, when the prompt was pushed as messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<PromptMessage>>,
    #[serde(default)]
    pub input_variables: Vec<String>,
    /// Model settings, e.g. `{"model": "openai:gpt-4.1-mini", "temperature": 0.2}`.
    #[serde(default)]
    pub model_config: Value,
    #[serde(default)]
    pub metadata: Value,
    /// Commit message.
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub parent_commit: Option<String>,
    pub created_ms: u64,
    /// Tags currently pointing at this version.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl PromptVersion {
    /// The template as a sinew-core [`PromptTemplate`].
    pub fn prompt_template(&self) -> Result<PromptTemplate, EvalError> {
        PromptTemplate::new(self.template.clone()).map_err(|e| EvalError::Prompt(e.to_string()))
    }

    /// Renders the prompt as chat messages: each message of a chat prompt,
    /// or a single user message for a plain template.
    pub fn render_messages(&self, vars: &HashMap<&str, String>) -> Result<Vec<Message>, EvalError> {
        let render = |t: &str| {
            PromptTemplate::new(t.to_string())
                .and_then(|p| p.render(vars))
                .map_err(|e| EvalError::Prompt(e.to_string()))
        };
        match &self.messages {
            Some(list) => list
                .iter()
                .map(|m| {
                    let text = render(&m.content)?;
                    Ok(match m.role.as_str() {
                        "system" => Message::system(text),
                        "assistant" => Message::assistant(text),
                        _ => Message::user(text),
                    })
                })
                .collect(),
            None => Ok(vec![Message::user(render(&self.template)?)]),
        }
    }
}

/// What to push. Build with [`PromptPush::new`] or [`PromptPush::chat`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptPush {
    pub name: String,
    pub template: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<PromptMessage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub model_config: Value,
    #[serde(default)]
    pub metadata: Value,
    /// Commit message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Tags to point at the pushed version.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl PromptPush {
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            template: template.into(),
            ..Default::default()
        }
    }

    /// A chat prompt: `(role, content template)` pairs.
    pub fn chat<R: Into<String>, C: Into<String>>(
        name: impl Into<String>,
        messages: impl IntoIterator<Item = (R, C)>,
    ) -> Self {
        let messages: Vec<PromptMessage> = messages
            .into_iter()
            .map(|(r, c)| PromptMessage {
                role: r.into(),
                content: c.into(),
            })
            .collect();
        Self {
            name: name.into(),
            template: messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
            messages: Some(messages),
            ..Default::default()
        }
    }

    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }

    pub fn model_config(mut self, c: Value) -> Self {
        self.model_config = c;
        self
    }

    pub fn metadata(mut self, m: Value) -> Self {
        self.metadata = m;
        self
    }

    pub fn message(mut self, m: impl Into<String>) -> Self {
        self.message = Some(m.into());
        self
    }

    pub fn tag(mut self, t: impl Into<String>) -> Self {
        self.tags.push(t.into());
        self
    }
}

/// Result of a push.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushResult {
    pub name: String,
    pub commit: String,
    pub version: u32,
    /// False when the content equals the latest version (nothing new stored).
    pub created: bool,
}

/// Splits `name@ref` (ref defaults to `latest`).
pub fn split_reference(reference: &str) -> (&str, &str) {
    match reference.split_once('@') {
        Some((n, r)) if !r.is_empty() => (n, r),
        Some((n, _)) => (n, "latest"),
        None => (reference, "latest"),
    }
}

fn is_commit_ref(r: &str) -> bool {
    r.len() >= 12 && r.chars().all(|c| c.is_ascii_hexdigit())
}

#[derive(Clone)]
struct Cached {
    at: Instant,
    version: PromptVersion,
}

/// Client for the scope prompt hub with a local cache.
///
/// ```no_run
/// # async fn demo() -> Result<(), sinew_eval::EvalError> {
/// use std::collections::HashMap;
/// use sinew_eval::{PromptHub, PromptPush};
///
/// let hub = PromptHub::new("http://127.0.0.1:7878");
/// hub.push(PromptPush::new("greeter", "Say hi to {name}.").tag("prod")).await?;
/// let template = hub.pull("greeter@prod").await?;          // sinew_core::PromptTemplate
/// let text = template.render(&HashMap::from([("name", "Ada".to_string())])).unwrap();
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct PromptHub {
    client: ScopeClient,
    ttl: Duration,
    cache: Arc<Mutex<HashMap<String, Cached>>>,
}

impl PromptHub {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_client(ScopeClient::new(base_url))
    }

    /// Reads `SINEW_SCOPE_URL` and `SINEW_SCOPE_TOKEN`.
    pub fn from_env() -> Self {
        Self::with_client(ScopeClient::from_env())
    }

    pub fn with_client(client: ScopeClient) -> Self {
        Self {
            client,
            ttl: Duration::from_secs(60),
            cache: Arc::default(),
        }
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.client = self.client.with_token(token);
        self
    }

    /// How long tag and `latest` lookups stay cached (default 60 s; zero
    /// disables caching for them).
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Pulls a prompt and returns it as a [`PromptTemplate`].
    pub async fn pull(&self, reference: &str) -> Result<PromptTemplate, EvalError> {
        self.pull_version(reference).await?.prompt_template()
    }

    /// Pulls the full version (template, messages, model config, tags).
    pub async fn pull_version(&self, reference: &str) -> Result<PromptVersion, EvalError> {
        let (name, r) = split_reference(reference);
        let key = format!("{name}@{r}");
        let cached = self.cache.lock().ok().and_then(|c| c.get(&key).cloned());
        if let Some(c) = &cached {
            if is_commit_ref(r) || c.at.elapsed() < self.ttl {
                return Ok(c.version.clone());
            }
        }
        let fetched: Result<PromptVersion, EvalError> = self
            .client
            .get_json(&format!("/api/prompts/{}/{}", encode(name), encode(r)))
            .await;
        match fetched {
            Ok(v) => {
                if let Ok(mut c) = self.cache.lock() {
                    let entry = Cached {
                        at: Instant::now(),
                        version: v.clone(),
                    };
                    c.insert(format!("{name}@{}", v.commit), entry.clone());
                    c.insert(key, entry);
                }
                Ok(v)
            }
            // Serve stale on network errors, never on "not found".
            Err(EvalError::Http(e)) => match cached {
                Some(c) => Ok(c.version),
                None => Err(EvalError::Http(e)),
            },
            Err(e) => Err(e),
        }
    }

    /// Pushes a new version (no-op when the content equals the latest one).
    pub async fn push(&self, push: PromptPush) -> Result<PushResult, EvalError> {
        let name = push.name.clone();
        let r: PushResult = self.client.post_json("/api/prompts", &push).await?;
        self.invalidate(&name);
        Ok(r)
    }

    /// Points `tag` at `commit` (a commit prefix, `vN` or another tag).
    pub async fn tag(&self, name: &str, tag: &str, commit: &str) -> Result<(), EvalError> {
        self.client
            .put_json::<Value>(
                &format!("/api/prompts/{}/tags/{}", encode(name), encode(tag)),
                &json!({ "commit": commit }),
            )
            .await?;
        self.invalidate(name);
        Ok(())
    }

    /// Every version of a prompt, newest first.
    pub async fn versions(&self, name: &str) -> Result<Vec<PromptVersion>, EvalError> {
        let v: Value = self
            .client
            .get_json(&format!("/api/prompts/{}", encode(name)))
            .await?;
        serde_json::from_value(v["versions"].clone())
            .map_err(|e| EvalError::Http(format!("bad response: {e}")))
    }

    /// Drops cached tag/`latest` entries of a prompt (commit entries stay).
    pub fn invalidate(&self, name: &str) {
        if let Ok(mut c) = self.cache.lock() {
            let prefix = format!("{name}@");
            c.retain(|k, _| !k.starts_with(&prefix) || is_commit_ref(&k[prefix.len()..]));
        }
    }

    /// Empties the cache.
    pub fn clear_cache(&self) {
        if let Ok(mut c) = self.cache.lock() {
            c.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_and_rendering() {
        assert_eq!(split_reference("a"), ("a", "latest"));
        assert_eq!(split_reference("a@prod"), ("a", "prod"));
        assert_eq!(split_reference("a@"), ("a", "latest"));
        assert!(is_commit_ref("0123456789ab"));
        assert!(!is_commit_ref("prod"));

        let p = PromptPush::chat("qa", [("system", "You answer in {lang}."), ("user", "{q}")]);
        let v = PromptVersion {
            name: p.name.clone(),
            commit: "0123456789ab".into(),
            version: 1,
            template: p.template.clone(),
            messages: p.messages.clone(),
            input_variables: vec!["lang".into(), "q".into()],
            model_config: Value::Null,
            metadata: Value::Null,
            message: None,
            parent_commit: None,
            created_ms: 0,
            tags: vec![],
        };
        let vars = HashMap::from([("lang", "French".to_string()), ("q", "2+2?".to_string())]);
        let msgs = v.render_messages(&vars).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "You answer in French.");
        assert_eq!(v.prompt_template().unwrap().variables(), vec!["lang", "q"]);
    }
}
