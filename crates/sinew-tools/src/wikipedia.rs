//! `wikipedia_search` and `wikipedia_summary`: Wikipedia's public REST APIs,
//! no API key.
//!
//! Both tools take a [`Wikipedia`] client as state, which fixes the language
//! edition (`en`, `pt`, `fr`, ...).
//!
//! ```no_run
//! # async fn demo() -> Result<(), sinew_tools::ToolError> {
//! use sinew_core::ToolBox;
//! use sinew_tools::wikipedia::{wikipedia_search, wikipedia_summary, Wikipedia};
//!
//! let wiki = Wikipedia::new("en");
//! let tools = ToolBox::new()
//!     .with(wikipedia_search::tool(wiki.clone()))
//!     .with(wikipedia_summary::tool(wiki.clone()));
//! let summary = wiki.summary("Rust (programming language)").await?;
//! println!("{}", summary.extract);
//! # let _ = tools;
//! # Ok(()) }
//! ```

use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

use crate::{strip_tags, ToolError, USER_AGENT};

/// A Wikipedia language edition. Cheap to clone.
///
/// ```
/// use sinew_core::Tool;
/// use sinew_tools::wikipedia::{wikipedia_summary, Wikipedia};
///
/// let tool = wikipedia_summary::tool(Wikipedia::new("fr"));
/// assert_eq!(tool.spec().name, "wikipedia_summary");
/// ```
#[derive(Debug, Clone)]
pub struct Wikipedia {
    client: reqwest::Client,
    base_url: String,
}

/// A search hit from [`wikipedia_search`](fn@wikipedia_search).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WikiHit {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Text around the match.
    pub excerpt: String,
    pub url: String,
}

/// The lead section of an article, from
/// [`wikipedia_summary`](fn@wikipedia_summary).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WikiSummary {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Plain-text summary (the article's first paragraphs).
    pub extract: String,
    pub url: String,
    /// `standard`, `disambiguation` (the title has several meanings; search
    /// for a more specific one) or `no-extract`.
    pub kind: String,
}

impl Wikipedia {
    /// The edition for a language code, e.g. `en` or `pt`.
    pub fn new(lang: &str) -> Self {
        let lang: String = lang
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
            .collect();
        let lang = if lang.is_empty() { "en".into() } else { lang };
        Self::with_base_url(format!("https://{lang}.wikipedia.org"))
    }

    /// A MediaWiki site with the same REST APIs (a mirror, a test server).
    pub fn with_base_url(base_url: impl AsRef<str>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent(USER_AGENT)
                .timeout(Duration::from_secs(20))
                .build()
                .unwrap_or_default(),
            base_url: base_url.as_ref().trim_end_matches('/').to_string(),
        }
    }

    /// Uses your own `reqwest` client. Wikipedia requires a descriptive
    /// `User-Agent`; keep one set.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    fn url(&self, path: &[&str]) -> Result<reqwest::Url, ToolError> {
        let mut url = reqwest::Url::parse(&self.base_url)
            .map_err(|e| ToolError::Config(format!("bad Wikipedia base URL: {e}")))?;
        url.path_segments_mut()
            .map_err(|_| ToolError::Config("bad Wikipedia base URL".into()))?
            .pop_if_empty()
            .extend(path);
        Ok(url)
    }

    fn page_url(&self, key: &str) -> String {
        self.url(&["wiki", key])
            .map(|u| u.to_string())
            .unwrap_or_default()
    }

    async fn get_json(&self, url: reqwest::Url) -> Result<(u16, Value), ToolError> {
        let resp = self
            .client
            .get(url)
            .header(reqwest::header::ACCEPT, "application/json")
            .send()
            .await
            .map_err(|e| ToolError::from_reqwest(&e))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(|e| ToolError::from_reqwest(&e))?;
        let v = serde_json::from_str(&text).unwrap_or(Value::Null);
        if !(200..300).contains(&status) && status != 404 {
            let (message, _) = crate::truncate_chars(text.trim(), 300);
            return Err(ToolError::Api {
                service: "wikipedia".into(),
                status,
                message,
            });
        }
        Ok((status, v))
    }

    /// Full-text search over article titles and bodies.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<WikiHit>, ToolError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(ToolError::Invalid("the query is empty".into()));
        }
        let mut url = self.url(&["w", "rest.php", "v1", "search", "page"])?;
        url.query_pairs_mut()
            .append_pair("q", query)
            .append_pair("limit", &limit.clamp(1, 20).to_string());
        let (_, v) = self.get_json(url).await?;
        let pages = v
            .get("pages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Ok(pages
            .iter()
            .map(|p| {
                let key = p.get("key").and_then(Value::as_str).unwrap_or_default();
                WikiHit {
                    title: p
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or(key)
                        .to_string(),
                    description: p
                        .get("description")
                        .and_then(Value::as_str)
                        .filter(|d| !d.is_empty())
                        .map(String::from),
                    excerpt: strip_tags(
                        p.get("excerpt").and_then(Value::as_str).unwrap_or_default(),
                    ),
                    url: self.page_url(key),
                }
            })
            .collect())
    }

    /// The lead section of the article with this exact title (redirects are
    /// followed).
    pub async fn summary(&self, title: &str) -> Result<WikiSummary, ToolError> {
        let title = title.trim();
        if title.is_empty() {
            return Err(ToolError::Invalid("the title is empty".into()));
        }
        let key = title.replace(' ', "_");
        let url = self.url(&["api", "rest_v1", "page", "summary", &key])?;
        let (status, v) = self.get_json(url).await?;
        if status == 404 {
            return Err(ToolError::Invalid(format!(
                "no Wikipedia article is titled `{title}`; search for the exact title first"
            )));
        }
        let get = |k: &str| {
            v.get(k)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        Ok(WikiSummary {
            title: Some(get("title"))
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| title.into()),
            description: Some(get("description")).filter(|d| !d.is_empty()),
            extract: get("extract"),
            url: v
                .pointer("/content_urls/desktop/page")
                .and_then(Value::as_str)
                .map(String::from)
                .unwrap_or_else(|| self.page_url(&key)),
            kind: Some(get("type"))
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| "standard".into()),
        })
    }
}

/// Search Wikipedia and return matching article titles with a short excerpt.
/// Use it to find the exact title of an article before reading its summary.
#[sinew_core::tool]
pub async fn wikipedia_search(
    #[state] wiki: Wikipedia,
    /// What to look for, e.g. "Treaty of Tordesillas".
    query: String,
    /// Number of results, 1 to 20 (default 5).
    limit: Option<usize>,
) -> Result<Vec<WikiHit>, ToolError> {
    wiki.search(&query, limit.unwrap_or(5)).await
}

/// Get the summary (lead section) of a Wikipedia article by its exact title,
/// with the article URL. If the result kind is "disambiguation", the title
/// has several meanings: search for a more specific title.
#[sinew_core::tool]
pub async fn wikipedia_summary(
    #[state] wiki: Wikipedia,
    /// Exact article title, e.g. "Rust (programming language)".
    title: String,
) -> Result<WikiSummary, ToolError> {
    wiki.summary(&title).await
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use sinew_core::Tool;

    use super::*;
    use crate::testutil::{fixture, serve, Reply};

    #[tokio::test]
    async fn searches() {
        let mock = serve(vec![Reply::json(
            "/w/rest.php/v1/search/page",
            fixture!("wikipedia_search.json"),
        )])
        .await;
        let t = wikipedia_search::tool(Wikipedia::with_base_url(&mock.base));
        assert!(!t.side_effects());
        let out = t
            .call(json!({"query": "rust programming language", "limit": 3}))
            .await
            .unwrap();
        assert_eq!(out[0]["title"], "Rust (programming language)");
        assert_eq!(
            out[0]["description"],
            "General-purpose programming language"
        );
        let excerpt = out[0]["excerpt"].as_str().unwrap();
        assert!(
            excerpt.starts_with("Rust is a general-purpose programming language"),
            "{excerpt}"
        );
        assert!(!excerpt.contains("searchmatch"));
        assert_eq!(
            out[0]["url"],
            format!("{}/wiki/Rust_(programming_language)", mock.base)
        );
        let req = &mock.requests()[0];
        assert_eq!(req.uri.query(), Some("q=rust+programming+language&limit=3"));
        assert!(req.headers["user-agent"]
            .to_str()
            .unwrap()
            .starts_with("sinew-tools/"));
    }

    #[tokio::test]
    async fn summarizes_and_reports_missing_pages() {
        let mock = serve(vec![
            Reply::json(
                "/api/rest_v1/page/summary/Rust_(programming_language)",
                fixture!("wikipedia_summary.json"),
            ),
            Reply::new(
                "/api/rest_v1/page/summary/No_such_page",
                404,
                "application/json",
                fixture!("wikipedia_summary_404.json"),
            ),
        ])
        .await;
        let t = wikipedia_summary::tool(Wikipedia::with_base_url(&mock.base));
        let out = t
            .call(json!({"title": "Rust (programming language)"}))
            .await
            .unwrap();
        assert_eq!(out["kind"], "standard");
        assert_eq!(
            out["url"],
            "https://en.wikipedia.org/wiki/Rust_(programming_language)"
        );
        assert!(out["extract"].as_str().unwrap().contains("borrow checker"));
        let err = t.call(json!({"title": "No such page"})).await.unwrap_err();
        assert!(
            err.to_string().contains("search for the exact title"),
            "{err}"
        );
    }

    #[test]
    fn builds_language_urls() {
        let w = Wikipedia::new("pt");
        assert_eq!(
            w.page_url("São_Paulo"),
            "https://pt.wikipedia.org/wiki/S%C3%A3o_Paulo"
        );
        let u = w
            .url(&["api", "rest_v1", "page", "summary", "AC/DC"])
            .unwrap();
        assert_eq!(
            u.as_str(),
            "https://pt.wikipedia.org/api/rest_v1/page/summary/AC%2FDC"
        );
        assert_eq!(
            Wikipedia::new("../evil").base_url,
            "https://evil.wikipedia.org"
        );
    }
}
