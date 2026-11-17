//! `web_search`: web search through Tavily, Brave Search or a SearXNG
//! instance.
//!
//! The provider and its API key are state on the tool; the model only sends
//! the query. Keys come from your configuration or from the environment:
//!
//! | Provider | Constructor | Environment |
//! |---|---|---|
//! | [Tavily](https://tavily.com) | [`WebSearch::tavily`] | `TAVILY_API_KEY` |
//! | [Brave Search API](https://brave.com/search/api/) | [`WebSearch::brave`] | `BRAVE_SEARCH_API_KEY` (or `BRAVE_API_KEY`) |
//! | [SearXNG](https://docs.searxng.org) (self-hosted, JSON format enabled) | [`WebSearch::searxng`] | `SEARXNG_URL` |
//!
//! [`WebSearch::from_env`] picks the first one configured, in that order.
//!
//! The provider endpoint is configuration, not model input, so it is not
//! subject to the [`net`](crate::net) guard; a SearXNG instance on your own
//! network works.
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use sinew_core::ToolBox;
//! use sinew_tools::search::{web_search, WebSearch};
//!
//! let search = WebSearch::from_env()?.max_results(5);
//! let tools = ToolBox::new().with(web_search::tool(search.clone()));
//! // Or call the client directly:
//! let hits = search.search("rust async runtime", None).await?;
//! # let _ = (tools, hits);
//! # Ok(()) }
//! ```

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{strip_tags, ToolError, USER_AGENT};

/// Largest number of results a single call returns.
pub const MAX_RESULTS: usize = 20;

#[derive(Clone, PartialEq, Eq)]
enum Provider {
    Tavily { api_key: String },
    Brave { api_key: String },
    SearXng,
}

/// Never prints the API key.
impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Provider::Tavily { .. } => "Tavily",
            Provider::Brave { .. } => "Brave",
            Provider::SearXng => "SearXng",
        })
    }
}

/// A configured search provider. Cheap to clone.
///
/// ```
/// use sinew_tools::search::WebSearch;
///
/// let s = WebSearch::searxng("http://localhost:8888").max_results(8);
/// assert_eq!(s.provider(), "searxng");
/// assert!(!format!("{:?}", WebSearch::tavily("tvly-secret")).contains("secret"));
/// ```
#[derive(Debug, Clone)]
pub struct WebSearch {
    client: reqwest::Client,
    provider: Provider,
    base_url: String,
    max_results: usize,
}

/// One search hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    /// Text excerpt from the page, as returned by the provider.
    pub snippet: String,
    /// Publication date when the provider knows it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published: Option<String>,
}

fn default_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_default()
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

impl WebSearch {
    fn with(provider: Provider, base_url: &str) -> Self {
        Self {
            client: default_client(),
            provider,
            base_url: base_url.trim_end_matches('/').to_string(),
            max_results: 5,
        }
    }

    /// Tavily with an API key (`tvly-...`).
    pub fn tavily(api_key: impl Into<String>) -> Self {
        Self::with(
            Provider::Tavily {
                api_key: api_key.into(),
            },
            "https://api.tavily.com",
        )
    }

    /// Brave Search API with a subscription token.
    pub fn brave(api_key: impl Into<String>) -> Self {
        Self::with(
            Provider::Brave {
                api_key: api_key.into(),
            },
            "https://api.search.brave.com",
        )
    }

    /// A SearXNG instance, e.g. `http://localhost:8888`. The instance must
    /// allow the `json` format (`search.formats` in `settings.yml`).
    pub fn searxng(base_url: impl AsRef<str>) -> Self {
        Self::with(Provider::SearXng, base_url.as_ref())
    }

    /// The first provider configured in the environment: `TAVILY_API_KEY`,
    /// then `BRAVE_SEARCH_API_KEY` / `BRAVE_API_KEY`, then `SEARXNG_URL`.
    pub fn from_env() -> Result<Self, ToolError> {
        if let Some(k) = env("TAVILY_API_KEY") {
            return Ok(Self::tavily(k));
        }
        if let Some(k) = env("BRAVE_SEARCH_API_KEY").or_else(|| env("BRAVE_API_KEY")) {
            return Ok(Self::brave(k));
        }
        if let Some(u) = env("SEARXNG_URL") {
            return Ok(Self::searxng(u));
        }
        Err(ToolError::Config(
            "no web search provider configured; set TAVILY_API_KEY, BRAVE_SEARCH_API_KEY or SEARXNG_URL".into(),
        ))
    }

    /// Overrides the provider's endpoint (a proxy, a regional endpoint, a
    /// test server).
    pub fn base_url(mut self, url: impl AsRef<str>) -> Self {
        self.base_url = url.as_ref().trim_end_matches('/').to_string();
        self
    }

    /// Results returned when the model does not ask for a number (default 5,
    /// at most [`MAX_RESULTS`]).
    pub fn max_results(mut self, n: usize) -> Self {
        self.max_results = n.clamp(1, MAX_RESULTS);
        self
    }

    /// Uses your own `reqwest` client (proxy, timeouts, TLS settings).
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// The provider's name: `tavily`, `brave` or `searxng`.
    pub fn provider(&self) -> &'static str {
        match self.provider {
            Provider::Tavily { .. } => "tavily",
            Provider::Brave { .. } => "brave",
            Provider::SearXng => "searxng",
        }
    }

    /// Runs a search; `limit` defaults to [`max_results`](Self::max_results).
    pub async fn search(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<SearchResult>, ToolError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(ToolError::Invalid("the query is empty".into()));
        }
        let n = limit.unwrap_or(self.max_results).clamp(1, MAX_RESULTS);
        let req = match &self.provider {
            Provider::Tavily { api_key } => self
                .client
                .post(format!("{}/search", self.base_url))
                .bearer_auth(api_key)
                .json(&json!({
                    "query": query,
                    "max_results": n,
                    "search_depth": "basic",
                    "include_answer": false,
                })),
            Provider::Brave { api_key } => self
                .client
                .get(format!("{}/res/v1/web/search", self.base_url))
                .header("X-Subscription-Token", api_key)
                .header(reqwest::header::ACCEPT, "application/json")
                .query(&[("q", query), ("count", &n.to_string())]),
            Provider::SearXng => self
                .client
                .get(format!("{}/search", self.base_url))
                .query(&[("q", query), ("format", "json")]),
        };
        let resp = req.send().await.map_err(|e| ToolError::from_reqwest(&e))?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| ToolError::from_reqwest(&e))?;
        if !status.is_success() {
            let (message, _) = crate::truncate_chars(body.trim(), 500);
            let hint = match (status.as_u16(), &self.provider) {
                (401 | 403, Provider::SearXng) => {
                    " (is the json format enabled in the instance's settings.yml?)"
                }
                (401 | 403, _) => " (check the API key)",
                _ => "",
            };
            return Err(ToolError::Api {
                service: self.provider().into(),
                status: status.as_u16(),
                message: format!("{message}{hint}"),
            });
        }
        let v: Value = serde_json::from_str(&body).map_err(|e| ToolError::Api {
            service: self.provider().into(),
            status: status.as_u16(),
            message: format!("response is not the expected JSON: {e}"),
        })?;
        let mut out = parse_results(&self.provider, &v);
        out.truncate(n);
        Ok(out)
    }
}

fn s(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn opt(v: &Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
}

fn parse_results(provider: &Provider, v: &Value) -> Vec<SearchResult> {
    let list = match provider {
        Provider::Brave { .. } => v.pointer("/web/results"),
        _ => v.get("results"),
    };
    let Some(items) = list.and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|r| {
            let url = s(r, "url");
            if url.is_empty() {
                return None;
            }
            let (snippet, published) = match provider {
                Provider::Tavily { .. } => (s(r, "content"), opt(r, "published_date")),
                Provider::Brave { .. } => (
                    strip_tags(&s(r, "description")),
                    opt(r, "page_age").or_else(|| opt(r, "age")),
                ),
                Provider::SearXng => (s(r, "content"), opt(r, "publishedDate")),
            };
            Some(SearchResult {
                title: strip_tags(&s(r, "title")),
                url,
                snippet,
                published,
            })
        })
        .collect()
}

/// Search the web and return the top results with title, URL and a short
/// snippet. Use it for current events, recent releases, prices and any fact
/// you are not sure of. Write the query as you would type it into a search
/// engine (keywords, names, quoted phrases); the snippets are short, so open a
/// result's URL when you need the full page.
#[sinew_core::tool]
pub async fn web_search(
    #[state] search: WebSearch,
    /// Search query, e.g. "tokio 1.40 release notes".
    query: String,
    /// Number of results, 1 to 20 (default set by the application).
    max_results: Option<usize>,
) -> Result<Vec<SearchResult>, ToolError> {
    search.search(&query, max_results).await
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use sinew_core::Tool;

    use super::*;
    use crate::testutil::{fixture, serve, Reply};

    #[tokio::test]
    async fn tavily_maps_request_and_response() {
        let mock = serve(vec![Reply::json("/search", fixture!("tavily_search.json"))]).await;
        let t = web_search::tool(WebSearch::tavily("tvly-test").base_url(&mock.base));
        assert!(!t.side_effects());
        let out = t
            .call(json!({"query": "rust async runtime", "max_results": 2}))
            .await
            .unwrap();
        assert_eq!(out[0]["title"], "Tokio - An asynchronous Rust runtime");
        assert_eq!(out[0]["url"], "https://tokio.rs/");
        assert!(out[0]["snippet"].as_str().unwrap().starts_with("Tokio is"));
        assert!(out[0].get("published").is_none());
        assert_eq!(out[1]["published"], "2025-03-02");
        let req = &mock.requests()[0];
        assert_eq!(req.method, "POST");
        assert_eq!(req.headers["authorization"], "Bearer tvly-test");
        let body: Value = serde_json::from_str(&req.body).unwrap();
        assert_eq!(body["query"], "rust async runtime");
        assert_eq!(body["max_results"], 2);
    }

    #[tokio::test]
    async fn brave_maps_request_and_response() {
        let mock = serve(vec![Reply::json(
            "/res/v1/web/search",
            fixture!("brave_search.json"),
        )])
        .await;
        let search = WebSearch::brave("brave-key")
            .base_url(&mock.base)
            .max_results(3);
        let out = search.search("rust async runtime", None).await.unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].title, "Tokio - An asynchronous Rust runtime");
        assert_eq!(
            out[1].snippet,
            "A small and fast async runtime for Rust & friends."
        );
        assert_eq!(out[1].published.as_deref(), Some("2025-09-14T08:00:00"));
        let req = &mock.requests()[0];
        assert_eq!(req.headers["x-subscription-token"], "brave-key");
        assert_eq!(req.uri.query(), Some("q=rust+async+runtime&count=3"));
    }

    #[tokio::test]
    async fn searxng_maps_request_and_response() {
        let mock = serve(vec![Reply::json(
            "/search",
            fixture!("searxng_search.json"),
        )])
        .await;
        let search = WebSearch::searxng(format!("{}/", mock.base));
        assert_eq!(search.provider(), "searxng");
        let out = search.search("rust async runtime", Some(2)).await.unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].url, "https://docs.rs/async-std");
        assert_eq!(out[1].published.as_deref(), Some("2024-06-01T00:00:00"));
        assert_eq!(
            mock.requests()[0].uri.query(),
            Some("q=rust+async+runtime&format=json")
        );
    }

    #[tokio::test]
    async fn reports_api_errors() {
        let mock = serve(vec![Reply::new(
            "/search",
            403,
            "text/html",
            "<h1>403 Forbidden</h1>",
        )])
        .await;
        let err = WebSearch::searxng(&mock.base)
            .search("x", None)
            .await
            .unwrap_err();
        assert!(
            matches!(&err, ToolError::Api { status: 403, message, .. } if message.contains("json format")),
            "{err}"
        );
        let err = WebSearch::tavily("bad")
            .base_url(&mock.base)
            .search("  ", None)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Invalid(_)));
    }

    #[test]
    fn schema_hides_state() {
        let spec = web_search::tool(WebSearch::searxng("http://localhost")).spec();
        assert_eq!(spec.name, "web_search");
        let props = spec.parameters["properties"].as_object().unwrap();
        assert!(props.contains_key("query") && props.contains_key("max_results"));
        assert_eq!(props.len(), 2);
    }
}
