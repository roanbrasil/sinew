//! `web_fetch`: download a page chosen by the model and return its main text.
//!
//! HTML goes through [dom_smoothie](https://docs.rs/dom_smoothie), a port of
//! Mozilla's Readability, and comes back as Markdown without navigation,
//! sidebars, scripts or footers. Plain text and JSON are returned as they
//! are; other content types are refused.
//!
//! Limits: the body is read up to [`FetchOptions::max_bytes`] and the text is
//! cut at [`FetchOptions::max_chars`] (the model may ask for less). Every URL,
//! including each redirect, passes the [`HostPolicy`] SSRF guard: by default
//! only public addresses, after DNS resolution.
//!
//! ```no_run
//! # async fn demo() -> Result<(), sinew_tools::ToolError> {
//! use sinew_core::ToolBox;
//! use sinew_tools::{fetch::{web_fetch, FetchOptions, WebFetch}, net::HostPolicy};
//!
//! let fetch = WebFetch::new(
//!     FetchOptions::default()
//!         .policy(HostPolicy::new().deny_host("internal.example.com"))
//!         .max_chars(10_000),
//! )?;
//! let tools = ToolBox::new().with(web_fetch::tool(fetch.clone()));
//! let page = fetch.fetch("https://www.rust-lang.org/", None).await?;
//! println!("{}", page.text);
//! # let _ = tools;
//! # Ok(()) }
//! ```

use std::{sync::Arc, time::Duration};

use serde::Serialize;

use crate::{net::HostPolicy, truncate_chars, ToolError, USER_AGENT};

/// Settings for [`WebFetch`].
#[derive(Debug, Clone, PartialEq)]
pub struct FetchOptions {
    /// Which hosts and addresses may be fetched (default: any public host).
    pub policy: HostPolicy,
    /// Bytes read from the body at most (default 5 MiB); the rest is dropped.
    pub max_bytes: usize,
    /// Characters of text returned at most (default 20 000).
    pub max_chars: usize,
    /// Whole-request timeout (default 20 s).
    pub timeout: Duration,
    pub user_agent: String,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            policy: HostPolicy::new(),
            max_bytes: 5 * 1024 * 1024,
            max_chars: 20_000,
            timeout: Duration::from_secs(20),
            user_agent: USER_AGENT.into(),
        }
    }
}

impl FetchOptions {
    pub fn policy(mut self, policy: HostPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn max_bytes(mut self, n: usize) -> Self {
        self.max_bytes = n;
        self
    }

    pub fn max_chars(mut self, n: usize) -> Self {
        self.max_chars = n;
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }
}

/// A guarded HTTP client with extraction settings. Cheap to clone.
///
/// ```
/// use sinew_tools::fetch::{FetchOptions, WebFetch};
///
/// let fetch = WebFetch::new(FetchOptions::default().max_chars(5_000))?;
/// assert_eq!(fetch.options().max_chars, 5_000);
/// # Ok::<(), sinew_tools::ToolError>(())
/// ```
#[derive(Debug, Clone)]
pub struct WebFetch {
    client: reqwest::Client,
    options: Arc<FetchOptions>,
}

/// What [`web_fetch`](fn@web_fetch) returns.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FetchedPage {
    /// The final URL, after redirects.
    pub url: String,
    pub status: u16,
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byline: Option<String>,
    /// Main text (Markdown for HTML pages).
    pub text: String,
    /// True when the body or the text was cut at a limit.
    pub truncated: bool,
}

impl WebFetch {
    /// Builds the guarded client. Fails only if the TLS backend cannot be
    /// initialized.
    pub fn new(options: FetchOptions) -> Result<Self, ToolError> {
        let client = options
            .policy
            .client_builder()
            .timeout(options.timeout)
            .user_agent(options.user_agent.clone())
            .build()
            .map_err(|e| ToolError::Config(format!("cannot build the HTTP client: {e}")))?;
        Ok(Self {
            client,
            options: Arc::new(options),
        })
    }

    /// The settings in use.
    pub fn options(&self) -> &FetchOptions {
        &self.options
    }

    /// Fetches `url` and extracts its text; `max_chars` can only lower the
    /// configured limit.
    pub async fn fetch(
        &self,
        url: &str,
        max_chars: Option<usize>,
    ) -> Result<FetchedPage, ToolError> {
        let url = self.options.policy.check_url_str(url.trim())?;
        let max_chars = max_chars.map_or(self.options.max_chars, |n| n.min(self.options.max_chars));
        let resp = self
            .client
            .get(url)
            .header(
                reqwest::header::ACCEPT,
                "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.5",
            )
            .send()
            .await
            .map_err(|e| ToolError::from_reqwest(&e))?;
        let final_url = resp.url().to_string();
        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let (body, cut) = crate::net::read_limited(resp, self.options.max_bytes).await?;
        let raw = String::from_utf8_lossy(&body).into_owned();
        let looks_html = content_type.contains("html")
            || (content_type.is_empty() && raw.trim_start().starts_with('<'));
        let is_text = content_type.is_empty()
            || content_type.starts_with("text/")
            || content_type.contains("json")
            || content_type.contains("xml")
            || content_type.contains("javascript");
        let (title, byline, text) = if looks_html {
            let page_url = final_url.clone();
            tokio::task::spawn_blocking(move || extract_html(&raw, &page_url))
                .await
                .map_err(|e| ToolError::Request(format!("extraction failed: {e}")))?
        } else if is_text {
            (None, None, raw)
        } else {
            return Err(ToolError::Invalid(format!(
                "unsupported content type `{content_type}`; web_fetch reads HTML, text and JSON"
            )));
        };
        let (text, text_cut) = truncate_chars(text.trim(), max_chars);
        Ok(FetchedPage {
            url: final_url,
            status,
            content_type: if content_type.is_empty() {
                "unknown".into()
            } else {
                content_type
            },
            title,
            byline,
            text,
            truncated: cut || text_cut,
        })
    }
}

/// Readability extraction to Markdown, with a whole-body fallback for pages
/// the algorithm cannot score (very short pages, app shells).
fn extract_html(html: &str, url: &str) -> (Option<String>, Option<String>, String) {
    use dom_smoothie::{Config, Readability, TextMode};
    let cfg = Config {
        text_mode: TextMode::Markdown,
        max_elements_to_parse: 50_000,
        ..Default::default()
    };
    let Ok(mut r) = Readability::new(html, Some(url), Some(cfg.clone()))
        .or_else(|_| Readability::new(html, None, Some(cfg)))
    else {
        return (None, None, String::new());
    };
    match r.parse() {
        Ok(article) => {
            let title = Some(article.title.trim().to_string()).filter(|t| !t.is_empty());
            let byline = article
                .byline
                .map(|b| b.trim().to_string())
                .filter(|b| !b.is_empty());
            (title, byline, tidy(&article.text_content))
        }
        Err(_) => {
            let title = Some(r.get_article_title().trim().to_string()).filter(|t| !t.is_empty());
            let doc = &r.doc;
            doc.select("script, style, noscript, template, svg")
                .remove();
            let text = doc.select("body").text();
            (title, None, tidy(&text))
        }
    }
}

/// Trims lines and collapses runs of blank lines.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0;
    for line in text.lines() {
        let line = line.trim_end();
        if line.trim().is_empty() {
            blank += 1;
            continue;
        }
        if !out.is_empty() {
            out.push_str(if blank > 0 { "\n\n" } else { "\n" });
        }
        blank = 0;
        out.push_str(line);
    }
    out
}

/// Download a web page and return its main text as Markdown, with navigation,
/// menus, ads and footers removed. Also reads plain text and JSON URLs. Use
/// it to read a page you found or were given; long pages are cut at
/// max_chars. Only public http(s) URLs work: local, private-network and
/// blocked hosts are refused.
#[sinew_core::tool]
pub async fn web_fetch(
    #[state] fetch: WebFetch,
    /// Full http(s) URL, e.g. "https://example.com/post".
    url: String,
    /// Maximum characters to return (cannot exceed the application's limit).
    max_chars: Option<usize>,
) -> Result<FetchedPage, ToolError> {
    fetch.fetch(&url, max_chars).await
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use sinew_core::Tool;

    use super::*;
    use crate::testutil::{fixture, serve, Reply};

    fn local() -> WebFetch {
        WebFetch::new(FetchOptions::default().policy(HostPolicy::new().allow_private(true)))
            .unwrap()
    }

    #[tokio::test]
    async fn extracts_main_text_from_html() {
        let mock = serve(vec![Reply::new(
            "/post",
            200,
            "text/html; charset=utf-8",
            fixture!("article.html"),
        )])
        .await;
        let t = web_fetch::tool(local());
        assert!(!t.side_effects());
        let out = t
            .call(json!({"url": format!("{}/post", mock.base)}))
            .await
            .unwrap();
        let text = out["text"].as_str().unwrap();
        assert!(text.contains("counting boats"), "{text}");
        assert!(text.contains("What happens next"), "{text}");
        assert!(text.contains("Moving ownership"), "{text}");
        for noise in [
            "Subscribe now",
            "Cookie policy",
            "All rights reserved",
            "trackingId",
            "console.log",
        ] {
            assert!(!text.contains(noise), "`{noise}` leaked: {text}");
        }
        assert_eq!(out["status"], 200);
        assert_eq!(out["truncated"], false);
        assert!(out["title"]
            .as_str()
            .unwrap()
            .contains("Ferris goes to the harbor"));
    }

    #[tokio::test]
    async fn enforces_size_limits_and_content_types() {
        let big = format!(
            "<html><body><p>{}</p></body></html>",
            "word ".repeat(10_000)
        );
        let mock = serve(vec![
            Reply::new("/big", 200, "text/html", &big),
            Reply::new("/data.json", 200, "application/json", r#"{"ok": true}"#),
            Reply::new("/img.png", 200, "image/png", "\u{89}PNG"),
            Reply::new("/missing", 404, "text/plain", "not here"),
        ])
        .await;
        let fetch = WebFetch::new(
            FetchOptions::default()
                .policy(HostPolicy::new().allow_private(true))
                .max_bytes(2_000)
                .max_chars(500),
        )
        .unwrap();
        let page = fetch
            .fetch(&format!("{}/big", mock.base), None)
            .await
            .unwrap();
        assert!(page.truncated);
        assert!(page.text.chars().count() <= 500);
        let page = fetch
            .fetch(&format!("{}/big", mock.base), Some(10_000))
            .await
            .unwrap();
        assert!(
            page.text.chars().count() <= 500,
            "max_chars cannot be raised"
        );
        let page = fetch
            .fetch(&format!("{}/data.json", mock.base), None)
            .await
            .unwrap();
        assert_eq!(page.text, r#"{"ok": true}"#);
        assert!(!page.truncated);
        let err = fetch
            .fetch(&format!("{}/img.png", mock.base), None)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("unsupported content type"),
            "{err}"
        );
        let page = fetch
            .fetch(&format!("{}/missing", mock.base), None)
            .await
            .unwrap();
        assert_eq!((page.status, page.text.as_str()), (404, "not here"));
    }

    #[tokio::test]
    async fn blocks_ssrf_by_default() {
        let mock = serve(vec![
            Reply::new("/ok", 200, "text/plain", "secret"),
            Reply::new("/hop", 302, "text/plain", "")
                .header("location", "http://169.254.169.254/latest/meta-data/"),
        ])
        .await;
        let port = mock.base.rsplit(':').next().unwrap().to_string();
        let t = web_fetch::tool(WebFetch::new(FetchOptions::default()).unwrap());
        for url in [
            format!("{}/ok", mock.base),
            format!("http://localhost:{port}/ok"),
            format!("http://[::1]:{port}/ok"),
            format!("http://10.0.0.8:{port}/ok"),
            "http://169.254.169.254/latest/meta-data/".to_string(),
            "file:///etc/passwd".to_string(),
        ] {
            let err = t.call(json!({ "url": url })).await.unwrap_err().to_string();
            assert!(err.contains("blocked"), "{url}: {err}");
        }
        // A public-looking name that resolves to a private address.
        let fetch = WebFetch::new(FetchOptions::default().policy(
            HostPolicy::new().pin_host("intranet.example.com", ["127.0.0.1".parse().unwrap()]),
        ))
        .unwrap();
        let err = fetch
            .fetch(&format!("http://intranet.example.com:{port}/ok"), None)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Blocked(_)), "{err}");
        // A redirect from an allowed host to a denied one.
        let fetch = WebFetch::new(
            FetchOptions::default().policy(
                HostPolicy::new()
                    .allow_private(true)
                    .allow_host("127.0.0.1"),
            ),
        )
        .unwrap();
        let err = fetch
            .fetch(&format!("{}/hop", mock.base), None)
            .await
            .unwrap_err();
        assert!(
            matches!(&err, ToolError::Blocked(m) if m.contains("redirect")),
            "{err}"
        );
        assert_eq!(
            mock.requests().len(),
            1,
            "only the first hop reached the server"
        );
    }

    #[test]
    fn falls_back_for_short_pages() {
        let (title, _, text) = extract_html(
            "<html><head><title>Tiny</title><script>x()</script></head><body><p>Hello <b>there</b></p></body></html>",
            "https://example.com/",
        );
        assert_eq!(title.as_deref(), Some("Tiny"));
        assert!(text.contains("Hello") && text.contains("there"), "{text}");
        assert!(!text.contains("x()"));
    }
}
