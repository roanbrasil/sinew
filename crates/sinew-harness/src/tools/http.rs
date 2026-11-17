//! `http_fetch`: HTTP GET with a size limit and HTML-to-text extraction.

use std::{sync::LazyLock, time::Duration};

use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};
use sinew_core::{Tool, ToolSpec};
use sinew_tools::{net::HostPolicy, ToolError};

use super::{arg_str, arg_u64, tool_err, truncate_middle};

/// `http_fetch` settings.
#[derive(Debug, Clone, PartialEq)]
pub struct HttpOptions {
    /// Bytes read from the body at most (default 2 MB); the rest is dropped.
    pub max_bytes: usize,
    /// Characters returned to the model (default 20 000).
    pub max_chars: usize,
    pub timeout: Duration,
    /// Allow `localhost`, private, link-local and other non-public addresses
    /// (default false). Checked on IP literals, on every DNS answer at
    /// connection time and on every redirect, with the guard shared with
    /// `sinew-tools` ([`sinew_tools::net::HostPolicy`]). Environment proxies
    /// are not used.
    pub allow_private: bool,
    /// If non-empty, only these hosts (and their subdomains) may be fetched.
    pub allowed_hosts: Vec<String>,
    pub user_agent: String,
}

impl Default for HttpOptions {
    fn default() -> Self {
        Self {
            max_bytes: 2 * 1024 * 1024,
            max_chars: 20_000,
            timeout: Duration::from_secs(30),
            allow_private: false,
            allowed_hosts: Vec::new(),
            user_agent: concat!("sinew-harness/", env!("CARGO_PKG_VERSION")).into(),
        }
    }
}

impl HttpOptions {
    pub fn allow_private(mut self, on: bool) -> Self {
        self.allow_private = on;
        self
    }

    pub fn allow_host(mut self, host: impl Into<String>) -> Self {
        self.allowed_hosts.push(host.into());
        self
    }

    pub fn max_bytes(mut self, n: usize) -> Self {
        self.max_bytes = n;
        self
    }
}

/// The `http_fetch` tool.
#[derive(Debug, Clone)]
pub struct HttpFetchTool {
    /// `Err` only if the TLS backend could not be initialized; then every
    /// call fails instead of falling back to an unguarded client.
    client: Result<reqwest::Client, String>,
    policy: HostPolicy,
    options: HttpOptions,
}

impl HttpFetchTool {
    pub fn new(options: HttpOptions) -> Self {
        let policy = HostPolicy::new()
            .allow_hosts(&options.allowed_hosts)
            .allow_private(options.allow_private);
        let client = policy
            .client_builder()
            .timeout(options.timeout)
            .user_agent(options.user_agent.clone())
            .build()
            .map_err(|e| format!("cannot build the HTTP client: {e}"));
        Self {
            client,
            policy,
            options,
        }
    }

    fn check_url(&self, raw: &str) -> Result<reqwest::Url, String> {
        self.policy.check_url_str(raw).map_err(|e| e.to_string())
    }
}

static DROP_BLOCKS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?is)<script\b.*?</script>|<style\b.*?</style>|<noscript\b.*?</noscript>|<!--.*?-->|<head\b.*?</head>").unwrap()
});
static TITLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<title[^>]*>(.*?)</title>").unwrap());
static BREAKS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)<br\s*/?>|</(?:p|div|li|h[1-6]|tr|section|article|header|footer|pre|blockquote|table)>|<(?:li|h[1-6])\b[^>]*>").unwrap()
});
static TAGS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<[^>]*>").unwrap());
static NUM_ENTITY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"&#(x[0-9a-fA-F]+|[0-9]+);").unwrap());

fn decode_entities(s: &str) -> String {
    let s = NUM_ENTITY.replace_all(s, |c: &regex::Captures<'_>| {
        let v = &c[1];
        let code = if let Some(hex) = v.strip_prefix('x') {
            u32::from_str_radix(hex, 16).ok()
        } else {
            v.parse().ok()
        };
        code.and_then(char::from_u32)
            .map(String::from)
            .unwrap_or_default()
    });
    s.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Extracts readable text from HTML: drops scripts, styles and tags, keeps
/// block structure as line breaks, decodes entities, collapses whitespace.
pub fn html_to_text(html: &str) -> String {
    let title = TITLE
        .captures(html)
        .map(|c| decode_entities(c[1].trim()))
        .filter(|t| !t.is_empty());
    let body = DROP_BLOCKS.replace_all(html, " ");
    let body = BREAKS.replace_all(&body, "\n");
    let body = TAGS.replace_all(&body, " ");
    let body = decode_entities(&body);
    let mut lines = Vec::new();
    let mut blank = false;
    for line in body.lines() {
        let l = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if l.is_empty() {
            if !blank && !lines.is_empty() {
                lines.push(String::new());
            }
            blank = true;
        } else {
            lines.push(l);
            blank = false;
        }
    }
    let text = lines.join("\n").trim().to_string();
    match title {
        Some(t) => format!("Title: {t}\n\n{text}"),
        None => text,
    }
}

#[async_trait]
impl Tool for HttpFetchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "http_fetch".into(),
            description: "Fetch a web page or API response with HTTP GET. HTML is converted to plain text; JSON and text are returned as-is (truncated if long).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s) URL"},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return"}
                },
                "required": ["url"]
            }),
        }
    }

    async fn call(&self, args: Value) -> sinew_core::Result<Value> {
        let name = "http_fetch";
        let url = self
            .check_url(arg_str(name, &args, "url")?)
            .map_err(|e| tool_err(name, e))?;
        let max_chars = arg_u64(&args, "max_chars").map_or(self.options.max_chars, |n| {
            (n as usize).min(self.options.max_chars)
        });
        let client = self.client.as_ref().map_err(|e| tool_err(name, e))?;
        let mut resp = client
            .get(url.clone())
            .send()
            .await
            .map_err(|e| tool_err(name, ToolError::from_reqwest(&e)))?;
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mut body = Vec::new();
        let mut cut = false;
        while let Some(chunk) = resp.chunk().await.map_err(|e| tool_err(name, e))? {
            let room = self.options.max_bytes.saturating_sub(body.len());
            if chunk.len() > room {
                body.extend_from_slice(&chunk[..room]);
                cut = true;
                break;
            }
            body.extend_from_slice(&chunk);
        }
        let raw = String::from_utf8_lossy(&body);
        let looks_html = content_type.contains("html")
            || (content_type.is_empty() && raw.trim_start().starts_with('<'));
        let text = if looks_html {
            html_to_text(&raw)
        } else {
            raw.into_owned()
        };
        let mut out = format!(
            "URL: {url}\nStatus: {}\nContent-Type: {}\n\n{}",
            status.as_u16(),
            if content_type.is_empty() {
                "unknown"
            } else {
                &content_type
            },
            truncate_middle(&text, max_chars)
        );
        if cut {
            out.push_str(&format!("\n[body cut at {} bytes]", self.options.max_bytes));
        }
        Ok(Value::String(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_text_from_html() {
        let html = r#"<html><head><title>Hi &amp; bye</title><style>p{}</style></head>
            <body><script>alert(1)</script><h1>Header</h1><p>One&nbsp;two &lt;3&gt; &#233;</p>
            <ul><li>a</li><li>b</li></ul><!-- hidden --></body></html>"#;
        let t = html_to_text(html);
        assert!(t.starts_with("Title: Hi & bye"), "{t}");
        assert!(t.contains("Header"));
        assert!(t.contains("One two <3> é"), "{t}");
        assert!(!t.contains("alert") && !t.contains("hidden") && !t.contains("p{}"));
        assert!(t.contains("a\n") || t.contains("\na"), "{t}");
    }

    #[test]
    fn rejects_private_and_bad_urls() {
        let t = HttpFetchTool::new(HttpOptions::default());
        assert!(t.check_url("http://127.0.0.1:8080/").is_err());
        assert!(t.check_url("http://localhost/").is_err());
        assert!(t.check_url("http://192.168.1.2/").is_err());
        assert!(t.check_url("http://[::1]/").is_err());
        assert!(t.check_url("http://[::ffff:127.0.0.1]/").is_err());
        assert!(t
            .check_url("http://169.254.169.254/latest/meta-data/")
            .is_err());
        assert!(t.check_url("http://100.64.0.1/").is_err());
        assert!(t.check_url("file:///etc/passwd").is_err());
        assert!(t.check_url("https://example.com/").is_ok());
        let t = HttpFetchTool::new(HttpOptions::default().allow_host("docs.rs"));
        assert!(t.check_url("https://docs.rs/x").is_ok());
        assert!(t.check_url("https://sub.docs.rs/x").is_ok());
        assert!(t.check_url("https://evil.com/docs.rs").is_err());
    }

    #[tokio::test]
    async fn fetches_with_size_limit() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = sock.read(&mut buf).await;
            let body = format!("<html><body><p>{}</p></body></html>", "word ".repeat(1000));
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
        });
        let t = HttpFetchTool::new(HttpOptions::default().allow_private(true).max_bytes(500));
        let out = t
            .call(json!({"url": format!("http://{addr}/page")}))
            .await
            .unwrap();
        let s = out.as_str().unwrap();
        assert!(s.contains("Status: 200"), "{s}");
        assert!(s.contains("word word"));
        assert!(s.contains("[body cut at 500 bytes]"));
    }
}
