//! `http_request`: HTTP calls restricted to an allow-list of hosts.
//!
//! Give an agent access to one or a few APIs without writing a tool per
//! endpoint. The application fixes which hosts and methods are allowed and
//! which headers (credentials) are sent; the model chooses the URL, query,
//! extra headers and body.
//!
//! - An allow-list is required: [`HttpRequestBuilder::build`] fails without
//!   one. The [`HostPolicy`] SSRF guard applies too (private addresses stay
//!   blocked unless the policy allows them).
//! - [`Tool::side_effects`] is `true` as soon as a method other than `GET`,
//!   `HEAD` or `OPTIONS` is enabled, so durable runs never repeat a `POST`
//!   whose outcome is unknown. Unsafe methods carry the call's
//!   [`idempotency_key`](sinew_core::tool::idempotency_key) in an
//!   `Idempotency-Key` header when the runtime provides one.
//! - Headers set by the application cannot be overridden by the model.
//!
//! ```
//! use reqwest::Method;
//! use sinew_core::Tool;
//! use sinew_tools::http::HttpRequestTool;
//!
//! let github = HttpRequestTool::builder()
//!     .name("github_api")
//!     .description("GitHub REST API v3. Repository data is under /repos/{owner}/{repo}.")
//!     .allow_host("api.github.com")
//!     .methods([Method::GET, Method::POST])
//!     .header("Authorization", "Bearer <token>")?
//!     .header("Accept", "application/vnd.github+json")?
//!     .build()?;
//! assert!(github.side_effects()); // POST is enabled
//! # Ok::<(), sinew_tools::ToolError>(())
//! ```

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue},
    Method,
};
use serde_json::{json, Map, Value};
use sinew_core::{Tool, ToolSpec};

use crate::{net::HostPolicy, truncate_chars, ToolError, USER_AGENT};

/// Headers the model may never set.
const RESERVED_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "upgrade",
    "te",
    "trailer",
    "keep-alive",
    "proxy-authorization",
    "proxy-connection",
];

/// Builder for [`HttpRequestTool`].
#[derive(Debug, Clone)]
pub struct HttpRequestBuilder {
    name: String,
    description: Option<String>,
    policy: HostPolicy,
    methods: Vec<Method>,
    headers: HeaderMap,
    max_bytes: usize,
    max_chars: usize,
    timeout: Duration,
    idempotency_header: Option<HeaderName>,
}

impl Default for HttpRequestBuilder {
    fn default() -> Self {
        Self {
            name: "http_request".into(),
            description: None,
            policy: HostPolicy::new(),
            methods: vec![Method::GET],
            headers: HeaderMap::new(),
            max_bytes: 2 * 1024 * 1024,
            max_chars: 20_000,
            timeout: Duration::from_secs(30),
            idempotency_header: Some(HeaderName::from_static("idempotency-key")),
        }
    }
}

impl HttpRequestBuilder {
    /// Tool name (default `http_request`). Use one tool per API with a
    /// descriptive name, e.g. `github_api`.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Text added to the tool description: what the API does, its base URL
    /// and main endpoints. The model reads it to decide how to call the tool.
    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.description = Some(text.into());
        self
    }

    /// Allows a host and its subdomains.
    pub fn allow_host(mut self, host: impl AsRef<str>) -> Self {
        self.policy = self.policy.allow_host(host);
        self
    }

    /// Replaces the whole policy (allow-list, deny-list, private addresses).
    pub fn policy(mut self, policy: HostPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Allowed methods (default `GET` only).
    pub fn methods(mut self, methods: impl IntoIterator<Item = Method>) -> Self {
        self.methods = methods.into_iter().collect();
        self.methods.dedup();
        self
    }

    /// A header sent on every request and hidden from the model, typically
    /// credentials.
    pub fn header(mut self, name: &str, value: &str) -> Result<Self, ToolError> {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|e| ToolError::Config(format!("bad header name `{name}`: {e}")))?;
        let mut value = HeaderValue::from_str(value)
            .map_err(|e| ToolError::Config(format!("bad value for header `{name}`: {e}")))?;
        value.set_sensitive(true);
        self.headers.insert(name, value);
        Ok(self)
    }

    /// Bytes read from a response body at most (default 2 MiB).
    pub fn max_bytes(mut self, n: usize) -> Self {
        self.max_bytes = n;
        self
    }

    /// Characters of body text returned at most (default 20 000).
    pub fn max_chars(mut self, n: usize) -> Self {
        self.max_chars = n;
        self
    }

    /// Whole-request timeout (default 30 s).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Header that carries the idempotency key on unsafe methods (default
    /// `Idempotency-Key`; `None` disables it).
    pub fn idempotency_header(mut self, name: Option<&str>) -> Result<Self, ToolError> {
        self.idempotency_header = name
            .map(|n| {
                HeaderName::from_bytes(n.as_bytes())
                    .map_err(|e| ToolError::Config(format!("bad header name `{n}`: {e}")))
            })
            .transpose()?;
        Ok(self)
    }

    /// Builds the tool. Fails without an allowed host or method.
    pub fn build(self) -> Result<HttpRequestTool, ToolError> {
        if self.policy.allowed_hosts().is_empty() {
            return Err(ToolError::Config(
                "http_request needs at least one allowed host (allow_host)".into(),
            ));
        }
        if self.methods.is_empty() {
            return Err(ToolError::Config(
                "http_request needs at least one method".into(),
            ));
        }
        let client = self
            .policy
            .client_builder()
            .timeout(self.timeout)
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| ToolError::Config(format!("cannot build the HTTP client: {e}")))?;
        Ok(HttpRequestTool {
            inner: Arc::new(Inner {
                client,
                spec: spec(&self),
                b: self,
            }),
        })
    }
}

struct Inner {
    client: reqwest::Client,
    spec: ToolSpec,
    b: HttpRequestBuilder,
}

/// The `http_request` tool. Cheap to clone.
#[derive(Clone)]
pub struct HttpRequestTool {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for HttpRequestTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpRequestTool")
            .field("name", &self.inner.b.name)
            .field("allowed_hosts", &self.inner.b.policy.allowed_hosts())
            .field("methods", &self.inner.b.methods)
            .finish_non_exhaustive()
    }
}

fn is_safe(m: &Method) -> bool {
    matches!(*m, Method::GET | Method::HEAD | Method::OPTIONS)
}

fn spec(b: &HttpRequestBuilder) -> ToolSpec {
    let methods: Vec<&str> = b.methods.iter().map(Method::as_str).collect();
    let hosts = b.policy.allowed_hosts().join(", ");
    let mut description = format!(
        "Send an HTTP request and return the status and body (JSON bodies are parsed). \
         Allowed hosts: {hosts} (and their subdomains). Allowed methods: {}.",
        methods.join(", ")
    );
    if b.methods.iter().any(|m| !is_safe(m)) {
        description.push_str(
            " Methods other than GET and HEAD change data on the server: call them only when the user asked for that change.",
        );
    }
    if let Some(extra) = &b.description {
        description.push(' ');
        description.push_str(extra.trim());
    }
    let mut method = json!({"type": "string", "enum": methods, "description": "HTTP method"});
    let mut required = vec!["url"];
    if b.methods.contains(&Method::GET) {
        method["description"] = json!("HTTP method (default GET)");
    } else {
        required.insert(0, "method");
    }
    ToolSpec {
        name: b.name.clone(),
        description,
        parameters: json!({
            "type": "object",
            "properties": {
                "method": method,
                "url": {"type": "string", "description": "Full https URL on an allowed host, e.g. https://api.example.com/v1/items"},
                "query": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Query parameters, added to the URL"},
                "headers": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Extra request headers"},
                "json": {"description": "JSON request body (sets Content-Type: application/json)"},
                "body": {"type": "string", "description": "Raw request body; use `json` for JSON"}
            },
            "required": required,
            "additionalProperties": false
        }),
    }
}

impl HttpRequestTool {
    /// Starts a builder: `GET` only, no hosts allowed yet.
    pub fn builder() -> HttpRequestBuilder {
        HttpRequestBuilder::default()
    }

    async fn run(&self, args: Value) -> Result<Value, ToolError> {
        let b = &self.inner.b;
        let args = match args {
            Value::Object(m) => m,
            Value::Null => Map::new(),
            _ => return Err(ToolError::Invalid("arguments must be a JSON object".into())),
        };
        if let Some(k) = args
            .keys()
            .find(|k| !["method", "url", "query", "headers", "json", "body"].contains(&k.as_str()))
        {
            return Err(ToolError::Invalid(format!("unknown argument `{k}`")));
        }
        let method = match args.get("method").and_then(Value::as_str) {
            Some(m) => Method::from_bytes(m.trim().to_ascii_uppercase().as_bytes())
                .map_err(|_| ToolError::Invalid(format!("bad method `{m}`")))?,
            None if b.methods.contains(&Method::GET) => Method::GET,
            None => return Err(ToolError::Invalid("`method` is required".into())),
        };
        if !b.methods.contains(&method) {
            return Err(ToolError::Blocked(format!(
                "method {method} is not allowed; allowed: {}",
                b.methods
                    .iter()
                    .map(Method::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let url = args
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::Invalid("`url` (string) is required".into()))?;
        let mut url = b.policy.check_url_str(url.trim())?;
        if let Some(q) = args.get("query") {
            let q = q
                .as_object()
                .ok_or_else(|| ToolError::Invalid("`query` must be an object".into()))?;
            let mut pairs = url.query_pairs_mut();
            for (k, v) in q {
                let v = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                pairs.append_pair(k, &v);
            }
        }
        let mut req = self.inner.client.request(method.clone(), url);
        let mut headers = HeaderMap::new();
        if let Some(h) = args.get("headers") {
            let h = h
                .as_object()
                .ok_or_else(|| ToolError::Invalid("`headers` must be an object".into()))?;
            for (k, v) in h {
                let name = HeaderName::from_bytes(k.as_bytes())
                    .map_err(|_| ToolError::Invalid(format!("bad header name `{k}`")))?;
                if RESERVED_HEADERS.contains(&name.as_str()) || name.as_str().starts_with("proxy-")
                {
                    return Err(ToolError::Blocked(format!("header `{name}` cannot be set")));
                }
                if b.headers.contains_key(&name) {
                    return Err(ToolError::Blocked(format!(
                        "header `{name}` is set by the application"
                    )));
                }
                let v = v
                    .as_str()
                    .ok_or_else(|| ToolError::Invalid(format!("header `{k}` must be a string")))?;
                let value = HeaderValue::from_str(v)
                    .map_err(|_| ToolError::Invalid(format!("bad value for header `{k}`")))?;
                headers.insert(name, value);
            }
        }
        if !is_safe(&method) {
            if let (Some(name), Some(key)) =
                (&b.idempotency_header, sinew_core::tool::idempotency_key())
            {
                if !headers.contains_key(name) {
                    if let Ok(v) = HeaderValue::from_str(&key) {
                        headers.insert(name.clone(), v);
                    }
                }
            }
        }
        headers.extend(b.headers.clone());
        req = req.headers(headers);
        match (args.get("json"), args.get("body")) {
            (Some(_), Some(_)) => {
                return Err(ToolError::Invalid(
                    "send either `json` or `body`, not both".into(),
                ))
            }
            (Some(j), None) => req = req.json(j),
            (None, Some(Value::String(s))) => req = req.body(s.clone()),
            (None, Some(_)) => return Err(ToolError::Invalid("`body` must be a string".into())),
            (None, None) => {}
        }
        let resp = req.send().await.map_err(|e| ToolError::from_reqwest(&e))?;
        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let location = resp
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .map(String::from);
        let (bytes, cut) = crate::net::read_limited(resp, b.max_bytes).await?;
        let text = String::from_utf8_lossy(&bytes);
        let (text, text_cut) = truncate_chars(&text, b.max_chars);
        let truncated = cut || text_cut;
        let body = if content_type.contains("json") && !truncated {
            serde_json::from_str(&text).unwrap_or(Value::String(text))
        } else {
            Value::String(text)
        };
        let mut out = json!({
            "status": status,
            "content_type": content_type,
            "body": body,
            "truncated": truncated,
        });
        if let Some(l) = location {
            out["location"] = Value::String(l);
        }
        Ok(out)
    }
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn spec(&self) -> ToolSpec {
        self.inner.spec.clone()
    }

    async fn call(&self, arguments: Value) -> sinew_core::Result<Value> {
        self.run(arguments)
            .await
            .map_err(|e| e.into_core(&self.inner.b.name))
    }

    fn side_effects(&self) -> bool {
        self.inner.b.methods.iter().any(|m| !is_safe(m))
    }
}

#[cfg(test)]
mod tests {
    use sinew_core::tool::with_idempotency_key;

    use super::*;
    use crate::testutil::{serve, Reply};

    fn local(b: HttpRequestBuilder) -> HttpRequestBuilder {
        b.policy(
            HostPolicy::new()
                .allow_private(true)
                .allow_host("127.0.0.1"),
        )
    }

    #[test]
    fn requires_an_allow_list() {
        let err = HttpRequestTool::builder().build().unwrap_err();
        assert!(matches!(err, ToolError::Config(_)));
        let t = HttpRequestTool::builder()
            .allow_host("api.example.com")
            .build()
            .unwrap();
        assert!(!t.side_effects());
        let spec = t.spec();
        assert_eq!(spec.name, "http_request");
        assert!(spec.description.contains("api.example.com"));
        assert_eq!(
            spec.parameters["properties"]["method"]["enum"],
            json!(["GET"])
        );
        assert_eq!(spec.parameters["required"], json!(["url"]));
        let t = HttpRequestTool::builder()
            .allow_host("api.example.com")
            .methods([Method::GET, Method::HEAD])
            .build()
            .unwrap();
        assert!(!t.side_effects());
        let t = HttpRequestTool::builder()
            .allow_host("api.example.com")
            .methods([Method::POST])
            .build()
            .unwrap();
        assert!(t.side_effects());
        assert!(t.spec().description.contains("change data"));
        assert_eq!(t.spec().parameters["required"], json!(["method", "url"]));
    }

    #[tokio::test]
    async fn sends_requests_on_allowed_hosts() {
        let mock = serve(vec![
            Reply::json("/v1/items", r#"{"items": [1, 2]}"#),
            Reply::new("/v1/create", 201, "application/json", r#"{"id": 7}"#),
        ])
        .await;
        let t = local(HttpRequestTool::builder())
            .name("items_api")
            .methods([Method::GET, Method::POST])
            .header("Authorization", "Bearer secret")
            .unwrap()
            .build()
            .unwrap();
        let out = t
            .call(json!({"url": format!("{}/v1/items", mock.base), "query": {"page": "2", "n": 5}}))
            .await
            .unwrap();
        assert_eq!(out["status"], 200);
        assert_eq!(out["body"]["items"], json!([1, 2]));
        let out = with_idempotency_key(
            "thread/tools/1/call_9".into(),
            t.call(json!({
                "method": "post",
                "url": format!("{}/v1/create", mock.base),
                "json": {"name": "x"},
                "headers": {"X-Trace": "abc"}
            })),
        )
        .await
        .unwrap();
        assert_eq!(out["status"], 201);
        let reqs = mock.requests();
        assert_eq!(reqs[0].uri.query(), Some("n=5&page=2"));
        assert_eq!(reqs[0].headers["authorization"], "Bearer secret");
        assert!(reqs[0].headers.get("idempotency-key").is_none());
        assert_eq!(reqs[1].method, "POST");
        assert_eq!(reqs[1].body, r#"{"name":"x"}"#);
        assert_eq!(reqs[1].headers["x-trace"], "abc");
        assert_eq!(reqs[1].headers["idempotency-key"], "thread/tools/1/call_9");
        assert_eq!(reqs[1].headers["content-type"], "application/json");
    }

    #[tokio::test]
    async fn refuses_what_the_policy_does_not_allow() {
        let mock = serve(vec![Reply::json("/x", "{}")]).await;
        let port = mock.base.rsplit(':').next().unwrap().to_string();
        let t = local(HttpRequestTool::builder())
            .header("Authorization", "Bearer secret")
            .unwrap()
            .build()
            .unwrap();
        for args in [
            json!({"method": "DELETE", "url": format!("{}/x", mock.base)}),
            json!({"url": "https://api.other.com/x"}),
            json!({"url": format!("http://localhost:{port}/x")}),
            json!({"url": "file:///etc/passwd"}),
            json!({"url": format!("{}/x", mock.base), "headers": {"Authorization": "Bearer mine"}}),
            json!({"url": format!("{}/x", mock.base), "headers": {"Host": "evil.com"}}),
        ] {
            let err = t.call(args.clone()).await.unwrap_err().to_string();
            assert!(err.contains("blocked"), "{args}: {err}");
        }
        for args in [
            json!({"url": format!("{}/x", mock.base), "json": {}, "body": "x"}),
            json!({"url": format!("{}/x", mock.base), "extra": 1}),
            json!({"method": "GET"}),
        ] {
            let err = t.call(args.clone()).await.unwrap_err().to_string();
            assert!(err.contains("invalid input"), "{args}: {err}");
        }
        assert!(mock.requests().is_empty());
        // Allow-listed but private: blocked unless the policy allows private
        // addresses.
        let t = HttpRequestTool::builder()
            .allow_host("127.0.0.1")
            .build()
            .unwrap();
        let err = t
            .call(json!({"url": format!("{}/x", mock.base)}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("blocked"), "{err}");
    }

    #[tokio::test]
    async fn truncates_long_bodies() {
        let long = "a".repeat(5_000);
        let mock = serve(vec![Reply::new("/long", 200, "text/plain", &long)]).await;
        let t = local(HttpRequestTool::builder())
            .max_bytes(1_000)
            .max_chars(100)
            .build()
            .unwrap();
        let out = t
            .call(json!({"url": format!("{}/long", mock.base)}))
            .await
            .unwrap();
        assert_eq!(out["truncated"], true);
        assert_eq!(out["body"].as_str().unwrap().len(), 100);
    }
}
