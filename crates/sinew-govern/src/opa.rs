//! [`OpaPolicy`]: asks an Open Policy Agent server (feature `opa`).
//!
//! The policy sends `POST {base}/v1/data/{path}` with
//! `{"input": <Request>}` (the [`Request`] serialized as JSON) and reads
//! `result`:
//!
//! | `result` | Decision |
//! |---|---|
//! | absent (undefined) | none: the other policies and the defaults decide |
//! | `true` / `false` | `Allow` / `Deny` |
//! | `{"decision": "allow" \| "ask" \| "deny" \| "redact", "reason": "...", "fields": [...]}` | that decision |
//! | `{"allow": true \| false, "reason": "..."}` | `Allow` / `Deny` |
//!
//! A Rego policy for it:
//!
//! ```text
//! package sinew.tools
//!
//! default decision := {"decision": "allow"}
//!
//! decision := {"decision": "deny", "reason": "refunds over 100 need a supervisor"} if {
//!     input.tool == "refund"
//!     input.arguments.amount > 100
//! }
//! ```
//!
//! with `OpaPolicy::new("opa", "http://localhost:8181", "sinew/tools/decision")`.
//! Network errors, timeouts, non-2xx answers and unexpected results are
//! errors, which the control plane counts as `Deny` (fail closed).

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::{decision::Decision, policy::Policy, request::Request, GovernError};

/// A [`Policy`] answered by an OPA server's data API; see the
/// [module docs](self).
#[derive(Debug, Clone)]
pub struct OpaPolicy {
    id: String,
    url: String,
    path: String,
    client: reqwest::Client,
    timeout: Duration,
    bearer: Option<String>,
}

impl OpaPolicy {
    /// `base` is the server (`http://localhost:8181`), `path` the document
    /// to query, with `/` or `.` separators (`sinew/tools/decision`).
    pub fn new(id: impl Into<String>, base: &str, path: &str) -> Self {
        let path = path.trim_matches('/').replace('.', "/");
        Self {
            id: id.into(),
            url: format!("{}/v1/data/{path}", base.trim_end_matches('/')),
            path,
            client: reqwest::Client::new(),
            timeout: Duration::from_secs(2),
            bearer: None,
        }
    }

    /// Per-request timeout (default 2 s).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sends `Authorization: Bearer <token>` (OPA's token authentication).
    pub fn bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Uses this HTTP client (proxies, TLS roots).
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    fn err(&self, message: impl std::fmt::Display) -> GovernError {
        GovernError::Evaluation {
            policy: self.id.clone(),
            message: message.to_string(),
        }
    }

    fn parse(&self, result: &Value) -> Result<Option<Decision>, GovernError> {
        let reason = |v: &Value| {
            v.get("reason")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("denied by OPA policy `{}`", self.path))
        };
        Ok(Some(match result {
            Value::Null => return Ok(None),
            Value::Bool(true) => Decision::Allow,
            Value::Bool(false) => Decision::deny(format!("denied by OPA policy `{}`", self.path)),
            Value::Object(o) => match (o.get("decision"), o.get("allow")) {
                (Some(Value::String(d)), _) => match d.as_str() {
                    "allow" => Decision::Allow,
                    "deny" => Decision::deny(reason(result)),
                    "ask" => Decision::ask(
                        o.get("reason")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .unwrap_or_else(|| {
                                format!("OPA policy `{}` requires approval", self.path)
                            }),
                    ),
                    "redact" => Decision::redact(
                        o.get("fields")
                            .and_then(Value::as_array)
                            .map(|a| {
                                a.iter()
                                    .filter_map(Value::as_str)
                                    .map(str::to_string)
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default(),
                    ),
                    other => return Err(self.err(format!("unknown decision `{other}` from OPA"))),
                },
                (_, Some(Value::Bool(true))) => Decision::Allow,
                (_, Some(Value::Bool(false))) => Decision::deny(reason(result)),
                _ => {
                    return Err(self.err(format!(
                    "unexpected OPA result (expected a boolean, `decision` or `allow`): {result}"
                )))
                }
            },
            other => return Err(self.err(format!("unexpected OPA result: {other}"))),
        }))
    }
}

#[async_trait]
impl Policy for OpaPolicy {
    fn id(&self) -> &str {
        &self.id
    }

    async fn evaluate(&self, request: &Request) -> Result<Option<Decision>, GovernError> {
        let mut req = self
            .client
            .post(&self.url)
            .timeout(self.timeout)
            .json(&json!({ "input": request }));
        if let Some(t) = &self.bearer {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await.map_err(|e| self.err(e))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(self.err(format!("OPA answered HTTP {status}: {body}")));
        }
        let body: Value = resp.json().await.map_err(|e| self.err(e))?;
        self.parse(body.get("result").unwrap_or(&Value::Null))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::Json, http::HeaderMap, routing::post, Router};

    /// A stand-in for OPA: the decision depends on the input.
    async fn opa(
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> (axum::http::StatusCode, Json<Value>) {
        if headers.get("authorization").and_then(|h| h.to_str().ok()) != Some("Bearer t0ken") {
            return (axum::http::StatusCode::UNAUTHORIZED, Json(json!({})));
        }
        let input = &body["input"];
        let result = match input["tool"].as_str() {
            Some("refund") if input["arguments"]["amount"].as_i64() > Some(100) => {
                json!({"decision": "deny", "reason": "refunds over 100 need a supervisor"})
            }
            Some("refund") => json!(true),
            Some("email") => json!({"decision": "ask"}),
            Some("lookup") => json!({"decision": "redact", "fields": ["/ssn"]}),
            Some("legacy") => json!({"allow": false}),
            Some("weird") => json!({"decision": "maybe"}),
            _ => return (axum::http::StatusCode::OK, Json(json!({}))),
        };
        (
            axum::http::StatusCode::OK,
            Json(json!({ "result": result })),
        )
    }

    #[tokio::test]
    async fn maps_opa_results() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route("/v1/data/sinew/tools/decision", post(opa));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");
        let p = OpaPolicy::new("opa", &base, "sinew.tools.decision").bearer_token("t0ken");
        let req = |tool: &str, args: Value| Request::tool("bot", tool, args);

        assert_eq!(
            p.evaluate(&req("refund", json!({"amount": 5})))
                .await
                .unwrap(),
            Some(Decision::Allow)
        );
        assert_eq!(
            p.evaluate(&req("refund", json!({"amount": 500})))
                .await
                .unwrap(),
            Some(Decision::deny("refunds over 100 need a supervisor"))
        );
        assert_eq!(
            p.evaluate(&req("email", json!({}))).await.unwrap(),
            Some(Decision::ask(
                "OPA policy `sinew/tools/decision` requires approval"
            ))
        );
        assert_eq!(
            p.evaluate(&req("lookup", json!({}))).await.unwrap(),
            Some(Decision::redact(["/ssn"]))
        );
        assert!(p
            .evaluate(&req("legacy", json!({})))
            .await
            .unwrap()
            .unwrap()
            .is_denied());
        assert_eq!(p.evaluate(&req("other", json!({}))).await.unwrap(), None);
        assert!(p.evaluate(&req("weird", json!({}))).await.is_err());

        // Wrong token, unreachable server: errors (the control plane denies).
        let unauth = OpaPolicy::new("opa", &base, "sinew/tools/decision");
        assert!(unauth
            .evaluate(&req("refund", json!({})))
            .await
            .unwrap_err()
            .to_string()
            .contains("HTTP 401"));
        let down =
            OpaPolicy::new("opa", "http://127.0.0.1:1", "x").timeout(Duration::from_millis(300));
        assert!(down.evaluate(&req("refund", json!({}))).await.is_err());
    }
}
