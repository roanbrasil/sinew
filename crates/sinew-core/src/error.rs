use std::time::Duration;

use thiserror::Error;

/// Errors from models, tools, prompts and parsers. Every variant links to
/// its entry in the [troubleshooting guide](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md).
#[derive(Debug, Error)]
pub enum Error {
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-http).
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-provider).
    #[error("{provider} returned HTTP {status}: {body}")]
    Provider {
        provider: String,
        status: u16,
        body: String,
    },
    /// The provider rejected the request with HTTP 429, or sent a
    /// `Retry-After` header with an error status (typically 503).
    ///
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-rate-limited).
    #[error("{provider} returned HTTP {status} (rate limited): {body}")]
    RateLimited {
        provider: String,
        status: u16,
        /// Parsed `Retry-After` header (seconds or HTTP date), if any.
        retry_after: Option<Duration>,
        body: String,
    },
    /// A [`timeout`](crate::resilience::WithTimeout) elapsed.
    ///
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-timeout).
    #[error("timed out after {0:?}")]
    Timeout(Duration),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-invalid-response).
    #[error("invalid provider response: {0}")]
    InvalidResponse(String),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-json).
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-tool-not-found).
    #[error("tool `{0}` is not registered")]
    ToolNotFound(String),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-tool).
    #[error("tool `{name}` failed: {message}")]
    Tool { name: String, message: String },
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-missing-variable).
    #[error("prompt template is missing variable `{0}`")]
    MissingVariable(String),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-template).
    #[error("malformed prompt template: {0}")]
    Template(String),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-config).
    #[error("missing configuration: {0}")]
    Config(String),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-structured).
    #[error("structured output failed: {message}")]
    Structured { message: String, raw: String },
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-stream).
    #[error("stream error: {0}")]
    Stream(String),
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-script-exhausted).
    #[error("scripted model has no responses left")]
    ScriptExhausted,
    /// Free-form error raised by a runnable or user code.
    ///
    /// See [troubleshooting](https://github.com/roanbrasil/sinew/blob/main/docs/troubleshooting.md#error-other).
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// The HTTP status for provider errors.
    pub fn status(&self) -> Option<u16> {
        match self {
            Error::Provider { status, .. } | Error::RateLimited { status, .. } => Some(*status),
            Error::Http(e) => e.status().map(|s| s.as_u16()),
            _ => None,
        }
    }

    /// How long the provider asked us to wait, if it said.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Error::RateLimited { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Transient failures worth retrying: rate limits, 408/409/5xx, timeouts
    /// and connection errors.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::RateLimited { .. } | Error::Timeout(_) => true,
            Error::Provider { status, .. } => matches!(*status, 408 | 409 | 429 | 500..=599),
            Error::Http(e) => {
                e.is_timeout()
                    || e.is_connect()
                    || e.is_request()
                    || e.status()
                        .is_some_and(|s| s.as_u16() == 429 || s.is_server_error())
            }
            Error::Stream(_) => true,
            _ => false,
        }
    }
}

/// Parses a `Retry-After` value: delta-seconds (possibly fractional) or an
/// IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`).
pub(crate) fn parse_retry_after(v: &str) -> Option<Duration> {
    let v = v.trim();
    if let Ok(secs) = v.parse::<f64>() {
        return (secs.is_finite() && secs >= 0.0).then(|| Duration::from_secs_f64(secs));
    }
    let target = parse_http_date(v)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(Duration::from_secs(target.saturating_sub(now)))
}

/// IMF-fixdate to Unix seconds.
fn parse_http_date(s: &str) -> Option<u64> {
    // "Sun, 06 Nov 1994 08:49:37 GMT"
    let rest = s.split_once(", ")?.1;
    let mut it = rest.split_whitespace();
    let day: u64 = it.next()?.parse().ok()?;
    let month = match it.next()? {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i64 = it.next()?.parse().ok()?;
    let mut hms = it.next()?.split(':');
    let h: u64 = hms.next()?.parse().ok()?;
    let m: u64 = hms.next()?.parse().ok()?;
    let sec: u64 = hms.next()?.parse().ok()?;
    // Days from civil (Howard Hinnant).
    let (y, mo) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * mo + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    let days = u64::try_from(days).ok()?;
    Some(days * 86400 + h * 3600 + m * 60 + sec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_parses_seconds_and_dates() {
        assert_eq!(parse_retry_after("2"), Some(Duration::from_secs(2)));
        assert_eq!(parse_retry_after("0.5"), Some(Duration::from_millis(500)));
        assert_eq!(
            parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(784111777)
        );
        assert_eq!(
            parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(Duration::ZERO)
        );
        assert_eq!(parse_retry_after("soon"), None);
    }

    #[test]
    fn retryable_classification() {
        let p = |status| Error::Provider {
            provider: "x".into(),
            status,
            body: String::new(),
        };
        assert!(p(500).is_retryable());
        assert!(p(429).is_retryable());
        assert!(!p(400).is_retryable());
        assert!(!p(401).is_retryable());
        assert!(Error::Timeout(Duration::from_secs(1)).is_retryable());
        assert!(!Error::Config("x".into()).is_retryable());
        assert_eq!(p(503).status(), Some(503));
    }
}
