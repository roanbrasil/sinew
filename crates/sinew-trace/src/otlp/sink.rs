//! [`OtlpSink`]: background batching export to an OTLP/HTTP collector.

use std::time::Duration;

use serde_json::{Map, Value};

use super::{base64_encode, export_request};
use crate::{
    batch::{BatchSettings, Batcher},
    Run, TraceSink,
};

/// Wire encoding for [`OtlpSink`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OtlpProtocol {
    /// `application/json` (OTLP/HTTP JSON).
    #[default]
    HttpJson,
    /// `application/x-protobuf` (needs the `otlp-proto` feature).
    #[cfg(feature = "otlp-proto")]
    HttpProtobuf,
}

/// Settings for [`OtlpSink`].
#[derive(Debug, Clone)]
pub struct OtlpSinkConfig {
    /// Full traces endpoint, e.g. `http://localhost:4318/v1/traces`.
    pub endpoint: String,
    /// Extra request headers (auth, tenant ids...).
    pub headers: Vec<(String, String)>,
    /// `service.name` resource attribute.
    pub service_name: String,
    /// Extra resource attributes (`deployment.environment`, ...).
    pub resource: Map<String, Value>,
    pub protocol: OtlpProtocol,
    /// Whether inputs, outputs and messages are exported (default true).
    pub capture_content: bool,
    pub batch_size: usize,
    pub flush_interval: Duration,
    pub capacity: usize,
    pub request_timeout: Duration,
    pub flush_timeout: Duration,
}

/// Percent-decodes `%XX` sequences (OTEL header values are URL-encoded).
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Some(b) = s
                .get(i + 1..i + 3)
                .and_then(|h| u8::from_str_radix(h, 16).ok())
            {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Parses `k1=v1,k2=v2` (the `OTEL_EXPORTER_OTLP_HEADERS` format).
pub fn parse_key_values(s: &str) -> Vec<(String, String)> {
    s.split(',')
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            let k = percent_decode(k.trim());
            (!k.is_empty()).then(|| (k, percent_decode(v.trim())))
        })
        .collect()
}

impl OtlpSinkConfig {
    /// `endpoint` is the full URL of the traces endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            headers: Vec::new(),
            service_name: "sinew-app".into(),
            resource: Map::new(),
            protocol: OtlpProtocol::default(),
            capture_content: true,
            batch_size: 256,
            flush_interval: Duration::from_secs(2),
            capacity: 10_000,
            request_timeout: Duration::from_secs(10),
            flush_timeout: Duration::from_secs(5),
        }
    }

    /// Reads the standard OpenTelemetry variables:
    /// `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (else `OTEL_EXPORTER_OTLP_ENDPOINT`
    /// with `/v1/traces` appended, else `http://localhost:4318/v1/traces`),
    /// `OTEL_EXPORTER_OTLP_HEADERS` / `..._TRACES_HEADERS`, `OTEL_SERVICE_NAME`,
    /// `OTEL_RESOURCE_ATTRIBUTES`, `OTEL_EXPORTER_OTLP_(TRACES_)PROTOCOL` and
    /// `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.
    pub fn from_env() -> Self {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let endpoint = var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            .or_else(|| {
                var("OTEL_EXPORTER_OTLP_ENDPOINT")
                    .map(|e| format!("{}/v1/traces", e.trim_end_matches('/')))
            })
            .unwrap_or_else(|| "http://localhost:4318/v1/traces".into());
        let mut c = Self::new(endpoint);
        for k in [
            "OTEL_EXPORTER_OTLP_HEADERS",
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        ] {
            if let Some(h) = var(k) {
                c.headers.extend(parse_key_values(&h));
            }
        }
        if let Some(s) = var("OTEL_SERVICE_NAME") {
            c.service_name = s;
        }
        if let Some(r) = var("OTEL_RESOURCE_ATTRIBUTES") {
            for (k, v) in parse_key_values(&r) {
                if k == "service.name" && var("OTEL_SERVICE_NAME").is_none() {
                    c.service_name = v.clone();
                }
                c.resource.insert(k, Value::String(v));
            }
        }
        #[cfg(feature = "otlp-proto")]
        {
            let proto = var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
                .or_else(|| var("OTEL_EXPORTER_OTLP_PROTOCOL"))
                .is_some_and(|p| p.trim() == "http/protobuf");
            if proto {
                c.protocol = OtlpProtocol::HttpProtobuf;
            }
        }
        if let Some(v) = var("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT") {
            c.capture_content = !matches!(v.trim().to_ascii_lowercase().as_str(), "false" | "0");
        }
        c
    }

    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Adds a resource attribute.
    pub fn resource_attribute(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.resource.insert(key.into(), value.into());
        self
    }

    /// Sets `sinew.project` (read by `sinew-scope` as the project).
    pub fn project(self, project: impl Into<String>) -> Self {
        self.resource_attribute("sinew.project", project.into())
    }

    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }

    /// `Authorization: Bearer <token>`.
    pub fn bearer(self, token: impl AsRef<str>) -> Self {
        let v = format!("Bearer {}", token.as_ref());
        self.header("authorization", v)
    }

    /// `Authorization: Basic base64(user:password)` (Langfuse uses the public
    /// key as user and the secret key as password).
    pub fn basic_auth(self, user: impl AsRef<str>, password: impl AsRef<str>) -> Self {
        let raw = format!("{}:{}", user.as_ref(), password.as_ref());
        let v = format!("Basic {}", base64_encode(raw.as_bytes()));
        self.header("authorization", v)
    }

    pub fn protocol(mut self, protocol: OtlpProtocol) -> Self {
        self.protocol = protocol;
        self
    }

    pub fn capture_content(mut self, on: bool) -> Self {
        self.capture_content = on;
        self
    }

    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    pub fn flush_interval(mut self, d: Duration) -> Self {
        self.flush_interval = d;
        self
    }

    pub fn capacity(mut self, n: usize) -> Self {
        self.capacity = n.max(1);
        self
    }

    pub fn request_timeout(mut self, d: Duration) -> Self {
        self.request_timeout = d;
        self
    }

    pub fn flush_timeout(mut self, d: Duration) -> Self {
        self.flush_timeout = d;
        self
    }
}

/// Exports finished runs as OpenTelemetry spans (GenAI semantic conventions)
/// to an OTLP/HTTP endpoint, in background batches. Like `HttpSink`, `record`
/// never blocks and failures are counted, never raised.
///
/// ```no_run
/// use sinew_trace::{OtlpSink, OtlpSinkConfig, Tracer};
/// // Jaeger / Tempo / an OpenTelemetry Collector on the default OTLP/HTTP port:
/// let tracer = Tracer::new().with_sink(OtlpSink::new(
///     OtlpSinkConfig::new("http://localhost:4318/v1/traces").service_name("my-agent"),
/// ));
/// // ... run graphs ...
/// tracer.flush();
/// ```
pub struct OtlpSink {
    batcher: Batcher,
}

impl OtlpSink {
    pub fn new(config: OtlpSinkConfig) -> Self {
        let settings = BatchSettings {
            batch_size: config.batch_size,
            flush_interval: config.flush_interval,
            capacity: config.capacity,
            flush_timeout: config.flush_timeout,
        };
        let batcher = Batcher::spawn("sinew-otlp-sink", settings, move || {
            let client = reqwest::blocking::Client::builder()
                .timeout(config.request_timeout)
                .build()
                .ok();
            move |batch: &[Run]| {
                let Some(client) = client.as_ref() else {
                    return false;
                };
                let body = export_request(
                    batch,
                    &config.service_name,
                    &config.resource,
                    config.capture_content,
                );
                let mut req = client.post(&config.endpoint);
                for (k, v) in &config.headers {
                    req = req.header(k.as_str(), v.as_str());
                }
                let req = match config.protocol {
                    OtlpProtocol::HttpJson => req.json(&body),
                    #[cfg(feature = "otlp-proto")]
                    OtlpProtocol::HttpProtobuf => req
                        .header("content-type", "application/x-protobuf")
                        .body(super::proto::encode_request(&body)),
                };
                req.send().is_ok_and(|r| r.status().is_success())
            }
        });
        Self { batcher }
    }

    /// Shortcut for `OtlpSink::new(OtlpSinkConfig::from_env())`.
    pub fn from_env() -> Self {
        Self::new(OtlpSinkConfig::from_env())
    }

    /// Spans accepted by the collector so far.
    pub fn sent(&self) -> u64 {
        self.batcher.sent()
    }

    /// Spans lost to a full queue or a failed request.
    pub fn dropped(&self) -> u64 {
        self.batcher.dropped()
    }
}

impl TraceSink for OtlpSink {
    fn record(&self, run: &Run) {
        self.batcher.record(run);
    }

    fn flush(&self) {
        self.batcher.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_otel_header_lists() {
        let h = parse_key_values("authorization=Basic%20abc==, x-tenant = t1,bad");
        assert_eq!(
            h,
            vec![
                ("authorization".to_string(), "Basic abc==".to_string()),
                ("x-tenant".to_string(), "t1".to_string())
            ]
        );
        let c = OtlpSinkConfig::new("http://x/v1/traces").basic_auth("pk", "sk");
        assert_eq!(c.headers[0].1, "Basic cGs6c2s=");
    }

    #[test]
    fn unreachable_collector_counts_drops() {
        let sink = OtlpSink::new(
            OtlpSinkConfig::new("http://127.0.0.1:9/v1/traces")
                .request_timeout(Duration::from_millis(300)),
        );
        sink.record(&Run {
            id: "a".into(),
            trace_id: "t".into(),
            parent_id: None,
            name: "x".into(),
            kind: crate::RunKind::Chain,
            start_ms: 0,
            end_ms: Some(1),
            inputs: Value::Null,
            outputs: None,
            error: None,
            input_tokens: None,
            output_tokens: None,
            metadata: Value::Null,
        });
        sink.flush();
        assert_eq!(sink.dropped(), 1);
        assert_eq!(sink.sent(), 0);
    }
}
