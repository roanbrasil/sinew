//! [`HttpSink`]: ships finished runs to a `sinew-scope serve` instance.
//!
//! `record` never blocks: runs go into a bounded channel and a background
//! thread POSTs them to `/api/runs` in batches, either when a batch is full or
//! when the flush interval elapses. When the channel is full, runs are dropped
//! and counted. Network errors are counted, never raised.

use std::time::Duration;

use serde_json::json;

use crate::{
    batch::{BatchSettings, Batcher},
    Run, TraceSink,
};

/// Settings for [`HttpSink`].
#[derive(Debug, Clone)]
pub struct HttpSinkConfig {
    /// Base URL of the scope server, e.g. `http://127.0.0.1:7878`.
    pub base_url: String,
    /// Project the runs are filed under (`default` on the server if `None`).
    pub project: Option<String>,
    /// Bearer token, if the server was started with `--token`.
    pub token: Option<String>,
    /// Maximum runs per POST.
    pub batch_size: usize,
    /// How long a partial batch may wait before it is sent.
    pub flush_interval: Duration,
    /// Channel capacity; runs beyond it are dropped.
    pub capacity: usize,
    /// HTTP request timeout.
    pub request_timeout: Duration,
    /// How long [`TraceSink::flush`] and drop wait for delivery.
    pub flush_timeout: Duration,
}

impl HttpSinkConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            project: None,
            token: None,
            batch_size: 100,
            flush_interval: Duration::from_secs(1),
            capacity: 10_000,
            request_timeout: Duration::from_secs(5),
            flush_timeout: Duration::from_secs(5),
        }
    }

    /// Reads `SINEW_SCOPE_URL` (default `http://127.0.0.1:7878`),
    /// `SINEW_SCOPE_TOKEN` and `SINEW_PROJECT`.
    pub fn from_env() -> Self {
        let url =
            std::env::var("SINEW_SCOPE_URL").unwrap_or_else(|_| "http://127.0.0.1:7878".into());
        let mut c = Self::new(url);
        c.token = std::env::var("SINEW_SCOPE_TOKEN")
            .ok()
            .filter(|t| !t.is_empty());
        c.project = std::env::var("SINEW_PROJECT")
            .ok()
            .filter(|p| !p.is_empty());
        c
    }

    pub fn project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
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

/// Background batching sink for a `sinew-scope` server.
///
/// ```no_run
/// use sinew_trace::{HttpSink, HttpSinkConfig, Tracer};
/// let tracer = Tracer::new().with_sink(HttpSink::new(
///     HttpSinkConfig::new("http://127.0.0.1:7878").project("demo"),
/// ));
/// // ... run graphs ...
/// tracer.flush(); // deliver the last batch before exiting
/// ```
pub struct HttpSink {
    batcher: Batcher,
}

impl HttpSink {
    pub fn new(config: HttpSinkConfig) -> Self {
        let settings = BatchSettings {
            batch_size: config.batch_size,
            flush_interval: config.flush_interval,
            capacity: config.capacity,
            flush_timeout: config.flush_timeout,
        };
        let batcher = Batcher::spawn("sinew-http-sink", settings, move || {
            let client = reqwest::blocking::Client::builder()
                .timeout(config.request_timeout)
                .build()
                .ok();
            let url = format!("{}/api/runs", config.base_url);
            move |batch: &[Run]| {
                client.as_ref().is_some_and(|client| {
                    let mut req = client.post(&url).json(&json!({
                        "project": config.project,
                        "runs": batch,
                    }));
                    if let Some(t) = &config.token {
                        req = req.bearer_auth(t);
                    }
                    req.send().is_ok_and(|r| r.status().is_success())
                })
            }
        });
        Self { batcher }
    }

    /// Shortcut for `HttpSink::new(HttpSinkConfig::from_env())`.
    pub fn from_env() -> Self {
        Self::new(HttpSinkConfig::from_env())
    }

    /// Runs accepted by the server so far.
    pub fn sent(&self) -> u64 {
        self.batcher.sent()
    }

    /// Runs lost to a full queue or a failed request.
    pub fn dropped(&self) -> u64 {
        self.batcher.dropped()
    }
}

impl TraceSink for HttpSink {
    fn record(&self, run: &Run) {
        self.batcher.record(run);
    }

    /// Blocks the calling thread (up to `flush_timeout`) until every queued
    /// run has been POSTed.
    fn flush(&self) {
        self.batcher.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RunKind, Tracer};
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::{
            mpsc::{sync_channel, Receiver},
            Arc,
        },
    };

    /// Minimal HTTP server that answers 200 and returns each request body.
    fn fake_server() -> (String, Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = sync_channel(64);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                // Read headers, then the body by Content-Length.
                let body = loop {
                    let n = s.read(&mut chunk).unwrap_or(0);
                    if n == 0 {
                        break String::new();
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    if let Some(pos) = text.find("\r\n\r\n") {
                        let len = text
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                            })
                            .unwrap_or(0);
                        if buf.len() >= pos + 4 + len {
                            break String::from_utf8_lossy(&buf[pos + 4..pos + 4 + len])
                                .to_string();
                        }
                    }
                };
                let _ = s.write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                );
                let _ = tx.send(body);
            }
        });
        (format!("http://{addr}"), rx)
    }

    #[test]
    fn batches_and_flushes() {
        let (url, bodies) = fake_server();
        let sink = Arc::new(HttpSink::new(
            HttpSinkConfig::new(url)
                .project("unit")
                .flush_interval(Duration::from_secs(60)),
        ));
        let tracer = Tracer::new().with_shared_sink(sink.clone());
        tracer
            .start("a", RunKind::Chain, None, json!({}))
            .finish(json!(1));
        tracer
            .start("b", RunKind::Chain, None, json!({}))
            .finish(json!(2));
        tracer.flush();
        let body: serde_json::Value =
            serde_json::from_str(&bodies.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap();
        assert_eq!(body["project"], "unit");
        assert_eq!(body["runs"].as_array().unwrap().len(), 2);
        assert_eq!(sink.sent(), 2);
        assert_eq!(sink.dropped(), 0);
    }

    #[test]
    fn unreachable_server_never_panics() {
        // Port 9 (discard) on localhost is almost never listening.
        let sink = HttpSink::new(
            HttpSinkConfig::new("http://127.0.0.1:9")
                .request_timeout(Duration::from_millis(300))
                .capacity(2),
        );
        for i in 0..10 {
            sink.record(&Run {
                id: i.to_string(),
                trace_id: "t".into(),
                parent_id: None,
                name: "x".into(),
                kind: RunKind::Tool,
                start_ms: 0,
                end_ms: Some(1),
                inputs: json!(null),
                outputs: None,
                error: None,
                input_tokens: None,
                output_tokens: None,
                metadata: json!({}),
            });
        }
        sink.flush();
        assert!(sink.dropped() > 0);
        assert_eq!(sink.sent(), 0);
    }
}
