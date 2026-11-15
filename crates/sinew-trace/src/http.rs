//! [`HttpSink`]: ships finished runs to a `sinew-scope serve` instance.
//!
//! `record` never blocks: runs go into a bounded channel and a background
//! thread POSTs them to `/api/runs` in batches, either when a batch is full or
//! when the flush interval elapses. When the channel is full, runs are dropped
//! and counted. Network errors are counted, never raised.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError},
        Arc, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use serde_json::json;

use crate::{Run, TraceSink};

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

enum Msg {
    Run(Box<Run>),
    Flush(SyncSender<()>),
    Shutdown(SyncSender<()>),
}

#[derive(Default)]
struct Counters {
    sent: AtomicU64,
    dropped: AtomicU64,
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
    tx: SyncSender<Msg>,
    counters: Arc<Counters>,
    worker: Mutex<Option<JoinHandle<()>>>,
    flush_timeout: Duration,
}

impl HttpSink {
    pub fn new(config: HttpSinkConfig) -> Self {
        let (tx, rx) = sync_channel(config.capacity);
        let counters = Arc::new(Counters::default());
        let flush_timeout = config.flush_timeout;
        let c = counters.clone();
        let worker = std::thread::Builder::new()
            .name("sinew-http-sink".into())
            .spawn(move || worker(config, rx, c))
            .ok();
        Self {
            tx,
            counters,
            worker: Mutex::new(worker),
            flush_timeout,
        }
    }

    /// Shortcut for `HttpSink::new(HttpSinkConfig::from_env())`.
    pub fn from_env() -> Self {
        Self::new(HttpSinkConfig::from_env())
    }

    /// Runs accepted by the server so far.
    pub fn sent(&self) -> u64 {
        self.counters.sent.load(Ordering::Relaxed)
    }

    /// Runs lost to a full queue or a failed request.
    pub fn dropped(&self) -> u64 {
        self.counters.dropped.load(Ordering::Relaxed)
    }

    /// Enqueues a control message and waits for the worker to acknowledge it,
    /// both within `flush_timeout`. Returns whether it was acknowledged.
    fn control(&self, make: fn(SyncSender<()>) -> Msg) -> bool {
        let deadline = Instant::now() + self.flush_timeout;
        let (ack_tx, ack_rx) = sync_channel(1);
        let mut msg = make(ack_tx);
        loop {
            match self.tx.try_send(msg) {
                Ok(()) => break,
                Err(TrySendError::Disconnected(_)) => return false,
                Err(TrySendError::Full(m)) => {
                    if Instant::now() >= deadline {
                        return false;
                    }
                    msg = m;
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
        let left = deadline.saturating_duration_since(Instant::now());
        ack_rx.recv_timeout(left).is_ok()
    }
}

impl TraceSink for HttpSink {
    fn record(&self, run: &Run) {
        if self.tx.try_send(Msg::Run(Box::new(run.clone()))).is_err() {
            self.counters.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Blocks the calling thread (up to `flush_timeout`) until every queued
    /// run has been POSTed.
    fn flush(&self) {
        self.control(Msg::Flush);
    }
}

impl Drop for HttpSink {
    fn drop(&mut self) {
        let acked = self.control(Msg::Shutdown);
        if let Some(handle) = self.worker.lock().ok().and_then(|mut w| w.take()) {
            // Only join a worker that confirmed it is exiting, so drop never hangs.
            if acked {
                let _ = handle.join();
            }
        }
    }
}

fn worker(config: HttpSinkConfig, rx: Receiver<Msg>, counters: Arc<Counters>) {
    let client = reqwest::blocking::Client::builder()
        .timeout(config.request_timeout)
        .build()
        .ok();
    let url = format!("{}/api/runs", config.base_url);
    let mut batch: Vec<Run> = Vec::new();
    let mut next_flush = Instant::now() + config.flush_interval;

    let send = |batch: &mut Vec<Run>| {
        if batch.is_empty() {
            return;
        }
        let n = batch.len() as u64;
        let ok = client.as_ref().is_some_and(|client| {
            let mut req = client.post(&url).json(&json!({
                "project": config.project,
                "runs": batch,
            }));
            if let Some(t) = &config.token {
                req = req.bearer_auth(t);
            }
            req.send().is_ok_and(|r| r.status().is_success())
        });
        let counter = if ok {
            &counters.sent
        } else {
            &counters.dropped
        };
        counter.fetch_add(n, Ordering::Relaxed);
        batch.clear();
    };

    loop {
        let wait = next_flush.saturating_duration_since(Instant::now());
        match rx.recv_timeout(wait) {
            Ok(Msg::Run(run)) => {
                batch.push(*run);
                if batch.len() >= config.batch_size {
                    send(&mut batch);
                    next_flush = Instant::now() + config.flush_interval;
                }
            }
            Ok(Msg::Flush(ack)) => {
                send(&mut batch);
                let _ = ack.try_send(());
            }
            Ok(Msg::Shutdown(ack)) => {
                send(&mut batch);
                let _ = ack.try_send(());
                return;
            }
            Err(RecvTimeoutError::Timeout) => {
                send(&mut batch);
                next_flush = Instant::now() + config.flush_interval;
            }
            Err(RecvTimeoutError::Disconnected) => {
                send(&mut batch);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RunKind, Tracer};
    use std::{
        io::{Read, Write},
        net::TcpListener,
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
