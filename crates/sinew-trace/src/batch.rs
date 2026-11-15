//! Shared background batching used by [`crate::HttpSink`] and
//! [`crate::OtlpSink`].
//!
//! `record` only does a `try_send` on a bounded channel; a worker thread calls
//! the `send` closure with a batch when it is full or when the flush interval
//! elapses. The closure returns whether the batch was delivered; runs are
//! counted as sent or dropped accordingly.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError},
        Arc, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crate::Run;

/// Batching settings shared by the network sinks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BatchSettings {
    pub batch_size: usize,
    pub flush_interval: Duration,
    pub capacity: usize,
    pub flush_timeout: Duration,
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

pub(crate) struct Batcher {
    tx: SyncSender<Msg>,
    counters: Arc<Counters>,
    worker: Mutex<Option<JoinHandle<()>>>,
    flush_timeout: Duration,
}

impl Batcher {
    /// `make_send` runs on the worker thread and builds the delivery closure
    /// there (so it may own a blocking HTTP client).
    pub fn spawn<S, F>(thread_name: &str, settings: BatchSettings, make_send: S) -> Self
    where
        S: FnOnce() -> F + Send + 'static,
        F: FnMut(&[Run]) -> bool,
    {
        let (tx, rx) = sync_channel(settings.capacity.max(1));
        let counters = Arc::new(Counters::default());
        let c = counters.clone();
        let worker = std::thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || worker(settings, rx, c, make_send()))
            .ok();
        Self {
            tx,
            counters,
            worker: Mutex::new(worker),
            flush_timeout: settings.flush_timeout,
        }
    }

    pub fn record(&self, run: &Run) {
        if self.tx.try_send(Msg::Run(Box::new(run.clone()))).is_err() {
            self.counters.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn flush(&self) {
        self.control(Msg::Flush);
    }

    pub fn sent(&self) -> u64 {
        self.counters.sent.load(Ordering::Relaxed)
    }

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

impl Drop for Batcher {
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

fn worker<F: FnMut(&[Run]) -> bool>(
    settings: BatchSettings,
    rx: Receiver<Msg>,
    counters: Arc<Counters>,
    mut deliver: F,
) {
    let batch_size = settings.batch_size.max(1);
    let mut batch: Vec<Run> = Vec::new();
    let mut next_flush = Instant::now() + settings.flush_interval;

    let mut send = |batch: &mut Vec<Run>| {
        if batch.is_empty() {
            return;
        }
        let n = batch.len() as u64;
        let counter = if deliver(batch) {
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
                if batch.len() >= batch_size {
                    send(&mut batch);
                    next_flush = Instant::now() + settings.flush_interval;
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
                next_flush = Instant::now() + settings.flush_interval;
            }
            Err(RecvTimeoutError::Disconnected) => {
                send(&mut batch);
                return;
            }
        }
    }
}
