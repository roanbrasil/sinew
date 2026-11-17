//! Run registry: lifecycle, fan-out of SSE frames to subscribers, per-thread
//! concurrency (multitask strategies), cancellation and shutdown.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::{mpsc, watch, Notify, OwnedMutexGuard};
use tokio_util::sync::CancellationToken;

use crate::{
    api::{now_ms, MultitaskStrategy, Run, RunStatus},
    sse::FrameWriter,
};

/// One run, live or finished.
pub(crate) struct RunEntry {
    pub id: String,
    pub thread_id: Option<String>,
    pub cancel: CancellationToken,
    inner: Mutex<Inner>,
    done: watch::Sender<bool>,
}

struct Inner {
    run: Run,
    subscribers: Vec<mpsc::UnboundedSender<Bytes>>,
    /// Terminal frames, replayed to late subscribers.
    tail: Vec<Bytes>,
    /// The `metadata` frame every subscriber gets first.
    head: Bytes,
}

/// How a run ended.
pub(crate) struct Final {
    pub status: RunStatus,
    pub output: Option<Value>,
    pub error: Option<String>,
    pub next: Vec<String>,
}

impl Final {
    pub(crate) fn error(msg: impl Into<String>) -> Self {
        Self {
            status: RunStatus::Error,
            output: None,
            error: Some(msg.into()),
            next: Vec::new(),
        }
    }
    pub(crate) fn cancelled(output: Option<Value>) -> Self {
        Self {
            status: RunStatus::Cancelled,
            output,
            error: None,
            next: Vec::new(),
        }
    }
}

impl RunEntry {
    pub(crate) fn new(run: Run) -> Arc<Self> {
        let head = FrameWriter::default().event(
            "metadata",
            &json!({"run_id": run.run_id, "thread_id": run.thread_id}),
        );
        Arc::new(Self {
            id: run.run_id.clone(),
            thread_id: run.thread_id.clone(),
            cancel: CancellationToken::new(),
            inner: Mutex::new(Inner {
                run,
                subscribers: Vec::new(),
                tail: Vec::new(),
                head,
            }),
            done: watch::channel(false).0,
        })
    }

    pub(crate) fn snapshot(&self) -> Run {
        self.inner.lock().unwrap().run.clone()
    }

    pub(crate) fn status(&self) -> RunStatus {
        self.inner.lock().unwrap().run.status
    }

    pub(crate) fn set_running(&self) {
        let mut i = self.inner.lock().unwrap();
        i.run.status = RunStatus::Running;
        i.run.updated_at = now_ms();
    }

    /// Receives the run's frames from now on: `metadata` first, then live
    /// frames; a finished run yields its terminal frames and closes.
    pub(crate) fn subscribe(&self) -> mpsc::UnboundedReceiver<Bytes> {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut i = self.inner.lock().unwrap();
        let _ = tx.send(i.head.clone());
        if i.run.status.is_terminal() {
            for f in &i.tail {
                let _ = tx.send(f.clone());
            }
        } else {
            i.subscribers.push(tx);
        }
        rx
    }

    pub(crate) fn broadcast(&self, frame: Bytes) {
        let mut i = self.inner.lock().unwrap();
        i.subscribers.retain(|tx| tx.send(frame.clone()).is_ok());
    }

    /// Records the outcome, emits the terminal frames and closes every
    /// subscriber. Returns the final record.
    pub(crate) fn finish(&self, f: Final) -> Run {
        let mut w = FrameWriter::default();
        match f.status {
            RunStatus::Interrupted => {
                w.push_event("interrupt", &json!({"next": f.next, "values": f.output}))
            }
            RunStatus::Error => w.push_event("error", &json!({"error": f.error})),
            _ => {}
        }
        let run = {
            let mut i = self.inner.lock().unwrap();
            i.run.status = f.status;
            i.run.output = f.output;
            i.run.error = f.error;
            i.run.interrupted_before = f.next;
            i.run.updated_at = now_ms();
            w.push_event(
                "end",
                &json!({"run_id": i.run.run_id, "status": i.run.status}),
            );
            let tail = w.take();
            for tx in i.subscribers.drain(..) {
                let _ = tx.send(tail.clone());
            }
            i.tail = vec![tail];
            i.run.clone()
        };
        self.done.send_replace(true);
        run
    }

    /// Resolves once the run is over.
    pub(crate) async fn wait(&self) -> Run {
        let mut rx = self.done.subscribe();
        let _ = rx.wait_for(|d| *d).await;
        self.snapshot()
    }
}

#[derive(Default)]
struct Slot {
    gate: Arc<tokio::sync::Mutex<()>>,
    running: Option<String>,
    pending: Vec<String>,
}

/// Why a run could not be scheduled.
pub(crate) enum Refused {
    Busy,
    Closing,
}

/// All runs of the server.
pub(crate) struct Runs {
    runs: Mutex<HashMap<String, Arc<RunEntry>>>,
    finished: Mutex<VecDeque<String>>,
    threads: Mutex<HashMap<String, Slot>>,
    max_finished: usize,
    closing: AtomicBool,
    active: AtomicUsize,
    idle: Notify,
}

/// Decrements the active-run count when dropped.
pub(crate) struct ActiveGuard(Arc<Runs>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        if self.0.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.idle.notify_waiters();
        }
    }
}

impl Runs {
    pub(crate) fn new(max_finished: usize) -> Arc<Self> {
        Arc::new(Self {
            runs: Mutex::new(HashMap::new()),
            finished: Mutex::new(VecDeque::new()),
            threads: Mutex::new(HashMap::new()),
            max_finished,
            closing: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            idle: Notify::new(),
        })
    }

    pub(crate) fn is_closing(&self) -> bool {
        self.closing.load(Ordering::Acquire)
    }

    pub(crate) fn get(&self, id: &str) -> Option<Arc<RunEntry>> {
        self.runs.lock().unwrap().get(id).cloned()
    }

    /// Runs of a thread, newest first.
    pub(crate) fn of_thread(&self, thread_id: &str) -> Vec<Run> {
        let mut v: Vec<Run> = self
            .runs
            .lock()
            .unwrap()
            .values()
            .filter(|e| e.thread_id.as_deref() == Some(thread_id))
            .map(|e| e.snapshot())
            .collect();
        v.sort_by_key(|r| std::cmp::Reverse(r.created_at));
        v
    }

    /// `(pending, running)` runs right now.
    pub(crate) fn live_counts(&self) -> (usize, usize) {
        let runs = self.runs.lock().unwrap();
        let (mut pending, mut running) = (0, 0);
        for e in runs.values() {
            match e.status() {
                RunStatus::Pending => pending += 1,
                RunStatus::Running => running += 1,
                _ => {}
            }
        }
        (pending, running)
    }

    /// Whether the thread has a run executing or queued.
    pub(crate) fn thread_busy(&self, thread_id: &str) -> bool {
        self.threads
            .lock()
            .unwrap()
            .get(thread_id)
            .is_some_and(|s| s.running.is_some() || !s.pending.is_empty())
    }

    /// Registers a run and, for thread runs, applies the multitask strategy.
    /// Returns the guard that keeps the shutdown waiting for it.
    pub(crate) fn admit(
        self: &Arc<Self>,
        entry: &Arc<RunEntry>,
        strategy: MultitaskStrategy,
    ) -> Result<ActiveGuard, Refused> {
        if self.is_closing() {
            return Err(Refused::Closing);
        }
        if let Some(tid) = &entry.thread_id {
            let mut threads = self.threads.lock().unwrap();
            let slot = threads.entry(tid.clone()).or_default();
            let busy = slot.running.is_some() || !slot.pending.is_empty();
            match strategy {
                MultitaskStrategy::Reject if busy => return Err(Refused::Busy),
                MultitaskStrategy::Interrupt if busy => {
                    let runs = self.runs.lock().unwrap();
                    for id in slot.running.iter().chain(slot.pending.iter()) {
                        if let Some(e) = runs.get(id) {
                            e.cancel.cancel();
                        }
                    }
                }
                _ => {}
            }
            slot.pending.push(entry.id.clone());
        }
        self.runs
            .lock()
            .unwrap()
            .insert(entry.id.clone(), entry.clone());
        self.active.fetch_add(1, Ordering::AcqRel);
        Ok(ActiveGuard(self.clone()))
    }

    /// Waits for the thread's turn (FIFO). `None` if cancelled meanwhile.
    pub(crate) async fn acquire(&self, entry: &RunEntry) -> Option<Option<OwnedMutexGuard<()>>> {
        let Some(tid) = &entry.thread_id else {
            return Some(None);
        };
        let gate = self
            .threads
            .lock()
            .unwrap()
            .entry(tid.clone())
            .or_default()
            .gate
            .clone();
        let guard = tokio::select! {
            biased;
            _ = entry.cancel.cancelled() => None,
            g = gate.lock_owned() => Some(g),
        };
        let mut threads = self.threads.lock().unwrap();
        let slot = threads.entry(tid.clone()).or_default();
        slot.pending.retain(|id| id != &entry.id);
        if guard.is_some() {
            slot.running = Some(entry.id.clone());
        }
        guard.map(Some)
    }

    /// Marks the run as no longer executing on its thread and schedules its
    /// record for eviction.
    pub(crate) fn release(&self, entry: &RunEntry) {
        if let Some(tid) = &entry.thread_id {
            let mut threads = self.threads.lock().unwrap();
            if let Some(slot) = threads.get_mut(tid) {
                if slot.running.as_deref() == Some(entry.id.as_str()) {
                    slot.running = None;
                }
                slot.pending.retain(|id| id != &entry.id);
            }
        }
        let evict = {
            let mut f = self.finished.lock().unwrap();
            f.push_back(entry.id.clone());
            if f.len() > self.max_finished {
                f.pop_front()
            } else {
                None
            }
        };
        if let Some(id) = evict {
            self.runs.lock().unwrap().remove(&id);
        }
    }

    /// Cancels every run of a thread and forgets the thread.
    pub(crate) fn drop_thread(&self, thread_id: &str) {
        for e in self.runs.lock().unwrap().values() {
            if e.thread_id.as_deref() == Some(thread_id) {
                e.cancel.cancel();
            }
        }
        let mut threads = self.threads.lock().unwrap();
        if threads
            .get(thread_id)
            .is_some_and(|s| s.running.is_none() && s.pending.is_empty())
        {
            threads.remove(thread_id);
        }
    }

    /// Stops admitting runs, lets running ones finish for up to `grace`,
    /// then cancels the rest and waits for them to wind down.
    pub(crate) async fn shutdown(&self, grace: Duration) {
        self.closing.store(true, Ordering::Release);
        let drained = async {
            loop {
                let notified = self.idle.notified();
                if self.active.load(Ordering::Acquire) == 0 {
                    return;
                }
                notified.await;
            }
        };
        if tokio::time::timeout(grace, drained).await.is_err() {
            for e in self.runs.lock().unwrap().values() {
                if !e.status().is_terminal() {
                    e.cancel.cancel();
                }
            }
            let _ = tokio::time::timeout(Duration::from_secs(5), async {
                while self.active.load(Ordering::Acquire) > 0 {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await;
        }
    }
}
