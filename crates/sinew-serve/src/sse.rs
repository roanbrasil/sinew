//! Server-Sent Events framing without per-event allocations.
//!
//! Frames are serialized once, straight into a shared [`BytesMut`] arena,
//! and handed out as reference-counted [`Bytes`]: every subscriber of a run
//! gets the same buffer, and the HTTP body writes it as is.

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use bytes::{BufMut, Bytes, BytesMut};
use futures_core::Stream;
use serde::Serialize;
use tokio::{sync::mpsc, time::Interval};

/// Size of the arena blocks frames are carved from.
const ARENA: usize = 16 * 1024;

/// Keep-alive comment sent on idle streams.
pub(crate) const PING: &[u8] = b": ping\n\n";

/// Serializes SSE frames into one growing arena.
pub(crate) struct FrameWriter {
    buf: BytesMut,
}

impl Default for FrameWriter {
    fn default() -> Self {
        Self {
            buf: BytesMut::with_capacity(ARENA),
        }
    }
}

impl FrameWriter {
    fn reserve(&mut self, n: usize) {
        if self.buf.capacity() - self.buf.len() < n {
            self.buf.reserve(ARENA.max(n));
        }
    }

    /// `event: {name}\ndata: {json}\n\n` as its own buffer.
    pub(crate) fn event<T: Serialize + ?Sized>(&mut self, name: &str, data: &T) -> Bytes {
        self.push_event(name, data);
        self.take()
    }

    /// Everything pushed since the last `take`.
    pub(crate) fn take(&mut self) -> Bytes {
        self.buf.split().freeze()
    }

    /// Whether nothing was pushed since the last `take`.
    pub(crate) fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Appends a frame without splitting it off, so several frames can be
    /// written out as one chunk.
    pub(crate) fn push_event<T: Serialize + ?Sized>(&mut self, name: &str, data: &T) {
        self.reserve(64 + name.len());
        self.buf.put_slice(b"event: ");
        self.buf.put_slice(name.as_bytes());
        self.buf.put_slice(b"\ndata: ");
        let mut w = (&mut self.buf).writer();
        // Serializing plain data into memory cannot fail except for maps
        // with non-string keys, which graph states don't produce.
        if serde_json::to_writer(&mut w, data).is_err() {
            self.buf.put_slice(b"null");
        }
        self.buf.put_slice(b"\n\n");
    }

    /// `data: {raw}\n\n` with pre-encoded content (OpenAI chunks).
    pub(crate) fn data_with(&mut self, f: impl FnOnce(&mut BytesMut)) -> Bytes {
        self.reserve(256);
        self.buf.put_slice(b"data: ");
        f(&mut self.buf);
        self.buf.put_slice(b"\n\n");
        self.buf.split().freeze()
    }
}

/// Writes `s` as a JSON string literal (with quotes) into `out`.
pub(crate) fn write_json_str(out: &mut BytesMut, s: &str) {
    out.reserve(s.len() + 2);
    out.put_u8(b'"');
    let bytes = s.as_bytes();
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let esc: &[u8] = match b {
            b'"' => b"\\\"",
            b'\\' => b"\\\\",
            b'\n' => b"\\n",
            b'\r' => b"\\r",
            b'\t' => b"\\t",
            0..=0x1f => {
                out.put_slice(&bytes[start..i]);
                out.put_slice(format!("\\u{b:04x}").as_bytes());
                start = i + 1;
                continue;
            }
            _ => continue,
        };
        out.put_slice(&bytes[start..i]);
        out.put_slice(esc);
        start = i + 1;
    }
    out.put_slice(&bytes[start..]);
    out.put_u8(b'"');
}

/// HTTP body stream: frames from a channel, coalesced when several are
/// ready, with a keep-alive comment after `keep_alive` of silence. Runs
/// `on_drop` when the client goes away before the channel closes.
pub(crate) struct SseBody {
    rx: mpsc::UnboundedReceiver<Bytes>,
    ping: Option<Interval>,
    on_drop: Option<Box<dyn FnOnce() + Send + Sync>>,
    done: bool,
    /// Open-stream gauge, decremented on drop.
    counter: Option<Arc<AtomicUsize>>,
}

impl SseBody {
    pub(crate) fn new(
        rx: mpsc::UnboundedReceiver<Bytes>,
        keep_alive: Option<Duration>,
        on_drop: Option<Box<dyn FnOnce() + Send + Sync>>,
    ) -> Self {
        let ping = keep_alive.map(|d| {
            let mut i = tokio::time::interval_at(tokio::time::Instant::now() + d, d);
            i.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            i
        });
        Self {
            rx,
            ping,
            on_drop,
            done: false,
            counter: None,
        }
    }

    /// Counts this body in `counter` while it is alive.
    pub(crate) fn counted(mut self, counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        self.counter = Some(counter);
        self
    }
}

impl Drop for SseBody {
    fn drop(&mut self) {
        if let Some(c) = &self.counter {
            c.fetch_sub(1, Ordering::Relaxed);
        }
        if !self.done {
            if let Some(f) = self.on_drop.take() {
                f();
            }
        }
    }
}

/// At most this many frames are merged into one write.
const COALESCE: usize = 64;

impl Stream for SseBody {
    type Item = Result<Bytes, std::convert::Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(None);
        }
        match this.rx.poll_recv(cx) {
            Poll::Ready(Some(first)) => {
                if let Some(p) = this.ping.as_mut() {
                    p.reset();
                }
                // Coalesce whatever else is already queued into one chunk.
                let mut more: Option<BytesMut> = None;
                for _ in 0..COALESCE {
                    match this.rx.try_recv() {
                        Ok(b) => more
                            .get_or_insert_with(|| {
                                let mut m = BytesMut::with_capacity(first.len() * 4);
                                m.put_slice(&first);
                                m
                            })
                            .put_slice(&b),
                        Err(_) => break,
                    }
                }
                Poll::Ready(Some(Ok(match more {
                    Some(m) => m.freeze(),
                    None => first,
                })))
            }
            Poll::Ready(None) => {
                this.done = true;
                Poll::Ready(None)
            }
            Poll::Pending => {
                let ticked = this
                    .ping
                    .as_mut()
                    .is_some_and(|p| p.poll_tick(cx).is_ready());
                if ticked {
                    Poll::Ready(Some(Ok(Bytes::from_static(PING))))
                } else {
                    Poll::Pending
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn frames_and_escaping() {
        let mut w = FrameWriter::default();
        let a = w.event("values", &json!({"x": 1}));
        let b = w.event("end", &json!(null));
        assert_eq!(&a[..], b"event: values\ndata: {\"x\":1}\n\n");
        assert_eq!(&b[..], b"event: end\ndata: null\n\n");

        let mut out = BytesMut::new();
        let s = "a\"b\\c\nd\u{1}é";
        write_json_str(&mut out, s);
        let back: String = serde_json::from_slice(&out).unwrap();
        assert_eq!(back, s);
    }

    #[tokio::test]
    async fn body_sends_keep_alive_when_idle() {
        use futures_util::StreamExt;
        let (tx, rx) = mpsc::unbounded_channel::<Bytes>();
        let mut body = SseBody::new(rx, Some(Duration::from_millis(20)), None);
        tx.send(Bytes::from_static(b"x")).unwrap();
        assert_eq!(&body.next().await.unwrap().unwrap()[..], b"x");
        assert_eq!(&body.next().await.unwrap().unwrap()[..], PING);
        drop(tx);
        assert!(body.next().await.is_none());
    }

    #[tokio::test]
    async fn body_coalesces_and_runs_on_drop() {
        use futures_util::StreamExt;
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(Bytes::from_static(b"a")).unwrap();
        tx.send(Bytes::from_static(b"b")).unwrap();
        let dropped = Arc::new(AtomicBool::new(false));
        let d = dropped.clone();
        let mut body = SseBody::new(
            rx,
            None,
            Some(Box::new(move || d.store(true, Ordering::SeqCst))),
        );
        assert_eq!(&body.next().await.unwrap().unwrap()[..], b"ab");
        drop(body);
        assert!(dropped.load(Ordering::SeqCst));
    }
}
