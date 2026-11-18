//! Streaming chat responses.
//!
//! [`ChatModel::chat_stream`] yields
//! [`StreamEvent`]s: text deltas, tool-call deltas, complete tool calls, usage
//! and a final `Done`. [`StreamAccumulator`] folds them back into a
//! [`ChatResponse`], and [`SseParser`] turns raw Server-Sent Events bytes into
//! events for the provider decoders.

use std::{collections::BTreeMap, pin::Pin};

use futures_core::Stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ChatModel, ChatRequest, ChatResponse, Message, Result, ToolCall, Usage};

/// A boxed, sendable stream.
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

/// The stream returned by [`ChatModel::chat_stream`].
pub type ChatStream = BoxStream<'static, Result<StreamEvent>>;

/// One incremental piece of a model response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// A chunk of assistant text.
    TextDelta { text: String },
    /// A chunk of a tool call. `index` identifies the call within the
    /// response; `id` and `name` usually arrive only in the first chunk and
    /// `arguments` is a fragment of the JSON argument string.
    ToolCallDelta {
        index: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        arguments: String,
    },
    /// A fully assembled tool call with parsed arguments. Providers emit it
    /// after the call's deltas, so consumers that don't care about partial
    /// arguments can just watch for this.
    ToolCall { index: usize, call: ToolCall },
    /// Token usage for the whole response (emitted once, near the end).
    Usage { usage: Usage },
    /// The response is finished. Always the last event of a successful stream.
    Done {
        model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
    },
}

impl StreamEvent {
    pub fn text(text: impl Into<String>) -> Self {
        StreamEvent::TextDelta { text: text.into() }
    }
}

/// Turns a complete response into the event sequence a streaming provider
/// would have produced. Used by the default `chat_stream`.
pub fn events_from_response(resp: ChatResponse) -> Vec<StreamEvent> {
    let mut out = Vec::new();
    if !resp.message.content.is_empty() {
        out.push(StreamEvent::text(resp.message.content));
    }
    for (index, call) in resp.message.tool_calls.into_iter().enumerate() {
        out.push(StreamEvent::ToolCall { index, call });
    }
    out.push(StreamEvent::Usage { usage: resp.usage });
    out.push(StreamEvent::Done {
        model: resp.model,
        stop_reason: None,
    });
    out
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

/// Folds [`StreamEvent`]s into a [`ChatResponse`].
///
/// Complete `ToolCall` events win over deltas with the same index; calls that
/// only ever arrived as deltas are assembled and their arguments parsed at
/// [`finish`](Self::finish).
#[derive(Default)]
pub struct StreamAccumulator {
    text: String,
    partial: BTreeMap<usize, PartialCall>,
    complete: BTreeMap<usize, ToolCall>,
    usage: Usage,
    model: String,
    stop_reason: Option<String>,
    done: bool,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::TextDelta { text } => self.text.push_str(text),
            StreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments,
            } => {
                let p = self.partial.entry(*index).or_default();
                if let Some(id) = id.as_ref().filter(|s| !s.is_empty()) {
                    p.id.clone_from(id);
                }
                if let Some(name) = name.as_ref().filter(|s| !s.is_empty()) {
                    p.name.clone_from(name);
                }
                p.arguments.push_str(arguments);
            }
            StreamEvent::ToolCall { index, call } => {
                self.complete.insert(*index, call.clone());
            }
            StreamEvent::Usage { usage } => {
                // Providers may split usage over several events; a non-zero
                // field always replaces what came before.
                let set = |dst: &mut u32, src: u32| {
                    if src > 0 {
                        *dst = src;
                    }
                };
                set(&mut self.usage.input_tokens, usage.input_tokens);
                set(&mut self.usage.output_tokens, usage.output_tokens);
                set(
                    &mut self.usage.cached_input_tokens,
                    usage.cached_input_tokens,
                );
                set(
                    &mut self.usage.cache_creation_input_tokens,
                    usage.cache_creation_input_tokens,
                );
                set(&mut self.usage.reasoning_tokens, usage.reasoning_tokens);
            }
            StreamEvent::Done { model, stop_reason } => {
                self.model.clone_from(model);
                self.stop_reason.clone_from(stop_reason);
                self.done = true;
            }
        }
    }

    /// Text received so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether a `Done` event was seen.
    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn stop_reason(&self) -> Option<&str> {
        self.stop_reason.as_deref()
    }

    pub fn finish(self) -> ChatResponse {
        let mut calls = self.complete;
        for (index, p) in self.partial {
            calls.entry(index).or_insert_with(|| ToolCall {
                id: p.id,
                name: p.name,
                arguments: parse_arguments(&p.arguments),
            });
        }
        ChatResponse {
            message: Message::assistant_with_tools(self.text, calls.into_values().collect()),
            usage: self.usage,
            model: self.model,
        }
    }
}

/// Parses a tool-call argument string; empty means `{}`, invalid JSON is kept
/// as a string so nothing is lost.
pub(crate) fn parse_arguments(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return Value::Object(Default::default());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Drains a stream into a [`ChatResponse`].
pub async fn collect_stream(mut stream: ChatStream) -> Result<ChatResponse> {
    let mut acc = StreamAccumulator::new();
    while let Some(ev) = stream.next().await {
        acc.push(&ev?);
    }
    Ok(acc.finish())
}

/// Streams a chat request, calling `on_event` for every event, and returns the
/// accumulated response.
pub async fn chat_streaming<M, F>(
    model: &M,
    request: ChatRequest,
    mut on_event: F,
) -> Result<ChatResponse>
where
    M: ChatModel + ?Sized,
    F: FnMut(&StreamEvent),
{
    let mut stream = model.chat_stream(request).await?;
    let mut acc = StreamAccumulator::new();
    while let Some(ev) = stream.next().await {
        let ev = ev?;
        on_event(&ev);
        acc.push(&ev);
    }
    Ok(acc.finish())
}

/// One Server-Sent Event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, if any.
    pub event: Option<String>,
    /// All `data:` lines joined with `\n`.
    pub data: String,
}

/// Incremental Server-Sent Events parser. Feed it raw bytes in whatever
/// chunks the network delivers; it returns each event once its terminating
/// blank line has arrived. Handles `\n`, `\r\n` and `\r` line endings,
/// comments, multi-line data and UTF-8 split across chunks.
///
/// Work is linear in the input: bytes of an unfinished line are scanned once
/// (the scan resumes where the previous chunk stopped) and the buffer is
/// compacted once per chunk, so a long line delivered in many small chunks
/// does not cause quadratic rescanning.
#[derive(Default)]
pub struct SseParser {
    buf: Vec<u8>,
    /// Bytes of `buf` already known not to contain a line break.
    scanned: usize,
    event: Option<String>,
    data: Vec<String>,
    /// A `\r` ended the last chunk; a following `\n` belongs to it.
    pending_cr: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        // An empty chunk carries nothing; returning early also keeps a
        // pending `\r` pending, so a following `\n` still pairs with it.
        if chunk.is_empty() {
            return Vec::new();
        }
        let mut chunk = chunk;
        if self.pending_cr && chunk.first() == Some(&b'\n') {
            chunk = &chunk[1..];
        }
        self.pending_cr = false;
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        let mut start = 0;
        let mut i = self.scanned;
        while i < self.buf.len() {
            let b = self.buf[i];
            if b == b'\n' || b == b'\r' {
                let line = String::from_utf8_lossy(&self.buf[start..i]).into_owned();
                if b == b'\r' {
                    if i + 1 < self.buf.len() {
                        if self.buf[i + 1] == b'\n' {
                            i += 1;
                        }
                    } else {
                        self.pending_cr = true;
                    }
                }
                i += 1;
                start = i;
                if let Some(ev) = self.line(&line) {
                    out.push(ev);
                }
            } else {
                i += 1;
            }
        }
        self.buf.drain(..start);
        self.scanned = self.buf.len();
        out
    }

    /// Flushes a trailing event that was not followed by a blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        self.scanned = 0;
        if !self.buf.is_empty() {
            let line = String::from_utf8_lossy(&std::mem::take(&mut self.buf)).into_owned();
            self.line(&line);
        }
        self.dispatch()
    }

    fn line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return None;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => self.data.push(value.to_string()),
            _ => {}
        }
        None
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        let event = self.event.take();
        if self.data.is_empty() {
            return None;
        }
        Some(SseEvent {
            event,
            data: std::mem::take(&mut self.data).join("\n"),
        })
    }
}

/// Provider-specific translation from SSE events to [`StreamEvent`]s.
pub(crate) trait SseDecoder: Send + 'static {
    fn on_event(&mut self, ev: &SseEvent) -> Result<Vec<StreamEvent>>;
    /// Called once when the provider signalled the end or the body ended.
    fn finish(&mut self) -> Vec<StreamEvent>;
    /// The provider sent its end-of-stream marker.
    fn is_done(&self) -> bool;
}

/// Feeds a whole recorded SSE body through a decoder. Used by tests.
#[cfg(test)]
pub(crate) fn decode_all<D: SseDecoder>(decoder: D, body: &str) -> Result<Vec<StreamEvent>> {
    decode_chunks(decoder, [body.as_bytes()])
}

/// Feeds an SSE body, delivered in `chunks`, through a decoder the way
/// [`sse_stream`] does: events are decoded as their chunk arrives, the first
/// decoder error ends the stream, and the provider's end marker stops reading.
pub(crate) fn decode_chunks<'a, D: SseDecoder>(
    mut decoder: D,
    chunks: impl IntoIterator<Item = &'a [u8]>,
) -> Result<Vec<StreamEvent>> {
    let mut parser = SseParser::new();
    let mut out = Vec::new();
    let mut feed = |events: Vec<SseEvent>, out: &mut Vec<StreamEvent>| -> Result<bool> {
        for ev in events {
            out.extend(decoder.on_event(&ev)?);
            if decoder.is_done() {
                return Ok(true);
            }
        }
        Ok(false)
    };
    let mut done = false;
    for chunk in chunks {
        if feed(parser.push(chunk), &mut out)? {
            done = true;
            break;
        }
    }
    if !done {
        feed(parser.finish().into_iter().collect(), &mut out)?;
    }
    out.extend(decoder.finish());
    Ok(out)
}

/// Turns an HTTP response with an SSE body into a [`ChatStream`].
pub(crate) fn sse_stream<D: SseDecoder>(resp: reqwest::Response, decoder: D) -> ChatStream {
    use std::collections::VecDeque;

    struct St<D> {
        resp: Option<reqwest::Response>,
        parser: SseParser,
        decoder: D,
        queue: VecDeque<Result<StreamEvent>>,
        finished: bool,
    }

    let state = St {
        resp: Some(resp),
        parser: SseParser::new(),
        decoder,
        queue: VecDeque::new(),
        finished: false,
    };

    futures_util::stream::unfold(state, |mut st| async move {
        loop {
            if let Some(item) = st.queue.pop_front() {
                return Some((item, st));
            }
            if st.finished {
                return None;
            }
            let Some(resp) = st.resp.as_mut() else {
                st.finished = true;
                st.queue.extend(st.decoder.finish().into_iter().map(Ok));
                continue;
            };
            let events = match resp.chunk().await {
                Ok(Some(bytes)) => st.parser.push(&bytes),
                Ok(None) => {
                    st.resp = None;
                    st.parser.finish().into_iter().collect()
                }
                Err(e) => {
                    st.finished = true;
                    st.resp = None;
                    return Some((Err(e.into()), st));
                }
            };
            for ev in events {
                match st.decoder.on_event(&ev) {
                    Ok(evs) => st.queue.extend(evs.into_iter().map(Ok)),
                    Err(e) => {
                        st.queue.push_back(Err(e));
                        st.resp = None;
                        st.finished = true;
                        break;
                    }
                }
                if st.decoder.is_done() {
                    st.resp = None;
                    break;
                }
            }
        }
    })
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sse_parser_handles_split_chunks_and_line_endings() {
        let mut p = SseParser::new();
        let mut evs = p.push(b"event: a\r\nda");
        assert!(evs.is_empty());
        evs.extend(p.push(b"ta: hel"));
        evs.extend(p.push(b"lo\r"));
        evs.extend(p.push(b"\n\r\n: comment\n\ndata: x\ndata: y\n\n"));
        evs.extend(p.push("data: ol\u{e1}".as_bytes().split_at(9).0));
        evs.extend(p.push(&"data: ol\u{e1}".as_bytes()[9..]));
        evs.extend(p.push(b"\n\n"));
        assert_eq!(
            evs,
            vec![
                SseEvent {
                    event: Some("a".into()),
                    data: "hello".into()
                },
                SseEvent {
                    event: None,
                    data: "x\ny".into()
                },
                SseEvent {
                    event: None,
                    data: "ol\u{e1}".into()
                },
            ]
        );
        assert!(p.push(b"data: tail").is_empty());
        assert_eq!(p.finish().unwrap().data, "tail");
    }

    #[test]
    fn sse_parser_long_line_in_tiny_chunks() {
        let payload = "x".repeat(200_000);
        let body = format!("data: {payload}\n\n");
        let mut p = SseParser::new();
        let mut evs = Vec::new();
        for c in body.as_bytes().chunks(7) {
            evs.extend(p.push(c));
        }
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data.len(), payload.len());
        // A `\r\n` split across chunks still counts as one line ending.
        let mut p = SseParser::new();
        let mut evs = p.push(b"data: a\r");
        evs.extend(p.push(b"\n\r\n"));
        assert_eq!(evs.len(), 1);
    }

    /// Found by the `sse_parser` fuzz target: an empty chunk between a `\r`
    /// and its `\n` made the `\n` a second line ending, which split one event
    /// in two.
    #[test]
    fn sse_parser_empty_chunk_keeps_pending_cr() {
        let body: &[u8] = b"data: a\r\ndata: b\n\n";
        let mut one_shot = SseParser::new();
        let expected = one_shot.push(body);
        assert_eq!(expected.len(), 1);
        assert_eq!(expected[0].data, "a\nb");
        let mut p = SseParser::new();
        let mut evs = p.push(b"data: a\r");
        evs.extend(p.push(b""));
        evs.extend(p.push(b"\ndata: b\n\n"));
        assert_eq!(evs, expected);
    }

    #[test]
    fn accumulator_assembles_deltas_and_prefers_complete_calls() {
        let mut acc = StreamAccumulator::new();
        for ev in [
            StreamEvent::text("Hel"),
            StreamEvent::text("lo"),
            StreamEvent::ToolCallDelta {
                index: 0,
                id: Some("c1".into()),
                name: Some("add".into()),
                arguments: "{\"a\":".into(),
            },
            StreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments: "1}".into(),
            },
            StreamEvent::ToolCallDelta {
                index: 1,
                id: Some("c2".into()),
                name: Some("neg".into()),
                arguments: "{\"x\":".into(),
            },
            StreamEvent::ToolCall {
                index: 1,
                call: ToolCall {
                    id: "c2".into(),
                    name: "neg".into(),
                    arguments: json!({"x": 5}),
                },
            },
            StreamEvent::Usage {
                usage: Usage::new(7, 0),
            },
            StreamEvent::Usage {
                usage: Usage::new(0, 3),
            },
            StreamEvent::Done {
                model: "m".into(),
                stop_reason: Some("tool_calls".into()),
            },
        ] {
            acc.push(&ev);
        }
        assert!(acc.is_done());
        let r = acc.finish();
        assert_eq!(r.message.content, "Hello");
        assert_eq!(r.message.tool_calls[0].arguments, json!({"a": 1}));
        assert_eq!(r.message.tool_calls[1].arguments, json!({"x": 5}));
        assert_eq!(r.usage.input_tokens, 7);
        assert_eq!(r.usage.output_tokens, 3);
        assert_eq!(r.model, "m");
    }

    #[tokio::test]
    async fn default_chat_stream_wraps_chat() {
        use crate::mock::ScriptedModel;
        struct Plain(ScriptedModel);
        #[async_trait::async_trait]
        impl ChatModel for Plain {
            fn name(&self) -> &str {
                "plain"
            }
            async fn chat(&self, r: ChatRequest) -> Result<ChatResponse> {
                self.0.chat(r).await
            }
        }
        let m = Plain(ScriptedModel::new([Message::assistant("one two")]));
        let s = m.chat_stream(ChatRequest::new(vec![])).await.unwrap();
        let events: Vec<_> = s.map(|e| e.unwrap()).collect().await;
        assert_eq!(events[0], StreamEvent::text("one two"));
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
    }
}
