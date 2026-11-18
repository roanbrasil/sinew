//! SSE parsing and the provider stream decoders.

use sinew_core::{
    __fuzz::{decode_stream, StreamDecoder},
    stream::{SseEvent, SseParser},
    StreamAccumulator, StreamEvent,
};

use crate::split_chunks;

fn parse(chunks: &[&[u8]]) -> Vec<SseEvent> {
    let mut p = SseParser::new();
    let mut out = Vec::new();
    for c in chunks {
        out.extend(p.push(c));
    }
    out.extend(p.finish());
    out
}

/// Chunked parsing never panics and yields exactly the events of one-shot
/// and byte-by-byte parsing; event data never contains a carriage return.
pub fn sse_parser(data: &[u8]) {
    let (chunks, body) = split_chunks(data);
    let one_shot = parse(&[body]);
    assert_eq!(parse(&chunks), one_shot, "chunked != one-shot");
    let bytes: Vec<&[u8]> = body.chunks(1).collect();
    assert_eq!(parse(&bytes), one_shot, "byte-by-byte != one-shot");
    for ev in &one_shot {
        assert!(!ev.data.contains('\r'), "CR in data: {ev:?}");
        assert!(
            ev.event
                .as_deref()
                .is_none_or(|e| !e.contains(['\r', '\n'])),
            "line break in event name: {ev:?}"
        );
    }
}

/// The decoder never panics, gives the same result however the body is
/// chunked, ends a successful stream with `Done`, and its events fold into
/// a response.
fn decoder(kind: StreamDecoder, data: &[u8]) {
    let (chunks, body) = split_chunks(data);
    let one_shot = decode_stream(kind, &[body]);
    let chunked = decode_stream(kind, &chunks);
    match (&one_shot, &chunked) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "chunked != one-shot"),
        (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
        _ => panic!("chunked and one-shot disagree: {one_shot:?} vs {chunked:?}"),
    }
    let Ok(events) = one_shot else {
        return;
    };
    assert!(
        matches!(events.last(), Some(StreamEvent::Done { .. })),
        "stream does not end with Done: {events:?}"
    );
    let mut acc = StreamAccumulator::new();
    for ev in &events {
        acc.push(ev);
        // Every event survives a serde round trip (they are sent over
        // sinew-serve's SSE endpoint).
        let json = serde_json::to_string(ev).expect("serialize event");
        let _: StreamEvent = serde_json::from_str(&json).expect("deserialize event");
    }
    assert!(acc.is_done());
    let resp = acc.finish();
    serde_json::to_string(&resp.message).expect("serialize message");
}

pub fn anthropic(data: &[u8]) {
    decoder(StreamDecoder::Anthropic, data);
}

pub fn openai(data: &[u8]) {
    decoder(StreamDecoder::OpenAi, data);
}

pub fn gemini(data: &[u8]) {
    decoder(StreamDecoder::Gemini, data);
}
