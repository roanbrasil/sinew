//! Structured output: JSON extraction from replies, schema validation,
//! list parsing and tool-call arguments assembled from partial JSON deltas.

use serde_json::{json, Value};
use sinew_core::{
    __fuzz::extract_json,
    parsers::ListFormat,
    structured::{validate, StructuredOutput},
    JsonOutputParser, ListOutputParser, Message, StreamAccumulator, StreamEvent,
};

use crate::split_chunks;

pub fn structured_json(data: &[u8]) {
    let (chunks, body) = split_chunks(data);
    let text = String::from_utf8_lossy(body);

    // The extracted JSON is a slice of the reply.
    if let Some(found) = extract_json(&text) {
        let start = found.as_ptr() as usize - text.as_ptr() as usize;
        assert!(start + found.len() <= text.len());
    }

    let parsed = JsonOutputParser::<Value>::new().parse(&text);
    if let Ok(v) = &parsed {
        // Whatever the parser accepts can be validated against any schema.
        let _ = validate(&json!({"type": "object", "required": ["a"]}), v, "$");
    }

    let so = StructuredOutput::new(
        "respond",
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
                "n": {"type": ["integer", "null"]}
            },
            "required": ["name"],
            "additionalProperties": false
        }),
    );
    let from_text: Result<Value, _> = so.parse_message(&Message::assistant(text.as_ref()));
    if let Ok(v) = from_text {
        assert!(v["name"].is_string(), "schema not enforced: {v}");
    }

    // A JSON document `[schema, value]` exercises the validator with an
    // arbitrary schema.
    if let Ok(Value::Array(pair)) = serde_json::from_str::<Value>(&text) {
        if let [schema, value] = pair.as_slice() {
            let _ = validate(schema, value, "$");
        }
    }

    for format in [ListFormat::Auto, ListFormat::Comma, ListFormat::Lines] {
        for item in ListOutputParser::new(format).parse(&text) {
            assert!(!item.is_empty());
            assert_eq!(item, item.trim(), "item not trimmed");
        }
    }

    // Partial JSON: the arguments of a streamed tool call are the same
    // however the argument string is split into deltas.
    let mut pieces: Vec<String> = chunks
        .iter()
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect();
    if pieces.concat() != text {
        // Lossy decoding of a chunk cut inside a character differs from
        // decoding the whole; fall back to char-boundary pieces.
        pieces = text.chars().map(String::from).collect();
    }
    if pieces.is_empty() {
        pieces.push(String::new());
    }
    let whole = assemble(std::iter::once(text.to_string()));
    let split = assemble(pieces.into_iter());
    assert_eq!(whole, split, "deltas assemble differently");
}

/// Feeds argument fragments as tool-call deltas and returns the call.
fn assemble(fragments: impl Iterator<Item = String>) -> Value {
    let mut acc = StreamAccumulator::new();
    for (i, arguments) in fragments.enumerate() {
        acc.push(&StreamEvent::ToolCallDelta {
            index: 0,
            id: (i == 0).then(|| "call_1".to_string()),
            name: (i == 0).then(|| "f".to_string()),
            arguments,
        });
    }
    let resp = acc.finish();
    let call = resp.message.tool_calls.first().expect("one tool call");
    assert_eq!(call.id, "call_1");
    call.arguments.clone()
}
