//! OTLP/HTTP trace ingest: the JSON mapping and the protobuf decoder.

use serde_json::Value;
use sinew_scope::otlp::otlp_to_runs;
use sinew_trace::otlp::proto::{decode_request, encode_request};

/// Maps a request body to runs and checks every run is storable.
fn check_runs(body: &Value) {
    let Ok(runs) = otlp_to_runs(body, None) else {
        return;
    };
    let with_header = otlp_to_runs(body, Some("p")).expect("same body, same result");
    assert_eq!(runs.len(), with_header.len());
    for ((_, run), (project, _)) in runs.iter().zip(&with_header) {
        assert_eq!(project, "p");
        assert!(!run.id.is_empty() && !run.trace_id.is_empty());
        if let Some(end) = run.end_ms {
            assert!(end >= run.start_ms, "run ends before it starts");
        }
        // Runs are stored as JSON and read back by the UI.
        let json = serde_json::to_string(run).expect("serialize run");
        let _: Value = serde_json::from_str(&json).expect("stored run parses again");
    }
}

/// OTLP/JSON: any JSON body maps to runs or a clean error.
pub fn json(data: &[u8]) {
    if let Ok(body) = serde_json::from_slice::<Value>(data) {
        check_runs(&body);
    }
}

/// OTLP protobuf: decoding never panics or overflows the stack; what it
/// decodes maps to runs and survives an encode / decode round trip.
pub fn proto(data: &[u8]) {
    let Ok(body) = decode_request(data) else {
        return;
    };
    check_runs(&body);
    let again = decode_request(&encode_request(&body)).expect("re-encoded request decodes");
    check_runs(&again);
}
