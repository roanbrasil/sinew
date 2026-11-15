//! OpenTelemetry export: maps Sinew [`Run`]s to OTLP spans following the
//! OpenTelemetry GenAI semantic conventions (`gen_ai.*` attributes).
//!
//! The mapping functions are always available. With the `otlp` feature,
//! [`OtlpSink`] ships spans to any OTLP/HTTP collector (Jaeger, Grafana Tempo,
//! Langfuse, Arize Phoenix, an OpenTelemetry Collector, `sinew-scope`) in
//! background batches. With `otlp-proto`, [`proto`] encodes and decodes the
//! binary protobuf form (hand-written, no extra dependencies) and the sink can
//! send `application/x-protobuf`.
//!
//! | Run | Span |
//! |---|---|
//! | `trace_id` | `traceId`: the UUID's 32 hex digits, else a 128-bit hash |
//! | `id`, `parent_id` | `spanId`, `parentSpanId`: 64-bit fold of the UUID, else a hash |
//! | `llm` run | name `chat {model}`, kind `CLIENT`, `gen_ai.operation.name = chat`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.input.messages` / `gen_ai.output.messages` |
//! | `tool` run | name `execute_tool {tool}`, `gen_ai.operation.name = execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.arguments` / `result` |
//! | `graph` run | name `invoke_agent {name}`, `gen_ai.operation.name = invoke_agent`, `gen_ai.agent.name` |
//! | other runs | the run name, `input.value` / `output.value` (JSON) |
//! | `error` | status `ERROR` with the message, `error.type` |
//! | every run | `sinew.run.kind`, `sinew.run.name`, `sinew.run.id`, `sinew.metadata.<key>` |

use serde_json::{json, Map, Value};

use crate::{Run, RunKind};

#[cfg(feature = "otlp-proto")]
pub mod proto;

#[cfg(feature = "otlp")]
mod sink;
#[cfg(feature = "otlp")]
pub use sink::{OtlpProtocol, OtlpSink, OtlpSinkConfig};

/// OTLP span kinds (`SpanKind` enum values).
pub const SPAN_KIND_INTERNAL: i64 = 1;
pub const SPAN_KIND_CLIENT: i64 = 3;
/// OTLP status code for errors.
pub const STATUS_CODE_ERROR: i64 = 2;

fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn hex_digits(id: &str) -> Option<String> {
    let s: String = id.chars().filter(|c| *c != '-').collect();
    (!s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit())).then(|| s.to_ascii_lowercase())
}

fn nonzero(s: String, len: usize) -> String {
    if s.chars().all(|c| c == '0') {
        format!("{}1", "0".repeat(len - 1))
    } else {
        s
    }
}

/// A 32-hex-digit OTLP trace id for a Sinew trace id: a UUID keeps its
/// digits, anything else is hashed.
pub fn trace_id_hex(id: &str) -> String {
    let s = match hex_digits(id) {
        Some(h) if h.len() == 32 => h,
        _ => format!(
            "{:016x}{:016x}",
            fnv1a(id.as_bytes(), 0xcbf2_9ce4_8422_2325),
            fnv1a(id.as_bytes(), 0x6c62_272e_07bb_0142)
        ),
    };
    nonzero(s, 32)
}

/// A 16-hex-digit OTLP span id for a Sinew run id: 16 hex digits are kept,
/// a UUID is folded to 64 bits, anything else is hashed.
pub fn span_id_hex(id: &str) -> String {
    let s = match hex_digits(id) {
        Some(h) if h.len() == 16 => h,
        Some(h) if h.len() == 32 => {
            let hi = u64::from_str_radix(&h[..16], 16).unwrap_or(0);
            let lo = u64::from_str_radix(&h[16..], 16).unwrap_or(0);
            format!("{:016x}", hi ^ lo)
        }
        _ => format!("{:016x}", fnv1a(id.as_bytes(), 0xcbf2_9ce4_8422_2325)),
    };
    nonzero(s, 16)
}

/// Converts plain JSON to an OTLP `AnyValue`.
pub fn any_value(v: &Value) -> Value {
    match v {
        Value::Null => json!({ "stringValue": "" }),
        Value::Bool(b) => json!({ "boolValue": b }),
        Value::Number(n) => match n.as_i64() {
            Some(i) => json!({ "intValue": i.to_string() }),
            None => json!({ "doubleValue": n.as_f64().unwrap_or(0.0) }),
        },
        Value::String(s) => json!({ "stringValue": s }),
        Value::Array(a) => {
            json!({ "arrayValue": { "values": a.iter().map(any_value).collect::<Vec<_>>() } })
        }
        Value::Object(o) => {
            json!({ "kvlistValue": { "values": o.iter().map(|(k, v)| kv(k, v.clone())).collect::<Vec<_>>() } })
        }
    }
}

/// One OTLP `KeyValue`.
pub fn kv(key: &str, value: Value) -> Value {
    json!({ "key": key, "value": any_value(&value) })
}

fn json_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Converts Sinew/OpenAI-style chat messages (`{role, content, tool_calls,
/// tool_call_id}`) to the GenAI semantic-convention message schema
/// (`{role, parts: [...]}`). Returns `None` when `v` is not a message list.
pub fn genai_messages(v: &Value) -> Option<Value> {
    let list: Vec<&Value> = match v {
        Value::Array(a) => a.iter().collect(),
        Value::Object(o) if o.contains_key("role") => vec![v],
        Value::Object(o) => match o.get("messages") {
            Some(Value::Array(a)) => a.iter().collect(),
            _ => return None,
        },
        _ => return None,
    };
    if list.is_empty() || !list.iter().all(|m| m.get("role").is_some()) {
        return None;
    }
    let out = list
        .into_iter()
        .map(|m| {
            let role = m["role"].as_str().unwrap_or("user").to_string();
            let content = &m["content"];
            let mut parts = Vec::new();
            if role == "tool" {
                parts.push(json!({
                    "type": "tool_call_response",
                    "id": m.get("tool_call_id").cloned().unwrap_or(Value::Null),
                    "response": content,
                }));
            } else {
                match content {
                    Value::String(s) if !s.is_empty() => {
                        parts.push(json!({ "type": "text", "content": s }))
                    }
                    Value::String(_) | Value::Null => {}
                    other => parts.push(json!({ "type": "text", "content": json_string(other) })),
                }
                if let Some(Value::Array(calls)) = m.get("tool_calls") {
                    for c in calls {
                        parts.push(json!({
                            "type": "tool_call",
                            "id": c.get("id").cloned().unwrap_or(Value::Null),
                            "name": c.get("name").cloned().unwrap_or(Value::Null),
                            "arguments": c.get("arguments").cloned().unwrap_or(Value::Null),
                        }));
                    }
                }
            }
            json!({ "role": role, "parts": parts })
        })
        .collect();
    Some(Value::Array(out))
}

fn model_of(run: &Run) -> String {
    run.metadata
        .get("model")
        .and_then(Value::as_str)
        .filter(|m| !m.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| run.name.clone())
}

fn kind_name(kind: RunKind) -> &'static str {
    match kind {
        RunKind::Graph => "graph",
        RunKind::Node => "node",
        RunKind::Llm => "llm",
        RunKind::Tool => "tool",
        RunKind::Chain => "chain",
    }
}

/// Maps one run to an OTLP/JSON span. With `capture_content` false, inputs,
/// outputs and messages are left out (only names, timings, usage, errors).
pub fn run_to_span(run: &Run, capture_content: bool) -> Value {
    let mut attrs: Vec<Value> = Vec::new();
    let mut push = |k: &str, v: Value| attrs.push(kv(k, v));
    let (name, span_kind) = match run.kind {
        RunKind::Llm => {
            let model = model_of(run);
            push("gen_ai.operation.name", json!("chat"));
            push("gen_ai.request.model", json!(run.name));
            push("gen_ai.response.model", json!(model));
            if let Some(p) = run.metadata.get("provider").and_then(Value::as_str) {
                push("gen_ai.provider.name", json!(p));
            }
            if let Some(t) = run.input_tokens {
                push("gen_ai.usage.input_tokens", json!(t));
            }
            if let Some(t) = run.output_tokens {
                push("gen_ai.usage.output_tokens", json!(t));
            }
            if capture_content {
                if let Some(m) = genai_messages(&run.inputs) {
                    push("gen_ai.input.messages", json!(json_string(&m)));
                } else {
                    push("input.value", json!(json_string(&run.inputs)));
                }
                if let Some(out) = &run.outputs {
                    match genai_messages(out) {
                        Some(Value::Array(mut msgs)) => {
                            for m in &mut msgs {
                                let calls = m["parts"]
                                    .as_array()
                                    .is_some_and(|p| p.iter().any(|x| x["type"] == "tool_call"));
                                m["finish_reason"] =
                                    json!(if calls { "tool_call" } else { "stop" });
                            }
                            push("gen_ai.output.messages", json!(json_string(&json!(msgs))));
                        }
                        _ => push("output.value", json!(json_string(out))),
                    }
                }
            }
            (format!("chat {model}"), SPAN_KIND_CLIENT)
        }
        RunKind::Tool => {
            push("gen_ai.operation.name", json!("execute_tool"));
            push("gen_ai.tool.name", json!(run.name));
            if capture_content {
                push(
                    "gen_ai.tool.call.arguments",
                    json!(json_string(&run.inputs)),
                );
                if let Some(out) = &run.outputs {
                    push("gen_ai.tool.call.result", json!(json_string(out)));
                }
            }
            (format!("execute_tool {}", run.name), SPAN_KIND_INTERNAL)
        }
        RunKind::Graph => {
            push("gen_ai.operation.name", json!("invoke_agent"));
            push("gen_ai.agent.name", json!(run.name));
            if capture_content {
                push("input.value", json!(json_string(&run.inputs)));
                push("input.mime_type", json!("application/json"));
                if let Some(out) = &run.outputs {
                    push("output.value", json!(json_string(out)));
                    push("output.mime_type", json!("application/json"));
                }
            }
            (format!("invoke_agent {}", run.name), SPAN_KIND_INTERNAL)
        }
        RunKind::Node | RunKind::Chain => {
            if capture_content {
                push("input.value", json!(json_string(&run.inputs)));
                push("input.mime_type", json!("application/json"));
                if let Some(out) = &run.outputs {
                    push("output.value", json!(json_string(out)));
                    push("output.mime_type", json!("application/json"));
                }
            }
            (run.name.clone(), SPAN_KIND_INTERNAL)
        }
    };
    push("sinew.run.kind", json!(kind_name(run.kind)));
    push("sinew.run.name", json!(run.name));
    push("sinew.run.id", json!(run.id));
    if let Some(e) = &run.error {
        push("error.type", json!(e.split(':').next().unwrap_or(e).trim()));
    }
    if let Value::Object(meta) = &run.metadata {
        for (k, v) in meta {
            if k == "model" || k == "provider" {
                continue;
            }
            let v = match v {
                Value::String(_) | Value::Bool(_) | Value::Number(_) => v.clone(),
                Value::Null => continue,
                other => json!(json_string(other)),
            };
            push(&format!("sinew.metadata.{k}"), v);
        }
    }
    let start_ns = run.start_ms.saturating_mul(1_000_000);
    let end_ns = run.end_ms.unwrap_or(run.start_ms).saturating_mul(1_000_000);
    let mut span = json!({
        "traceId": trace_id_hex(&run.trace_id),
        "spanId": span_id_hex(&run.id),
        "name": name,
        "kind": span_kind,
        "startTimeUnixNano": start_ns.to_string(),
        "endTimeUnixNano": end_ns.to_string(),
        "attributes": attrs,
    });
    if let Some(p) = run.parent_id.as_deref().filter(|p| !p.is_empty()) {
        span["parentSpanId"] = json!(span_id_hex(p));
    }
    if let Some(e) = &run.error {
        span["status"] = json!({ "code": STATUS_CODE_ERROR, "message": e });
    }
    span
}

/// Builds an OTLP `ExportTraceServiceRequest` (JSON form) for a batch.
/// `resource` holds resource attributes (`service.name` is added when
/// missing).
pub fn export_request(
    runs: &[Run],
    service_name: &str,
    resource: &Map<String, Value>,
    capture_content: bool,
) -> Value {
    let mut res = vec![kv("service.name", json!(service_name))];
    res.push(kv("telemetry.sdk.name", json!("sinew")));
    res.push(kv("telemetry.sdk.language", json!("rust")));
    res.push(kv(
        "telemetry.sdk.version",
        json!(env!("CARGO_PKG_VERSION")),
    ));
    for (k, v) in resource {
        if k != "service.name" {
            res.push(kv(k, v.clone()));
        }
    }
    json!({
        "resourceSpans": [{
            "resource": { "attributes": res },
            "scopeSpans": [{
                "scope": { "name": "sinew-trace", "version": env!("CARGO_PKG_VERSION") },
                "spans": runs.iter().map(|r| run_to_span(r, capture_content)).collect::<Vec<_>>(),
            }],
        }],
    })
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding (used for HTTP basic auth and OTLP bytes).
pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Decodes standard or URL-safe base64; padding optional.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llm_run() -> Run {
        Run {
            id: "4b1c1f5e-1d1a-4a7e-9b2e-0f8f2d6c1a11".into(),
            trace_id: "9f0a6c2e-2b3d-4c5e-8f70-112233445566".into(),
            parent_id: Some("root-run".into()),
            name: "gpt-4o-mini".into(),
            kind: RunKind::Llm,
            start_ms: 1_700_000_000_000,
            end_ms: Some(1_700_000_000_250),
            inputs: json!({"messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": "42", "tool_call_id": "c1"}
            ], "tools": ["multiply"]}),
            outputs: Some(json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "c2", "name": "multiply", "arguments": {"a": 2, "b": 3}}
            ]})),
            error: None,
            input_tokens: Some(12),
            output_tokens: Some(7),
            metadata: json!({"model": "gpt-4o-mini-2024-07-18", "tags": ["prod"], "ttft_ms": 80}),
        }
    }

    fn attr<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
        span["attributes"]
            .as_array()?
            .iter()
            .find(|a| a["key"] == key)
            .map(|a| &a["value"])
    }

    #[test]
    fn ids_are_valid_hex() {
        let t = trace_id_hex("9f0a6c2e-2b3d-4c5e-8f70-112233445566");
        assert_eq!(t, "9f0a6c2e2b3d4c5e8f70112233445566");
        assert_eq!(trace_id_hex("not-a-uuid").len(), 32);
        assert_eq!(span_id_hex("root-run").len(), 16);
        assert_eq!(span_id_hex("00f067aa0ba902b7"), "00f067aa0ba902b7");
        assert_eq!(span_id_hex("0000000000000000"), "0000000000000001");
        assert_eq!(span_id_hex("x"), span_id_hex("x"));
        assert_ne!(span_id_hex("x"), span_id_hex("y"));
    }

    #[test]
    fn llm_span_follows_genai_conventions() {
        let span = run_to_span(&llm_run(), true);
        assert_eq!(span["name"], "chat gpt-4o-mini-2024-07-18");
        assert_eq!(span["kind"], SPAN_KIND_CLIENT);
        assert_eq!(span["parentSpanId"], span_id_hex("root-run"));
        assert_eq!(span["startTimeUnixNano"], "1700000000000000000");
        assert_eq!(
            attr(&span, "gen_ai.operation.name").unwrap()["stringValue"],
            "chat"
        );
        assert_eq!(
            attr(&span, "gen_ai.usage.input_tokens").unwrap()["intValue"],
            "12"
        );
        let input: Value = serde_json::from_str(
            attr(&span, "gen_ai.input.messages").unwrap()["stringValue"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(input[0]["parts"][0]["content"], "be brief");
        assert_eq!(input[2]["parts"][0]["type"], "tool_call_response");
        let output: Value = serde_json::from_str(
            attr(&span, "gen_ai.output.messages").unwrap()["stringValue"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(output[0]["finish_reason"], "tool_call");
        assert_eq!(output[0]["parts"][0]["name"], "multiply");
        assert_eq!(
            attr(&span, "sinew.metadata.ttft_ms").unwrap()["intValue"],
            "80"
        );
        assert!(attr(&span, "sinew.metadata.tags").is_some());

        let lean = run_to_span(&llm_run(), false);
        assert!(attr(&lean, "gen_ai.input.messages").is_none());
        assert!(attr(&lean, "gen_ai.usage.output_tokens").is_some());
    }

    #[test]
    fn tool_errors_set_status() {
        let mut r = llm_run();
        r.kind = RunKind::Tool;
        r.name = "multiply".into();
        r.error = Some("timeout: took too long".into());
        let span = run_to_span(&r, true);
        assert_eq!(span["name"], "execute_tool multiply");
        assert_eq!(span["status"]["code"], STATUS_CODE_ERROR);
        assert_eq!(attr(&span, "error.type").unwrap()["stringValue"], "timeout");
        let req = export_request(&[r], "svc", &Map::new(), true);
        let res = &req["resourceSpans"][0]["resource"]["attributes"][0];
        assert_eq!(res["value"]["stringValue"], "svc");
    }

    #[test]
    fn base64_round_trip() {
        for s in ["", "f", "fo", "foo", "foob", "fooba", "foobar"] {
            let e = base64_encode(s.as_bytes());
            assert_eq!(base64_decode(&e).unwrap(), s.as_bytes());
        }
        assert_eq!(base64_encode(b"pk:sk"), "cGs6c2s=");
    }
}
