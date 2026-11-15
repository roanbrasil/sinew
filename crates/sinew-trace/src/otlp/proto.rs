//! OTLP protobuf codec for traces, hand-written so it needs no extra
//! dependencies.
//!
//! [`encode_request`] turns the JSON form of an `ExportTraceServiceRequest`
//! (as built by [`super::export_request`]) into protobuf bytes, and
//! [`decode_request`] turns protobuf bytes back into that JSON form (hex ids,
//! `intValue` as a string, enums as numbers), so one span mapping serves both
//! encodings. Unknown fields are skipped.

use serde_json::{json, Map, Value};

use super::{base64_decode, base64_encode};

/// Decoding error with a short description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtoError(pub String);

impl std::fmt::Display for ProtoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid OTLP protobuf: {}", self.0)
    }
}

impl std::error::Error for ProtoError {}

type Result<T> = std::result::Result<T, ProtoError>;

// ---------- encoding ----------

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn put_tag(out: &mut Vec<u8>, field: u32, wire: u8) {
    put_varint(out, (u64::from(field) << 3) | u64::from(wire));
}

fn put_bytes(out: &mut Vec<u8>, field: u32, data: &[u8]) {
    put_tag(out, field, 2);
    put_varint(out, data.len() as u64);
    out.extend_from_slice(data);
}

fn put_str(out: &mut Vec<u8>, field: u32, s: &str) {
    if !s.is_empty() {
        put_bytes(out, field, s.as_bytes());
    }
}

fn put_fixed64(out: &mut Vec<u8>, field: u32, v: u64) {
    put_tag(out, field, 1);
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_uint(out: &mut Vec<u8>, field: u32, v: u64) {
    if v != 0 {
        put_tag(out, field, 0);
        put_varint(out, v);
    }
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .filter_map(|i| u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok())
        .collect()
}

fn as_u64(v: &Value) -> u64 {
    match v {
        Value::String(s) => s.parse().unwrap_or(0),
        Value::Number(n) => n.as_u64().unwrap_or(0),
        _ => 0,
    }
}

fn enc_any(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(s) = v.get("stringValue") {
        put_bytes(&mut out, 1, s.as_str().unwrap_or("").as_bytes());
    } else if let Some(b) = v.get("boolValue") {
        put_tag(&mut out, 2, 0);
        put_varint(&mut out, u64::from(b.as_bool().unwrap_or(false)));
    } else if let Some(i) = v.get("intValue") {
        let n: i64 = match i {
            Value::String(s) => s.parse().unwrap_or(0),
            Value::Number(n) => n.as_i64().unwrap_or(0),
            _ => 0,
        };
        put_tag(&mut out, 3, 0);
        put_varint(&mut out, n as u64);
    } else if let Some(d) = v.get("doubleValue") {
        put_fixed64(&mut out, 4, d.as_f64().unwrap_or(0.0).to_bits());
    } else if let Some(a) = v.get("arrayValue") {
        let mut inner = Vec::new();
        for x in a["values"].as_array().into_iter().flatten() {
            put_bytes(&mut inner, 1, &enc_any(x));
        }
        put_bytes(&mut out, 5, &inner);
    } else if let Some(kvl) = v.get("kvlistValue") {
        put_bytes(&mut out, 6, &enc_kvs(&kvl["values"]));
    } else if let Some(b) = v.get("bytesValue") {
        put_bytes(
            &mut out,
            7,
            &base64_decode(b.as_str().unwrap_or("")).unwrap_or_default(),
        );
    }
    out
}

/// Encodes a list of KeyValue as repeated field 1 (attributes / values).
fn enc_kvs(list: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    enc_attrs_into(&mut out, 1, list);
    out
}

fn enc_attrs_into(out: &mut Vec<u8>, field: u32, list: &Value) {
    for a in list.as_array().into_iter().flatten() {
        let mut kv = Vec::new();
        put_str(&mut kv, 1, a["key"].as_str().unwrap_or(""));
        put_bytes(&mut kv, 2, &enc_any(&a["value"]));
        put_bytes(out, field, &kv);
    }
}

fn enc_span(s: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    put_bytes(
        &mut out,
        1,
        &hex_to_bytes(s["traceId"].as_str().unwrap_or("")),
    );
    put_bytes(
        &mut out,
        2,
        &hex_to_bytes(s["spanId"].as_str().unwrap_or("")),
    );
    put_str(&mut out, 3, s["traceState"].as_str().unwrap_or(""));
    if let Some(p) = s["parentSpanId"].as_str().filter(|p| !p.is_empty()) {
        put_bytes(&mut out, 4, &hex_to_bytes(p));
    }
    put_str(&mut out, 5, s["name"].as_str().unwrap_or(""));
    put_uint(&mut out, 6, as_u64(&s["kind"]));
    put_fixed64(&mut out, 7, as_u64(&s["startTimeUnixNano"]));
    put_fixed64(&mut out, 8, as_u64(&s["endTimeUnixNano"]));
    enc_attrs_into(&mut out, 9, &s["attributes"]);
    for e in s["events"].as_array().into_iter().flatten() {
        let mut ev = Vec::new();
        put_fixed64(&mut ev, 1, as_u64(&e["timeUnixNano"]));
        put_str(&mut ev, 2, e["name"].as_str().unwrap_or(""));
        enc_attrs_into(&mut ev, 3, &e["attributes"]);
        put_bytes(&mut out, 11, &ev);
    }
    if let Some(st) = s.get("status").filter(|v| v.is_object()) {
        let mut sb = Vec::new();
        put_str(&mut sb, 2, st["message"].as_str().unwrap_or(""));
        put_uint(&mut sb, 3, as_u64(&st["code"]));
        put_bytes(&mut out, 15, &sb);
    }
    out
}

/// Encodes the JSON form of an `ExportTraceServiceRequest` as protobuf.
pub fn encode_request(req: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    for rs in req["resourceSpans"].as_array().into_iter().flatten() {
        let mut rsb = Vec::new();
        if let Some(res) = rs.get("resource") {
            let mut rb = Vec::new();
            enc_attrs_into(&mut rb, 1, &res["attributes"]);
            put_bytes(&mut rsb, 1, &rb);
        }
        for ss in rs["scopeSpans"].as_array().into_iter().flatten() {
            let mut ssb = Vec::new();
            if let Some(scope) = ss.get("scope").filter(|v| v.is_object()) {
                let mut sc = Vec::new();
                put_str(&mut sc, 1, scope["name"].as_str().unwrap_or(""));
                put_str(&mut sc, 2, scope["version"].as_str().unwrap_or(""));
                put_bytes(&mut ssb, 1, &sc);
            }
            for span in ss["spans"].as_array().into_iter().flatten() {
                put_bytes(&mut ssb, 2, &enc_span(span));
            }
            put_str(&mut ssb, 3, ss["schemaUrl"].as_str().unwrap_or(""));
            put_bytes(&mut rsb, 2, &ssb);
        }
        put_str(&mut rsb, 3, rs["schemaUrl"].as_str().unwrap_or(""));
        put_bytes(&mut out, 1, &rsb);
    }
    out
}

// ---------- decoding ----------

enum Field<'a> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

fn varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *buf
            .get(*pos)
            .ok_or_else(|| ProtoError("truncated varint".into()))?;
        *pos += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b < 0x80 {
            return Ok(v);
        }
    }
    Err(ProtoError("varint too long".into()))
}

fn take<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(n)
        .filter(|e| *e <= buf.len())
        .ok_or_else(|| ProtoError("truncated field".into()))?;
    let s = &buf[*pos..end];
    *pos = end;
    Ok(s)
}

fn fields(buf: &[u8]) -> Result<Vec<(u32, Field<'_>)>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let key = varint(buf, &mut pos)?;
        let field = (key >> 3) as u32;
        let f = match key & 7 {
            0 => Field::Varint(varint(buf, &mut pos)?),
            1 => Field::Fixed64(u64::from_le_bytes(
                take(buf, &mut pos, 8)?.try_into().unwrap_or([0; 8]),
            )),
            2 => {
                let n = varint(buf, &mut pos)? as usize;
                Field::Bytes(take(buf, &mut pos, n)?)
            }
            5 => Field::Fixed32(u32::from_le_bytes(
                take(buf, &mut pos, 4)?.try_into().unwrap_or([0; 4]),
            )),
            w => return Err(ProtoError(format!("unsupported wire type {w}"))),
        };
        out.push((field, f));
    }
    Ok(out)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn utf8(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn dec_any(buf: &[u8]) -> Result<Value> {
    let mut out = json!({});
    for (n, f) in fields(buf)? {
        out = match (n, f) {
            (1, Field::Bytes(b)) => json!({ "stringValue": utf8(b) }),
            (2, Field::Varint(v)) => json!({ "boolValue": v != 0 }),
            (3, Field::Varint(v)) => json!({ "intValue": (v as i64).to_string() }),
            (4, Field::Fixed64(v)) => json!({ "doubleValue": f64::from_bits(v) }),
            (5, Field::Bytes(b)) => {
                let mut values = Vec::new();
                for (k, f) in fields(b)? {
                    if let (1, Field::Bytes(x)) = (k, f) {
                        values.push(dec_any(x)?);
                    }
                }
                json!({ "arrayValue": { "values": values } })
            }
            (6, Field::Bytes(b)) => json!({ "kvlistValue": { "values": dec_kvs(b, 1)? } }),
            (7, Field::Bytes(b)) => json!({ "bytesValue": base64_encode(b) }),
            _ => continue,
        };
    }
    Ok(out)
}

fn dec_kv(buf: &[u8]) -> Result<Value> {
    let (mut key, mut value) = (String::new(), json!({}));
    for (n, f) in fields(buf)? {
        match (n, f) {
            (1, Field::Bytes(b)) => key = utf8(b),
            (2, Field::Bytes(b)) => value = dec_any(b)?,
            _ => {}
        }
    }
    Ok(json!({ "key": key, "value": value }))
}

/// Decodes every KeyValue found in repeated field `field` of `buf`.
fn dec_kvs(buf: &[u8], field: u32) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    for (n, f) in fields(buf)? {
        if let (k, Field::Bytes(b)) = (n, f) {
            if k == field {
                out.push(dec_kv(b)?);
            }
        }
    }
    Ok(out)
}

fn dec_span(buf: &[u8]) -> Result<Value> {
    let mut s = Map::new();
    let mut attrs = Vec::new();
    let mut events = Vec::new();
    for (n, f) in fields(buf)? {
        match (n, f) {
            (1, Field::Bytes(b)) => {
                s.insert("traceId".into(), json!(hex(b)));
            }
            (2, Field::Bytes(b)) => {
                s.insert("spanId".into(), json!(hex(b)));
            }
            (3, Field::Bytes(b)) => {
                s.insert("traceState".into(), json!(utf8(b)));
            }
            (4, Field::Bytes(b)) if !b.is_empty() => {
                s.insert("parentSpanId".into(), json!(hex(b)));
            }
            (5, Field::Bytes(b)) => {
                s.insert("name".into(), json!(utf8(b)));
            }
            (6, Field::Varint(v)) => {
                s.insert("kind".into(), json!(v));
            }
            (7, Field::Fixed64(v)) => {
                s.insert("startTimeUnixNano".into(), json!(v.to_string()));
            }
            (8, Field::Fixed64(v)) => {
                s.insert("endTimeUnixNano".into(), json!(v.to_string()));
            }
            (9, Field::Bytes(b)) => attrs.push(dec_kv(b)?),
            (11, Field::Bytes(b)) => {
                let mut ev = Map::new();
                let mut ev_attrs = Vec::new();
                for (k, f) in fields(b)? {
                    match (k, f) {
                        (1, Field::Fixed64(v)) => {
                            ev.insert("timeUnixNano".into(), json!(v.to_string()));
                        }
                        (2, Field::Bytes(x)) => {
                            ev.insert("name".into(), json!(utf8(x)));
                        }
                        (3, Field::Bytes(x)) => ev_attrs.push(dec_kv(x)?),
                        _ => {}
                    }
                }
                ev.insert("attributes".into(), Value::Array(ev_attrs));
                events.push(Value::Object(ev));
            }
            (15, Field::Bytes(b)) => {
                let mut st = Map::new();
                for (k, f) in fields(b)? {
                    match (k, f) {
                        (2, Field::Bytes(x)) => {
                            st.insert("message".into(), json!(utf8(x)));
                        }
                        (3, Field::Varint(v)) => {
                            st.insert("code".into(), json!(v));
                        }
                        _ => {}
                    }
                }
                s.insert("status".into(), Value::Object(st));
            }
            (16, Field::Fixed32(v)) => {
                s.insert("flags".into(), json!(v));
            }
            _ => {}
        }
    }
    s.insert("attributes".into(), Value::Array(attrs));
    if !events.is_empty() {
        s.insert("events".into(), Value::Array(events));
    }
    Ok(Value::Object(s))
}

/// Decodes a protobuf `ExportTraceServiceRequest` into its JSON form.
pub fn decode_request(buf: &[u8]) -> Result<Value> {
    let mut resource_spans = Vec::new();
    for (n, f) in fields(buf)? {
        let (1, Field::Bytes(rs)) = (n, f) else {
            continue;
        };
        let mut resource = json!({ "attributes": [] });
        let mut scope_spans = Vec::new();
        for (k, f) in fields(rs)? {
            match (k, f) {
                (1, Field::Bytes(b)) => resource = json!({ "attributes": dec_kvs(b, 1)? }),
                (2, Field::Bytes(b)) => {
                    let mut scope = json!({});
                    let mut spans = Vec::new();
                    for (j, f) in fields(b)? {
                        match (j, f) {
                            (1, Field::Bytes(sc)) => {
                                let mut o = Map::new();
                                for (i, f) in fields(sc)? {
                                    match (i, f) {
                                        (1, Field::Bytes(x)) => {
                                            o.insert("name".into(), json!(utf8(x)));
                                        }
                                        (2, Field::Bytes(x)) => {
                                            o.insert("version".into(), json!(utf8(x)));
                                        }
                                        _ => {}
                                    }
                                }
                                scope = Value::Object(o);
                            }
                            (2, Field::Bytes(sp)) => spans.push(dec_span(sp)?),
                            _ => {}
                        }
                    }
                    scope_spans.push(json!({ "scope": scope, "spans": spans }));
                }
                _ => {}
            }
        }
        resource_spans.push(json!({ "resource": resource, "scopeSpans": scope_spans }));
    }
    Ok(json!({ "resourceSpans": resource_spans }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{otlp::export_request, Run, RunKind};

    #[test]
    fn round_trips_through_protobuf() {
        let run = Run {
            id: "4b1c1f5e-1d1a-4a7e-9b2e-0f8f2d6c1a11".into(),
            trace_id: "9f0a6c2e-2b3d-4c5e-8f70-112233445566".into(),
            parent_id: Some("p".into()),
            name: "multiply".into(),
            kind: RunKind::Tool,
            start_ms: 1_700_000_000_000,
            end_ms: Some(1_700_000_000_005),
            inputs: json!({"a": 2, "b": 3.5, "flag": true, "list": [1, "x"]}),
            outputs: Some(json!(7)),
            error: Some("boom".into()),
            input_tokens: None,
            output_tokens: None,
            metadata: json!({"n": -3, "f": 0.25, "nested": {"k": "v"}}),
        };
        let req = export_request(&[run], "svc", &Map::new(), true);
        let bytes = encode_request(&req);
        let back = decode_request(&bytes).unwrap();
        let a = &req["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        let b = &back["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        for k in [
            "traceId",
            "spanId",
            "parentSpanId",
            "name",
            "kind",
            "startTimeUnixNano",
            "endTimeUnixNano",
            "attributes",
            "status",
        ] {
            assert_eq!(a[k], b[k], "field {k}");
        }
        assert_eq!(
            req["resourceSpans"][0]["resource"],
            back["resourceSpans"][0]["resource"]
        );
        assert_eq!(
            back["resourceSpans"][0]["scopeSpans"][0]["scope"]["name"],
            "sinew-trace"
        );
    }

    #[test]
    fn any_values_round_trip() {
        for v in [
            json!({"stringValue": "héllo"}),
            json!({"boolValue": true}),
            json!({"intValue": "-42"}),
            json!({"doubleValue": 1.5}),
            json!({"arrayValue": {"values": [{"intValue": "1"}, {"stringValue": "a"}]}}),
            json!({"kvlistValue": {"values": [{"key": "k", "value": {"boolValue": false}}]}}),
            json!({"bytesValue": "AAEC"}),
        ] {
            assert_eq!(dec_any(&enc_any(&v)).unwrap(), v);
        }
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode_request(&[0x0a, 0xff]).is_err());
        assert!(decode_request(&[0x0f]).is_err());
        assert_eq!(decode_request(&[]).unwrap(), json!({"resourceSpans": []}));
    }
}
