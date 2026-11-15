//! OTLP/HTTP JSON trace ingest: maps OpenTelemetry spans to Sinew [`Run`]s.
//!
//! Mapping, following the OpenTelemetry GenAI semantic conventions:
//!
//! | Run field | Source |
//! |---|---|
//! | `id`, `trace_id`, `parent_id` | `spanId`, `traceId`, `parentSpanId` (hex) |
//! | `start_ms`, `end_ms` | `startTimeUnixNano`, `endTimeUnixNano` / 10^6 |
//! | `kind` | `gen_ai.operation.name`: `chat`, `text_completion`, `generate_content`, `embeddings` → `llm`; `execute_tool` (or a `gen_ai.tool.name`) → `tool`; a model plus token usage → `llm`; `invoke_agent` → `graph`; anything else → `chain`. A `sinew.run.kind` attribute (spans from `OtlpSink`) wins |
//! | `name` | llm: `gen_ai.response.model` / `gen_ai.request.model`; tool: `gen_ai.tool.name`; else the span name |
//! | `input_tokens`, `output_tokens` | `gen_ai.usage.input_tokens` / `output_tokens` (legacy `prompt_tokens` / `completion_tokens`) |
//! | `inputs` | `gen_ai.input.messages`, `gen_ai.prompt`, `gen_ai.tool.call.arguments`, or `input.value` |
//! | `outputs` | `gen_ai.output.messages`, `gen_ai.completion`, `gen_ai.tool.call.result`, or `output.value` |
//! | `error` | span status `ERROR` (code 2): its message, else `error.type`, else `"error"` |
//! | `metadata` | every span attribute, plus `resource`, `scope`, `span_kind`, `model`; `sinew.metadata.<key>` attributes are restored as top-level keys |
//!
//! The project comes from the `X-Sinew-Project` header, else the resource
//! attribute `sinew.project`, else `service.name`, else `default`.

use serde_json::{json, Map, Value};
use sinew_trace::{Run, RunKind};

/// Converts one OTLP AnyValue (`{"stringValue": ...}` etc.) to plain JSON.
pub fn any_value(v: &Value) -> Value {
    let Some(obj) = v.as_object() else {
        return Value::Null;
    };
    if let Some(s) = obj.get("stringValue") {
        return s.clone();
    }
    if let Some(b) = obj.get("boolValue") {
        return b.clone();
    }
    if let Some(i) = obj.get("intValue") {
        // int64 is a JSON string in the OTLP/JSON encoding, but accept numbers.
        return match i {
            Value::String(s) => s.parse::<i64>().map(Value::from).unwrap_or(i.clone()),
            other => other.clone(),
        };
    }
    if let Some(d) = obj.get("doubleValue") {
        return match d {
            Value::String(s) => s.parse::<f64>().map(Value::from).unwrap_or(d.clone()),
            other => other.clone(),
        };
    }
    if let Some(a) = obj.get("arrayValue") {
        return Value::Array(
            a["values"]
                .as_array()
                .map(|vs| vs.iter().map(any_value).collect())
                .unwrap_or_default(),
        );
    }
    if let Some(kv) = obj.get("kvlistValue") {
        return Value::Object(attributes(&kv["values"]));
    }
    if let Some(b) = obj.get("bytesValue") {
        return b.clone();
    }
    Value::Null
}

/// Converts an OTLP `attributes` array (`[{key, value}]`) to a JSON object.
pub fn attributes(list: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    for kv in list.as_array().into_iter().flatten() {
        if let Some(k) = kv["key"].as_str() {
            out.insert(k.to_string(), any_value(&kv["value"]));
        }
    }
    out
}

fn nanos_to_ms(v: &Value) -> Option<u64> {
    let n: u128 = match v {
        Value::String(s) => s.parse().ok()?,
        Value::Number(n) => n.as_u64()? as u128,
        _ => return None,
    };
    if n == 0 {
        None
    } else {
        Some((n / 1_000_000) as u64)
    }
}

fn as_u32(v: Option<&Value>) -> Option<u32> {
    match v? {
        Value::Number(n) => n.as_u64().map(|x| x.min(u32::MAX as u64) as u32),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn first_str<'a>(attrs: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|k| attrs.get(*k).and_then(Value::as_str))
        .filter(|s| !s.is_empty())
}

/// Attribute values that carry JSON as a string (message lists) are parsed.
fn first_payload(attrs: &Map<String, Value>, keys: &[&str]) -> Option<Value> {
    keys.iter().find_map(|k| {
        attrs.get(*k).map(|v| match v {
            Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| v.clone()),
            other => other.clone(),
        })
    })
}

fn span_kind_name(v: &Value) -> &'static str {
    let code = match v {
        Value::Number(n) => n.as_i64().unwrap_or(0),
        Value::String(s) => match s.as_str() {
            "SPAN_KIND_INTERNAL" => 1,
            "SPAN_KIND_SERVER" => 2,
            "SPAN_KIND_CLIENT" => 3,
            "SPAN_KIND_PRODUCER" => 4,
            "SPAN_KIND_CONSUMER" => 5,
            _ => 0,
        },
        _ => 0,
    };
    match code {
        1 => "internal",
        2 => "server",
        3 => "client",
        4 => "producer",
        5 => "consumer",
        _ => "unspecified",
    }
}

fn is_error_status(status: &Value) -> bool {
    match &status["code"] {
        Value::Number(n) => n.as_i64() == Some(2),
        Value::String(s) => s == "STATUS_CODE_ERROR" || s == "2",
        _ => false,
    }
}

/// Maps a single OTLP span (with its resource and scope context) to a run.
pub fn span_to_run(span: &Value, resource: &Map<String, Value>, scope: &Value) -> Option<Run> {
    let id = span["spanId"]
        .as_str()
        .filter(|s| !s.is_empty())?
        .to_string();
    let trace_id = span["traceId"]
        .as_str()
        .filter(|s| !s.is_empty())?
        .to_string();
    let parent_id = span["parentSpanId"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let attrs = attributes(&span["attributes"]);
    let span_name = span["name"].as_str().unwrap_or("span").to_string();

    let op = first_str(&attrs, &["gen_ai.operation.name"]).unwrap_or("");
    let tool_name = first_str(&attrs, &["gen_ai.tool.name"]);
    let model = first_str(&attrs, &["gen_ai.response.model", "gen_ai.request.model"]);
    let kind = match op {
        "chat" | "text_completion" | "generate_content" | "embeddings" | "completion" => {
            RunKind::Llm
        }
        "execute_tool" => RunKind::Tool,
        _ if tool_name.is_some() => RunKind::Tool,
        _ if model.is_some()
            && (attrs.contains_key("gen_ai.usage.input_tokens")
                || attrs.contains_key("gen_ai.usage.prompt_tokens")) =>
        {
            RunKind::Llm
        }
        _ if op == "invoke_agent" || op == "create_agent" => RunKind::Graph,
        _ => RunKind::Chain,
    };
    // Spans exported by sinew-trace's OtlpSink carry the original kind/name.
    let kind = first_str(&attrs, &["sinew.run.kind"])
        .and_then(|k| serde_json::from_value(json!(k)).ok())
        .unwrap_or(kind);
    let name = match first_str(&attrs, &["sinew.run.name"]) {
        Some(n) => n.to_string(),
        None => match kind {
            RunKind::Llm => model.map(str::to_string).unwrap_or(span_name.clone()),
            RunKind::Tool => tool_name.map(str::to_string).unwrap_or(span_name.clone()),
            RunKind::Graph => first_str(&attrs, &["gen_ai.agent.name"])
                .map(str::to_string)
                .unwrap_or(span_name.clone()),
            _ => span_name.clone(),
        },
    };

    let input_tokens = as_u32(
        attrs
            .get("gen_ai.usage.input_tokens")
            .or_else(|| attrs.get("gen_ai.usage.prompt_tokens")),
    );
    let output_tokens = as_u32(
        attrs
            .get("gen_ai.usage.output_tokens")
            .or_else(|| attrs.get("gen_ai.usage.completion_tokens")),
    );

    let mut inputs = first_payload(
        &attrs,
        &[
            "gen_ai.input.messages",
            "gen_ai.prompt",
            "gen_ai.tool.call.arguments",
            "input.value",
        ],
    );
    let mut outputs = first_payload(
        &attrs,
        &[
            "gen_ai.output.messages",
            "gen_ai.completion",
            "gen_ai.tool.call.result",
            "output.value",
        ],
    );
    // Older instrumentations put messages in span events.
    for ev in span["events"].as_array().into_iter().flatten() {
        let ev_name = ev["name"].as_str().unwrap_or("");
        let ev_attrs = Value::Object(attributes(&ev["attributes"]));
        if ev_name == "gen_ai.choice" && outputs.is_none() {
            outputs = Some(ev_attrs);
        } else if ev_name.starts_with("gen_ai.") && ev_name.ends_with(".message") {
            let entry = json!({ "event": ev_name, "attributes": ev_attrs });
            match &mut inputs {
                Some(Value::Array(a)) => a.push(entry),
                None => inputs = Some(Value::Array(vec![entry])),
                _ => {}
            }
        }
    }

    let status = &span["status"];
    let error = is_error_status(status).then(|| {
        status["message"]
            .as_str()
            .filter(|m| !m.is_empty())
            .or_else(|| first_str(&attrs, &["error.type"]))
            .unwrap_or("error")
            .to_string()
    });

    let mut metadata = Map::new();
    metadata.insert("otel".into(), json!(true));
    metadata.insert("span_name".into(), json!(span_name));
    metadata.insert("span_kind".into(), json!(span_kind_name(&span["kind"])));
    if let Some(m) = model {
        metadata.insert("model".into(), json!(m));
    }
    if let Some(s) = scope["name"].as_str() {
        metadata.insert("scope".into(), json!(s));
    }
    // `sinew.metadata.<key>` attributes become top-level metadata again
    // (JSON-encoded arrays/objects are decoded).
    for (k, v) in &attrs {
        if let Some(key) = k.strip_prefix("sinew.metadata.") {
            let v = match v {
                Value::String(s) if s.starts_with('[') || s.starts_with('{') => {
                    serde_json::from_str(s).unwrap_or_else(|_| v.clone())
                }
                other => other.clone(),
            };
            metadata.insert(key.to_string(), v);
        }
    }
    metadata.insert("attributes".into(), Value::Object(attrs.clone()));
    metadata.insert("resource".into(), Value::Object(resource.clone()));

    let start_ms = nanos_to_ms(&span["startTimeUnixNano"]).unwrap_or(0);
    Some(Run {
        id,
        trace_id,
        parent_id,
        name,
        kind,
        start_ms,
        end_ms: nanos_to_ms(&span["endTimeUnixNano"]).map(|e| e.max(start_ms)),
        inputs: inputs.unwrap_or_else(|| json!({})),
        outputs,
        error,
        input_tokens,
        output_tokens,
        metadata: Value::Object(metadata),
    })
}

/// Maps an `ExportTraceServiceRequest` (JSON) to `(project, run)` pairs.
/// `project_override` (from the request header) wins over resource attributes.
pub fn otlp_to_runs(
    body: &Value,
    project_override: Option<&str>,
) -> Result<Vec<(String, Run)>, String> {
    let resource_spans = body
        .get("resourceSpans")
        .or_else(|| body.get("resource_spans"))
        .and_then(Value::as_array)
        .ok_or("expected an object with a `resourceSpans` array")?;
    let mut out = Vec::new();
    for rs in resource_spans {
        let resource = attributes(&rs["resource"]["attributes"]);
        let project = project_override
            .map(str::to_string)
            .or_else(|| {
                first_str(&resource, &["sinew.project", "service.name"]).map(str::to_string)
            })
            .unwrap_or_else(|| "default".into());
        let scope_spans = rs
            .get("scopeSpans")
            .or_else(|| rs.get("scope_spans"))
            .and_then(Value::as_array);
        for ss in scope_spans.into_iter().flatten() {
            for span in ss["spans"].as_array().into_iter().flatten() {
                if let Some(run) = span_to_run(span, &resource, &ss["scope"]) {
                    out.push((project.clone(), run));
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> Value {
        json!({"stringValue": v})
    }
    fn i(v: i64) -> Value {
        json!({"intValue": v.to_string()})
    }

    /// A typical GenAI agent export: agent span → chat span → tool span.
    fn sample() -> Value {
        json!({
          "resourceSpans": [{
            "resource": {"attributes": [
              {"key": "service.name", "value": s("support-bot")},
              {"key": "deployment.environment", "value": s("dev")}
            ]},
            "scopeSpans": [{
              "scope": {"name": "opentelemetry.instrumentation.openai", "version": "0.1"},
              "spans": [
                {
                  "traceId": "5b8efff798038103d269b633813fc60c",
                  "spanId": "aaaaaaaaaaaaaaaa",
                  "name": "invoke_agent support",
                  "kind": 1,
                  "startTimeUnixNano": "1700000000000000000",
                  "endTimeUnixNano":   "1700000001500000000",
                  "attributes": [
                    {"key": "gen_ai.operation.name", "value": s("invoke_agent")},
                    {"key": "gen_ai.agent.name", "value": s("support")}
                  ],
                  "status": {}
                },
                {
                  "traceId": "5b8efff798038103d269b633813fc60c",
                  "spanId": "bbbbbbbbbbbbbbbb",
                  "parentSpanId": "aaaaaaaaaaaaaaaa",
                  "name": "chat gpt-4o-mini",
                  "kind": "SPAN_KIND_CLIENT",
                  "startTimeUnixNano": "1700000000100000000",
                  "endTimeUnixNano":   "1700000000900000000",
                  "attributes": [
                    {"key": "gen_ai.operation.name", "value": s("chat")},
                    {"key": "gen_ai.system", "value": s("openai")},
                    {"key": "gen_ai.request.model", "value": s("gpt-4o-mini")},
                    {"key": "gen_ai.response.model", "value": s("gpt-4o-mini-2024-07-18")},
                    {"key": "gen_ai.usage.input_tokens", "value": i(120)},
                    {"key": "gen_ai.usage.output_tokens", "value": {"intValue": 42}},
                    {"key": "gen_ai.request.temperature", "value": {"doubleValue": 0.2}},
                    {"key": "gen_ai.input.messages", "value": s("[{\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"content\":\"Where is my order?\"}]}]")},
                    {"key": "gen_ai.output.messages", "value": s("[{\"role\":\"assistant\",\"parts\":[{\"type\":\"text\",\"content\":\"Checking.\"}]}]")}
                  ]
                },
                {
                  "traceId": "5b8efff798038103d269b633813fc60c",
                  "spanId": "cccccccccccccccc",
                  "parentSpanId": "aaaaaaaaaaaaaaaa",
                  "name": "execute_tool lookup_order",
                  "startTimeUnixNano": "1700000001000000000",
                  "endTimeUnixNano":   "1700000001200000000",
                  "attributes": [
                    {"key": "gen_ai.operation.name", "value": s("execute_tool")},
                    {"key": "gen_ai.tool.name", "value": s("lookup_order")},
                    {"key": "gen_ai.tool.call.arguments", "value": s("{\"id\": 7}")},
                    {"key": "error.type", "value": s("TimeoutError")}
                  ],
                  "status": {"code": 2}
                }
              ]
            }]
          }]
        })
    }

    #[test]
    fn maps_genai_spans() {
        let runs = otlp_to_runs(&sample(), None).unwrap();
        assert_eq!(runs.len(), 3);
        assert!(runs.iter().all(|(p, _)| p == "support-bot"));

        let (_, agent) = &runs[0];
        assert_eq!(agent.kind, RunKind::Graph);
        assert_eq!(agent.name, "support");
        assert_eq!(agent.parent_id, None);
        assert_eq!(agent.start_ms, 1_700_000_000_000);
        assert_eq!(agent.duration_ms(), Some(1500));
        assert_eq!(agent.trace_id, "5b8efff798038103d269b633813fc60c");

        let (_, llm) = &runs[1];
        assert_eq!(llm.kind, RunKind::Llm);
        assert_eq!(llm.name, "gpt-4o-mini-2024-07-18");
        assert_eq!(llm.parent_id.as_deref(), Some("aaaaaaaaaaaaaaaa"));
        assert_eq!(llm.input_tokens, Some(120));
        assert_eq!(llm.output_tokens, Some(42));
        assert_eq!(llm.duration_ms(), Some(800));
        assert_eq!(llm.inputs[0]["role"], "user");
        assert_eq!(
            llm.outputs.as_ref().unwrap()[0]["parts"][0]["content"],
            "Checking."
        );
        assert_eq!(llm.metadata["model"], "gpt-4o-mini-2024-07-18");
        assert_eq!(llm.metadata["span_kind"], "client");
        assert_eq!(
            llm.metadata["attributes"]["gen_ai.request.temperature"],
            0.2
        );
        assert_eq!(llm.metadata["resource"]["deployment.environment"], "dev");
        assert_eq!(llm.error, None);

        let (_, tool) = &runs[2];
        assert_eq!(tool.kind, RunKind::Tool);
        assert_eq!(tool.name, "lookup_order");
        assert_eq!(tool.inputs["id"], 7);
        assert_eq!(tool.error.as_deref(), Some("TimeoutError"));
    }

    #[test]
    fn header_project_wins_and_legacy_attributes_work() {
        let body = json!({"resourceSpans": [{
            "resource": {"attributes": [{"key": "sinew.project", "value": s("from-resource")}]},
            "scopeSpans": [{"spans": [{
                "traceId": "t1", "spanId": "s1", "name": "openai.chat",
                "startTimeUnixNano": 2_000_000_000u64, "endTimeUnixNano": 3_000_000_000u64,
                "attributes": [
                    {"key": "gen_ai.request.model", "value": s("llama3")},
                    {"key": "gen_ai.usage.prompt_tokens", "value": i(10)},
                    {"key": "gen_ai.usage.completion_tokens", "value": i(5)},
                    {"key": "gen_ai.prompt", "value": s("hello")}
                ],
                "events": [{"name": "gen_ai.choice", "attributes": [{"key": "message", "value": s("hi!")}]}],
                "status": {"code": "STATUS_CODE_ERROR", "message": "rate limited"}
            }]}]
        }]});
        let runs = otlp_to_runs(&body, None).unwrap();
        assert_eq!(runs[0].0, "from-resource");
        let r = &runs[0].1;
        // No operation name: an LLM is inferred from model + usage.
        assert_eq!(r.kind, RunKind::Llm);
        assert_eq!(r.name, "llama3");
        assert_eq!(r.input_tokens, Some(10));
        assert_eq!(r.output_tokens, Some(5));
        assert_eq!(r.inputs, json!("hello"));
        assert_eq!(r.outputs.as_ref().unwrap()["message"], "hi!");
        assert_eq!(r.error.as_deref(), Some("rate limited"));
        assert_eq!((r.start_ms, r.end_ms), (2000, Some(3000)));

        let runs = otlp_to_runs(&body, Some("hdr")).unwrap();
        assert_eq!(runs[0].0, "hdr");
    }

    #[test]
    fn rejects_non_otlp_and_skips_spans_without_ids() {
        assert!(otlp_to_runs(&json!({"foo": 1}), None).is_err());
        let body = json!({"resourceSpans": [{"scopeSpans": [{"spans": [{"name": "x"}]}]}]});
        assert!(otlp_to_runs(&body, None).unwrap().is_empty());
        assert!(otlp_to_runs(&json!({"resourceSpans": []}), None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn any_value_handles_nested_types() {
        let v = any_value(&json!({"kvlistValue": {"values": [
            {"key": "a", "value": {"arrayValue": {"values": [i(1), {"boolValue": true}]}}}
        ]}}));
        assert_eq!(v, json!({"a": [1, true]}));
    }
}
