//! JSON round trips against the examples of the A2A 0.3.0 specification.

use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};
use sinew_a2a::{
    jsonrpc::{decode_response, error_code, RpcError},
    types::*,
};

fn round_trip<T: Serialize + DeserializeOwned + std::fmt::Debug + PartialEq>(v: Value) -> T {
    let parsed: T = serde_json::from_value(v.clone()).expect("parses");
    let back = serde_json::to_value(&parsed).unwrap();
    assert_eq!(back, v, "re-serialized JSON differs");
    let again: T = serde_json::from_value(back).unwrap();
    assert_eq!(again, parsed);
    parsed
}

#[test]
fn sample_agent_card() {
    let card: AgentCard = round_trip(json!({
      "protocolVersion": "0.2.9",
      "name": "GeoSpatial Route Planner Agent",
      "description": "Provides advanced route planning.",
      "url": "https://georoute-agent.example.com/a2a/v1",
      "preferredTransport": "JSONRPC",
      "additionalInterfaces" : [
        {"url": "https://georoute-agent.example.com/a2a/v1", "transport": "JSONRPC"},
        {"url": "https://georoute-agent.example.com/a2a/grpc", "transport": "GRPC"},
        {"url": "https://georoute-agent.example.com/a2a/json", "transport": "HTTP+JSON"}
      ],
      "provider": {
        "organization": "Example Geo Services Inc.",
        "url": "https://www.examplegeoservices.com"
      },
      "iconUrl": "https://georoute-agent.example.com/icon.png",
      "version": "1.2.0",
      "documentationUrl": "https://docs.examplegeoservices.com/georoute-agent/api",
      "capabilities": {
        "streaming": true,
        "pushNotifications": true,
        "stateTransitionHistory": false
      },
      "securitySchemes": {
        "google": {
          "type": "openIdConnect",
          "openIdConnectUrl": "https://accounts.google.com/.well-known/openid-configuration"
        }
      },
      "security": [{ "google": ["openid", "profile", "email"] }],
      "defaultInputModes": ["application/json", "text/plain"],
      "defaultOutputModes": ["application/json", "image/png"],
      "skills": [
        {
          "id": "route-optimizer-traffic",
          "name": "Traffic-Aware Route Optimizer",
          "description": "Calculates the optimal driving route.",
          "tags": ["maps", "routing", "navigation", "directions", "traffic"],
          "examples": [
            "Plan a route from A to B avoiding tolls.",
            "{\"origin\": {\"lat\": 37.422, \"lng\": -122.084}}"
          ],
          "inputModes": ["application/json", "text/plain"],
          "outputModes": ["application/json", "application/vnd.geo+json", "text/html"]
        },
        {
          "id": "custom-map-generator",
          "name": "Personalized Map Generator",
          "description": "Creates custom map images.",
          "tags": ["maps", "customization", "visualization", "cartography"],
          "examples": ["Generate a map of my upcoming road trip."],
          "inputModes": ["application/json"],
          "outputModes": ["image/png", "image/jpeg", "application/json", "text/html"]
        }
      ],
      "supportsAuthenticatedExtendedCard": true,
      "signatures": [
        {
          "protected": "eyJhbGciOiJFUzI1NiJ9",
          "signature": "QFdkNLNszlGj3z3u0YQGt"
        }
      ]
    }));
    assert!(card.supports_streaming());
    assert!(matches!(
        card.security_schemes["google"],
        SecurityScheme::OpenIdConnect { .. }
    ));
    assert_eq!(card.skills.len(), 2);
}

#[test]
fn other_security_schemes() {
    let schemes: std::collections::BTreeMap<String, SecurityScheme> = round_trip(json!({
        "key": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
        "bearer": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
        "oauth": {"type": "oauth2", "flows": {"clientCredentials": {"tokenUrl": "https://t", "scopes": {}}}},
        "mtls": {"type": "mutualTLS", "description": "client certs"}
    }));
    assert_eq!(schemes.len(), 4);
}

#[test]
fn message_send_request_and_task_result() {
    let req = json!({
      "jsonrpc": "2.0",
      "id": 1,
      "method": "message/send",
      "params": {
        "message": {
          "role": "user",
          "parts": [{ "kind": "text", "text": "tell me a joke" }],
          "messageId": "9229e770-767c-417b-a0b0-f0741243c589"
        },
        "metadata": {}
      }
    });
    let mut params: MessageSendParams = serde_json::from_value(req["params"].clone()).unwrap();
    assert_eq!(params.message.text(), "tell me a joke");
    assert_eq!(params.message.role, Role::User);
    // We always write `kind`, the example omits it on the message.
    params.message.message_id = "m".into();
    let out = serde_json::to_value(&params).unwrap();
    assert_eq!(out["message"]["kind"], "message");

    let resp = json!({
      "jsonrpc": "2.0",
      "id": 1,
      "result": {
        "id": "363422be-b0f9-4692-a24d-278670e7c7f1",
        "contextId": "c295ea44-7543-4f78-b524-7a38915ad6e4",
        "status": { "state": "completed" },
        "artifacts": [{
          "artifactId": "9b6934dd-37e3-4eb1-8766-962efaab63a1",
          "name": "joke",
          "parts": [{ "kind": "text", "text": "Why did the chicken cross the road? To get to the other side!" }]
        }],
        "history": [{
          "role": "user",
          "parts": [{ "kind": "text", "text": "tell me a joke" }],
          "messageId": "9229e770-767c-417b-a0b0-f0741243c589",
          "taskId": "363422be-b0f9-4692-a24d-278670e7c7f1",
          "contextId": "c295ea44-7543-4f78-b524-7a38915ad6e4",
          "kind": "message"
        }],
        "kind": "task",
        "metadata": {}
      }
    });
    let result = decode_response(resp.to_string().as_bytes())
        .unwrap()
        .unwrap();
    let r: SendMessageResult = round_trip(result);
    let task = r.as_task().unwrap();
    assert_eq!(task.status.state, TaskState::Completed);
    assert_eq!(
        r.text(),
        "Why did the chicken cross the road? To get to the other side!"
    );
}

#[test]
fn direct_message_result() {
    let r: SendMessageResult = round_trip(json!({
        "role": "agent",
        "messageId": "363422be-b0f9-4692-a24d-278670e7c7f1",
        "contextId": "c295ea44-7543-4f78-b524-7a38915ad6e4",
        "parts": [{ "kind": "text", "text": "Why did the chicken cross the road?" }],
        "kind": "message",
        "metadata": {}
    }));
    assert!(matches!(r, SendMessageResult::Message(_)));
}

#[test]
fn streaming_events() {
    let events = [
        json!({
          "id": "225d6247-06ba-4cda-a08b-33ae35c8dcfa",
          "contextId": "05217e44-7e9f-473e-ab4f-2c2dde50a2b1",
          "status": { "state": "submitted", "timestamp": "2025-04-02T16:59:25.331844" },
          "history": [{
            "role": "user",
            "parts": [
              { "kind": "text", "text": "write a long paper describing the attached pictures" },
              { "kind": "file", "file": { "mimeType": "image/png", "bytes": "aGVsbG8=" } }
            ],
            "messageId": "bbb7dee1-cf5c-4683-8a6f-4114529da5eb",
            "taskId": "225d6247-06ba-4cda-a08b-33ae35c8dcfa",
            "contextId": "05217e44-7e9f-473e-ab4f-2c2dde50a2b1",
            "kind": "message"
          }],
          "kind": "task",
          "metadata": {}
        }),
        json!({
          "taskId": "225d6247-06ba-4cda-a08b-33ae35c8dcfa",
          "contextId": "05217e44-7e9f-473e-ab4f-2c2dde50a2b1",
          "artifact": {
            "artifactId": "9b6934dd-37e3-4eb1-8766-962efaab63a1",
            "parts": [{"kind":"text", "text": "<section 1...>"}]
          },
          "append": false,
          "lastChunk": false,
          "kind":"artifact-update"
        }),
        json!({
          "taskId": "225d6247-06ba-4cda-a08b-33ae35c8dcfa",
          "contextId": "05217e44-7e9f-473e-ab4f-2c2dde50a2b1",
          "artifact": {
            "artifactId": "9b6934dd-37e3-4eb1-8766-962efaab63a1",
            "parts": [{"kind":"text", "text": "<section 2...>"}]
          },
          "append": true,
          "lastChunk": true,
          "kind":"artifact-update"
        }),
        json!({
          "taskId": "225d6247-06ba-4cda-a08b-33ae35c8dcfa",
          "contextId": "05217e44-7e9f-473e-ab4f-2c2dde50a2b1",
          "status": { "state": "completed", "timestamp":"2025-04-02T16:59:35.331844" },
          "final": true,
          "kind":"status-update"
        }),
    ];
    let parsed: Vec<StreamResponse> = events.into_iter().map(round_trip).collect();
    assert!(matches!(parsed[0], StreamResponse::Task(_)));
    assert!(matches!(parsed[1], StreamResponse::ArtifactUpdate(_)));
    assert!(!parsed[2].is_final());
    assert!(parsed[3].is_final());
    assert_eq!(parsed[1].text().as_deref(), Some("<section 1...>"));
    let StreamResponse::Task(t) = &parsed[0] else {
        unreachable!()
    };
    assert_eq!(
        t.history[0].text(),
        "write a long paper describing the attached pictures\n[file image/png 8 bytes base64]"
    );
}

#[test]
fn input_required_task() {
    let t: Task = round_trip(json!({
      "id": "3f36680c-7f37-4a5f-945e-d78981fafd36",
      "contextId": "c295ea44-7543-4f78-b524-7a38915ad6e4",
      "status": {
        "state": "input-required",
        "message": {
          "role": "agent",
          "parts": [{ "kind": "text", "text": "Where would you like to fly to?" }],
          "messageId": "c2e1b2dd-f200-4b04-bc22-1b0c65a1aad2",
          "taskId": "3f36680c-7f37-4a5f-945e-d78981fafd36",
          "contextId": "c295ea44-7543-4f78-b524-7a38915ad6e4",
          "kind": "message"
        },
        "timestamp": "2024-03-15T10:10:00Z"
      },
      "history": [{
        "role": "user",
        "parts": [{ "kind": "text", "text": "I'd like to book a flight." }],
        "messageId": "c53ba666-3f97-433c-a87b-6084276babe2",
        "taskId": "3f36680c-7f37-4a5f-945e-d78981fafd36",
        "contextId": "c295ea44-7543-4f78-b524-7a38915ad6e4",
        "kind": "message"
      }],
      "kind": "task"
    }));
    assert!(t.status.state.is_interrupted());
    assert!(t.status.state.ends_turn());
    assert!(!t.status.state.is_terminal());
    assert_eq!(
        task_text(&t),
        "Where would you like to fly to?",
        "falls back to the status message"
    );
}

#[test]
fn data_part_and_file_uri() {
    let a: Artifact = round_trip(json!({
        "artifactId": "a1",
        "name": "FlightItinerary.json",
        "parts": [
            { "kind": "data", "data": { "confirmationId": "XYZ123", "from": "JFK" } },
            { "kind": "file", "file": { "name": "map.png", "mimeType": "image/png", "uri": "https://x/map.png" } }
        ]
    }));
    assert_eq!(
        parts_text(&a.parts),
        "{\"confirmationId\":\"XYZ123\",\"from\":\"JFK\"}\n[file map.png image/png https://x/map.png]"
    );
}

#[test]
fn task_states_use_kebab_case() {
    for (state, s) in [
        (TaskState::InputRequired, "input-required"),
        (TaskState::AuthRequired, "auth-required"),
        (TaskState::Canceled, "canceled"),
        (TaskState::Unknown, "unknown"),
    ] {
        assert_eq!(serde_json::to_value(state).unwrap(), json!(s));
    }
}

#[test]
fn error_responses() {
    let body = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "error": { "code": -32001, "message": "Task not found" }
    });
    let err = decode_response(body.to_string().as_bytes())
        .unwrap()
        .unwrap_err();
    assert_eq!(err.code, error_code::TASK_NOT_FOUND);
    assert_eq!(RpcError::task_not_cancelable("x").code, -32002);
    assert_eq!(RpcError::push_not_supported().code, -32003);
    assert_eq!(RpcError::unsupported("x").code, -32004);
    assert_eq!(RpcError::extended_card_not_configured().code, -32007);
}

#[test]
fn artifact_text_chunks_merge() {
    let mut a = Artifact::new("a", vec![Part::text("Hel")]);
    a.append_parts(vec![Part::text("lo"), Part::data(json!(1))]);
    a.append_parts(vec![Part::text("!")]);
    assert_eq!(a.parts.len(), 3);
    assert_eq!(a.parts[0].as_text(), Some("Hello"));
    assert_eq!(parts_text(&a.parts), "Hello\n{\"value\":1}\n!");
}

#[test]
fn timestamps_are_rfc3339() {
    let ts = now_rfc3339();
    assert_eq!(ts.len(), 24, "{ts}");
    assert!(ts.ends_with('Z') && ts.as_bytes()[10] == b'T', "{ts}");
    assert!(ts.starts_with("20"), "{ts}");
}
