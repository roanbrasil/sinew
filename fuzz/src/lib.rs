//! Checks shared by the cargo-fuzz targets in `fuzz_targets/` and by the
//! stable replay test in `tests/replay.rs`.
//!
//! Every target is one function `fn(&[u8])` that panics when an invariant is
//! broken. [`TARGETS`] lists them by name, so the replay test can run the
//! committed seeds and regression inputs through the same code on stable.

pub mod json_rpc;
pub mod loaders;
pub mod otlp;
pub mod redaction;
pub mod streams;
pub mod structured;
pub mod tools;

/// A fuzz target: its name (the `fuzz_targets/<name>.rs` binary) and check.
pub struct Target {
    pub name: &'static str,
    pub run: fn(&[u8]),
}

/// Every fuzz target.
pub const TARGETS: &[Target] = &[
    Target {
        name: "sse_parser",
        run: streams::sse_parser,
    },
    Target {
        name: "stream_anthropic",
        run: streams::anthropic,
    },
    Target {
        name: "stream_openai",
        run: streams::openai,
    },
    Target {
        name: "stream_gemini",
        run: streams::gemini,
    },
    Target {
        name: "structured_json",
        run: structured::structured_json,
    },
    Target {
        name: "json_rpc_mcp",
        run: json_rpc::mcp,
    },
    Target {
        name: "json_rpc_a2a",
        run: json_rpc::a2a,
    },
    Target {
        name: "loader_html",
        run: loaders::html,
    },
    Target {
        name: "loader_csv",
        run: loaders::csv,
    },
    Target {
        name: "loader_json",
        run: loaders::json,
    },
    Target {
        name: "loader_markdown",
        run: loaders::markdown,
    },
    Target {
        name: "otlp_json",
        run: otlp::json,
    },
    Target {
        name: "otlp_proto",
        run: otlp::proto,
    },
    Target {
        name: "redactor",
        run: redaction::redactor,
    },
    Target {
        name: "guardrails",
        run: redaction::guardrails,
    },
    Target {
        name: "tool_args",
        run: tools::tool_args,
    },
];

/// Splits fuzz input into network-style chunks and the body they cover.
///
/// The first byte `n` (taken modulo 9) says how many chunk-length bytes
/// follow; the body is the rest, cut into chunks whose lengths cycle through
/// those bytes. A length of 0 produces an empty chunk (real transports can
/// deliver those). With no length bytes the body is a single chunk.
pub fn split_chunks(data: &[u8]) -> (Vec<&[u8]>, &[u8]) {
    let Some((&first, rest)) = data.split_first() else {
        return (Vec::new(), &[]);
    };
    let n = (first as usize % 9).min(rest.len());
    let (lengths, body) = rest.split_at(n);
    if lengths.is_empty() || lengths.iter().all(|&l| l == 0) {
        let mut chunks = vec![body];
        if !lengths.is_empty() {
            chunks.insert(0, &body[..0]);
        }
        return (chunks, body);
    }
    let mut chunks = Vec::new();
    let mut pos = 0;
    for &len in lengths.iter().cycle() {
        if pos >= body.len() {
            break;
        }
        let end = (pos + len as usize).min(body.len());
        chunks.push(&body[pos..end]);
        pos = end;
    }
    (chunks, body)
}

/// A single-threaded Tokio runtime for the async targets, created once per
/// fuzzing thread.
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    thread_local! {
        static RT: tokio::runtime::Runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
    }
    RT.with(|rt| rt.block_on(f))
}

/// Asserts `v` is a well-formed JSON-RPC 2.0 response object: `jsonrpc`
/// is `"2.0"`, `id` is a string, number or null, and exactly one of
/// `result` and `error` is present; an error has an integer `code` and a
/// string `message`.
pub fn assert_json_rpc_response(v: &serde_json::Value) {
    let obj = v
        .as_object()
        .unwrap_or_else(|| panic!("response is not an object: {v}"));
    assert_eq!(obj.get("jsonrpc"), Some(&"2.0".into()), "bad jsonrpc: {v}");
    let id = obj.get("id").unwrap_or_else(|| panic!("no id: {v}"));
    assert!(
        id.is_string() || id.is_number() || id.is_null(),
        "bad id: {v}"
    );
    match (obj.get("result"), obj.get("error")) {
        (Some(_), None) => {}
        (None, Some(e)) => {
            assert!(e["code"].is_i64(), "error without integer code: {v}");
            assert!(e["message"].is_string(), "error without message: {v}");
        }
        _ => panic!("response needs exactly one of result and error: {v}"),
    }
}
