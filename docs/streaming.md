# Streaming

Sinew streams at two levels:

1. **Model level** (`sinew-core`): `ChatModel::chat_stream` yields text deltas,
   tool-call deltas, complete tool calls, usage and a final `Done`.
2. **Graph level** (`sinew-graph`): `CompiledGraph::stream` yields per-node
   events, tokens and custom events emitted by nodes, and one terminal event.

Run `cargo run --example streaming` (offline) or
`cargo run --example streaming -- --ollama` to see both.

## Model streaming

```rust
use futures_util::StreamExt;
use sinew::core::StreamEvent;

let mut stream = model.chat_stream(ChatRequest::new(messages).with_tools(specs)).await?;
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta { text } => print!("{text}"),
        StreamEvent::ToolCallDelta { index, id, name, arguments } => { /* partial JSON */ }
        StreamEvent::ToolCall { index, call } => { /* complete, arguments parsed */ }
        StreamEvent::Usage { usage } => { /* once, near the end */ }
        StreamEvent::Done { model, stop_reason } => { /* always last */ }
    }
}
```

`chat_stream` returns `Result<ChatStream>`, where
`ChatStream = BoxStream<'static, Result<StreamEvent>>`. Errors that happen
before any output, such as an HTTP status error, come back from the call
itself. Errors in the middle of a stream arrive as an `Err` item.

### Event contract

| Event | When |
|---|---|
| `TextDelta { text }` | Each chunk of assistant text. |
| `ToolCallDelta { index, id, name, arguments }` | Each chunk of a tool call. `id` and `name` usually come only in the first chunk for an `index`, and `arguments` is a fragment of the JSON string. |
| `ToolCall { index, call }` | After a call's deltas, with the arguments parsed. If you don't need partial arguments, watch only for this event. |
| `Usage { usage }` | Once, with the totals for the response. |
| `Done { model, stop_reason }` | Last event of every successful stream. |

### Default implementation

`chat_stream` has a default implementation that calls `chat()` and replays
the response as one `TextDelta` plus complete `ToolCall`s. Every existing
`ChatModel` can therefore be streamed without changes. A model only overrides
`chat_stream` to get real incremental output.

### Providers

- **`OpenAiCompat`** (OpenAI, Ollama, vLLM, llama.cpp) sends `stream: true` and
  `stream_options.include_usage`. It decodes `choices[].delta.content` and
  `choices[].delta.tool_calls[]`: the deltas are keyed by `index`, and a chunk
  without `index` falls back to its position in the array. It also reads the
  trailing usage chunk and the `[DONE]` sentinel. If a server rejects
  `stream_options`, call `.with_stream_usage(false)`.
- **`Anthropic`** decodes the Messages streaming events: `message_start`
  (model, input tokens), `content_block_start/delta/stop` (`text_delta`,
  `input_json_delta`), `message_delta` (stop reason, output tokens),
  `message_stop`, `ping` and `error`.
- **`ScriptedModel`** streams its scripted text word by word. Each tool call
  comes as two argument deltas followed by the complete call. This lets
  offline tests and examples exercise the streaming path.

SSE parsing is done by pure, synchronous code with no I/O: `SseParser` takes
bytes in arbitrary chunks, and each provider has a decoder. They are
unit-tested with recorded event streams.

### Back to a `ChatResponse`

```rust
use sinew::core::{collect_stream, chat_streaming, StreamAccumulator};

let resp = collect_stream(model.chat_stream(req).await?).await?;           // drain
let resp = chat_streaming(&model, req, |ev| { /* observe */ }).await?;     // observe + drain
```

`StreamAccumulator` is the fold behind both functions. Use it directly when
you drive the stream yourself. If a `ToolCall` event and deltas share an
index, the complete call wins. Calls that only arrived as deltas are assembled
and parsed in `finish()`.

`sinew_trace::traced_chat_stream` is the streaming version of `traced_chat`.
It records an `llm` run with usage and a `streamed_chunks` count.

## Graph streaming

```rust
use futures_util::StreamExt;
use sinew::graph::GraphEvent;

let mut events = graph.stream("thread-1", Some(input));   // None resumes the thread
while let Some(ev) = events.next().await {
    match ev {
        GraphEvent::NodeStart { node, step, task } => {}
        GraphEvent::Token { node, delta, .. } => print!("{delta}"),
        GraphEvent::Custom { node, value, .. } => {}
        GraphEvent::NodeEnd { node, step, task, state } => {}
        GraphEvent::Interrupted { next, state } => {}
        GraphEvent::Completed { state } => {}
        GraphEvent::Error(e) => {}
    }
}
```

- Every stream ends with exactly one terminal event: `Completed`,
  `Interrupted` or `Error`. `GraphEvent::is_terminal()` checks for one.
- `step` is the superstep index. `task` is the index of the node execution
  within its superstep. Without fan-out, `task` is always 0. With fan-out, use
  `task` to tell parallel branches apart, including several `SendTo` runs of
  the same node.
- `NodeEnd.state` is that node's own output. After a fan-out this is the
  branch output before the reducer merges it. The merged state shows up in
  `Completed` and `Interrupted`.
- The stream itself drives the run, so nothing runs in the background.
  Dropping the stream cancels the run, including any parallel branches still
  in flight. Checkpoints already written stay, so you can resume the thread.
- `stream(thread_id, Some(input))` starts a run and `stream(thread_id, None)`
  resumes from the latest checkpoint, exactly like `run`. `invoke` and `run`
  behave as before and pay nothing for streaming.

### Emitting from nodes

Nodes registered with `add_node_with_context` receive a `NodeContext`:

```rust
g.add_node_with_context("work", |s: S, ctx: NodeContext| async move {
    ctx.emit_token("partial text");              // GraphEvent::Token
    ctx.emit(serde_json::json!({"progress": 0.5})); // GraphEvent::Custom
    Ok(s)
});
```

Both calls are no-ops unless the graph is being consumed with `stream`.
`ctx.is_streaming()` lets a node choose a streaming model call only when
someone is listening. Use `NodeContext::new(node, step, thread_id)` to build a
detached context and call a node function directly in a unit test.

### `react_agent`

When the agent graph is streamed, the `agent` node calls `chat_stream` and
forwards every text delta as `GraphEvent::Token`. It then accumulates the
stream, tool calls included, into the assistant message stored in the state.
With `invoke` or `run`, the node makes a plain `chat` call as before.
