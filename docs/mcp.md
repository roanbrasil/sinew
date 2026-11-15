# Model Context Protocol (`sinew-mcp`)

`sinew-mcp` connects Sinew to the [Model Context Protocol](https://modelcontextprotocol.io).
It has three parts:

| Part | What it does |
|---|---|
| `McpClient` | Connects to an MCP server and turns its tools into Sinew `Tool`s, so a `react_agent` can call them like any local tool. |
| `McpServer` | Serves any Sinew `ToolBox`, and optionally an agent graph as one tool, to MCP clients such as Claude Code. |
| `sinew-mcp` binary | An MCP server that lets an assistant read your Sinew traces (`list_traces`, `get_trace`, `search_runs`). |

The crate speaks protocol version `2025-06-18` and also accepts `2025-03-26` and `2024-11-05`.
Stdio is always available. The Streamable HTTP client sits behind the `http` cargo feature.

## Using MCP tools in an agent

```rust
use std::sync::Arc;
use sinew_core::{providers::OpenAiCompat, Message};
use sinew_graph::prebuilt::{react_agent, AgentState};
use sinew_mcp::McpClient;

let client = McpClient::connect_stdio("my-mcp-server", ["--some-flag"]).await?;
let tools = client.toolbox().await?;          // every server tool as a Sinew Tool
let model = Arc::new(OpenAiCompat::ollama("qwen3:4b-instruct-2507-q4_K_M"));
let agent = react_agent(model, tools)?;
let out = agent.invoke(AgentState::new(vec![Message::user("...")])).await?;
client.close().await;                          // optional; dropping the last handle also stops the server
```

Other constructors:

- `McpClient::connect_command(tokio::process::Command)`: set env vars, the working directory or stderr handling yourself.
- `McpClient::connect(reader, writer)`: any pair of async byte streams, such as in-process pipes.
- `McpClient::connect_http(url)` (feature `http`): a Streamable HTTP endpoint. The client tracks the
  `Mcp-Session-Id` and `MCP-Protocol-Version` headers and reads JSON or SSE replies.

Lower-level calls: `server_info()`, `list_tools()` (follows pagination), `call_tool(name, args)`,
`ping()`, `request(method, params)`, `notify(method, params)` and `set_request_timeout(d)`. The
default timeout is 60 s. A request that times out is cancelled on the server with
`notifications/cancelled`.

### How results are mapped

| MCP result | Sinew value |
|---|---|
| `text` content | the text; several items are joined with newlines |
| `image` / `audio` | `[image image/png, 1234 bytes base64]` (a summary, never the raw base64) |
| `resource_link` | `[resource name: uri]` |
| embedded `resource` | its `text`, or `[resource uri, mime]` for blobs |
| unknown content type | the raw JSON |
| only `structuredContent`, no content | that JSON value |
| `isError: true` | `Err(Error::Tool)`, which `ToolBox::run` turns into an "error: tool ... failed: ..." message |
| JSON-RPC error (unknown tool, bad params, closed connection) | `Err(Error::Tool)` as well |

Because failures become tool messages instead of aborting the graph, the model sees the error and can
retry or explain it. In traces, the `tool` run is marked as failed.

## Serving Sinew tools over MCP

```rust
use sinew_mcp::McpServer;

McpServer::new("calculator", "0.1.0")
    .with_tools(my_toolbox)                                   // any ToolBox
    .with_agent("math_agent", "Answers arithmetic questions", react_agent(model, tools)?)
    .with_instructions("Use math_agent for multi-step arithmetic.")
    .serve_stdio()
    .await?;
```

- `with_agent` wraps a `CompiledGraph<AgentState>` as one tool that takes `{"input": "..."}` and returns
  the last assistant message. You can also use `AgentTool` directly, with `.with_system(prompt)`.
- A Sinew tool's string output becomes one text item. Other JSON is sent as text, and objects are also
  sent as `structuredContent`.
- A tool that returns `Err` produces a result with `isError: true` and the message as text.
- Unknown tools and invalid params are JSON-RPC errors with code `-32602`. Unknown methods get
  `-32601`, unparseable lines `-32700`, and invalid messages and batches `-32600`.
- Tool calls run concurrently and can be cancelled. Other requests are answered in order.
- When stdin closes, the server finishes the calls still in flight and exits.
- Only JSON-RPC goes to stdout. Write logs to stderr.

`serve(reader, writer)` serves any byte streams. `handle_message(json)` answers a single message, if
you want to plug in your own transport.

## The `sinew-mcp` trace server

```bash
cargo install --path crates/sinew-mcp          # or: cargo build --release -p sinew-mcp
sinew-mcp --traces /abs/path/to/traces.jsonl   # speaks MCP on stdio
```

It reads the JSONL file that `sinew_trace::JsonlSink` writes. The file is read again on every call, so
traces from a running app show up right away. The file does not have to exist yet.

| Tool | Arguments | Returns |
|---|---|---|
| `list_traces` | `limit` (default 20), `errors_only` | newest traces with start time (UTC), root run, duration, run count, tokens in→out, error count and first error |
| `get_trace` | `id` (full id or unique prefix), `io`, `io_chars` (default 300) | the run tree (graph → node → llm/tool) with timings, tokens and errors, optionally with truncated inputs and outputs |
| `search_runs` | `name` (substring, case-insensitive), `kind` (`graph`/`node`/`llm`/`tool`/`chain`), `error` (`true`, `false` or a substring of the message), `limit` (default 50) | matching runs across all traces, newest first |

An unknown or ambiguous id, a bad `kind`, or an unreadable file comes back as a tool error (`isError`)
that explains the problem.

### Registering it with Claude Code

Build the binary first, then register it with an absolute path to the trace file:

```bash
cargo build --release -p sinew-mcp

# Current project only (stored in your local Claude Code settings):
claude mcp add sinew-traces -- /path/to/sinew/target/release/sinew-mcp --traces /path/to/app/traces.jsonl

# Share it with the team through the project's .mcp.json:
claude mcp add --scope project sinew-traces -- sinew-mcp --traces ./traces.jsonl

# Check it:
claude mcp list
```

Everything after `--` is the command Claude Code spawns. For the project scope, `.mcp.json` ends up as:

```json
{
  "mcpServers": {
    "sinew-traces": {
      "command": "sinew-mcp",
      "args": ["--traces", "./traces.jsonl"]
    }
  }
}
```

Then ask things like "list my recent traces", "why did the last run fail?" or "which tool calls errored
today?". Claude will call `list_traces`, `get_trace` and `search_runs`. Remove the server with
`claude mcp remove sinew-traces`.

## Design: our own JSON-RPC layer instead of `rmcp`

The official Rust SDK, `rmcp`, implements the whole protocol: resources, prompts, sampling, elicitation
and OAuth. With it come proc macros, `schemars` and a large dependency tree. Sinew only needs the tool
surface: `initialize`, `tools/list`, `tools/call`, `ping` and cancellation. That is a few hundred lines
of JSON-RPC over tokio. Writing it ourselves means:

- no new dependencies: everything the crate uses is already in the workspace;
- an exact mapping to Sinew's `Tool` and `ToolBox`;
- a small surface to keep in sync with the spec.

Resources or prompts would be added in `protocol.rs`, `client.rs` and `server.rs`.

Known limits: the client does not offer sampling, roots or elicitation (it answers such server requests
with `-32601`). The HTTP client does not open the optional server-to-client GET stream. There is no HTTP
server transport yet.

## Examples and tests

```bash
cargo run -p sinew-mcp --example mcp_agent                    # offline, scripted model, tools over MCP
SINEW_MODEL=qwen3:4b-instruct-2507-q4_K_M \
  cargo run -p sinew-mcp --example mcp_agent -- --ollama      # local Ollama
cargo test -p sinew-mcp                                       # unit + stdio + in-process tests
cargo test -p sinew-mcp --features http                       # adds the Streamable HTTP test
```

`mcp_agent` runs both sides. It spawns itself with `--serve` as an MCP server, connects with `McpClient`,
and gives the remote tools to a `react_agent`.
