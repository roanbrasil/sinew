# Model Context Protocol (`sinew-mcp`)

`sinew-mcp` connects Sinew to the [Model Context Protocol](https://modelcontextprotocol.io).
It has three parts:

| Part | What it does |
|---|---|
| `McpClient` | Connects to an MCP server and turns its tools into Sinew `Tool`s, so a `react_agent` can call them like any local tool. Reads resources, subscribes to their changes, and loads prompts as `Message`s or `PromptTemplate`s. |
| `McpServer` | Serves any Sinew `ToolBox`, optionally an agent graph as one tool, and resources, resource templates and prompts, to MCP clients such as Claude Code. |
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

### Vetting server tools (tool poisoning, rug pulls)

A server's tool descriptions go to the model as instructions. Vet them before registering
(`sinew_core::tool_vetting`, OWASP ASI04):

```rust
use sinew_core::tool_vetting::ToolVetter;

let vetter = ToolVetter::recommended()        // description guardrails + pins
    .allow_server("crm-notes")                // tools from other servers are dropped
    .lock_file("tools.lock.json");            // fingerprints per server, kept across runs
let report = client.vetted_toolbox(&vetter).await?;
let agent = react_agent(model, report.toolbox())?;
```

`McpTool::source()` reports the server's name and version from the `initialize` handshake; pins are keyed
by that name, and traced tool runs carry it as `metadata.tool_source`. A tool whose description (or a
parameter description) fails a guardrail is dropped, and a tool whose name, description or schema changed
since it was pinned is refused unless `approve_changes` accepts it. With `VetAction::Fail`,
`vetted_toolbox` returns `McpError::Vetting`. The harness does the same with `AgentBuilder::vet_tools`.
See [guardrails](guardrails.md#tool-descriptions).

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

## Resources and prompts

Besides tools, an MCP server can offer **resources** (data a client reads by URI, such as files, records
or documentation) and **prompts** (named message templates with arguments, which clients usually show as
slash commands). `McpServer` serves both; `McpClient` reads them.

### Serving resources

```rust
use sinew_mcp::{McpResource, McpResourceTemplate, McpServer, ResourceContents, ResourceRequest, RpcError};

let server = McpServer::new("docs", "1.0")
    // Fixed text.
    .with_text_resource(
        McpResource::new("docs://readme", "readme").with_mime_type("text/markdown"),
        "# My service",
    )
    // Computed on every read; text or binary (`ResourceContents::blob_bytes`).
    .with_resource(McpResource::new("docs://status", "status"), |r: ResourceRequest| async move {
        Ok(vec![ResourceContents::text(r.uri, "all systems normal")])
    })
    // A URI template: any `docs://pages/<slug>` URI calls the handler with `params["slug"]`.
    .with_resource_template(
        McpResourceTemplate::new("docs://pages/{slug}", "page").with_mime_type("text/plain"),
        |r: ResourceRequest| async move {
            match load_page(&r.params["slug"]).await {
                Some(text) => Ok(vec![ResourceContents::text(r.uri, text)]),
                None => Err(RpcError::resource_not_found(&r.uri)),
            }
        },
    );
```

| Method | Behavior |
|---|---|
| `resources/list` | Static resources. Paginated with `nextCursor` when `with_page_size(n)` is set; otherwise one page |
| `resources/templates/list` | Resource templates, paginated the same way |
| `resources/read` | Static resources match by exact URI first, then templates in registration order. An unknown URI is error `-32002` with `data: {"uri": ...}`; a handler can return the same with `RpcError::resource_not_found`, or `RpcError::internal` (`-32603`) |
| `resources/subscribe` / `unsubscribe` | Per connection (stdio) or per session (HTTP). Unknown URIs are `-32002` |

URI templates follow RFC 6570 for the forms resource templates use: `{name}` matches one value without
`/`, `?` or `#`; `{+name}` and `{#name}` match anything, including `/`. Matched values are
percent-decoded. `UriTemplate` is public if you need the matcher elsewhere.

### Serving prompts

```rust
use sinew_core::PromptTemplate;
use sinew_mcp::{GetPromptResult, McpPrompt, PromptMessage, PromptRequest};

let server = server
    // Every template variable becomes a required argument; the result is one user message.
    .with_prompt_template(
        "summarize",
        "Summarize a text",
        PromptTemplate::new("Summarize in a {style} style:\n{text}")?,
    )
    // Full control: optional arguments, several messages, assistant turns.
    .with_prompt(
        McpPrompt::new("review")
            .with_argument("code", "The code to review", true)
            .with_argument("focus", "What to focus on", false),
        |r: PromptRequest| async move {
            let focus = r.arguments.get("focus").map_or("bugs", String::as_str).to_string();
            Ok(GetPromptResult::new([
                PromptMessage::user(format!("Review for {focus}:\n{}", r.arguments["code"])),
                PromptMessage::assistant("I will review it."),
            ]))
        },
    );
```

The server checks arguments before the handler runs: an unknown prompt, a non-string argument value or a
missing required argument is error `-32602` naming the problem.

The `resources` capability (`{"subscribe": true, "listChanged": true}`) and the `prompts` capability
(`{"listChanged": true}`) are advertised in `initialize` only when at least one resource (or template) or
prompt is registered at that moment. Servers without them answer the methods with `-32601`, as before.

### Changing resources and prompts at runtime

Clones of an `McpServer` share their resources, prompts and connected clients (tools are copied). Keep a
clone before calling `serve_*`:

```rust
let handle = server.clone();
tokio::spawn(server.serve_stdio());

handle.add_resource(McpResource::new("docs://new", "new"), handler);  // sends resources/list_changed
handle.remove_prompt("review");                                       // sends prompts/list_changed
handle.notify_resource_updated("docs://status");                      // to subscribers of that URI
```

`add_resource`, `add_resource_template` and `add_prompt` replace an entry with the same URI, template or
name. `notify_resource_updated`, `notify_resource_list_changed` and `notify_prompt_list_changed` return how
many clients received the notification.

### Reading resources and prompts from a client

```rust
let resources = client.list_resources().await?;            // follows pagination
let templates = client.list_resource_templates().await?;
let contents = client.read_resource("docs://readme").await?; // Vec<ResourceContents> (text or base64 blob)

let prompts = client.list_prompts().await?;
let result = client.get_prompt("review", json!({"code": "fn f() {}"})).await?;   // GetPromptResult
let messages = client.get_prompt_messages("review", json!({"code": "..."})).await?; // Vec<Message>
let template = client.prompt_template("summarize").await?;         // sinew_core::PromptTemplate
let chat = client.chat_prompt_template("review").await?;           // sinew_core::ChatPromptTemplate
```

Errors from the server arrive as `McpError::Rpc` with the code (`-32002` for an unknown resource,
`-32602` for an unknown prompt or a missing argument).

`get_prompt_messages` maps user and assistant messages to Sinew `Message`s; an image in a user message
becomes an image part, and other non-text content becomes text as in the tool result table above.

`prompt_template` and `chat_prompt_template` turn a server prompt into a local template: they fetch the
prompt once with a placeholder for each argument, escape literal braces, and put `{argument}` where the
placeholders came back. This works when the server inserts argument values verbatim, which most prompts
do. A server that validates or transforms arguments (enumerations, numbers, case changes) can reject or
alter the placeholders; call `get_prompt` with real values for those.

With the `rag` feature (enabled by the `sinew` umbrella's `rag` feature when `mcp` is on),
`client.read_resource_documents(uri)` returns `sinew_rag::Document`s, one per text item, with the URI as
id and `source` metadata and the MIME type as `mime_type` metadata. Blob items are skipped.
`ResourceContents::to_document()` converts a single item.

### Subscriptions and notifications

```rust
let mut rx = client.notifications().await?;    // tokio broadcast receiver
client.subscribe_resource("docs://status").await?;
while let Ok(n) = rx.recv().await {
    match n {
        McpNotification::ResourceUpdated { uri } => { let fresh = client.read_resource(&uri).await?; }
        McpNotification::ResourceListChanged | McpNotification::PromptListChanged => { /* list again */ }
        _ => {}
    }
}
```

Over stdio the notifications arrive on the same pipe. Over Streamable HTTP, `notifications()` and
`subscribe_resource` open the server-to-client `GET` stream first (once per client) and return after the
server accepted it, so a notification sent afterwards is delivered. A server that answers that `GET` with
`405` (stateless servers, or servers without the stream) simply delivers no notifications. Each
`notifications()` call returns an independent receiver; a receiver that falls more than 256 messages
behind gets `RecvError::Lagged`.

## MCP in a microservice (Streamable HTTP)

With the `http-server` feature (umbrella: `mcp-http`), `McpServer::http_router` returns an axum
`Router` serving the MCP Streamable HTTP transport, so a service can expose its operations as MCP
tools next to its REST routes. Tools are usually async (they hit a database or another API), and
`TypedTool` (feature `schemars` on `sinew-core`) derives the JSON Schema from the argument struct:

```rust
use schemars::JsonSchema;
use serde::Deserialize;
use sinew_core::{ToolBox, TypedTool};
use sinew_mcp::{McpHttpConfig, McpServer};

#[derive(Deserialize, JsonSchema)]
struct GetOrder {
    /// Order id, e.g. "A-1001".
    id: String,
}

let tools = ToolBox::new().with(TypedTool::new("get_order", "Look up an order", |a: GetOrder| async move {
    Ok(serde_json::json!({ "id": a.id, "status": "paid" })) // query your database here
}));

let mcp = McpServer::new("orders", "1.0.0")
    .with_tools(tools)
    .http_router(McpHttpConfig::default().with_bearer_token(std::env::var("MCP_TOKEN")?));

let app = axum::Router::new()
    .route("/orders", axum::routing::get(list_orders))
    .merge(mcp); // POST/DELETE /mcp and GET /health
axum::serve(tokio::net::TcpListener::bind("0.0.0.0:8080").await?, app).await?;
```

| Behavior | Details |
|---|---|
| `POST /mcp` | One JSON-RPC message (batches from older clients are accepted). Requests get `application/json`; notifications and responses get `202` |
| Sessions | `initialize` returns `Mcp-Session-Id`; later requests without it get `400`, with an unknown or expired one `404`. Idle TTL (default 1 h) and a cap on open sessions |
| `stateless()` | No sessions, so any replica behind a load balancer can answer any request |
| `GET /mcp` | With `Mcp-Session-Id` and `Accept: text/event-stream`: the session's SSE stream for server notifications (resource updates for the session's subscriptions, list changes), with keep-alive comments. One stream per session; a new `GET` replaces the previous one. Notifications sent while no stream is open are dropped. `400` without a session, `404` for an unknown one, `406` without `text/event-stream`, `405` in stateless mode |
| `DELETE /mcp` | Ends the session, its subscriptions and its stream |
| Auth | `with_bearer_token` (repeatable, for rotation; constant-time compare, `401` + `WWW-Authenticate`), or OAuth 2.1 with `with_oauth` ([below](#oauth-21-public-mcp-servers)) |
| Origin | `with_allowed_origins([...])` rejects browser requests from other origins (`403`), against DNS rebinding |
| Version | An unsupported `MCP-Protocol-Version` header gets `400` |
| `GET /health` | Unauthenticated, for probes |
| Path | `with_path("/tools/mcp")` |

Other services, or agents, connect with the HTTP client:

```rust
let client = sinew_mcp::McpClient::connect_http_with_token("http://orders:8080/mcp", token).await?;
let agent = react_agent(model, client.toolbox().await?)?; // remote tools, local agent
```

`connect_http_with_headers` adds any other header (API keys, tenant ids, trace context). Claude Code
can use the same service: `claude mcp add --transport http orders http://127.0.0.1:8080/mcp --header "Authorization: Bearer <token>"`.
See [sample 22](../samples/22-mcp-microservice/src/main.rs).

## OAuth 2.1 (public MCP servers)

Static bearer tokens are fine between services inside a cluster. An MCP server that users reach from
their own clients needs the [MCP authorization specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization):
the server is an OAuth 2.1 resource server, an identity provider (Keycloak, Auth0, Entra ID, Okta, ...)
issues the tokens, and clients discover everything from the server's `401`. The `oauth-server` and
`oauth-client` features (both: `oauth`; umbrella: `mcp-oauth`) implement the two sides.

### Server: `McpHttpConfig::with_oauth`

```rust
use sinew_mcp::{oauth::OAuthConfig, McpHttpConfig, McpServer};

let oauth = OAuthConfig::new(
    "https://mcp.example.com/mcp",               // resource: the URL clients use; tokens need it in `aud`
    "https://auth.example.com/realms/mcp",       // issuer: tokens need it in `iss`
)
.with_scopes_supported(["mcp:tools", "mcp:admin"])
.with_required_scopes(["mcp:tools"])             // every request
.with_tool_scopes("delete_order", ["mcp:admin"]); // tools/call of this tool only

let mcp = McpServer::new("orders", "1.0.0")
    .with_tools(tools)
    .http_router(McpHttpConfig::default().with_oauth(oauth).with_allowed_origins(["https://app.example.com"]));
```

| Behavior | Details |
|---|---|
| Metadata | `GET /.well-known/oauth-protected-resource/mcp` and `GET /.well-known/oauth-protected-resource` serve the RFC 9728 document: `resource`, `authorization_servers` (default: the issuer), `scopes_supported`, `bearer_methods_supported: ["header"]`. Unauthenticated |
| No token | `401` with `WWW-Authenticate: Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource/mcp"` |
| Invalid token | `401` with `error="invalid_token"` and a short `error_description` (`token expired`, `wrong audience`, ...). The token itself is never echoed or logged |
| Missing scope | `403` with `error="insufficient_scope"` and `scope="..."` listing what the request needs. Required scopes apply to every request; tool scopes to `tools/call` of that tool (also inside a batch). Listing tools is not restricted |
| Keys unavailable | `503` when the JWKS cannot be fetched and nothing is cached |
| Static tokens | `with_bearer_token` is ignored on an OAuth endpoint: the specification forbids accepting tokens not issued for the resource |

Token validation (`oauth::JwtValidator`, usable on its own for other routes):

- Signature with a key from the issuer's JWKS, selected by `kid`. RS256/384/512, PS256/384/512, ES256/384 and
  EdDSA are accepted by default (`with_algorithms` narrows the list). `alg: none` and HMAC algorithms are always
  refused, and the key type, curve and the JWK's `alg` must match the header.
- `iss` equal to the issuer; `aud` (string or array) containing the resource URL, or an extra value from
  `with_audience` for providers that cannot put the resource URL there.
- `exp` required and in the future, `nbf` (when present) in the past, with 60 seconds of leeway
  (`with_leeway`). Date claims that are not numbers are rejected rather than skipped.
- The JWKS comes from `with_jwks_uri`, or from the `jwks_uri` in the issuer's metadata (RFC 8414
  `/.well-known/oauth-authorization-server`, then OpenID Connect discovery). It is cached for an hour
  (`with_jwks_cache_ttl`) and fetched again when a token names an unknown `kid`, at most once per 30 seconds
  (`with_jwks_refresh_interval`), so key rotation needs no restart and forged `kid`s cannot flood the provider.
- Issuer and JWKS URLs must be HTTPS except on loopback hosts; `allow_insecure_http()` lifts that for a
  development network.

Validated claims are available as `AccessClaims` (`subject`, `client_id`, `scopes`, `expires_at`, raw `claims`)
from `JwtValidator::validate`; tool handlers do not receive them yet.

### Client: authorization code + PKCE

```rust
use std::sync::Arc;
use sinew_mcp::{oauth::{InMemoryTokenStore, OAuthClient, OAuthClientConfig}, McpClient};

let url = "https://mcp.example.com/mcp";
let oauth = OAuthClient::discover(
    url,
    OAuthClientConfig::new("http://127.0.0.1:8765/callback") // HTTPS, or a loopback http URL
        .with_client_name("orders agent")
        .with_store(Arc::new(InMemoryTokenStore::new())),  // or your keychain-backed TokenStore
)
.await?;

let (authorize_url, state) = oauth.begin()?;   // open authorize_url in the user's browser
// ... your callback handler receives ?code=...&state=...
oauth.complete(&code, &returned_state).await?; // or oauth.complete_redirect(&full_callback_url)

let client = McpClient::connect_http_with_oauth(url, oauth.clone()).await?;
```

`discover` follows the specification's order: an unauthenticated request to the MCP server, the
`resource_metadata` URL from its `401` (or the RFC 9728 well-known URL), a check that the metadata's
`resource` covers the server URL, the authorization server's metadata (RFC 8414, then OpenID Connect discovery;
its `issuer` must match), and the client registration: `with_client_id` (plus `with_client_secret` for a
confidential client), else a registration kept in the `TokenStore`, else dynamic client registration
(RFC 7591) as a public client. Scopes default to the challenge's `scope`, then the resource's
`scopes_supported` (`with_scopes` overrides).

`begin` returns the authorization URL and its `state`; the PKCE verifier (S256, 256 random bits) stays inside the
client. The URL carries the `resource` parameter (RFC 8707) so the token is bound to this server, and the token
requests send it too. `complete` accepts each `state` once, within ten minutes, comparing in constant time.

`OAuthClient` is a `TokenProvider`. `connect_http_with_oauth` sends its token on every request, refreshes it when
it expires within 30 seconds (`with_refresh_skew`), and after a `401` refreshes once and retries the request.
Concurrent requests share one refresh. A rejected refresh token (`invalid_grant`) clears the stored tokens, so the
application starts a new authorization. Implement `TokenProvider` yourself for other grants (client credentials,
token exchange), and `TokenStore` to keep tokens and the registration outside process memory. `Debug` output of
`TokenSet`, `ClientRegistration` and `Pkce` redacts secrets.

### Example: Keycloak in docker compose

A local identity provider for development. This configuration is not exercised in CI; command names follow
Keycloak 26 and may differ in other versions.

```yaml
# compose.yaml
services:
  keycloak:
    image: quay.io/keycloak/keycloak:26.0
    command: start-dev
    environment:
      KC_BOOTSTRAP_ADMIN_USERNAME: admin
      KC_BOOTSTRAP_ADMIN_PASSWORD: admin
    ports: ["8080:8080"]
```

```bash
docker compose up -d
kc() { docker compose exec keycloak /opt/keycloak/bin/kcadm.sh "$@"; }
kc config credentials --server http://localhost:8080 --realm master --user admin --password admin
kc create realms -s realm=mcp -s enabled=true
# A scope whose tokens carry the MCP server URL in `aud` (Keycloak does not derive `aud` from RFC 8707 `resource`).
SCOPE=$(kc create client-scopes -r mcp -i -s name=mcp:tools -s protocol=openid-connect \
  -s 'attributes."include.in.token.scope"=true' | tr -d '\r')
kc create client-scopes/$SCOPE/protocol-mappers/models -r mcp -s name=mcp-audience \
  -s protocol=openid-connect -s protocolMapper=oidc-audience-mapper \
  -s 'config."included.custom.audience"=http://localhost:3000/mcp' -s 'config."access.token.claim"=true'
kc update realms/mcp/default-default-client-scopes/$SCOPE
kc create users -r mcp -s username=alice -s enabled=true
kc set-password -r mcp --username alice --new-password alice
```

```rust
// Server on http://localhost:3000
let oauth = OAuthConfig::new("http://localhost:3000/mcp", "http://localhost:8080/realms/mcp")
    .with_scopes_supported(["mcp:tools"])
    .with_required_scopes(["mcp:tools"]);
```

Keycloak publishes OpenID Connect discovery at `/realms/mcp/.well-known/openid-configuration`, including
`jwks_uri` and a `registration_endpoint`, so neither side needs more configuration. Anonymous dynamic
registration is subject to the realm's client registration policies (Trusted Hosts, Allowed Client Scopes): allow
your client's host there, or create a public client with PKCE (`S256`) and the loopback redirect URI and pass its id
with `OAuthClientConfig::with_client_id`. Hosts other than `localhost` need HTTPS, or `allow_insecure_http()` on
both sides for a private development network.

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
and OAuth. With it come proc macros, `schemars` and a large dependency tree. Sinew uses the server
features (tools, resources, prompts), `ping`, cancellation and server notifications. That is JSON-RPC
over tokio. Writing it ourselves means:

- no new dependencies: everything the crate uses is already in the workspace;
- an exact mapping to Sinew's `Tool`, `ToolBox`, `Message` and `PromptTemplate`;
- a small surface to keep in sync with the spec.

Message types live in `protocol.rs`; the server and client logic in `server.rs` and `client.rs`.

Known limits: the client does not offer sampling, roots or elicitation (it answers such server requests
with `-32601`). The server does not implement `completion/complete` (argument autocompletion) or
`logging/setLevel`. Over HTTP, server notifications go only to sessions with an open `GET` stream; there
is no event replay (`Last-Event-ID`) after a reconnect.

## Examples and tests

```bash
cargo run -p sinew-mcp --example mcp_agent                    # offline, scripted model, tools over MCP
SINEW_MODEL=qwen3:4b-instruct-2507-q4_K_M \
  cargo run -p sinew-mcp --example mcp_agent -- --ollama      # local Ollama
cargo run -p sample-11-mcp                                    # tools, a resource template and a prompt over stdio
cargo test -p sinew-mcp                                       # unit + stdio + in-process tests
cargo test -p sinew-mcp --all-features                        # adds the Streamable HTTP, OAuth and rag tests
```

`mcp_agent` runs both sides. It spawns itself with `--serve` as an MCP server, connects with `McpClient`,
and gives the remote tools to a `react_agent`. [Sample 11](../samples/11-mcp/src/main.rs) does the same
and then reads a resource and renders a prompt.
