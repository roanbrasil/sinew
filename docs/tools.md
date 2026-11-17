# Tools

A tool is something the model can ask your program to do: look up an order, run a query, call an API.
Sinew sends the tool's name, description and JSON Schema to the model, runs the call the model asks for,
and feeds the result back. This guide covers the four ways to define a tool, from the shortest to the
most flexible.

| Way | Use it when |
|---|---|
| [`#[sinew::tool]`](#the-sinewtool-attribute) | Almost always. A plain function becomes a tool; schema and description come from the code. |
| [`TypedTool`](#typedtool) | You build tools at runtime (names or descriptions from config) but still want typed arguments. |
| [`AsyncFnTool` / `FnTool`](#closures-with-a-hand-written-schema) | The schema is dynamic or comes from elsewhere (OpenAPI, MCP). |
| [`impl Tool`](#implementing-tool-by-hand) | Full control: custom spec per instance, streaming state, wrapping another system. |

## The `#[sinew::tool]` attribute

```rust
use sinew::prelude::*;

#[derive(serde::Serialize)]
struct Order { id: String, status: String, total: f64 }

/// Look up an order by id.
#[tool]
async fn get_order(
    /// Order id, e.g. "A-1001".
    id: String,
) -> Result<Order, String> {
    Ok(Order { id, status: "paid".into(), total: 54.0 })
}

let agent = react_agent(model, tools![get_order])?;
```

What the macro produces:

- **Name**: the function name (`get_order`). Override with `#[tool(name = "lookup")]`.
- **Description**: the function's doc comment. Override with `#[tool(description = "...")]`. A tool without
  either is a compile error, because the description is how the model decides when to call it.
- **Parameters**: a JSON Schema built from the argument types with `schemars`. Doc comments on arguments
  become parameter descriptions; `Option<T>` arguments are optional; enums become `enum` constraints;
  nested types are inlined (no `$ref`), which every provider accepts.
- **The tool**: a struct with the same name as the function. The function itself is unchanged and still
  callable, so you can unit-test it directly. Get the tool with `get_order::tool()`, or build a `ToolBox`
  with `tools![get_order, other_tool]`.

The macro is enabled by default in the `sinew` crate (feature `macros`). With `sinew-core` alone, enable
`features = ["macros"]`.

### Arguments

Arguments must be owned types that implement `serde::Deserialize` and `schemars::JsonSchema`: `String`,
numbers, `bool`, `Vec<T>`, `Option<T>`, `HashMap<String, T>`, or your own types with
`#[derive(Deserialize, JsonSchema)]`. Borrowed types (`&str`) are rejected at compile time with a hint.

`#[serde(...)]` and `#[schemars(...)]` attributes on an argument apply to that parameter:

```rust
/// Search the catalog.
#[tool]
fn search(
    query: String,
    /// Maximum number of results.
    #[serde(default = "ten")]
    limit: usize,
) -> Vec<String> { /* ... */ }
fn ten() -> usize { 10 }
```

Unknown argument names are rejected, so a model that invents a parameter gets an error it can correct
instead of a silent default.

### Return values and errors

Return any `T: Serialize`, or `Result<T, E>` where `E: Display` (`String`, `anyhow::Error`,
`sinew::core::Error`, your own error type). An `Err` becomes `Error::Tool` with the error's message. Inside
an agent, `ToolBox::run` turns that error into a tool message, so the model sees what went wrong and can
retry with different arguments.

### Shared state with `#[state]`

Real tools need a database pool, an HTTP client or a cache. Mark one argument `#[state]`: it is left out of
the schema (the model never sees it) and you pass it when building the tool. It is cloned on every call, so
use cheap-to-clone handles (`Arc<...>`, pools, clients):

```rust
type Db = std::sync::Arc<sqlx::PgPool>;

/// Change an order's status.
#[tool]
async fn update_order_status(
    #[state] db: Db,
    id: String,
    /// New status.
    status: Status,
) -> Result<Order, sqlx::Error> { /* ... */ }

let tools = ToolBox::new()
    .with(get_order::tool(db.clone()))
    .with(update_order_status::tool(db));
```

Group several pieces of state in one struct; only one `#[state]` argument is allowed.

### Compile-time checks

| Mistake | Error |
|---|---|
| No doc comment and no `description` | `#[tool] needs a description` |
| Generic function | `#[tool] functions cannot be generic` |
| `&str` / `&[T]` argument | `arguments ... must be owned` |
| Method with `self` | suggests `#[state]` |
| Destructuring pattern argument | `arguments must be plain names` |
| Two `#[state]` arguments | `supports one #[state] argument` |
| Unknown attribute | lists `name`, `description`, `crate` |

## `TypedTool`

Same typed arguments, built at runtime (feature `schemars`):

```rust
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Add { a: f64, b: f64 }

let add = TypedTool::new("add", "Add two numbers", |x: Add| async move { Ok(x.a + x.b) });
```

## Closures with a hand-written schema

`AsyncFnTool` (async) and `FnTool` (sync) take a name, a description, a JSON Schema value and a closure over
`serde_json::Value`. Use them when the schema is generated elsewhere, for example from an OpenAPI document.

## Implementing `Tool` by hand

```rust
#[async_trait::async_trait]
impl Tool for Inventory {
    fn spec(&self) -> ToolSpec { /* name, description, JSON Schema */ }
    async fn call(&self, args: serde_json::Value) -> sinew::core::Result<serde_json::Value> { /* ... */ }
}
```

## Built-in tools

Web search, web fetch, Wikipedia, a calculator, date/time, HTTP requests and e-mail are ready-made in
`sinew-tools`, one cargo feature per family: see [Built-in tools](builtin-tools.md).

## Using tools

- **Agents**: `react_agent(model, tools)`, the supervisor and swarm prebuilts, and the harness all take a
  `ToolBox`.
- **MCP**: `McpServer::new(..).with_tools(tools)` serves the same tools to Claude Code, Cursor or any MCP
  client, over stdio or HTTP ([MCP guide](mcp.md)).
- **A2A**: expose an agent that uses them as an Agent2Agent service ([A2A guide](a2a.md)).

Samples: [`03-tools-agent`](../samples/03-tools-agent) and
[`22-mcp-microservice`](../samples/22-mcp-microservice).
