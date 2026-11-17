# Adding a tool

There are two places a tool can go:

| Kind | Where | How |
|---|---|---|
| A tool for your agents, a sample or an integration crate | Your crate, or `samples/` | `#[sinew::tool]` on a function (preferred) |
| A built-in tool of the agent harness, opt-in on `AgentBuilder` | `crates/sinew-harness/src/tools/` | `impl Tool` by hand (the harness does not depend on the macro) |

Tools that wrap a third-party service are usually best published in your own crate on top of `sinew`; open a
**New integration** issue if you think one belongs in the workspace. Built-in harness tools are for general agent
work (files, shell, HTTP, planning) and need an issue first, because each one adds to every agent's prompt.

## The trait

From `crates/sinew-core/src/tool.rs`:

```rust
/// Tool description sent to the model. `parameters` is a JSON Schema object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call(&self, arguments: Value) -> Result<Value>;
}
```

`ToolBox::run` turns an `Err` into an `error: ...` tool message, so the model sees the failure and can retry.
Errors must therefore be readable by a model: say what was wrong and what a valid value looks like.

## Path A: `#[sinew::tool]`

```rust
use sinew::prelude::*;

/// Look up an order by id.
#[tool]
async fn get_order(
    /// Order id, e.g. "A-1001".
    id: String,
) -> Result<Order, String> {
    /* ... */
}

let tools = tools![get_order];            // or ToolBox::new().with(get_order::tool())
```

The name comes from the function, the description from the doc comment, the JSON Schema from the argument types
(argument doc comments become parameter descriptions; `Option<T>` is optional). Shared handles (a pool, a
client) go in one `#[state]` argument, passed to `get_order::tool(state)`. The full reference is
[docs/tools.md](../tools.md).

The macro needs feature `macros` (on by default in `sinew`; `features = ["macros"]` on `sinew-core`).

### Tests

- Call the function directly: the macro leaves it unchanged, so business logic is tested like any function.
- Check the generated spec: `get_order::tool().spec()` has the expected name, description, required fields and
  enum values. `spec_comes_from_signature_and_docs` in `crates/sinew-core/tests/tool_macro.rs` is the pattern.
- Run it in an agent with `ScriptedModel`: script a tool call and the final answer, then assert on the tool
  result message. No network.

Changes to the macro itself go in `crates/sinew-macros/src/lib.rs`, with a test in `tool_macro.rs` and, for a new
compile error, a `trybuild` case in `crates/sinew-core/tests/ui/` (`.rs` plus the expected `.stderr`; regenerate
the `.stderr` with `TRYBUILD=overwrite cargo test -p sinew-core --features macros --test tool_macro`).

## Path B: built-in harness tool

`crates/sinew-harness/src/tools/think.rs` is the smallest complete example.

1. **Module.** Create `crates/sinew-harness/src/tools/acme.rs` and implement `Tool`:

   ```rust
   use async_trait::async_trait;
   use serde_json::{json, Value};
   use sinew_core::{Tool, ToolSpec};

   use super::{arg_str, tool_err};

   pub struct AcmeTool { /* options */ }

   #[async_trait]
   impl Tool for AcmeTool {
       fn spec(&self) -> ToolSpec {
           ToolSpec {
               name: "acme".into(),
               description: "What it does, when to use it, what it returns.".into(),
               parameters: json!({
                   "type": "object",
                   "properties": { "query": {"type": "string", "description": "..."} },
                   "required": ["query"]
               }),
           }
       }

       async fn call(&self, args: Value) -> sinew_core::Result<Value> {
           let query = arg_str("acme", &args, "query")?;
           /* ... */
           Err(tool_err("acme", "explain what failed and how to fix the arguments"))
       }
   }
   ```

   Use the helpers in `tools/mod.rs` (`arg_str`, `opt_str`, `arg_u64`, `arg_bool`, `tool_err`,
   `truncate_middle`): models send numbers as strings and booleans as `"true"`, and the helpers accept both. Cap
   output size (`truncate_middle`) so one call cannot fill the context window.

2. **Register it** in `crates/sinew-harness/src/tools/mod.rs` (`pub mod acme;`, `pub use acme::AcmeTool;`), add an
   opt-in method on `AgentBuilder` in `crates/sinew-harness/src/agent.rs` next to `think()` / `http_fetch()`, and
   push the tool in `build()`. If the model needs a usage hint, add one where `build()` collects the
   `# Working style` hints.

3. **Permissions and safety.** A tool that writes, runs commands or reaches the network must be listed with `ask`
   in `PermissionPolicy::interactive()` (`crates/sinew-harness/src/permissions.rs`), the policy `sinew-agent`
   uses without `--yes`. If it takes a path, resolve it with `Workspace::resolve` so it cannot escape the
   workspace; if it takes a URL, apply the same private-address checks as `http_fetch`.

4. **Feature flag.** Only for a new dependency: add a feature to `crates/sinew-harness/Cargo.toml` and gate the
   module, the export and the builder method. Tools written with the existing dependencies need none.

5. **Tests.**
   - Unit tests in the module for argument handling, limits and errors (`fs.rs`, `shell.rs` and `http.rs` have
     examples). Use `tempfile` for files; never call external services from `cargo test`.
   - An agent test in `crates/sinew-harness/tests/agent.rs`: a `ScriptedModel` that calls the tool and then
     answers, checking the tool result the model received and, for risky tools, that the permission policy asks
     or denies.

## Docs to update

- Path A in the workspace: the sample's README, or [docs/tools.md](../tools.md) if the tool shows a new pattern.
- Path B: the **Tools** section of [docs/harness.md](../harness.md) (arguments and limits), the builder reference
  table in the same file, and the permissions paragraph if `interactive()` changed.

## CHANGELOG

```markdown
- `sinew-harness`: `acme` built-in tool (`AgentBuilder::acme`), asks for approval under `PermissionPolicy::interactive()` (#123).
```
