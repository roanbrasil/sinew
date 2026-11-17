<img src="https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png" alt="Sinew logo" width="64" align="right">

# sinew-macros

[![crates.io](https://img.shields.io/crates/v/sinew-macros.svg)](https://crates.io/crates/sinew-macros)
[![docs.rs](https://docs.rs/sinew-macros/badge.svg)](https://docs.rs/sinew-macros)

Procedural macros for [Sinew](https://github.com/roanbrasil/sinew). Use them through `sinew` (enabled by
default) or `sinew-core` with `features = ["macros"]`; do not depend on this crate directly.

```rust
use sinew::prelude::*;

/// Look up an order by id.
#[tool]
async fn get_order(
    #[state] db: Db,
    /// Order id, e.g. "A-1001".
    id: String,
) -> Result<Order, String> { /* ... */ }

let tools = ToolBox::new().with(get_order::tool(db));
```

- `#[tool]`: a function becomes a tool; name from the function, description from the doc comment, JSON
  Schema from the argument types, `#[state]` for injected application state.
- `prompt!` / `chat_prompt!`: prompt templates checked at compile time; a missing or misspelled variable is
  a compile error. `prompt!("Summarize {text}").text(doc).format()`.
- `#[derive(State)]`: field-wise reducers for graph state (re-exported by `sinew-graph`).

Guides: [docs/tools.md](https://github.com/roanbrasil/sinew/blob/main/docs/tools.md) ·
[docs/prompts.md](https://github.com/roanbrasil/sinew/blob/main/docs/prompts.md).

How this crate fits with the others: [architecture guide](https://github.com/roanbrasil/sinew/blob/main/docs/architecture.md).

## License

MIT OR Apache-2.0.
