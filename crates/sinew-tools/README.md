# sinew-tools

Built-in tools for [Sinew](https://github.com/roanbrasil/sinew) agents, one cargo feature per family:

| Feature | Tools | Side effects |
|---|---|---|
| `calculator` | `calculator` (expression parser, no code execution) | no |
| `datetime` | `current_time`, `date_add`, `date_diff` (IANA time zones, calendar-aware) | no |
| `web-search` | `web_search` (Tavily, Brave Search, SearXNG) | no |
| `web-fetch` | `web_fetch` (Readability to Markdown, size limits, SSRF guard) | no |
| `wikipedia` | `wikipedia_search`, `wikipedia_summary` | no |
| `http` | `http_request` (host allow-list, fixed credentials) | when a method other than GET/HEAD/OPTIONS is enabled |
| `email-smtp` | `send_email` (recipient allow-list) | yes |
| `email-imap` | `email_list`, `email_read` (read-only) | no |
| `full` | all of them | |

```toml
[dependencies]
sinew-tools = { version = "0.1", features = ["web-search", "web-fetch", "calculator"] }
```

```rust
use sinew_core::ToolBox;
use sinew_tools::{
    calculator::calculator,
    fetch::{web_fetch, FetchOptions, WebFetch},
    search::{web_search, WebSearch},
};

let tools = ToolBox::new()
    .with(web_search::tool(WebSearch::from_env()?))   // TAVILY_API_KEY, BRAVE_SEARCH_API_KEY or SEARXNG_URL
    .with(web_fetch::tool(WebFetch::new(FetchOptions::default())?))
    .with(calculator::tool());
```

API keys, clients and limits are `#[state]` of each tool: the model never sees them. Tools that open
model-chosen URLs go through `net::HostPolicy`, which refuses loopback, private, link-local and other non-public
addresses after DNS resolution and on every redirect.

Guide: [docs/builtin-tools.md](https://github.com/roanbrasil/sinew/blob/main/docs/builtin-tools.md).

License: MIT OR Apache-2.0.
