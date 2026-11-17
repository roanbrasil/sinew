# Built-in tools

`sinew-tools` is a library of ready-made tools for the cases most agents need: searching and reading the web,
looking things up on Wikipedia, arithmetic, dates, calling an HTTP API and e-mail. Each family is a cargo
feature, so you compile only what you use. Every tool is an ordinary `sinew_core::Tool`: put it in a `ToolBox`
and hand it to `react_agent`, the harness, an MCP server or anything else that takes tools.

```toml
[dependencies]
sinew-tools = { version = "0.1", features = ["web-search", "web-fetch", "calculator", "datetime"] }
# or, with the umbrella crate, every family at once (`sinew::builtin_tools`):
# sinew = { version = "0.1", features = ["tools"] }
```

| Feature | Tools | Side effects | Configuration (state) |
|---|---|---|---|
| `calculator` | `calculator` | no | none |
| `datetime` | `current_time`, `date_add`, `date_diff` | no | `Clock` (system clock and time zone, or a fixed instant) |
| `web-search` | `web_search` | no | `WebSearch` (Tavily, Brave Search or SearXNG) |
| `web-fetch` | `web_fetch` | no | `WebFetch` (SSRF guard, size limits) |
| `wikipedia` | `wikipedia_search`, `wikipedia_summary` | no | `Wikipedia` (language edition) |
| `http` | `http_request` (`HttpRequestTool`) | yes when a method other than GET/HEAD/OPTIONS is enabled | allow-listed hosts, methods, fixed headers |
| `email-smtp` | `send_email` | yes | `SmtpMailer` (SMTP URL, sender, recipient allow-list) |
| `email-imap` | `email_list`, `email_read` | no (read-only) | `ImapReader` (host, credentials, mailbox) |
| `email` | both e-mail families | | |
| `full` | everything | | |

Tools built with [`#[sinew::tool]`](tools.md) take their configuration as a `#[state]` argument: you pass it
once, with `name::tool(state)`, and the model never sees it. API keys, credentials and limits therefore stay in
your program.

```rust
use sinew::core::ToolBox;
use sinew_tools::{
    calculator::calculator,
    datetime::{current_time, date_add, date_diff, Clock},
    fetch::{web_fetch, FetchOptions, WebFetch},
    search::{web_search, WebSearch},
    wikipedia::{wikipedia_search, wikipedia_summary, Wikipedia},
};

let clock = Clock::system().time_zone("America/Toronto")?;
let wiki = Wikipedia::new("en");
let tools = ToolBox::new()
    .with(web_search::tool(WebSearch::from_env()?))
    .with(web_fetch::tool(WebFetch::new(FetchOptions::default())?))
    .with(wikipedia_search::tool(wiki.clone()))
    .with(wikipedia_summary::tool(wiki))
    .with(calculator::tool())
    .with(current_time::tool(clock.clone()))
    .with(date_add::tool(clock.clone()))
    .with(date_diff::tool(clock));
let agent = react_agent(model, tools)?;
```

All tools return `Result<_, sinew_tools::ToolError>`. Inside an agent the error message becomes the tool result,
so the model reads, for example, `blocked: address 10.0.0.5 is loopback, private, link-local or reserved` or
`invalid input: division by zero` and can
correct its call.

## Calculator

`calculator(expression)` evaluates arithmetic with a recursive-descent parser over `f64`: `+ - * / %`, `^` or
`**`, `!`, parentheses, scientific notation, `pi`, `e`, `tau` and the functions `sqrt cbrt abs exp ln log log2
log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil round trunc min max pow hypot` (radians for
trigonometry). No code is evaluated: anything outside this grammar is rejected with the position of the
offending character. Expressions longer than 1 000 characters or nested deeper than 64 levels are refused, as
are division by zero and non-finite results. `sinew_tools::calculator::evaluate(&str)` is the same evaluator
without the tool wrapper.

## Date and time

| Tool | Does |
|---|---|
| `current_time(timezone?)` | Current date, time, weekday, UTC offset and Unix time, in an IANA time zone. |
| `date_add(start, years?, months?, weeks?, days?, hours?, minutes?, seconds?, timezone?)` | Calendar arithmetic. `2024-01-31` plus one month is `2024-02-29`; zoned values account for daylight saving time. |
| `date_diff(start, end, timezone?)` | Years, months, days and time between two points, plus totals in days and hours. |

Inputs can be `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM[:SS]`, an RFC 3339 timestamp, a zoned value such as
`2026-11-17T14:00[Europe/Paris]`, or `now`, `today`, `tomorrow`, `yesterday`. The `Clock` state provides "now"
and the default time zone; `Clock::fixed(unix_seconds)` makes tests deterministic. Time zones come from
[jiff](https://docs.rs/jiff) and the system's IANA database.

## Web search

`web_search(query, max_results?)` returns `[{title, url, snippet, published?}]` from one of three providers:

| Provider | Constructor | Environment |
|---|---|---|
| Tavily | `WebSearch::tavily(key)` | `TAVILY_API_KEY` |
| Brave Search API | `WebSearch::brave(key)` | `BRAVE_SEARCH_API_KEY` or `BRAVE_API_KEY` |
| SearXNG | `WebSearch::searxng(url)` | `SEARXNG_URL` |

`WebSearch::from_env()` picks the first one configured, in that order. SearXNG must have the `json` format
enabled (`search.formats` in `settings.yml`). `base_url` points a provider at a proxy or a test server, and
`client` replaces the `reqwest` client. The provider endpoint is your configuration, not model input, so it is
not restricted by the SSRF guard below; a SearXNG instance on your own network works.

## Web fetch

`web_fetch(url, max_chars?)` downloads a page and returns `{url, status, content_type, title?, byline?, text,
truncated}`. HTML goes through [dom_smoothie](https://docs.rs/dom_smoothie), a port of Mozilla's Readability,
and comes back as Markdown without menus, sidebars, scripts and footers; pages the algorithm cannot score fall
back to the text of `<body>`. Plain text, JSON and XML are returned as they are; other content types are
refused.

`FetchOptions` sets the limits: `max_bytes` (5 MiB read at most; the rest is dropped), `max_chars` (20 000
characters returned; the model can ask for fewer, not more), `timeout` (20 s) and the host `policy`.

### SSRF protection

A model that can choose URLs can be steered, by a prompt injection in a page it read, to
`http://169.254.169.254/` (cloud credentials), `http://localhost:8080/admin` or a database on your network.
`sinew_tools::net::HostPolicy` blocks that:

- only `http` and `https`;
- a deny-list and an optional allow-list of hosts (a host also matches its subdomains);
- by default, no loopback, private (RFC 1918, IPv6 unique local), link-local (including `169.254.169.254`),
  unspecified, shared (`100.64.0.0/10`), benchmarking, documentation, multicast, broadcast or reserved address.
  IPv4-mapped, IPv4-compatible, NAT64 and 6to4 IPv6 addresses are judged by the IPv4 address they embed;
- names such as `localhost`, `*.localhost`, `*.local` and `*.internal` are refused before any lookup.

The check does not stop at the URL text. `HostPolicy::client_builder()` returns a `reqwest` client builder whose
DNS resolver drops non-public answers, so `internal.example.com` resolving to `10.0.0.5` is refused at
connection time (and a DNS rebinding between a check and the connection does not help). Each redirect target is
checked again, at most five redirects are followed, and proxies from the environment are ignored because a
proxy would resolve the name itself.

```rust
use sinew_tools::net::HostPolicy;

let policy = HostPolicy::new()
    .deny_host("example.org")          // never, even if allowed
    .allow_host("docs.rs")             // once a host is allowed, only allowed hosts work
    .allow_host("wikipedia.org");
let fetch = WebFetch::new(FetchOptions::default().policy(policy))?;
```

`allow_private(true)` turns the address check off (an intranet tool, a local service); combine it with an
allow-list. `pin_host(name, addrs)` resolves a name to fixed addresses, which still pass the check (split-horizon
DNS, tests). The harness's `http_fetch` tool (`sinew-harness`) uses the same guard.

## Wikipedia

`wikipedia_search(query, limit?)` returns titles, descriptions and excerpts; `wikipedia_summary(title)` returns
the lead section, the article URL and its kind (`standard`, or `disambiguation` when the title has several
meanings). Both use Wikipedia's public REST APIs without a key. `Wikipedia::new("pt")` selects the language
edition; `Wikipedia::with_base_url` points at another MediaWiki site.

## HTTP requests

`HttpRequestTool` gives the model one API without a tool per endpoint. The application decides the hosts, the
methods and the credentials; the model sends the URL, query, extra headers and a JSON or text body, and receives
`{status, content_type, body, truncated, location?}` (JSON bodies parsed).

```rust
use reqwest::Method;
use sinew_tools::http::HttpRequestTool;

let github = HttpRequestTool::builder()
    .name("github_api")
    .description("GitHub REST API. Issues are under /repos/{owner}/{repo}/issues.")
    .allow_host("api.github.com")
    .methods([Method::GET, Method::POST])
    .header("Authorization", &format!("Bearer {token}"))?
    .build()?;
```

- `build()` fails without an allowed host. The SSRF guard applies as for `web_fetch`.
- Headers set with `header` are sent on every request, are marked sensitive and cannot be overridden by the
  model; hop-by-hop headers (`Host`, `Content-Length`, `Connection`, ...) cannot be set at all.
- `side_effects()` is `true` as soon as `POST`, `PUT`, `PATCH`, `DELETE` or another unsafe method is enabled, so
  [durable runs](durable-execution.md) never repeat such a call blindly. Unsafe requests carry the call's
  `idempotency_key()` in an `Idempotency-Key` header (configurable with `idempotency_header`), which APIs that
  deduplicate requests understand.

## E-mail

`send_email(to, subject, body, cc?, reply_to?)` (feature `email-smtp`, [lettre](https://docs.rs/lettre) with
rustls) sends a plain-text message:

```rust
use sinew_tools::email::{send_email, SmtpMailer};

let mailer = SmtpMailer::from_url("smtps://bot%40example.com:app-password@smtp.example.com", "Bot <bot@example.com>")?
    .allow_recipient("@example.com")      // a domain (subdomains included)
    .allow_recipient("partner@vendor.com") // or one address
    .max_recipients(5);
let tool = send_email::tool(mailer);
```

- Recipients must be on the allow-list; with an empty list every recipient is refused unless you call
  `allow_any_recipient()`. Consider pairing the tool with an approval step in the harness.
- `side_effects()` is `true`. Inside durable runs the `Message-ID` is derived from the call's idempotency key, so
  a message sent twice after a crash carries the same ID and can be recognized as a duplicate.
- `SmtpMailer::from_env()` reads `SMTP_URL` and `SMTP_FROM`.

`email_list(mailbox?, unread_only?, limit?)` and `email_read(uid, mailbox?)` (feature `email-imap`,
[async-imap](https://docs.rs/async-imap) with rustls) read a mailbox. They open it with `EXAMINE` and fetch with
`BODY.PEEK`, so nothing is marked as read and nothing can be changed. `email_read` returns the plain-text body
(HTML-only messages converted to text) and the names of attachments, which are not downloaded.
`ImapReader::from_env()` reads `IMAP_HOST`, `IMAP_USERNAME`, `IMAP_PASSWORD` and `IMAP_PORT`.

## Testing

Every network tool in the crate is tested offline: search, Wikipedia, fetch and HTTP against a local axum
server that replays recorded responses (`crates/sinew-tools/tests/fixtures`), SMTP and IMAP against scripted
local servers. In your own tests, point the tools at a local server the same way (`WebSearch::base_url`,
`Wikipedia::with_base_url`, a `HostPolicy` with `allow_private(true)`), or record whole agent runs with
[`sinew_core::vcr`](testing.md), whose cassettes also capture tool results.

## Not included yet

- Sandboxed code execution: use the harness's `shell` tool with its permission policy for now.
- Calendar (CalDAV): no small, maintained async CalDAV client fits the dependency policy yet.
