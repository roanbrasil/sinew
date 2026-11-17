//! Built-in tools for [Sinew](https://github.com/roanbrasil/sinew) agents.
//!
//! Each tool family is a cargo feature, so an application compiles only the
//! families it uses. Every tool is a `sinew_core::Tool` with a description
//! written for the model, a JSON Schema for its arguments and an accurate
//! `side_effects()` flag (see the durable execution guide). Clients, API keys
//! and limits are passed as state when the tool is built and are never seen
//! by the model.
//!
//! | Feature | Tools | Side effects |
//! |---|---|---|
//! | `calculator` | [`calculator`](fn@calculator::calculator) | no |
//! | `datetime` | [`current_time`](fn@datetime::current_time), [`date_add`](fn@datetime::date_add), [`date_diff`](fn@datetime::date_diff) | no |
//! | `web-search` | [`web_search`](fn@search::web_search) (Tavily, Brave Search, SearXNG) | no |
//! | `web-fetch` | [`web_fetch`](fn@fetch::web_fetch) (main text by readability, SSRF guard) | no |
//! | `wikipedia` | [`wikipedia_search`](fn@wikipedia::wikipedia_search), [`wikipedia_summary`](fn@wikipedia::wikipedia_summary) | no |
//! | `http` | [`HttpRequestTool`](http::HttpRequestTool) (`http_request`, host allow-list) | yes, when a method other than GET/HEAD is enabled |
//! | `email-smtp` | [`send_email`](fn@email::send_email) | yes |
//! | `email-imap` | [`email_list`](fn@email::email_list), [`email_read`](fn@email::email_read) (read-only) | no |
//! | `full` | all of the above | |
//!
//! The [`net`] module (always compiled) holds the SSRF guard shared by the
//! tools that open model-chosen URLs and by `sinew-harness`'s `http_fetch`.
//!
//! ```
//! # #[cfg(feature = "calculator")] {
//! use sinew_core::{Tool, ToolBox};
//! use sinew_tools::calculator::calculator;
//!
//! let tools = ToolBox::new().with(calculator::tool());
//! assert_eq!(calculator::tool().spec().name, "calculator");
//! # let _ = tools;
//! # }
//! ```
//!
//! Guide: [docs/builtin-tools.md](https://github.com/roanbrasil/sinew/blob/main/docs/builtin-tools.md).

mod error;
pub mod net;

#[cfg(feature = "calculator")]
pub mod calculator;
#[cfg(feature = "datetime")]
pub mod datetime;
#[cfg(any(feature = "email-smtp", feature = "email-imap"))]
pub mod email;
#[cfg(feature = "web-fetch")]
pub mod fetch;
#[cfg(feature = "http")]
pub mod http;
#[cfg(feature = "web-search")]
pub mod search;
#[cfg(feature = "wikipedia")]
pub mod wikipedia;

#[cfg(test)]
#[allow(dead_code, unused_imports, unused_macros)]
mod testutil;

pub use error::ToolError;

/// User agent sent by the HTTP-based tools.
pub const USER_AGENT: &str = concat!(
    "sinew-tools/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/roanbrasil/sinew)"
);

/// Keeps the first `max_chars` characters; the flag is true when text was cut.
#[allow(dead_code)]
pub(crate) fn truncate_chars(text: &str, max_chars: usize) -> (String, bool) {
    match text.char_indices().nth(max_chars) {
        Some((i, _)) => (text[..i].to_string(), true),
        None => (text.to_string(), false),
    }
}

/// Removes HTML tags and decodes the common entities (search snippets).
#[allow(dead_code)]
pub(crate) fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        assert_eq!(truncate_chars("héllo", 2), ("hé".to_string(), true));
        assert_eq!(truncate_chars("hi", 5), ("hi".to_string(), false));
        assert_eq!(
            strip_tags("a <strong>b</strong> &amp; &quot;c&quot;"),
            "a b & \"c\""
        );
    }
}
