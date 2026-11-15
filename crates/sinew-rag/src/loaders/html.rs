use std::path::PathBuf;

use super::{read_to_string, source_metadata, Loader};
use crate::{Document, Result};

/// Loads an HTML file as plain text (see [`html_to_text`]); the `<title>`
/// goes to the `title` metadata field.
#[derive(Debug, Clone)]
pub struct HtmlLoader {
    path: PathBuf,
}

impl HtmlLoader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl Loader for HtmlLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let html = read_to_string(&self.path)?;
        let mut metadata = source_metadata(&self.path);
        if let Some(t) = html_title(&html) {
            metadata.insert("title".into(), t.into());
        }
        Ok(vec![
            Document::new(html_to_text(&html)).with_metadata_map(metadata)
        ])
    }
}

const SKIP: &[&str] = &[
    "script", "style", "head", "noscript", "template", "svg", "iframe",
];
/// Elements separated by a blank line.
const PARAGRAPH: &[&str] = &[
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "table",
    "section",
    "article",
    "blockquote",
    "pre",
];
/// Elements that start a new line.
const BLOCK: &[&str] = &[
    "div",
    "br",
    "tr",
    "header",
    "footer",
    "nav",
    "aside",
    "main",
    "hr",
    "dl",
    "dt",
    "dd",
    "figure",
    "figcaption",
    "form",
    "address",
];

fn html_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let open_end = start + lower[start..].find('>')? + 1;
    let end = open_end + lower[open_end..].find("</title")?;
    let t = decode_entities(html[open_end..end].trim());
    (!t.is_empty()).then_some(t)
}

/// Converts HTML to readable plain text without a DOM parser: drops
/// comments and `script` / `style` / `head` / ... elements, turns block
/// elements into line breaks and list items into `- ` lines, decodes
/// entities and collapses whitespace.
pub fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut rest = html;
    let mut skip_until: Option<String> = None;
    while !rest.is_empty() {
        let Some(lt) = rest.find('<') else {
            if skip_until.is_none() {
                push_text(&mut out, rest);
            }
            break;
        };
        if skip_until.is_none() {
            push_text(&mut out, &rest[..lt]);
        }
        rest = &rest[lt..];
        let starts_tag = rest[1..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || matches!(c, '/' | '!' | '?'));
        if !starts_tag {
            if skip_until.is_none() {
                out.push('<');
            }
            rest = &rest[1..];
            continue;
        }
        if let Some(after) = rest.strip_prefix("<!--") {
            rest = after.find("-->").map(|i| &after[i + 3..]).unwrap_or("");
            continue;
        }
        let Some(gt) = rest.find('>') else {
            // A stray `<`: keep it as text.
            if skip_until.is_none() {
                out.push('<');
            }
            rest = &rest[1..];
            continue;
        };
        let tag = &rest[1..gt];
        rest = &rest[gt + 1..];
        let closing = tag.starts_with('/');
        let name: String = tag
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        if let Some(until) = &skip_until {
            if closing && &name == until {
                skip_until = None;
            }
            continue;
        }
        if name.is_empty() {
            continue;
        }
        if !closing && SKIP.contains(&name.as_str()) && !tag.ends_with('/') {
            skip_until = Some(name);
            continue;
        }
        match name.as_str() {
            "li" if !closing => out.push_str("\n- "),
            "td" | "th" if !closing => out.push_str(" \t"),
            n if PARAGRAPH.contains(&n) => out.push_str("\n\n"),
            n if BLOCK.contains(&n) => out.push('\n'),
            _ => {}
        }
    }
    normalize_whitespace(&decode_entities(&out))
}

/// Source newlines are plain whitespace in HTML.
fn push_text(out: &mut String, text: &str) {
    out.extend(
        text.chars()
            .map(|c| if c == '\n' || c == '\r' { ' ' } else { c }),
    );
}

fn normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank_run = 0;
    for line in s.lines() {
        let mut collapsed = String::with_capacity(line.len());
        for word in line.split_whitespace() {
            if !collapsed.is_empty() {
                collapsed.push(' ');
            }
            collapsed.push_str(word);
        }
        if collapsed.is_empty() {
            blank_run += 1;
            continue;
        }
        if !out.is_empty() {
            out.push_str(if blank_run > 0 { "\n\n" } else { "\n" });
        }
        blank_run = 0;
        out.push_str(&collapsed);
    }
    out
}

fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let end = rest[1..]
            .char_indices()
            .take(10)
            .find(|(_, c)| *c == ';')
            .map(|(i, _)| i + 1);
        let decoded = end.and_then(|e| {
            let ent = &rest[1..e];
            let ch = match ent {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" | "#39" => Some('\''),
                "nbsp" => Some(' '),
                "mdash" => Some('—'),
                "ndash" => Some('–'),
                "hellip" => Some('…'),
                "copy" => Some('©'),
                _ => ent
                    .strip_prefix("#x")
                    .or_else(|| ent.strip_prefix("#X"))
                    .and_then(|h| u32::from_str_radix(h, 16).ok())
                    .or_else(|| ent.strip_prefix('#').and_then(|d| d.parse().ok()))
                    .and_then(char::from_u32),
            };
            ch.map(|c| (c, e + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_html() {
        let html = r#"<html><head><title>T &amp; U</title><style>p{}</style></head>
            <body><!-- hi --><h1>Title</h1><p>One&nbsp;two &lt;3 &#x41;&#66;</p>
            <script>alert(1)</script><ul><li>a</li><li>b</li></ul>x < y</body></html>"#;
        let t = html_to_text(html);
        assert_eq!(t, "Title\n\nOne two <3 AB\n\n- a\n- b\n\nx < y");
        assert_eq!(html_title(html).as_deref(), Some("T & U"));
        assert_eq!(decode_entities("&bogus; &amp"), "&bogus; &amp");
    }
}
