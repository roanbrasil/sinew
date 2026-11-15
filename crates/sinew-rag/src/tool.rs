use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use sinew_core::{Tool, ToolSpec};

use crate::{Document, Retriever};

/// Formats documents as numbered context blocks:
/// `[1] (source: docs/a.md)\ntext`, separated by blank lines. Each text is
/// cut to `max_chars` characters when given.
pub fn format_documents(docs: &[Document], max_chars: Option<usize>) -> String {
    let mut out = String::new();
    for (i, d) in docs.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        out.push_str(&format!("[{}]", i + 1));
        if let Some(src) = d.source() {
            out.push_str(&format!(" (source: {src})"));
        }
        out.push('\n');
        match max_chars {
            Some(n) if d.content.chars().count() > n => {
                out.extend(d.content.chars().take(n));
                out.push('…');
            }
            _ => out.push_str(&d.content),
        }
    }
    out
}

/// A retriever exposed to agents as a [`Tool`] taking `{"query": "..."}`
/// and returning the formatted documents (see [`retriever_tool`]).
#[derive(Clone)]
pub struct RetrieverTool {
    retriever: Arc<dyn Retriever>,
    name: String,
    description: String,
    max_chars: Option<usize>,
}

impl RetrieverTool {
    /// Cut each document to `n` characters in the tool output.
    pub fn with_max_chars(mut self, n: usize) -> Self {
        self.max_chars = Some(n);
        self
    }
}

/// Turns a retriever into a tool for `react_agent` and other tool-calling
/// loops (LangChain's `create_retriever_tool`).
///
/// ```
/// # use std::sync::Arc;
/// # use sinew_rag::*;
/// let r = Bm25Retriever::from_documents(vec![Document::new("Sinew is a Rust LLM framework")]);
/// let tool = retriever_tool(Arc::new(r), "search_docs", "Searches the Sinew documentation");
/// # use sinew_core::Tool;
/// assert_eq!(tool.spec().name, "search_docs");
/// ```
pub fn retriever_tool(
    retriever: Arc<dyn Retriever>,
    name: impl Into<String>,
    description: impl Into<String>,
) -> RetrieverTool {
    RetrieverTool {
        retriever,
        name: name.into(),
        description: description.into(),
        max_chars: None,
    }
}

#[async_trait]
impl Tool for RetrieverTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to search for" }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> sinew_core::Result<Value> {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .or_else(|| arguments.as_str())
            .ok_or_else(|| sinew_core::Error::Tool {
                name: self.name.clone(),
                message: "missing string argument `query`".into(),
            })?;
        let docs = self
            .retriever
            .retrieve(query)
            .await
            .map_err(|e| sinew_core::Error::Tool {
                name: self.name.clone(),
                message: e.to_string(),
            })?;
        if docs.is_empty() {
            return Ok(Value::String("No relevant documents found.".into()));
        }
        Ok(Value::String(format_documents(&docs, self.max_chars)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bm25Retriever;

    #[tokio::test]
    async fn tool_formats_documents() {
        let r = Bm25Retriever::from_documents(vec![
            Document::new("Checkpoints persist state")
                .with_metadata("source", "docs/checkpointing.md"),
            Document::new("unrelated"),
        ]);
        let t = retriever_tool(Arc::new(r), "search", "Search docs").with_max_chars(11);
        let out = t.call(json!({"query": "checkpoints"})).await.unwrap();
        assert_eq!(
            out,
            Value::String("[1] (source: docs/checkpointing.md)\nCheckpoints…".into())
        );
        let none = t.call(json!({"query": "zzz"})).await.unwrap();
        assert_eq!(none, Value::String("No relevant documents found.".into()));
        assert!(t.call(json!({})).await.is_err());
        assert_eq!(t.spec().parameters["required"][0], "query");
    }
}
