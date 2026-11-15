use std::sync::Arc;

use sinew_core::{ChatModel, ChatRequest, Message, Usage};

use crate::{format_documents, Document, Result, Retriever};

const SYSTEM: &str = "You answer questions using only the numbered context passages. \
Cite the passages you used with their numbers in square brackets, like [1] or [2][3]. \
If the context does not contain the answer, say that you don't know.";

const USER: &str = "Context:\n{context}\n\nQuestion: {question}";

/// The answer of a [`RagChain`] and the documents it was grounded on.
#[derive(Debug, Clone)]
pub struct RagAnswer {
    pub answer: String,
    /// Retrieved documents, numbered from 1 in the prompt in this order.
    pub sources: Vec<Document>,
    pub usage: Usage,
}

impl RagAnswer {
    /// Unique `source` metadata values, in order.
    pub fn source_names(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for s in self.sources.iter().filter_map(Document::source) {
            if !out.contains(&s) {
                out.push(s);
            }
        }
        out
    }

    /// The passages whose `[n]` marker appears in the answer.
    pub fn cited(&self) -> Vec<&Document> {
        self.sources
            .iter()
            .enumerate()
            .filter(|(i, _)| self.answer.contains(&format!("[{}]", i + 1)))
            .map(|(_, d)| d)
            .collect()
    }
}

/// Question → retrieve → prompt → chat model → answer with sources.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use sinew_rag::*;
/// # async fn demo(model: Arc<dyn sinew_core::ChatModel>, retriever: Arc<dyn Retriever>) -> sinew_rag::Result<()> {
/// let rag = RagChain::new(model, retriever);
/// let out = rag.invoke("How do I resume a paused graph?").await?;
/// println!("{}\nsources: {:?}", out.answer, out.source_names());
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct RagChain {
    model: Arc<dyn ChatModel>,
    retriever: Arc<dyn Retriever>,
    system: String,
    template: String,
    temperature: Option<f32>,
    max_chars: Option<usize>,
}

impl RagChain {
    pub fn new(model: Arc<dyn ChatModel>, retriever: Arc<dyn Retriever>) -> Self {
        Self {
            model,
            retriever,
            system: SYSTEM.into(),
            template: USER.into(),
            temperature: Some(0.0),
            max_chars: None,
        }
    }

    pub fn with_system_prompt(mut self, s: impl Into<String>) -> Self {
        self.system = s.into();
        self
    }

    /// User message template with `{context}` and `{question}` placeholders.
    pub fn with_template(mut self, t: impl Into<String>) -> Self {
        self.template = t.into();
        self
    }

    /// `None` leaves the provider default.
    pub fn with_temperature(mut self, t: Option<f32>) -> Self {
        self.temperature = t;
        self
    }

    /// Cut each passage to `n` characters in the prompt.
    pub fn with_max_chars(mut self, n: usize) -> Self {
        self.max_chars = Some(n);
        self
    }

    /// The messages that would be sent for `question` and `docs`.
    pub fn messages(&self, question: &str, docs: &[Document]) -> Vec<Message> {
        let context = if docs.is_empty() {
            "(no passages found)".to_string()
        } else {
            format_documents(docs, self.max_chars)
        };
        let user = self
            .template
            .replace("{context}", &context)
            .replace("{question}", question);
        vec![Message::system(self.system.clone()), Message::user(user)]
    }

    pub async fn invoke(&self, question: &str) -> Result<RagAnswer> {
        let sources = self.retriever.retrieve(question).await?;
        let mut req = ChatRequest::new(self.messages(question, &sources));
        if let Some(t) = self.temperature {
            req = req.with_temperature(t);
        }
        let resp = self.model.chat(req).await?;
        Ok(RagAnswer {
            answer: resp.message.content,
            sources,
            usage: resp.usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bm25Retriever;
    use sinew_core::mock::ScriptedModel;

    #[tokio::test]
    async fn answers_with_sources() {
        let retriever = Bm25Retriever::from_documents(vec![
            Document::new("Use SqliteCheckpointer for durable threads.")
                .with_metadata("source", "docs/checkpointing.md"),
            Document::new("MCP exposes tools over stdio.").with_metadata("source", "docs/mcp.md"),
        ]);
        let model = Arc::new(ScriptedModel::new([Message::assistant(
            "Use SqliteCheckpointer [1].",
        )]));
        let rag = RagChain::new(model.clone(), Arc::new(retriever));
        let out = rag.invoke("durable threads checkpointer").await.unwrap();
        assert_eq!(out.answer, "Use SqliteCheckpointer [1].");
        assert_eq!(out.source_names()[0], "docs/checkpointing.md");
        assert_eq!(out.cited().len(), 1);
        let req = &model.requests()[0];
        assert!(req.messages[0].content.contains("Cite"));
        assert!(req.messages[1]
            .content
            .contains("[1] (source: docs/checkpointing.md)"));
        assert!(req.messages[1]
            .content
            .ends_with("Question: durable threads checkpointer"));
        assert_eq!(req.temperature, Some(0.0));
    }
}
