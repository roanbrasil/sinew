use std::sync::Arc;

use async_trait::async_trait;
use sinew_core::{ChatModel, ChatRequest, Message};

use super::{dedup, Retriever};
use crate::{Document, Result};

/// Rewrites a question into search queries.
#[async_trait]
pub trait QueryExpander: Send + Sync {
    async fn expand(&self, query: &str) -> Result<Vec<String>>;
}

/// Asks a chat model for `n` alternative phrasings (LangChain's
/// `MultiQueryRetriever` prompt), one per line.
#[derive(Clone)]
pub struct LlmQueryExpander {
    model: Arc<dyn ChatModel>,
    n: usize,
    prompt: String,
}

const PROMPT: &str = "You are an AI language model assistant. Your task is to generate {n} \
different versions of the given user question to retrieve relevant documents from a vector \
database. By generating multiple perspectives on the user question, your goal is to help the \
user overcome some of the limitations of distance-based similarity search. Provide these \
alternative questions separated by newlines, with no numbering and no other text.\n\
Original question: {question}";

impl LlmQueryExpander {
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            model,
            n: 3,
            prompt: PROMPT.into(),
        }
    }

    /// How many alternatives to ask for (default 3).
    pub fn with_n(mut self, n: usize) -> Self {
        self.n = n.max(1);
        self
    }

    /// Custom prompt; `{question}` and `{n}` are substituted.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }
}

/// Parses one query per line, dropping bullets, numbering and blanks.
pub(crate) fn parse_lines(text: &str, max: usize) -> Vec<String> {
    text.lines()
        .map(|l| {
            l.trim()
                .trim_start_matches(|c: char| {
                    c.is_ascii_digit() || matches!(c, '-' | '*' | '•' | '.' | ')')
                })
                .trim()
                .trim_matches('"')
                .to_string()
        })
        .filter(|l| !l.is_empty())
        .take(max)
        .collect()
}

#[async_trait]
impl QueryExpander for LlmQueryExpander {
    async fn expand(&self, query: &str) -> Result<Vec<String>> {
        let prompt = self
            .prompt
            .replace("{n}", &self.n.to_string())
            .replace("{question}", query);
        let resp = self
            .model
            .chat(ChatRequest::new(vec![Message::user(prompt)]).with_temperature(0.0))
            .await?;
        Ok(parse_lines(&resp.message.content, self.n))
    }
}

/// Retrieves with several rewrites of the question and returns the unique
/// union of the results (original query first).
#[derive(Clone)]
pub struct MultiQueryRetriever {
    base: Arc<dyn Retriever>,
    expander: Arc<dyn QueryExpander>,
    include_original: bool,
}

impl MultiQueryRetriever {
    pub fn new(base: Arc<dyn Retriever>, expander: Arc<dyn QueryExpander>) -> Self {
        Self {
            base,
            expander,
            include_original: true,
        }
    }

    /// Shortcut for an [`LlmQueryExpander`] with 3 rewrites.
    pub fn from_llm(base: Arc<dyn Retriever>, model: Arc<dyn ChatModel>) -> Self {
        Self::new(base, Arc::new(LlmQueryExpander::new(model)))
    }

    /// Also search with the original question (default true).
    pub fn include_original(mut self, on: bool) -> Self {
        self.include_original = on;
        self
    }
}

#[async_trait]
impl Retriever for MultiQueryRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<Document>> {
        let mut queries = Vec::new();
        if self.include_original {
            queries.push(query.to_string());
        }
        for q in self.expander.expand(query).await? {
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let lists =
            futures_util::future::try_join_all(queries.iter().map(|q| self.base.retrieve(q)))
                .await?;
        Ok(dedup(lists.into_iter().flatten().collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sinew_core::mock::ScriptedModel;

    struct Echo;

    #[async_trait]
    impl Retriever for Echo {
        async fn retrieve(&self, q: &str) -> Result<Vec<Document>> {
            Ok(vec![
                Document::new(format!("doc for {q}")),
                Document::new("shared"),
            ])
        }
    }

    #[test]
    fn parses_numbered_lines() {
        assert_eq!(
            parse_lines("1. first\n\n- second\n3) \"third\"\nfourth", 3),
            ["first", "second", "third"]
        );
    }

    #[tokio::test]
    async fn merges_results_of_all_queries() {
        let model = Arc::new(ScriptedModel::new([Message::assistant(
            "what is x?\nexplain x",
        )]));
        let r = MultiQueryRetriever::from_llm(Arc::new(Echo), model.clone());
        let got: Vec<String> = r
            .retrieve("x?")
            .await
            .unwrap()
            .into_iter()
            .map(|d| d.content)
            .collect();
        assert_eq!(
            got,
            [
                "doc for x?",
                "shared",
                "doc for what is x?",
                "doc for explain x"
            ]
        );
        assert!(model.requests()[0].messages[0].content.contains("x?"));
    }
}
