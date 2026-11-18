//! Retrieval for Sinew: the LangChain retrieval stack in Rust.
//!
//! - [`embeddings`]: the [`Embeddings`] trait, OpenAI-compatible
//!   (OpenAI, Ollama, LM Studio, vLLM, ...) and Gemini providers, a
//!   deterministic [`FakeEmbeddings`] and a [`CachedEmbeddings`] wrapper;
//! - [`loaders`]: text, Markdown, HTML, JSON / JSONL, CSV, PDF (feature
//!   `pdf`) and a glob-aware [`DirectoryLoader`];
//! - [`splitter`]: [`RecursiveCharacterTextSplitter`] (with code presets and
//!   token-approximate lengths) and [`MarkdownHeaderTextSplitter`];
//! - [`vectorstore`]: the [`VectorStore`] trait with metadata [`Filter`]s and
//!   MMR, an in-memory store, SQLite (default feature), Qdrant (`qdrant`) and
//!   pgvector (`pgvector`);
//! - [`retriever`]: vector, BM25, hybrid (reciprocal rank fusion),
//!   multi-query, contextual compression / LLM rerank and parent-document
//!   retrievers;
//! - [`retriever_tool`] exposes a retriever to agents and [`RagChain`]
//!   answers questions with sources.
//!
//! ```
//! use std::sync::Arc;
//! use sinew_rag::*;
//!
//! # tokio_test();
//! # fn tokio_test() { let rt = tokio::runtime::Runtime::new().unwrap(); rt.block_on(async {
//! let docs = vec![
//!     Document::new("Checkpointers make graph threads durable.").with_metadata("source", "a.md"),
//!     Document::new("Streaming yields tokens as they arrive.").with_metadata("source", "b.md"),
//! ];
//! let chunks = RecursiveCharacterTextSplitter::new(500, 50).split_documents(&docs);
//! let store = InMemoryVectorStore::new(FakeEmbeddings::new(256));
//! store.add_documents(chunks).await.unwrap();
//! let hits = store.similarity_search("durable threads", 1, None).await.unwrap();
//! assert_eq!(hits[0].source(), Some("a.md"));
//! # }); }
//! ```
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png",
    html_favicon_url = "https://raw.githubusercontent.com/roanbrasil/sinew/main/docs/assets/logo-128.png"
)]

mod chain;
mod document;
pub mod embeddings;
mod error;
pub mod loaders;
pub mod retriever;
pub mod splitter;
mod text;
mod tool;
pub mod vectorstore;

pub use chain::{RagAnswer, RagChain};
pub use document::{Document, Metadata};
pub use embeddings::{
    embeddings_from_spec, CachedEmbeddings, EmbeddingCache, Embeddings, FakeEmbeddings,
    GeminiEmbeddings, InMemoryEmbeddingCache, OpenAiEmbeddings,
};
pub use error::{Error, Result};
#[cfg(feature = "pdf")]
pub use loaders::PdfLoader;
pub use loaders::{
    html_to_text, CsvLoader, DirectoryLoader, HtmlLoader, JsonLoader, Loader, MarkdownLoader,
    TextLoader,
};
pub use retriever::{
    reciprocal_rank_fusion, Bm25Retriever, ContextualCompressionRetriever, DocStore,
    DocumentCompressor, EmbeddingsFilter, EnsembleRetriever, InMemoryDocStore, LlmQueryExpander,
    LlmReranker, MultiQueryRetriever, ParentDocumentRetriever, QueryExpander, Retriever,
    SearchType, VectorStoreRetriever,
};
pub use splitter::{
    approx_token_len, char_len, KeepSeparator, Language, MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter, TextSplitter,
};
pub use text::english_tokenize;
pub use tool::{format_documents, retriever_tool, RetrieverTool};
#[cfg(feature = "pgvector")]
pub use vectorstore::PgVectorStore;
#[cfg(feature = "qdrant")]
pub use vectorstore::QdrantVectorStore;
#[cfg(feature = "sqlite")]
pub use vectorstore::SqliteVectorStore;
pub use vectorstore::{
    maximal_marginal_relevance, Filter, InMemoryVectorStore, ScoredDocument, VectorStore,
};

/// Entry points for the fuzz targets in `fuzz/`: the loaders' parsers on
/// in-memory input. Not public API: no stability guarantees.
#[doc(hidden)]
pub mod __fuzz {
    use crate::{
        loaders::{CsvLoader, JsonLoader},
        Document, Metadata, Result,
    };

    /// The `<title>` an [`HtmlLoader`](crate::HtmlLoader) would store.
    pub fn html_title(html: &str) -> Option<String> {
        crate::loaders::html::html_title(html)
    }

    /// What a [`MarkdownLoader`](crate::MarkdownLoader) does with a file:
    /// the body and the front matter / title metadata.
    pub fn markdown(text: &str) -> (String, Metadata) {
        let mut metadata = Metadata::new();
        let body = crate::loaders::parse_markdown(text, &mut metadata);
        (body, metadata)
    }

    /// Runs `loader` on CSV `bytes` instead of its file.
    pub fn csv(loader: &CsvLoader, bytes: &[u8]) -> Result<Vec<Document>> {
        loader.documents_from_bytes(bytes)
    }

    /// Runs `loader` on `text` instead of its file.
    pub fn json(loader: &JsonLoader, text: &str) -> Result<Vec<Document>> {
        loader.documents(text)
    }
}
