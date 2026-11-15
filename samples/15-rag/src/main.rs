//! Retrieval-augmented generation over this repository's `docs/*.md`:
//! load → split by Markdown headers and size → embed → hybrid retrieval
//! (BM25 + vectors, reciprocal rank fusion) → answer with cited sources.
//! Then the same retriever is handed to a ReAct agent as a tool.
//!
//! ```bash
//! cargo run -p sample-15-rag                                   # offline: fake embeddings + scripted model
//! cargo run -p sample-15-rag -- --ollama                       # Ollama chat + Ollama embeddings (bge-m3)
//! cargo run -p sample-15-rag -- --model gemini:gemini-2.5-flash --embed gemini:gemini-embedding-001
//! cargo run -p sample-15-rag -- --ollama "How do I trace a run?"
//! ```
//!
//! Embeddings: `--embed provider:model` (anything `embeddings_from_spec`
//! accepts), else Ollama `SINEW_EMBED_MODEL` (default `bge-m3`) with
//! `--ollama`, else deterministic offline `FakeEmbeddings`.

use std::{path::PathBuf, sync::Arc, time::Instant};

use sample_common::{is_live, pick_model, BoxError};
use serde_json::json;
use sinew::prelude::*;
use sinew_rag::{
    embeddings_from_spec, english_tokenize, retriever_tool, Bm25Retriever, CachedEmbeddings,
    DirectoryLoader, Document, Embeddings, EnsembleRetriever, FakeEmbeddings, InMemoryVectorStore,
    Loader, MarkdownHeaderTextSplitter, RagChain, RecursiveCharacterTextSplitter, Retriever,
    TextSplitter, VectorStore, VectorStoreRetriever,
};

const DEFAULT_QUESTION: &str = "How do I persist graph threads so they survive a restart?";

fn arg_after(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// The first argument that is neither a flag nor a flag's value.
fn question() -> String {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut skip = false;
    for a in &args {
        if skip {
            skip = false;
            continue;
        }
        if a == "--model" || a == "--embed" {
            skip = true;
            continue;
        }
        if !a.starts_with("--") {
            return a.clone();
        }
    }
    DEFAULT_QUESTION.to_string()
}

fn pick_embeddings() -> Result<Arc<dyn Embeddings>, BoxError> {
    if let Some(spec) = arg_after("--embed") {
        return Ok(embeddings_from_spec(&spec)?);
    }
    if std::env::args().any(|a| a == "--ollama") {
        let model = std::env::var("SINEW_EMBED_MODEL").unwrap_or_else(|_| "bge-m3".into());
        return Ok(embeddings_from_spec(&format!("ollama:{model}"))?);
    }
    Ok(Arc::new(FakeEmbeddings::new(512)))
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let docs_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs");

    // 1. Load every Markdown file in docs/.
    let docs = DirectoryLoader::new(&docs_dir).glob("*.md").load()?;

    // 2. One chunk per section, then bound the size.
    let by_header =
        MarkdownHeaderTextSplitter::new([("#", "title"), ("##", "section")]).split_documents(&docs);
    let chunks = RecursiveCharacterTextSplitter::new(1200, 150).split_documents(&by_header);
    // Stable ids (re-indexing replaces instead of duplicating), a short
    // `source`, and the section path prepended so it is searchable too.
    let chunks: Vec<Document> = chunks
        .into_iter()
        .enumerate()
        .map(|(i, mut c)| {
            let src = format!(
                "docs/{}",
                c.metadata["relative_path"].as_str().unwrap_or("?")
            );
            let heading: Vec<&str> = ["title", "section"]
                .iter()
                .filter_map(|k| c.metadata.get(*k).and_then(|v| v.as_str()))
                .collect();
            c.content = format!("{}\n{}", heading.join(" › "), c.content);
            c.id = Some(format!("{src}#{i}"));
            c.metadata.insert("source".into(), src.into());
            c
        })
        .collect();

    // 3. Embed (cached) into an in-memory store.
    let embeddings = CachedEmbeddings::new(pick_embeddings()?);
    println!(
        "Indexing {} chunks from {} files with `{}` ...",
        chunks.len(),
        docs.len(),
        embeddings.model()
    );
    let t = Instant::now();
    let store = InMemoryVectorStore::new(embeddings);
    store.add_documents(chunks.clone()).await?;
    println!("  indexed in {:.0?}\n", t.elapsed());
    let store: Arc<dyn VectorStore> = Arc::new(store);

    // 4. Hybrid retrieval: keywords + vectors, fused by reciprocal rank.
    let hybrid: Arc<dyn Retriever> = Arc::new(
        EnsembleRetriever::new(vec![
            Arc::new(
                Bm25Retriever::from_documents_with_tokenizer(chunks, english_tokenize).with_k(6),
            ),
            Arc::new(VectorStoreRetriever::new(store).with_k(6)),
        ])
        .with_k(4),
    );

    // 5. RAG chain: retrieve, prompt with numbered passages, answer with citations.
    let model = pick_model(vec![
        Message::assistant(
            "Give the graph a durable checkpointer such as `SqliteCheckpointer` (or \
             `PostgresCheckpointer` for several hosts): every step is saved, so after a restart \
             you reconnect to the same database and call `graph.run(thread, None)` to resume [1].",
        ),
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "1".into(),
                name: "search_docs".into(),
                arguments: json!({"query": "stream tokens as they arrive"}),
            }],
        ),
        Message::assistant(
            "Use `chat_stream` (or the graph's streaming API) to receive `StreamEvent`s as the \
             provider produces them.",
        ),
    ])?;
    let question = question();
    let rag = RagChain::new(model.clone(), hybrid.clone()).with_max_chars(1500);
    let t = Instant::now();
    let out = rag.invoke(&question).await?;
    println!("Q: {question}\n");
    println!("A: {}\n", out.answer);
    println!("Sources ({:.0?}):", t.elapsed());
    for (i, d) in out.sources.iter().enumerate() {
        let file = d.metadata["relative_path"].as_str().unwrap_or("?");
        let section = d
            .metadata
            .get("section")
            .or_else(|| d.metadata.get("title"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let cited = if out.answer.contains(&format!("[{}]", i + 1)) {
            "  (cited)"
        } else {
            ""
        };
        println!("  [{}] {file} › {section}{cited}", i + 1);
    }

    // 6. The same retriever as an agent tool.
    let tools = ToolBox::new().with(
        retriever_tool(
            hybrid,
            "search_docs",
            "Searches the Sinew documentation and returns relevant passages",
        )
        .with_max_chars(600),
    );
    let agent = react_agent(model, tools)?;
    let state = agent
        .invoke(AgentState::new(vec![
            Message::system("Answer from the documentation. Search before answering."),
            Message::user("How can I show tokens to the user while the model is still writing?"),
        ]))
        .await?;
    println!("\nAgent:");
    for m in &state.messages[2..] {
        match m.role {
            Role::Assistant if m.has_tool_calls() => {
                for c in &m.tool_calls {
                    println!("  → {}({})", c.name, c.arguments);
                }
            }
            Role::Tool => {
                let first = m.content.lines().next().unwrap_or("");
                println!("  ← {} lines, first: {first}", m.content.lines().count());
            }
            _ => println!("  {}", m.content),
        }
    }
    if !is_live() {
        println!("\n(offline run: add --ollama or --model provider:model for real answers)");
    }
    Ok(())
}
