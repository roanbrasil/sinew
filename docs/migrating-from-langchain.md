# Migrating from LangChain / LangGraph / LangSmith

This guide is for people who know the Python stack and want to write the same programs with Sinew.
It has three parts:

1. a [concept map](#concept-map) from LangChain, LangGraph and LangSmith names to Sinew types;
2. [side-by-side snippets](#side-by-side-patterns) for the ten most common patterns;
3. [what is intentionally different](#what-is-intentionally-different) and why.

All Rust snippets assume the umbrella crate:

```toml
[dependencies]
sinew = { version = "0.1", features = ["full"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
futures-util = "0.3"
schemars = "1"   # only for snippet 3 (schemas derived from Rust types)
```

`features = ["full"]` is the short path; the [feature table](../README.md#cargo-features-umbrella-crate)
lists the smaller ones (`rag`, `eval`, `http-trace`, `sqlite`, ...). Every Rust snippet below
compiles as written (they were checked with `cargo check` against the workspace); the ones that
call a model need that model at runtime.

## Concept map

Crate names in parentheses. With the umbrella crate, `sinew_core` is `sinew::core`,
`sinew_graph` is `sinew::graph`, `sinew_rag` is `sinew::rag`, and so on; the most used items are
in `sinew::prelude`.

### LangChain

| LangChain (Python) | Sinew | Notes |
|---|---|---|
| `init_chat_model("provider:model")` | `providers::from_spec("provider:model")` (`sinew-core`) | Returns `Arc<dyn ChatModel>`. 24 providers, local and hosted; see [providers.md](providers.md). `providers::catalog()` lists them. |
| `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`, `ChatOllama` | `OpenAiCompat`, `Anthropic`, `Gemini`, `OpenAiCompat::ollama(..)` (`sinew-core`) | Any OpenAI-compatible server goes through `OpenAiCompat::new(base_url, model)`. |
| `BaseChatModel`, `.invoke()`, `.stream()` | `ChatModel` trait: `chat(ChatRequest)`, `chat_stream(ChatRequest)` (`sinew-core`) | Implement `ChatModel` for a custom model ([sample 07](../samples/07-custom-model/src/main.rs)). |
| `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage` | `Message::system`, `Message::user`, `Message::assistant`, `Message::tool_result` with `Role` (`sinew-core`) | Multimodal parts are `ContentPart` (text, image, PDF, audio); see [multimodal.md](multimodal.md). |
| `AIMessage.tool_calls`, `usage_metadata` | `Message::tool_calls: Vec<ToolCall>`, `ChatResponse::usage: Usage` | `Usage::cost(model)` prices a call. |
| `@tool` decorator | `#[sinew::tool]` on a sync or async `fn` (`sinew-macros`, re-exported) | Name from the function, description from the doc comment, schema from the argument types; `#[state]` injects application state. See [tools.md](tools.md). |
| `StructuredTool.from_function`, `BaseTool` subclass | `FnTool`, `AsyncFnTool`, `TypedTool`, or `impl Tool` (`sinew-core`) | |
| list of tools, `bind_tools` | `ToolBox` / `tools![a, b]`; tools travel in `ChatRequest` | `react_agent` and `ReactAgent` bind them for you. |
| `model.with_structured_output(Schema)` | `StructuredOutput::new(name, schema)` / `StructuredOutput::for_type::<T>()` then `.generate::<T>(model, request)` (`sinew-core`) | Tool mode by default, `StructuredMode::Json` on request; invalid output is retried with the validation error. |
| `PromptTemplate`, `ChatPromptTemplate.from_messages` | `PromptTemplate::new`, `ChatPromptTemplate::from_messages` (`sinew-core`) | Missing variables are errors, not empty strings. |
| LCEL `prompt \| model \| parser`, `Runnable` | `Runnable<I, O>` trait and `RunnableExt::pipe` (`sinew-core`) | `invoke`, `batch`, `stream`, `transform`. See [runnables.md](runnables.md). |
| `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, `RunnableBranch` | `runnable::RunnableMap`, `runnable::passthrough()`, `runnable::lambda` / `lambda_sync`, `runnable::RunnableBranch` | |
| `StrOutputParser`, `JsonOutputParser`, `CommaSeparatedListOutputParser` | `StrOutputParser`, `JsonOutputParser<T>`, `ListOutputParser::comma()` (`sinew-core`) | `JsonOutputParser<T>` parses into any `Deserialize` type. |
| `.with_retry()`, `.with_fallbacks()`, rate limiter, `set_llm_cache` | `ChatModelExt::with_retry(RetryPolicy)`, `with_fallbacks`, `with_rate_limit(RateLimiter)`, `with_timeout`, `with_cache(InMemoryCache)` (`sinew-core`) | `SqliteCache` behind the `sqlite-cache` feature. See [resilience.md](resilience.md). |
| Document loaders | `TextLoader`, `DirectoryLoader`, Markdown, HTML, JSON, CSV and PDF loaders (`sinew-rag`) | See [rag.md](rag.md). |
| Text splitters | `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`, ... (`sinew-rag`) | |
| Embeddings | `Embeddings` trait, `embeddings_from_spec`, `OpenAiEmbeddings`, `CachedEmbeddings` (`sinew-rag`) | |
| Vector stores | `VectorStore` trait: `InMemoryVectorStore`, SQLite, pgvector, Qdrant (`sinew-rag`) | Metadata filters with `Filter`. |
| Retrievers, `EnsembleRetriever`, `MultiQueryRetriever`, `ContextualCompressionRetriever` | `Retriever` trait, `VectorStoreRetriever`, `Bm25Retriever`, `EnsembleRetriever`, `MultiQueryRetriever`, `ContextualCompressionRetriever` (`sinew-rag`) | |
| `create_retrieval_chain` / RAG chain | `RagChain::new(model, retriever)` (`sinew-rag`) | Returns the answer and the source documents. |
| `create_retriever_tool` | `retriever_tool(retriever, name, description)` (`sinew-rag`) | |
| `langchain-mcp-adapters` | `McpClient::connect_stdio` / `connect_http` then `.toolbox()` (`sinew-mcp`) | `McpServer` serves a `ToolBox` or an agent. See [mcp.md](mcp.md). |

### LangGraph

| LangGraph (Python) | Sinew | Notes |
|---|---|---|
| `StateGraph(State)` with a `TypedDict` | `StateGraph::<S>::new()` where `S: Clone + Serialize + DeserializeOwned` (`sinew-graph`) | State is a Rust struct. |
| `add_node`, `add_edge`, `START`, `END`, `set_entry_point` | `add_node`, `add_edge`, `START`, `END`, `set_entry_point` | Nodes are `async fn(S) -> Result<S, BoxError>`. |
| node with `config` / `runtime` argument | `add_node_with_context(name, \|s, ctx: NodeContext\| ...)` | `ctx` gives `interrupt`, `emit`, `emit_token`, `store()`. |
| `add_conditional_edges(src, router, path_map)` | `add_conditional_edges(src, router, targets)`; `add_conditional_edges_multi` for several targets | The target list is checked by `compile()`. |
| `Send("node", arg)` (map-reduce) | `SendTo::new("node", state)` with `add_send_edges` | Also `add_fan_out` for static parallel branches. See [fan-out.md](fan-out.md). |
| `Annotated[list, operator.add]` reducers | `StateGraph::with_reducer` / `with_reducer_from_base` | A `#[derive(State)]` with per-field reducers is planned ([#5](https://github.com/roanbrasil/sinew/issues/5)). |
| `Command(update=..., goto=...)` | `Command::goto(..).with_update(state)`, `with_json_update`, `to_parent()`, via `add_command_node` | See [commands-and-interrupts.md](commands-and-interrupts.md). |
| `interrupt(value)` | `ctx.interrupt::<T>(value)?` inside `add_node_with_context` | Returns the resume value on the second run. |
| `Command(resume=...)` | `Command::resume(value)` with `run_with`, or `graph.resume(thread, value)` | |
| `interrupt_before` / `interrupt_after` at compile | `CompiledGraph::interrupt_before([..])` / `interrupt_after([..])` | |
| `InMemorySaver` / `MemorySaver` | `MemoryCheckpointer` (`sinew-graph`) | |
| `SqliteSaver`, `PostgresSaver` | `SqliteCheckpointer::connect`, `PostgresCheckpointer::connect` (`sinew-checkpoint`, features `sqlite` / `postgres`) | See [checkpointing.md](checkpointing.md). |
| `config={"configurable": {"thread_id": ...}}` | the `thread_id` argument of `run`, `stream`, `get_state`, or `RunConfig::new(thread)` | |
| `InMemoryStore`, `PostgresStore`, `store.search` | `InMemoryStore`, `SqliteStore` (`sinew-checkpoint`), `Store` trait with `put` / `get` / `search(SearchQuery)` | Attach with `with_store`; see [store.md](store.md). |
| subgraphs (compiled graph as a node) | `add_subgraph(name, compiled)`; `add_subgraph_mapped` for a different state type | See [subgraphs.md](subgraphs.md). |
| `get_state`, `get_state_history`, `update_state` | `get_state`, `get_state_history`, `get_state_at`, `update_state`, `update_state_with` | |
| replay / fork from a checkpoint | `run_with(RunConfig::new(t).at_checkpoint(id), RunInput::Continue)`; fork with `update_state_with(RunConfig .. at_checkpoint(id), state, None)` | See [time-travel.md](time-travel.md). |
| `stream_mode="values" \| "updates" \| "messages" \| "custom" \| "debug"`, `subgraphs=True` | `stream_with(thread, input, StreamOptions::new([StreamMode::..]).with_subgraphs(true))` yielding `GraphEvent` | `StreamMode::{Values, Updates, Messages, Custom, Tasks, Debug}`; see [streaming.md](streaming.md). |
| `RetryPolicy`, `CachePolicy` on nodes | `set_node_retry`, `set_node_cache`, `set_node_timeout` | |
| `recursion_limit` | `CompiledGraph::with_recursion_limit` | |
| `graph.get_graph().draw_mermaid()` | `CompiledGraph::to_mermaid()`, `to_dot()`, `view()` | See [visualization.md](visualization.md). |
| `create_react_agent` / `create_agent` | `react_agent(model, tools)` or the `ReactAgent` builder (`system_prompt`, `response_format`, hooks, `max_iterations`, `loop_guard`) | State is `AgentState { messages }`. |
| `langgraph-supervisor` `create_supervisor` | `Supervisor::new(model).agent(name, description, graph)` or `create_supervisor(model, agents)` | See [multi-agent.md](multi-agent.md). |
| `langgraph-swarm` `create_swarm`, `create_handoff_tool` | `Swarm`, `create_swarm`, `Handoff` / `create_handoff_tool` | |
| Deep Agents (`deepagents`) | `sinew_harness::Agent::builder()` (`sinew-harness`, feature `harness`) | File, shell, HTTP and todo tools, subagents, permissions, compaction, budgets. See [harness.md](harness.md). |
| LangGraph Platform / LangServe | `sinew_serve::Server` and `ServeClient` (`sinew-serve`, feature `serve`) | Assistants, threads, runs, SSE streaming, plus `/v1/chat/completions`. See [serve.md](serve.md). |

### LangSmith

| LangSmith (Python) | Sinew | Notes |
|---|---|---|
| `LANGSMITH_TRACING=true`, `@traceable`, run tree | `Tracer` with sinks; `tracer.start(name, RunKind, parent, inputs)` returns a `RunHandle` (`sinew-trace`) | Graphs and agents trace themselves after `.with_tracer(tracer)`. |
| LangSmith backend | `HttpSink` to a `sinew-scope serve` instance (feature `http-trace`); `JsonlSink`, `MemorySink`, `OtlpSink` | See [scope.md](scope.md) and [otel.md](otel.md). |
| LangSmith UI (traces, monitoring, feedback, annotation queues) | `sinew-scope` binary (self-hosted, one SQLite file) | `ScopeClient::create_feedback`, `create_queue`; see [monitoring.md](monitoring.md). |
| Datasets and examples | `Dataset`, `Example` (`sinew-eval`, feature `eval`) | JSONL import/export. |
| Evaluators, LLM-as-judge | `Evaluator` trait: `ExactMatch`, `Contains`, `RegexMatch`, `JsonFieldEquals`, `NumericTolerance`, `LlmJudge` | See [evaluation.md](evaluation.md). |
| `evaluate(target, data=..., evaluators=...)`, experiments | `evaluate(target, &dataset, &evaluators, concurrency)` returns an `ExperimentReport`; `evaluate_pairwise` for A/B | Upload with `ScopeClient::upload_experiment`. |
| Prompt hub (`client.push_prompt`, `pull_prompt`) | `PromptHub::push(PromptPush)`, `pull("name@tag")`, `pull_version` (`sinew-eval`) | Served by `sinew-scope`; see [prompt-hub.md](prompt-hub.md). |

## Side-by-side patterns

Each pattern shows the Python version first and the Rust version below it.

### 1. First chat call

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-4-5")
reply = model.invoke([
    ("system", "Answer in one sentence."),
    ("user", "Why write LLM infrastructure in Rust?"),
])
print(reply.content)
print(reply.usage_metadata)
```

```rust
use sinew::prelude::*;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = from_spec("anthropic:claude-sonnet-5-5")?;
    let reply = model
        .chat(
            ChatRequest::new(vec![
                Message::system("Answer in one sentence."),
                Message::user("Why write LLM infrastructure in Rust?"),
            ])
            .with_temperature(0.2),
        )
        .await?;
    println!("{}", reply.message.content);
    println!("{} tokens in, {} out", reply.usage.input_tokens, reply.usage.output_tokens);
    Ok(())
}
```

### 2. Tool-calling agent

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def unit_price(product: str) -> float:
    """Returns the unit price in dollars of a product."""
    return {"keyboard": 89.0, "mouse": 35.0}.get(product.lower(), 0.0)

agent = create_react_agent(model, [unit_price])
out = agent.invoke({"messages": [("user", "How much do 3 keyboards cost?")]})
print(out["messages"][-1].content)
```

```rust
use sinew::prelude::*;

/// Returns the unit price in dollars of a product.
#[tool]
fn unit_price(
    /// Product name, e.g. "keyboard".
    product: String,
) -> f64 {
    match product.to_lowercase().as_str() {
        "keyboard" => 89.0,
        "mouse" => 35.0,
        _ => 0.0,
    }
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = from_spec("ollama:qwen3:4b")?;
    let agent = react_agent(model, tools![unit_price])?;
    let out = agent
        .invoke(AgentState::new(vec![Message::user("How much do 3 keyboards cost?")]))
        .await?;
    println!("{}", out.last().map(|m| m.content.as_str()).unwrap_or_default());
    Ok(())
}
```

### 3. Structured output

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str

extractor = model.with_structured_output(Invoice)
invoice = extractor.invoke("Northwind Supplies billed us $1,740.00 USD.")
print(invoice.vendor, invoice.total)
```

```rust
use schemars::JsonSchema;
use serde::Deserialize;
use sinew::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct Invoice {
    vendor: String,
    total: f64,
    /// ISO 4217 code.
    currency: String,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = from_spec("openai:gpt-4.1-mini")?;
    let invoice: Invoice = StructuredOutput::for_type::<Invoice>()
        .with_description("Record the invoice fields")
        .generate(
            model.as_ref(),
            ChatRequest::new(vec![Message::user(
                "Northwind Supplies billed us $1,740.00 USD.",
            )]),
        )
        .await?;
    println!("{} {} {}", invoice.vendor, invoice.total, invoice.currency);
    Ok(())
}
```

`for_type` needs the `schemars` feature (included in `full`) and `schemars` in your own
`Cargo.toml` for the derive. Without it, pass a JSON Schema: `StructuredOutput::new("invoice", json!({...}))`.

### 4. RAG chain

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

docs = DirectoryLoader("docs", glob="*.md").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
store = InMemoryVectorStore.from_documents(chunks, OllamaEmbeddings(model="bge-m3"))
retriever = store.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)
print(chain.invoke("How do I persist graph threads?"))
```

```rust
use std::sync::Arc;

use sinew::prelude::*;
use sinew::rag::{
    embeddings_from_spec, DirectoryLoader, InMemoryVectorStore, Loader, RagChain,
    RecursiveCharacterTextSplitter, TextSplitter, VectorStore, VectorStoreRetriever,
};

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let docs = DirectoryLoader::new("docs").glob("*.md").load()?;
    let chunks = RecursiveCharacterTextSplitter::new(1000, 150).split_documents(&docs);

    let store = InMemoryVectorStore::new(embeddings_from_spec("ollama:bge-m3")?);
    store.add_documents(chunks).await?;
    let retriever = VectorStoreRetriever::new(Arc::new(store)).with_k(4);

    let rag = RagChain::new(from_spec("ollama:qwen3:4b")?, Arc::new(retriever));
    let out = rag.invoke("How do I persist graph threads?").await?;
    println!("{}", out.answer);
    for doc in out.cited() {
        println!("  source: {}", doc.metadata["relative_path"]);
    }
    Ok(())
}
```

`RagChain` formats numbered passages, asks for citations and returns the sources. To assemble
the chain yourself, call `retriever.retrieve(question)` and pipe a `PromptTemplate` into the model
as in pattern 5. [Sample 15](../samples/15-rag/src/main.rs) adds BM25 + vector hybrid retrieval.

### 5. LCEL pipeline

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are terse."),
    ("user", "Tell me a joke about {topic}"),
])
chain = prompt | model.with_retry(stop_after_attempt=3) | StrOutputParser()
print(chain.invoke({"topic": "Rust"}))
print(chain.batch([{"topic": "Go"}, {"topic": "Zig"}]))
```

```rust
use std::sync::Arc;

use sinew::core::runnable::vars;
use sinew::core::RetryPolicy;
use sinew::prelude::*;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model: Arc<dyn ChatModel> = Arc::new(
        OpenAiCompat::ollama("qwen3:4b").with_retry(RetryPolicy::new(3)),
    );
    let chain = ChatPromptTemplate::from_messages([
        (Role::System, "You are terse."),
        (Role::User, "Tell me a joke about {topic}"),
    ])?
    .pipe(model)
    .pipe(StrOutputParser);

    println!("{}", chain.invoke(vars([("topic", "Rust")])).await?);
    for joke in chain.batch(vec![vars([("topic", "Go")]), vars([("topic", "Zig")])]).await {
        println!("{}", joke?);
    }
    Ok(())
}
```

The pipe is checked at compile time: the output type of each step must be the input type of the
next. `RunnableMap` (`RunnableParallel`), `RunnableBranch` and `lambda` cover the other LCEL
building blocks; see [runnables.md](runnables.md) and [sample 16](../samples/16-runnables/src/main.rs).

### 6. Custom StateGraph with a conditional edge

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class Ticket(TypedDict):
    text: str
    category: str
    reply: str

def classify(state: Ticket):
    return {"category": "billing" if "invoice" in state["text"] else "support"}

def billing(state: Ticket):
    return {"reply": "Billing will contact you."}

def support(state: Ticket):
    return {"reply": "Support will contact you."}

g = StateGraph(Ticket)
g.add_node("classify", classify)
g.add_node("billing", billing)
g.add_node("support", support)
g.add_edge(START, "classify")
g.add_conditional_edges("classify", lambda s: s["category"], ["billing", "support"])
g.add_edge("billing", END)
g.add_edge("support", END)
graph = g.compile()
print(graph.invoke({"text": "Where is my invoice?"})["reply"])
```

```rust
use serde::{Deserialize, Serialize};
use sinew::prelude::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Ticket {
    text: String,
    category: String,
    reply: String,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let mut g = StateGraph::<Ticket>::new();
    g.add_node("classify", |mut s: Ticket| async move {
        s.category = if s.text.contains("invoice") { "billing" } else { "support" }.into();
        Ok(s)
    });
    g.add_node("billing", |mut s: Ticket| async move {
        s.reply = "Billing will contact you.".into();
        Ok(s)
    });
    g.add_node("support", |mut s: Ticket| async move {
        s.reply = "Support will contact you.".into();
        Ok(s)
    });
    g.set_entry_point("classify");
    g.add_conditional_edges(
        "classify",
        |s: &Ticket| s.category.clone(),
        ["billing", "support"],
    );
    g.add_edge("billing", END);
    g.add_edge("support", END);

    let graph = g.compile()?; // reports every wiring problem at once
    let out = graph
        .invoke(Ticket { text: "Where is my invoice?".into(), ..Default::default() })
        .await?;
    println!("{}", out.reply);
    Ok(())
}
```

Nodes take and return the whole state. The router returns the name of the next node; returning a
name that is not in the target list is a runtime error, and a target that is not a node is a
`compile()` error.

### 7. Human-in-the-loop: interrupt and resume

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

def approve(state):
    answer = interrupt({"question": "Send this?", "draft": state["body"]})
    return {"approved": answer["approved"]}

graph = g.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "email-42"}}
out = graph.invoke({"body": "The release slips to Friday."}, config)
print(out["__interrupt__"][0].value)
graph.invoke(Command(resume={"approved": True}), config)
```

```rust
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew::prelude::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Email {
    body: String,
    approved: bool,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let mut g = StateGraph::<Email>::new();
    g.add_node_with_context("approve", |mut s: Email, ctx: NodeContext| async move {
        // First run: pauses here. After resume: returns the human's answer.
        let answer: Value = ctx.interrupt(json!({ "question": "Send this?", "draft": s.body }))?;
        s.approved = answer["approved"].as_bool().unwrap_or(false);
        Ok(s)
    });
    g.set_entry_point("approve");
    g.add_edge("approve", END);
    let graph = g.compile()?.with_checkpointer(MemoryCheckpointer::new());

    let thread = "email-42";
    let input = Email { body: "The release slips to Friday.".into(), ..Default::default() };
    let out = graph.run(thread, Some(input)).await?;
    if let RunOutcome::Interrupted { interrupts, .. } = &out {
        println!("paused: {}", interrupts[0].value);
    }

    let done = graph
        .run_with(thread, Command::resume(json!({ "approved": true })))
        .await?
        .into_state();
    println!("approved = {}", done.approved);
    Ok(())
}
```

`graph.resume(thread, value)` is shorthand for `run_with(thread, Command::resume(value))`. For
static breakpoints use `.interrupt_before(["send"])` and continue with `graph.run(thread, None)`
([sample 05](../samples/05-human-in-the-loop/src/main.rs)). Swap `MemoryCheckpointer` for
`SqliteCheckpointer::connect("sqlite://threads.db")` to survive restarts.

### 8. Supervisor multi-agent

```python
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

researcher = create_react_agent(model, [lookup_population], name="researcher",
                                prompt="You look up facts. No math.")
math = create_react_agent(model, [add], name="math", prompt="You do arithmetic.")
graph = create_supervisor([researcher, math], model=model).compile()
out = graph.invoke({"messages": [("user", "Combined population of Montreal and Quebec City?")]})
```

```rust
use serde_json::{json, Value};
use sinew::graph::prebuilt::{ReactAgent, Supervisor};
use sinew::prelude::*;

/// Population of a city (2021 census).
#[tool]
fn lookup_population(city: String) -> Value {
    match city.to_lowercase().as_str() {
        "montreal" => json!(1_762_949),
        "quebec city" => json!(549_459),
        _ => json!(null),
    }
}

/// Adds two integers.
#[tool]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = from_spec("ollama:qwen3:4b")?;
    let researcher = ReactAgent::new(model.clone(), tools![lookup_population])
        .system_prompt("You look up facts. No math.")
        .compile()?;
    let math = ReactAgent::new(model.clone(), tools![add])
        .system_prompt("You do arithmetic.")
        .compile()?;
    let graph = Supervisor::new(model)
        .agent("researcher", "looks up city populations", researcher)
        .agent("math", "adds numbers exactly", math)
        .compile()?;

    let out = graph
        .invoke(AgentState::new(vec![Message::user(
            "Combined population of Montreal and Quebec City?",
        )]))
        .await?;
    println!("{}", out.last().map(|m| m.content.as_str()).unwrap_or_default());
    Ok(())
}
```

Each worker is an ordinary compiled graph running as a subgraph, so its events show up with a
namespace when streamed with `with_subgraphs(true)`. `Swarm` / `create_swarm` give the
peer-to-peer variant ([multi-agent.md](multi-agent.md), [sample 17](../samples/17-multi-agent/src/main.rs)).

### 9. Streaming tokens

```python
for chunk in model.stream("Explain Rust ownership in one sentence."):
    print(chunk.content, end="", flush=True)

for token, meta in agent.stream({"messages": [("user", "Hi")]}, stream_mode="messages"):
    print(token.content, end="")
```

```rust
use std::io::Write;

use futures_util::StreamExt;
use sinew::prelude::*;
use sinew::graph::StreamOptions;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = from_spec("ollama:qwen3:4b")?;

    // A model, token by token.
    let mut stream = model
        .chat_stream(ChatRequest::new(vec![Message::user(
            "Explain Rust ownership in one sentence.",
        )]))
        .await?;
    while let Some(event) = stream.next().await {
        if let StreamEvent::TextDelta { text } = event? {
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }

    // An agent graph, messages mode.
    let agent = react_agent(model, ToolBox::new())?;
    let mut events = agent.stream_with(
        "chat-1",
        AgentState::new(vec![Message::user("Hi")]),
        StreamOptions::new([StreamMode::Messages]),
    );
    while let Some(event) = events.next().await {
        match event {
            GraphEvent::Token { delta, .. } => print!("{delta}"),
            GraphEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }
    Ok(())
}
```

Dropping a graph stream cancels the run. See [streaming.md](streaming.md) for the other modes.

### 10. Tracing to a self-hosted UI

```python
# export LANGSMITH_TRACING=true
# export LANGSMITH_ENDPOINT=https://my-langsmith.example.com
# export LANGSMITH_PROJECT=my-agent
from langsmith import traceable

@traceable(run_type="chain")
def nightly_report(date: str):
    ...

agent.invoke({"messages": [("user", "Hi")]})  # traced automatically
```

```bash
cargo install sinew-scope
sinew-scope serve            # UI at http://127.0.0.1:7878/
```

```rust
use serde_json::json;
use sinew::prelude::*;
use sinew::trace::{HttpSink, HttpSinkConfig, RunKind};

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let tracer = Tracer::new().with_sink(HttpSink::new(
        HttpSinkConfig::new("http://127.0.0.1:7878").project("my-agent"),
    ));

    // Graphs and agents record graph -> node -> llm/tool runs.
    let agent = react_agent(from_spec("ollama:qwen3:4b")?, ToolBox::new())?
        .with_tracer(tracer.clone());
    agent.invoke(AgentState::new(vec![Message::user("Hi")])).await?;

    // Your own code: the equivalent of @traceable.
    let mut run = tracer.start("nightly-report", RunKind::Chain, None, json!({ "date": "2026-11-15" }));
    run.set_metadata("owner", json!("data-team"));
    run.finish(json!({ "rows": 1280 }));

    // HttpSink batches; deliver the last batch before exit.
    tokio::task::spawn_blocking(move || tracer.flush()).await?;
    Ok(())
}
```

The same `Tracer` can also write JSONL (`JsonlSink`) or export OpenTelemetry (`OtlpSink`) to
Langfuse, Phoenix, Jaeger or any collector ([otel.md](otel.md)). Evaluation experiments and the
prompt hub use the same server ([evaluation.md](evaluation.md), [prompt-hub.md](prompt-hub.md),
[sample 13](../samples/13-scope-and-evals/src/main.rs)).

## What is intentionally different

**Typed state instead of dictionaries.** LangGraph state is a `TypedDict` or Pydantic model that
nodes read and patch by key. In Sinew it is a Rust struct (`Clone + Serialize + DeserializeOwned`).
A misspelled field or a wrong type is a compile error, not a `KeyError` at step 40, and IDEs can
follow every read and write of a field. The `serde` bounds are what lets checkpointers, `Command`
JSON updates and subgraph events serialize the state.

**Whole-state nodes (`S -> S`).** A Python node returns a partial update that reducers merge into
the state. A Sinew node receives the state by value and returns the new state. For sequential
steps this is the whole story: there is nothing to merge and no implicit reducer to remember.
Merging only matters when branches run in parallel (`add_fan_out`, `add_send_edges`,
`add_conditional_edges_multi`); there you register one reducer with `with_reducer`, and
`compile()` refuses a graph that can fan out without one. Without a checkpointer the state is
moved between nodes, not cloned. When a partial update is the natural shape (for example from a
human), `Command::with_json_update` merges a JSON patch.

**Checks before the first token.** `compile()` validates the whole graph and reports every problem
at once: unknown edge targets, unreachable nodes, missing entry point, fan-out without a reducer.
Pipelines (`pipe`) are type-checked: `JsonOutputParser<City>` produces a `City`, and the next
step must accept a `City`. `#[sinew::tool]` derives the JSON Schema from the argument types
at compile time, and the tool's code receives typed arguments.

**Cargo features instead of extra packages.** Python splits the stack into `langchain-core`,
`langchain-community`, `langgraph-checkpoint-sqlite`, `langchain-mcp-adapters` and so on. Sinew
is one workspace with one version; heavy dependencies (SQLite, Postgres, Qdrant, PDF parsing,
OpenTelemetry protobuf, the HTTP server) sit behind features of the umbrella crate, so a binary
contains only what it uses. Code that needs a feature you did not enable does not compile, instead
of failing with `ImportError` at runtime.

**Errors are values.** Every fallible call returns `Result`. Model and tool errors are
`sinew_core::Error` variants (`Error::Tool`, `Error::Timeout`, `Error::Structured`,
`Error::Provider`, `Error::RateLimited`) that retry policies inspect to decide what is transient; graph errors are `GraphError`;
nodes return `Result<S, BoxError>` so `?` works on any error type. A failed tool call is reported
back to the model as a tool message instead of unwinding the agent. Tracing never fails the
application: sink errors are swallowed and unfinished runs are recorded as failed.

**Async without a global runtime.** Everything is `async` on Tokio. There is no `invoke` /
`ainvoke` pair: one `async fn` serves both, and `batch` runs inputs concurrently with a bound.

**No implicit configuration.** There are no environment variables that switch tracing on and no
global LLM cache. A tracer is attached with `.with_tracer(tracer)`, a cache with
`.with_cache(cache)`, a checkpointer with `.with_checkpointer(cp)`. Provider API keys are still read
from the usual variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...) by `from_spec`.
