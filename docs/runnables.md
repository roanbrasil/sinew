# Runnables

`sinew_core::runnable` is Sinew's take on LangChain Expression Language
(LCEL): small async steps that compose into pipelines. A
`Runnable<I, O>` turns an `I` into an `O`:

```rust
#[async_trait]
pub trait Runnable<I, O>: Send + Sync {
    async fn invoke(&self, input: I) -> Result<O>;
    async fn batch(&self, inputs: Vec<I>) -> Vec<Result<O>>;               // bounded concurrency
    async fn batch_with_concurrency(&self, inputs: Vec<I>, n: usize) -> Vec<Result<O>>;
    async fn stream(&self, input: I) -> Result<RunnableStream<O>>;         // output chunks
    async fn transform(&self, input: RunnableStream<I>) -> Result<RunnableStream<O>>;
}
```

Only `invoke` is required. The other methods have default implementations.
Run `cargo run -p sample-16-runnables` (offline) or add `--ollama` to see
everything on this page in one program.

## What is already a runnable

| Type | Input | Output |
|---|---|---|
| any `ChatModel` (including `Arc<dyn ChatModel>` and every wrapper) | anything `Into<ChatRequest>`: `ChatRequest`, `Vec<Message>`, `Message`, `String`, `&'static str` (one user message) | `ChatResponse` |
| `PromptTemplate` | anything `TemplateVars`: `Vars` (`HashMap<String, String>`), `HashMap<&'static str, String>`, a JSON object | `String` |
| `ChatPromptTemplate` | same as `PromptTemplate` | `ChatRequest` |
| `StructuredOutput::runnable::<T, _>(model)` | anything `Into<ChatRequest>` | `T` (schema-validated, auto-repaired) |
| `StrOutputParser` | `ChatResponse`, `Message` or `String` | `String` |
| `JsonOutputParser<T = Value>` | same | `T` (fences and prose stripped) |
| `ListOutputParser` (`comma()`, `lines()`, auto) | same | `Vec<String>` |
| `lambda(async fn)` / `lambda_sync(fn)` | your closure's argument | your closure's output |
| `RunnablePassthrough` | `I` | `I` |

Each type has one generic `Runnable` impl, so the compiler can always infer
the input and output types along a chain.

### Compile-time checked prompts

`prompt!` and `chat_prompt!` (feature `macros`) check a template's variables
at compile time and convert into `PromptTemplate` / `ChatPromptTemplate`, so
they slot into the same chains:

```rust
use sinew::prelude::*;

// Render directly: a missing `.lang(...)` is a compile error.
let text = prompt!("Summarize {text} in {lang}").text(doc).lang("pt-BR").format();

// Or convert and compose like any PromptTemplate.
let chain = PromptTemplate::from(prompt!("Tell me a joke about {topic}"))
    .pipe(model.clone())
    .pipe(StrOutputParser);
```

See [Prompt templates](prompts.md) for the builder, file templates and chat
prompts.

## Composing

`RunnableExt` adds combinators to every runnable:

```rust
use sinew_core::{runnable::*, *};

// prompt | model | parser
let chain = PromptTemplate::new("Tell me a joke about {topic}")?
    .pipe(model.clone())
    .pipe(StrOutputParser);
let joke = chain.invoke(vars([("topic", "Rust")])).await?;
```

| Combinator | LangChain | What it does |
|---|---|---|
| `a.pipe(b)` | `a \| b` | Feeds `a`'s output into `b`. |
| `a.map(f)` | `a \| RunnableLambda(f)` | Applies a sync function to the output. |
| `a.then(async_f)` | | Feeds the output into an async closure. |
| `parallel((a, b, ...))` | `RunnableParallel` | Runs 2 to 5 runnables concurrently on clones of the input and returns a tuple. The first error cancels the rest. |
| `RunnableMap::new().with("k", r)...` | `RunnableParallel({...})` | Same, but the outputs (anything `Serialize`) go into a JSON object under their keys. |
| `RunnablePassthrough::assign().with("k", r)` | `RunnablePassthrough.assign` | Keeps the input JSON object and adds keys computed concurrently from it. |
| `RunnableBranch::new(default).when(pred, r)` | `RunnableBranch` | Routes to the first branch whose predicate matches. |
| `a.retry(RetryPolicy)` | `with_retry` | Retries failed invocations (input must be `Clone`). |
| `a.fallbacks([b.boxed(), ...])` | `with_fallbacks` | Tries the others in order when `a` fails. |
| `a.boxed()` | | Type-erases into a cloneable `BoxRunnable<I, O>`. |

Parallel branches feeding a final prompt:

```rust
let facts = PromptTemplate::new("One fact about {topic}.")?.pipe(model.clone()).pipe(StrOutputParser);
let joke  = PromptTemplate::new("A one-line joke about {topic}.")?.pipe(model.clone()).pipe(StrOutputParser);

let pipeline = RunnableMap::new()
    .with("facts", facts)
    .with("joke", joke)
    .with("topic", lambda_sync(|v: Value| Ok(v["topic"].clone())))
    .pipe(PromptTemplate::new("Intro to {topic}.\nFact: {facts}\nJoke: {joke}")?)
    .pipe(model.clone())
    .pipe(StrOutputParser);

let intro = pipeline.invoke(json!({"topic": "Rust"})).await?;
```

Typed outputs at the end of a chain:

```rust
#[derive(Deserialize)]
struct City { name: String, population: u64 }

let chain = PromptTemplate::new("The largest city in {country}?")?
    .pipe(StructuredOutput::new("city", schema).runnable::<City, _>(model.clone()));
let city = chain.invoke(json!({"country": "Brazil"})).await?;
```

## Batch

`batch` runs inputs concurrently, at most `DEFAULT_BATCH_CONCURRENCY` (8) at a
time, and returns results in input order. Each input gets its own
`Result`, so one failure doesn't discard the others.
`batch_with_concurrency(inputs, n)` sets the bound. Concurrency comes from
polling futures together on the current task, so nothing is spawned and no
particular runtime is required.

## Streaming

`stream` yields output *chunks*:

- A chat model streams `ChatResponse` chunks. Each chunk holds a text delta,
  a complete tool call, the usage, or the final model name.
- `StrOutputParser` turns each incoming chunk into its text delta as soon as
  it arrives.
- `Pipe::stream` calls `first.stream` and passes the chunk stream to
  `second.transform`.

So this streams tokens:

```rust
let mut s = PromptTemplate::new("Explain {x}")?
    .pipe(model)
    .pipe(StrOutputParser)
    .stream(vars([("x", "ownership")]))
    .await?;
while let Some(delta) = s.next().await {
    print!("{}", delta?);
}
```

A step that needs its whole input, such as a JSON parser, a closure or a
prompt, uses the default `transform`: it merges the incoming chunks and then
runs. Merging works like this:

- `String` chunks are concatenated.
- `ChatResponse` and `Message` chunks are folded into one value: text
  concatenated, tool calls collected, usage and model name kept.
- For any other type, the last chunk wins.

## Performance notes

- Combinators are generic structs, so `prompt.pipe(model).pipe(parser)` is a
  concrete type with static dispatch between steps. The only allocation per
  step is the boxed future that `async_trait` creates for `invoke`, which is
  noise next to an HTTP round trip.
- Use `BoxRunnable` where you need type erasure, such as heterogeneous lists,
  `RunnableBranch` arms or struct fields. It is an `Arc<dyn Runnable>`, so
  cloning it is cheap.
- `parallel` and `RunnableMap` clone the input once per branch. Wrap large
  inputs in an `Arc`, or pass a JSON value.

## Writing your own

```rust
struct Shout;

#[async_trait]
impl Runnable<String, String> for Shout {
    async fn invoke(&self, s: String) -> Result<String> {
        Ok(s.to_uppercase())
    }
}

let chain = PromptTemplate::new("Say hi to {name}")?.pipe(model).pipe(StrOutputParser).pipe(Shout);
```

Override `transform` if your step can work on chunks incrementally.
