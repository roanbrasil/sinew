//! Runnable pipelines (the LCEL equivalent), end to end:
//!
//! 1. Resilience: a primary endpoint that is down, retried with backoff,
//!    then a fallback model takes over.
//! 2. A pipeline with two branches that run in parallel (`RunnableMap`),
//!    merged by a final prompt.
//! 3. Routing with `RunnableBranch`, run over a batch of inputs.
//! 4. Token streaming through `prompt | model | StrOutputParser`.
//! 5. Token usage and cost per model, plus what the same tokens would cost
//!    on a few hosted models.
//!
//! ```bash
//! cargo run -p sample-16-runnables                      # offline, scripted
//! cargo run -p sample-16-runnables -- --ollama          # local model
//! cargo run -p sample-16-runnables -- --model openai:gpt-4.1-mini
//! ```

use std::{
    collections::BTreeMap,
    io::Write,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures_util::StreamExt;
use sample_common::{pick_model, BoxError};
use serde_json::{json, Value};
use sinew::core::{
    providers::OpenAiCompat,
    runnable::{lambda_sync, vars, RunnableBranch, RunnableMap},
    stream::ChatStream,
    ChatModel, ChatModelExt, ChatRequest, ChatResponse, ListOutputParser, Message, PromptTemplate,
    Result, RetryPolicy, Runnable, RunnableExt, StrOutputParser, StreamEvent, Usage,
};

/// Wraps a model and adds up token usage per reported model name.
struct CostTracker {
    inner: Arc<dyn ChatModel>,
    totals: Arc<Mutex<BTreeMap<String, Usage>>>,
}

impl CostTracker {
    fn record(totals: &Mutex<BTreeMap<String, Usage>>, model: &str, usage: Usage) {
        *totals.lock().unwrap().entry(model.to_string()).or_default() += usage;
    }
}

#[async_trait]
impl ChatModel for CostTracker {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let resp = self.inner.chat(request).await?;
        Self::record(&self.totals, &resp.model, resp.usage);
        Ok(resp)
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let totals = self.totals.clone();
        let name = self.inner.name().to_string();
        let usage = Arc::new(Mutex::new(Usage::default()));
        let stream = self.inner.chat_stream(request).await?;
        Ok(stream
            .inspect(move |ev| match ev {
                Ok(StreamEvent::Usage { usage: u }) => *usage.lock().unwrap() = *u,
                Ok(StreamEvent::Done { model, .. }) => {
                    let m = if model.is_empty() { &name } else { model };
                    Self::record(&totals, m, *usage.lock().unwrap());
                }
                _ => {}
            })
            .boxed())
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), BoxError> {
    // What the offline model answers, in call order.
    let llm = pick_model(vec![
        Message::assistant("Rust's borrow checker prevents data races at compile time."),
        Message::assistant("Rust guarantees memory safety without a garbage collector, and its ownership model makes data races a compile error."),
        Message::assistant("Why did the Rust developer never get lost? They always followed the borrow checker."),
        Message::assistant("Rust gives you memory safety without a garbage collector, and the borrow checker turns data races into compile errors. As the joke goes, Rust developers never get lost: they always follow the borrow checker."),
        Message::assistant("#rustlang, #ownership, #memorysafety"),
        Message::assistant("Sunlight scatters off air molecules, and blue light scatters the most because of its short wavelength."),
        Message::assistant("Ownership: each value has one owner, and it is dropped when the owner goes out of scope."),
    ])?;
    let totals = Arc::new(Mutex::new(BTreeMap::new()));
    let tracked: Arc<dyn ChatModel> = Arc::new(CostTracker {
        inner: llm,
        totals: totals.clone(),
    });

    // 1. Resilience. Nothing listens on port 9, so every attempt fails fast
    //    with a connection error (retryable). After the retries, the
    //    fallback answers.
    println!("== 1. retry + fallback ==");
    let resilient = Arc::new(
        OpenAiCompat::new("http://127.0.0.1:9/v1", "primary-down")
            .with_timeout(Duration::from_secs(5))
            .with_retry(
                RetryPolicy::new(2)
                    .with_initial_backoff(Duration::from_millis(50))
                    .with_max_backoff(Duration::from_millis(200)),
            )
            .with_fallbacks([tracked.clone()]),
    );
    let t = Instant::now();
    let answer = PromptTemplate::new("In one sentence: what does {thing} prevent?")?
        .pipe(resilient)
        .pipe(StrOutputParser)
        .invoke(vars([("thing", "Rust's borrow checker")]))
        .await?;
    println!(
        "{answer}\n(primary down: 3 attempts, then fallback, in {:?})",
        t.elapsed()
    );

    // 2. Two branches in parallel, merged by a final prompt.
    println!("\n== 2. parallel branches ==");
    let facts = PromptTemplate::new("Give one key fact about {topic} in one sentence.")?
        .pipe(tracked.clone())
        .pipe(StrOutputParser);
    let joke = PromptTemplate::new("Tell a one-line programmer joke about {topic}.")?
        .pipe(tracked.clone())
        .pipe(StrOutputParser);
    let pipeline = RunnableMap::new()
        .with("facts", facts)
        .with("joke", joke)
        .with("topic", lambda_sync(|v: Value| Ok(v["topic"].clone())))
        .pipe(PromptTemplate::new(
            "Write a two-sentence intro about {topic}.\nFact: {facts}\nJoke: {joke}",
        )?)
        .pipe(tracked.clone())
        .pipe(StrOutputParser);
    let intro = pipeline.invoke(json!({"topic": "Rust"})).await?;
    println!("{intro}");

    // 3. Routing: questions get an answer, everything else gets hashtags.
    println!("\n== 3. branch + batch ==");
    let hashtags =
        PromptTemplate::new("Three hashtags for {topic}, comma-separated, nothing else.")?
            .pipe(tracked.clone())
            .pipe(ListOutputParser::comma());
    let answer = PromptTemplate::new("Answer in one sentence: {topic}")?
        .pipe(tracked.clone())
        .pipe(StrOutputParser)
        .map(|a: String| vec![a]);
    let router = RunnableBranch::new(hashtags).when(
        |v: &Value| v["topic"].as_str().is_some_and(|t| t.ends_with('?')),
        answer,
    );
    let inputs = vec![
        json!({"topic": "Rust ownership"}),
        json!({"topic": "Why is the sky blue?"}),
    ];
    for (input, out) in inputs.iter().zip(router.batch(inputs.clone()).await) {
        println!("{} -> {:?}", input["topic"], out?);
    }

    // 4. Streaming: the parser turns model chunks into text deltas.
    println!("\n== 4. streaming ==");
    let explain = PromptTemplate::new("Explain {concept} in one sentence.")?
        .pipe(tracked.clone())
        .pipe(StrOutputParser);
    let mut stream = explain
        .stream(vars([("concept", "Rust ownership")]))
        .await?;
    while let Some(delta) = stream.next().await {
        print!("{}", delta?);
        std::io::stdout().flush()?;
    }
    println!();

    // 5. Usage and cost.
    println!("\n== 5. usage and cost ==");
    let totals = totals.lock().unwrap().clone();
    let mut all = Usage::default();
    for (model, usage) in &totals {
        all += *usage;
        let cost = usage
            .cost(model)
            .map_or("n/a (local or unpriced model)".to_string(), |c| {
                format!("${c:.6}")
            });
        println!(
            "{model}: {} in / {} out tokens, cost {cost}",
            usage.input_tokens, usage.output_tokens
        );
    }
    println!("the same {} tokens on hosted models:", all.total_tokens());
    for m in [
        "gpt-4.1-mini",
        "claude-sonnet-5-5",
        "gemini-2.5-flash",
        "grok-4-fast",
    ] {
        println!("  {m:>18}: ${:.6}", all.cost(m).unwrap_or(0.0));
    }
    Ok(())
}
