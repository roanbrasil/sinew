//! Live checks against a local Ollama (127.0.0.1:11434). Ignored by default:
//! `cargo test -p sinew-core --test live_ollama -- --ignored`
//! (`SINEW_MODEL` overrides the model, default `qwen3:4b-instruct-2507-q4_K_M`).

use std::{sync::Arc, time::Duration};

use futures_util::StreamExt;
use sinew_core::{
    providers::OpenAiCompat,
    runnable::{parallel, vars},
    ChatModel, ChatModelExt, InMemoryCache, ListOutputParser, PromptTemplate, RetryPolicy,
    Runnable, RunnableExt, StrOutputParser,
};

fn model() -> OpenAiCompat {
    let name = std::env::var("SINEW_MODEL")
        .unwrap_or_else(|_| "qwen3:4b-instruct-2507-q4_K_M".to_string());
    OpenAiCompat::ollama(name)
}

#[tokio::test]
#[ignore = "needs a local Ollama"]
async fn chain_with_parallel_branches_retry_and_fallback() {
    let unreachable = OpenAiCompat::new("http://127.0.0.1:9/v1", "nope");
    let llm = Arc::new(
        unreachable
            .with_retry(RetryPolicy::new(1).with_initial_backoff(Duration::from_millis(10)))
            .with_fallbacks([Arc::new(model()) as Arc<dyn ChatModel>])
            .with_timeout(Duration::from_secs(120)),
    );
    let capital = PromptTemplate::new("Reply with only the capital city of {country}.")
        .unwrap()
        .pipe(llm.clone())
        .pipe(StrOutputParser);
    let colours = PromptTemplate::new(
        "List the colours of the flag of {country}, comma-separated, nothing else.",
    )
    .unwrap()
    .pipe(llm.clone())
    .pipe(ListOutputParser::comma());
    let (city, flag) = parallel((capital, colours))
        .invoke(vars([("country", "France")]))
        .await
        .unwrap();
    assert!(city.to_lowercase().contains("paris"), "{city}");
    assert!(flag.len() >= 2, "{flag:?}");
}

#[tokio::test]
#[ignore = "needs a local Ollama"]
async fn streaming_chain_and_cache() {
    let cached = Arc::new(model().with_cache(InMemoryCache::new(16)));
    let chain = PromptTemplate::new("Count from 1 to {n}, separated by spaces.")
        .unwrap()
        .pipe(cached.clone())
        .pipe(StrOutputParser);
    let mut s = chain.stream(vars([("n", "5")])).await.unwrap();
    let mut chunks = 0;
    let mut text = String::new();
    while let Some(c) = s.next().await {
        text.push_str(&c.unwrap());
        chunks += 1;
    }
    assert!(chunks > 1, "expected several deltas");
    assert!(text.contains('5'), "{text}");
    // Same request again: served from the cache, identical text.
    let t = std::time::Instant::now();
    let again = chain.invoke(vars([("n", "5")])).await.unwrap();
    assert_eq!(again, text);
    assert!(t.elapsed() < Duration::from_millis(50));
}
