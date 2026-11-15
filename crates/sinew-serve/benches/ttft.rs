//! Time-to-first-token overhead of the server versus calling the graph
//! directly. Self-contained (no criterion):
//!
//! ```bash
//! cargo bench -p sinew-serve --bench ttft
//! SINEW_BENCH_ITERS=5000 cargo bench -p sinew-serve --bench ttft
//! ```
//!
//! Every iteration runs a react agent over an instant scripted model that
//! streams 32 tokens, so the numbers are pure framework + HTTP overhead:
//!
//! - `direct`: `CompiledGraph::stream`, time until the first `Token` event;
//! - `sse raw`: `POST /runs/stream` with a keep-alive reqwest client, time
//!   until the first `event: tokens` chunk arrives;
//! - `client`: the same through `ServeClient::stream`;
//! - `openai`: `POST /v1/chat/completions` with `stream: true`, time until
//!   the first content delta;
//! - `wait`: `POST /runs/wait` full round trip (all 32 tokens).

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::json;
use sinew_core::{
    BoxStream, ChatModel, ChatRequest, ChatResponse, ChatStream, Message, StreamEvent, ToolBox,
    Usage,
};
use sinew_graph::{
    prebuilt::{react_agent, AgentState},
    GraphEvent,
};
use sinew_serve::{Assistant, RunCreate, ServeClient, Server, StreamMode, StreamPart};

const TOKENS: usize = 32;

/// Answers every request instantly with `TOKENS` one-word deltas.
struct Instant32;

#[async_trait]
impl ChatModel for Instant32 {
    fn name(&self) -> &str {
        "instant"
    }
    async fn chat(&self, _: ChatRequest) -> sinew_core::Result<ChatResponse> {
        Ok(ChatResponse {
            message: Message::assistant("tok ".repeat(TOKENS)),
            usage: Usage::default(),
            model: "instant".into(),
        })
    }
    async fn chat_stream(&self, _: ChatRequest) -> sinew_core::Result<ChatStream> {
        let events = (0..TOKENS)
            .map(|_| Ok(StreamEvent::text("tok ")))
            .chain([Ok(StreamEvent::Done {
                model: "instant".into(),
                stop_reason: None,
            })]);
        let s: BoxStream<'static, _> = Box::pin(futures_util::stream::iter(events));
        Ok(s)
    }
}

struct Stats(Vec<Duration>);

impl Stats {
    fn line(mut self, name: &str) -> Duration {
        self.0.sort();
        let pct = |p: f64| self.0[((self.0.len() - 1) as f64 * p) as usize];
        let mean = self.0.iter().sum::<Duration>() / self.0.len() as u32;
        println!(
            "{name:<10} p50 {:>8.1} µs   p90 {:>8.1} µs   p99 {:>8.1} µs   mean {:>8.1} µs",
            us(pct(0.50)),
            us(pct(0.90)),
            us(pct(0.99)),
            us(mean)
        );
        pct(0.50)
    }
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

async fn measure<F, Fut>(iters: usize, mut f: F) -> Stats
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Duration>,
{
    for _ in 0..iters / 10 + 10 {
        f().await;
    }
    let mut v = Vec::with_capacity(iters);
    for _ in 0..iters {
        v.push(f().await);
    }
    Stats(v)
}

fn main() {
    let iters: usize = std::env::var("SINEW_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let rt = sinew_serve::runtime(None).expect("runtime");
    rt.block_on(async move {
        let model: Arc<dyn ChatModel> = Arc::new(Instant32);
        let graph = react_agent(model.clone(), ToolBox::new()).unwrap();
        let srv = Server::builder()
            .assistant(Assistant::agent("agent", graph.clone()))
            .build()
            .spawn("127.0.0.1:0")
            .await
            .unwrap();
        let base = srv.base_url();
        let http = reqwest::Client::builder()
            .tcp_nodelay(true)
            .pool_max_idle_per_host(4)
            .build()
            .unwrap();
        let client = ServeClient::new(&base).with_http_client(http.clone());
        let input = AgentState::new(vec![Message::user("hi")]);
        println!("ttft over {iters} iterations, {TOKENS} tokens per reply\n");

        let direct = measure(iters, || {
            let graph = graph.clone();
            let input = input.clone();
            async move {
                let t = Instant::now();
                let mut s = graph.stream("bench", Some(input));
                let mut first = None;
                while let Some(ev) = s.next().await {
                    if matches!(ev, GraphEvent::Token { .. }) && first.is_none() {
                        first = Some(t.elapsed());
                    }
                }
                first.unwrap()
            }
        })
        .await
        .line("direct");

        let body = serde_json::to_vec(&json!({
            "assistant_id": "agent",
            "input": "hi",
            "stream_mode": ["tokens"],
        }))
        .unwrap();
        let raw = measure(iters, || {
            let http = http.clone();
            let body = body.clone();
            let url = format!("{base}/runs/stream");
            async move {
                let t = Instant::now();
                let resp = http
                    .post(url)
                    .header("content-type", "application/json")
                    .body(body)
                    .send()
                    .await
                    .unwrap();
                let mut bytes = resp.bytes_stream();
                let mut first = None;
                let mut seen = Vec::new();
                while let Some(chunk) = bytes.next().await {
                    let chunk = chunk.unwrap();
                    if first.is_none() {
                        seen.extend_from_slice(&chunk);
                        if contains(&seen, b"event: tokens") {
                            first = Some(t.elapsed());
                        }
                    }
                }
                first.unwrap()
            }
        })
        .await
        .line("sse raw");

        let req = RunCreate::new("agent", json!("hi")).with_stream_mode([StreamMode::Tokens]);
        let typed = measure(iters, || {
            let client = client.clone();
            let req = req.clone();
            async move {
                let t = Instant::now();
                let mut s = client.stream(None, &req).await.unwrap();
                let mut first = None;
                while let Some(p) = s.next().await {
                    if matches!(p.unwrap(), StreamPart::Token { .. }) && first.is_none() {
                        first = Some(t.elapsed());
                    }
                }
                first.unwrap()
            }
        })
        .await
        .line("client");

        let oa_body = serde_json::to_vec(&json!({
            "model": "agent", "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        let openai = measure(iters, || {
            let http = http.clone();
            let body = oa_body.clone();
            let url = format!("{base}/v1/chat/completions");
            async move {
                let t = Instant::now();
                let resp = http
                    .post(url)
                    .header("content-type", "application/json")
                    .body(body)
                    .send()
                    .await
                    .unwrap();
                let mut bytes = resp.bytes_stream();
                let mut first = None;
                let mut seen = Vec::new();
                while let Some(chunk) = bytes.next().await {
                    let chunk = chunk.unwrap();
                    if first.is_none() {
                        seen.extend_from_slice(&chunk);
                        if contains(&seen, b"\"content\":\"t") {
                            first = Some(t.elapsed());
                        }
                    }
                }
                first.unwrap()
            }
        })
        .await
        .line("openai");

        let wait_req = RunCreate::new("agent", json!("hi"));
        let wait = measure(iters, || {
            let client = client.clone();
            let req = wait_req.clone();
            async move {
                let t = Instant::now();
                client.wait(None, &req).await.unwrap();
                t.elapsed()
            }
        })
        .await
        .line("wait");

        println!(
            "\nserver overhead on time-to-first-token (p50): sse raw +{:.1} µs, client +{:.1} µs, openai +{:.1} µs; full wait round trip {:.1} µs",
            us(raw.saturating_sub(direct)),
            us(typed.saturating_sub(direct)),
            us(openai.saturating_sub(direct)),
            us(wait),
        );
        srv.shutdown().await.unwrap();
    });
}
