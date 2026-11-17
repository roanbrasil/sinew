//! Wrappers that make any [`ChatModel`] more robust: retries with backoff,
//! fallbacks to other models, client-side rate limiting and timeouts.
//! Response caching lives in [`crate::cache`].
//!
//! Every wrapper is itself a `ChatModel`, so they stack:
//!
//! ```no_run
//! # use std::{sync::Arc, time::Duration};
//! # use sinew_core::{*, providers::OpenAiCompat};
//! # fn demo() -> Result<()> {
//! let model = OpenAiCompat::openai("gpt-4.1-mini")?
//!     .with_timeout(Duration::from_secs(60))
//!     .with_retry(RetryPolicy::default())
//!     .with_rate_limit(RateLimiter::per_minute(500).with_tokens_per_minute(200_000))
//!     .with_fallbacks([Arc::new(OpenAiCompat::ollama("qwen3:4b")) as Arc<dyn ChatModel>]);
//! # Ok(()) }
//! ```
//!
//! Order matters: wrappers apply from the inside out, so in the example a
//! timeout counts as a retryable failure, and the fallback only kicks in
//! once the primary's retries are exhausted.

use std::{
    future::Future,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures_util::StreamExt;

use crate::{
    cache::{CachedModel, ResponseCache},
    pricing::estimate_request_tokens,
    stream::ChatStream,
    ChatModel, ChatRequest, ChatResponse, Error, Result,
};

type ErrorPredicate = Arc<dyn Fn(&Error) -> bool + Send + Sync>;

/// Exponential backoff with jitter.
///
/// The delay before retry `n` (0-based) is
/// `min(initial_backoff * multiplier^n, max_backoff)`, with "equal jitter"
/// (a uniformly random value between half and all of it) when `jitter` is
/// on. A `Retry-After` from the provider ([`Error::retry_after`]) replaces
/// the computed delay, capped at `max_retry_after`. By default only
/// [`Error::is_retryable`] errors are retried.
#[derive(Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub multiplier: f64,
    pub jitter: bool,
    pub max_retry_after: Duration,
    retry_if: ErrorPredicate,
}

impl std::fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryPolicy")
            .field("max_retries", &self.max_retries)
            .field("initial_backoff", &self.initial_backoff)
            .field("max_backoff", &self.max_backoff)
            .field("multiplier", &self.multiplier)
            .field("jitter", &self.jitter)
            .field("max_retry_after", &self.max_retry_after)
            .finish()
    }
}

impl Default for RetryPolicy {
    /// 3 retries, 500 ms initial backoff doubling up to 30 s, jitter on.
    fn default() -> Self {
        Self::new(3)
    }
}

impl RetryPolicy {
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            multiplier: 2.0,
            jitter: true,
            max_retry_after: Duration::from_secs(60),
            retry_if: Arc::new(Error::is_retryable),
        }
    }

    pub fn with_initial_backoff(mut self, d: Duration) -> Self {
        self.initial_backoff = d;
        self
    }
    pub fn with_max_backoff(mut self, d: Duration) -> Self {
        self.max_backoff = d;
        self
    }
    pub fn with_multiplier(mut self, m: f64) -> Self {
        self.multiplier = m;
        self
    }
    pub fn with_jitter(mut self, on: bool) -> Self {
        self.jitter = on;
        self
    }
    /// Upper bound for honouring a provider's `Retry-After`.
    pub fn with_max_retry_after(mut self, d: Duration) -> Self {
        self.max_retry_after = d;
        self
    }
    /// Which errors to retry (default: [`Error::is_retryable`]).
    pub fn retry_if(mut self, f: impl Fn(&Error) -> bool + Send + Sync + 'static) -> Self {
        self.retry_if = Arc::new(f);
        self
    }

    pub fn should_retry(&self, attempt: u32, err: &Error) -> bool {
        attempt < self.max_retries && (self.retry_if)(err)
    }

    /// The wait before retry number `attempt` (0-based) after `err`.
    pub fn delay(&self, attempt: u32, err: &Error) -> Duration {
        if let Some(d) = err.retry_after() {
            return d.min(self.max_retry_after);
        }
        let base = self.initial_backoff.as_secs_f64() * self.multiplier.powi(attempt as i32);
        let base = base.min(self.max_backoff.as_secs_f64()).max(0.0);
        let secs = if self.jitter {
            base / 2.0 + random_unit() * base / 2.0
        } else {
            base
        };
        Duration::from_secs_f64(secs)
    }

    /// Runs `op` until it succeeds, fails with a non-retryable error or runs
    /// out of retries.
    pub async fn run<T, F, Fut>(&self, mut op: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut attempt = 0;
        loop {
            match op().await {
                Ok(v) => return Ok(v),
                Err(e) if self.should_retry(attempt, &e) => {
                    tokio::time::sleep(self.delay(attempt, &e)).await;
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// A uniform value in `[0, 1)` from a process-wide splitmix64 sequence
/// seeded by the clock. Good enough for jitter; not for anything secret.
fn random_unit() -> f64 {
    static STATE: AtomicU64 = AtomicU64::new(0);
    let mut s = STATE.load(Ordering::Relaxed);
    if s == 0 {
        s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0x9E37_79B9_7F4A_7C15, |d| d.as_nanos() as u64)
            | 1;
    }
    let next = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    STATE.store(next, Ordering::Relaxed);
    let mut z = next;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Retries failed requests; see [`ChatModelExt::with_retry`].
///
/// For streams, only failures before the stream starts (connection errors,
/// HTTP status errors) are retried; an error in the middle of a stream is
/// passed through, since part of the answer has already been delivered.
pub struct WithRetry<M> {
    inner: M,
    policy: RetryPolicy,
}

#[async_trait]
impl<M: ChatModel> ChatModel for WithRetry<M> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn capabilities(&self) -> crate::Capabilities {
        self.inner.capabilities()
    }
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        self.policy.run(|| self.inner.chat(request.clone())).await
    }
    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        self.policy
            .run(|| self.inner.chat_stream(request.clone()))
            .await
    }
}

/// Tries other models when the primary fails; see
/// [`ChatModelExt::with_fallbacks`].
pub struct WithFallbacks<M> {
    primary: M,
    fallbacks: Vec<Arc<dyn ChatModel>>,
    fallback_if: ErrorPredicate,
}

impl<M> WithFallbacks<M> {
    /// Which errors trigger a fallback (default: every error).
    pub fn fallback_if(mut self, f: impl Fn(&Error) -> bool + Send + Sync + 'static) -> Self {
        self.fallback_if = Arc::new(f);
        self
    }
}

#[async_trait]
impl<M: ChatModel> ChatModel for WithFallbacks<M> {
    fn name(&self) -> &str {
        self.primary.name()
    }
    fn capabilities(&self) -> crate::Capabilities {
        self.primary.capabilities()
    }
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let mut err = match self.primary.chat(request.clone()).await {
            Ok(r) => return Ok(r),
            Err(e) => e,
        };
        for m in &self.fallbacks {
            if !(self.fallback_if)(&err) {
                break;
            }
            match m.chat(request.clone()).await {
                Ok(r) => return Ok(r),
                Err(e) => err = e,
            }
        }
        Err(err)
    }
    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let mut err = match self.primary.chat_stream(request.clone()).await {
            Ok(s) => return Ok(s),
            Err(e) => e,
        };
        for m in &self.fallbacks {
            if !(self.fallback_if)(&err) {
                break;
            }
            match m.chat_stream(request.clone()).await {
                Ok(s) => return Ok(s),
                Err(e) => err = e,
            }
        }
        Err(err)
    }
}

struct Bucket {
    capacity: f64,
    tokens: f64,
    per_sec: f64,
}

impl Bucket {
    fn new(per_minute: f64, burst: Option<f64>) -> Self {
        let capacity = burst.unwrap_or(per_minute).max(1.0);
        Self {
            capacity,
            tokens: capacity,
            per_sec: per_minute / 60.0,
        }
    }
    fn refill(&mut self, elapsed: f64) {
        self.tokens = (self.tokens + elapsed * self.per_sec).min(self.capacity);
    }
    /// Seconds until `n` tokens are available (0 if they are).
    fn wait_for(&self, n: f64) -> f64 {
        let n = n.min(self.capacity);
        if self.tokens >= n {
            0.0
        } else {
            (n - self.tokens) / self.per_sec
        }
    }
}

struct LimiterState {
    requests: Bucket,
    tokens: Option<Bucket>,
    last: Instant,
}

/// Client-side token-bucket limiter for requests per minute and, optionally,
/// tokens per minute. Clones share the same buckets, so one limiter can guard
/// several models that share a provider quota.
///
/// Token accounting estimates the prompt size before the request
/// ([`estimate_request_tokens`]) and corrects the bucket with the real usage
/// afterwards, so sustained throughput converges on the configured TPM.
#[derive(Clone)]
pub struct RateLimiter {
    state: Arc<Mutex<LimiterState>>,
}

impl RateLimiter {
    /// `requests` per minute, bursting up to that many at once.
    pub fn per_minute(requests: u32) -> Self {
        Self::new(f64::from(requests), None)
    }

    /// `requests` per second (burst of one second's worth).
    pub fn per_second(requests: u32) -> Self {
        let rpm = f64::from(requests) * 60.0;
        Self::new(rpm, Some(f64::from(requests)))
    }

    fn new(rpm: f64, burst: Option<f64>) -> Self {
        Self {
            state: Arc::new(Mutex::new(LimiterState {
                requests: Bucket::new(rpm, burst),
                tokens: None,
                last: Instant::now(),
            })),
        }
    }

    /// Caps the request bucket size (how many requests may go out at once).
    pub fn with_burst(self, burst: u32) -> Self {
        {
            let mut s = self.lock();
            let b = f64::from(burst).max(1.0);
            s.requests.capacity = b;
            s.requests.tokens = s.requests.tokens.min(b);
        }
        self
    }

    /// Adds a tokens-per-minute budget.
    pub fn with_tokens_per_minute(self, tpm: u32) -> Self {
        self.lock().tokens = Some(Bucket::new(f64::from(tpm), None));
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LimiterState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Waits until one request carrying about `tokens` tokens may go out,
    /// then takes it from the buckets.
    pub async fn acquire(&self, tokens: u32) {
        loop {
            let wait = {
                let mut s = self.lock();
                let now = Instant::now();
                let elapsed = now.duration_since(s.last).as_secs_f64();
                s.last = now;
                s.requests.refill(elapsed);
                if let Some(t) = s.tokens.as_mut() {
                    t.refill(elapsed);
                }
                let need = f64::from(tokens);
                let wait = s
                    .requests
                    .wait_for(1.0)
                    .max(s.tokens.as_ref().map_or(0.0, |t| t.wait_for(need)));
                if wait <= 0.0 {
                    s.requests.tokens -= 1.0;
                    if let Some(t) = s.tokens.as_mut() {
                        t.tokens -= need.min(t.capacity);
                    }
                    return;
                }
                wait
            };
            tokio::time::sleep(Duration::from_secs_f64(wait.min(60.0))).await;
        }
    }

    /// Corrects the token bucket once the real usage is known. The bucket may
    /// go negative, which delays later requests.
    pub fn record(&self, estimated: u32, actual: u32) {
        let mut s = self.lock();
        if let Some(t) = s.tokens.as_mut() {
            t.tokens -= f64::from(actual) - f64::from(estimated);
            t.tokens = t.tokens.min(t.capacity);
        }
    }
}

/// Waits for a [`RateLimiter`] before every request; see
/// [`ChatModelExt::with_rate_limit`].
pub struct RateLimited<M> {
    inner: M,
    limiter: RateLimiter,
}

#[async_trait]
impl<M: ChatModel> ChatModel for RateLimited<M> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn capabilities(&self) -> crate::Capabilities {
        self.inner.capabilities()
    }
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let est = estimate_request_tokens(&request);
        self.limiter.acquire(est).await;
        let resp = self.inner.chat(request).await?;
        self.limiter.record(est, resp.usage.total_tokens());
        Ok(resp)
    }
    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        self.limiter
            .acquire(estimate_request_tokens(&request))
            .await;
        self.inner.chat_stream(request).await
    }
}

/// Fails requests that take too long; see [`ChatModelExt::with_timeout`].
pub struct WithTimeout<M> {
    inner: M,
    timeout: Duration,
    idle: Option<Duration>,
}

impl<M> WithTimeout<M> {
    /// For streams: fail if no event arrives for `idle` (the overall timeout
    /// only bounds the time to the start of the stream).
    pub fn with_stream_idle_timeout(mut self, idle: Duration) -> Self {
        self.idle = Some(idle);
        self
    }
}

#[async_trait]
impl<M: ChatModel> ChatModel for WithTimeout<M> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn capabilities(&self) -> crate::Capabilities {
        self.inner.capabilities()
    }
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        tokio::time::timeout(self.timeout, self.inner.chat(request))
            .await
            .map_err(|_| Error::Timeout(self.timeout))?
    }
    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let stream = tokio::time::timeout(self.timeout, self.inner.chat_stream(request))
            .await
            .map_err(|_| Error::Timeout(self.timeout))??;
        let Some(idle) = self.idle else {
            return Ok(stream);
        };
        Ok(
            futures_util::stream::unfold(Some(stream), move |st| async move {
                let mut s = st?;
                match tokio::time::timeout(idle, s.next()).await {
                    Ok(Some(item)) => Some((item, Some(s))),
                    Ok(None) => None,
                    Err(_) => Some((Err(Error::Timeout(idle)), None)),
                }
            })
            .boxed(),
        )
    }
}

/// Resilience and caching wrappers for every [`ChatModel`].
pub trait ChatModelExt: ChatModel + Sized {
    /// Retries transient failures with exponential backoff and jitter,
    /// honouring `Retry-After`.
    fn with_retry(self, policy: RetryPolicy) -> WithRetry<Self> {
        WithRetry {
            inner: self,
            policy,
        }
    }

    /// Tries `fallbacks` in order when this model fails.
    fn with_fallbacks(
        self,
        fallbacks: impl IntoIterator<Item = Arc<dyn ChatModel>>,
    ) -> WithFallbacks<Self> {
        WithFallbacks {
            primary: self,
            fallbacks: fallbacks.into_iter().collect(),
            fallback_if: Arc::new(|_| true),
        }
    }

    /// Waits for `limiter` before each request.
    fn with_rate_limit(self, limiter: RateLimiter) -> RateLimited<Self> {
        RateLimited {
            inner: self,
            limiter,
        }
    }

    /// Fails with [`Error::Timeout`] when a request takes longer than `d`.
    fn with_timeout(self, d: Duration) -> WithTimeout<Self> {
        WithTimeout {
            inner: self,
            timeout: d,
            idle: None,
        }
    }

    /// Serves repeated identical requests from `cache`.
    fn with_cache<C: ResponseCache + 'static>(self, cache: C) -> CachedModel<Self, C> {
        CachedModel::new(self, cache)
    }

    /// Checks each request's new user messages and tool results with
    /// `input`, and the response with `output`; see
    /// [`Guarded`](crate::guardrails::Guarded) and the
    /// [guardrails module](crate::guardrails).
    ///
    /// ```
    /// use sinew_core::guardrails::{guard, Guardrails, PiiRedactor};
    /// use sinew_core::{mock::ScriptedModel, ChatModelExt, Message};
    ///
    /// let model = ScriptedModel::new([Message::assistant("ok")])
    ///     .with_guardrails([guard(PiiRedactor::new())], Guardrails::new());
    /// ```
    fn with_guardrails(
        self,
        input: impl Into<crate::guardrails::Guardrails>,
        output: impl Into<crate::guardrails::Guardrails>,
    ) -> crate::guardrails::Guarded<Self> {
        crate::guardrails::Guarded::new(self, input, output)
    }
}

impl<M: ChatModel + Sized> ChatModelExt for M {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mock::ScriptedModel, Message};
    use std::sync::atomic::AtomicUsize;

    /// Fails the first `fail` calls with `make_err`, then answers.
    pub(crate) struct Flaky {
        pub fail: usize,
        pub calls: AtomicUsize,
        pub make_err: fn() -> Error,
    }

    #[async_trait]
    impl ChatModel for Flaky {
        fn name(&self) -> &str {
            "flaky"
        }
        async fn chat(&self, _r: ChatRequest) -> Result<ChatResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail {
                return Err((self.make_err)());
            }
            Ok(ChatResponse {
                message: Message::assistant("ok"),
                usage: crate::Usage::new(1, 1),
                model: "flaky".into(),
            })
        }
    }

    fn rate_limited() -> Error {
        Error::RateLimited {
            provider: "p".into(),
            status: 429,
            retry_after: Some(Duration::from_millis(20)),
            body: String::new(),
        }
    }

    fn bad_request() -> Error {
        Error::Provider {
            provider: "p".into(),
            status: 400,
            body: "bad".into(),
        }
    }

    fn req() -> ChatRequest {
        ChatRequest::new(vec![Message::user("hi")])
    }

    #[tokio::test]
    async fn retry_honours_retry_after_and_gives_up_on_4xx() {
        let m = Flaky {
            fail: 2,
            calls: AtomicUsize::new(0),
            make_err: rate_limited,
        }
        .with_retry(RetryPolicy::new(3).with_initial_backoff(Duration::from_secs(10)));
        let t = Instant::now();
        assert_eq!(m.chat(req()).await.unwrap().message.content, "ok");
        // Two Retry-After waits of 20 ms, not the 10 s backoff.
        let e = t.elapsed();
        assert!(
            e >= Duration::from_millis(40) && e < Duration::from_secs(2),
            "{e:?}"
        );

        let m = Flaky {
            fail: 5,
            calls: AtomicUsize::new(0),
            make_err: bad_request,
        };
        let m = m.with_retry(RetryPolicy::new(3));
        assert!(m.chat(req()).await.is_err());
        assert_eq!(m.inner.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_stops_after_max_retries() {
        let m = Flaky {
            fail: 10,
            calls: AtomicUsize::new(0),
            make_err: || Error::Timeout(Duration::from_millis(1)),
        }
        .with_retry(
            RetryPolicy::new(2)
                .with_initial_backoff(Duration::from_millis(1))
                .with_jitter(false),
        );
        assert!(matches!(m.chat(req()).await, Err(Error::Timeout(_))));
        assert_eq!(m.inner.calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn backoff_grows_and_is_capped() {
        let p = RetryPolicy::new(10)
            .with_initial_backoff(Duration::from_millis(100))
            .with_max_backoff(Duration::from_millis(1000))
            .with_jitter(false);
        let e = Error::Timeout(Duration::ZERO);
        assert_eq!(p.delay(0, &e), Duration::from_millis(100));
        assert_eq!(p.delay(2, &e), Duration::from_millis(400));
        assert_eq!(p.delay(9, &e), Duration::from_millis(1000));
        let j = p.with_jitter(true);
        for _ in 0..100 {
            let d = j.delay(1, &e);
            assert!(d >= Duration::from_millis(100) && d <= Duration::from_millis(200));
        }
        assert_eq!(
            RetryPolicy::default().delay(0, &rate_limited()),
            Duration::from_millis(20)
        );
    }

    #[tokio::test]
    async fn fallbacks_take_over() {
        let backup: Arc<dyn ChatModel> =
            Arc::new(ScriptedModel::new([Message::assistant("from backup")]));
        let m = Flaky {
            fail: 1,
            calls: AtomicUsize::new(0),
            make_err: bad_request,
        }
        .with_fallbacks([backup]);
        assert_eq!(m.chat(req()).await.unwrap().message.content, "from backup");

        let backup: Arc<dyn ChatModel> = Arc::new(ScriptedModel::new([]));
        let m = Flaky {
            fail: 1,
            calls: AtomicUsize::new(0),
            make_err: bad_request,
        }
        .with_fallbacks([backup])
        .fallback_if(|e| e.is_retryable());
        assert!(matches!(
            m.chat(req()).await,
            Err(Error::Provider { status: 400, .. })
        ));
    }

    #[tokio::test]
    async fn rate_limiter_spaces_requests() {
        let lim = RateLimiter::per_second(20).with_burst(1);
        let t = Instant::now();
        for _ in 0..4 {
            lim.acquire(0).await;
        }
        // 1 immediate + 3 spaced by 50 ms.
        let e = t.elapsed();
        assert!(e >= Duration::from_millis(140), "{e:?}");
    }

    #[tokio::test]
    async fn token_budget_blocks_until_refilled() {
        let lim = RateLimiter::per_minute(1_000_000).with_tokens_per_minute(6000); // 100 tok/s
        lim.acquire(6000).await; // drains the bucket
        let t = Instant::now();
        lim.acquire(10).await; // needs ~0.1 s of refill
        assert!(t.elapsed() >= Duration::from_millis(80));
        // Under-estimates are charged back.
        lim.record(10, 1010);
        assert!(lim.lock().tokens.as_ref().unwrap().tokens < -900.0);
    }

    #[tokio::test]
    async fn timeout_fires() {
        struct Slow;
        #[async_trait]
        impl ChatModel for Slow {
            fn name(&self) -> &str {
                "slow"
            }
            async fn chat(&self, _r: ChatRequest) -> Result<ChatResponse> {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Err(Error::Other("unreachable".into()))
            }
        }
        let m = Slow.with_timeout(Duration::from_millis(20));
        assert!(matches!(m.chat(req()).await, Err(Error::Timeout(_))));
        // Timeout + retry + fallback stack.
        let backup: Arc<dyn ChatModel> = Arc::new(ScriptedModel::new([Message::assistant("b")]));
        let m = Slow
            .with_timeout(Duration::from_millis(10))
            .with_retry(RetryPolicy::new(1).with_initial_backoff(Duration::from_millis(1)))
            .with_fallbacks([backup]);
        assert_eq!(m.chat(req()).await.unwrap().message.content, "b");
    }
}
