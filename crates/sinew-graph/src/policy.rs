//! Per-node execution policies: retries with backoff, result caching and
//! timeouts.

use std::{
    collections::HashMap,
    fmt,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde_json::Value;

use crate::{BoxError, GraphInterrupt};

type RetryPredicate = Arc<dyn Fn(&BoxError) -> bool + Send + Sync>;
type CacheKeyFn = Arc<dyn Fn(&Value) -> String + Send + Sync>;

/// How a failing node is retried. Set it with
/// [`StateGraph::set_node_retry`](crate::StateGraph::set_node_retry).
///
/// Attempt `n` (starting at 1) that fails waits
/// `min(initial_interval * backoff_factor^(n-1), max_interval)` (with up to
/// +50% random jitter when `jitter` is on) before the next attempt, until
/// `max_attempts` attempts were made or the predicate rejects the error.
/// Interrupts are never retried.
#[derive(Clone)]
pub struct RetryPolicy {
    /// Total attempts, including the first (default 3).
    pub max_attempts: usize,
    /// Wait before the second attempt (default 500 ms).
    pub initial_interval: Duration,
    /// Multiplier applied to the wait after each attempt (default 2.0).
    pub backoff_factor: f64,
    /// Upper bound for a single wait (default 128 s).
    pub max_interval: Duration,
    /// Add random jitter to waits (default true).
    pub jitter: bool,
    retry_on: Option<RetryPredicate>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_interval: Duration::from_millis(500),
            backoff_factor: 2.0,
            max_interval: Duration::from_secs(128),
            jitter: true,
            retry_on: None,
        }
    }
}

impl fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetryPolicy")
            .field("max_attempts", &self.max_attempts)
            .field("initial_interval", &self.initial_interval)
            .field("backoff_factor", &self.backoff_factor)
            .field("max_interval", &self.max_interval)
            .field("jitter", &self.jitter)
            .field("retry_on", &self.retry_on.as_ref().map(|_| "fn"))
            .finish()
    }
}

impl RetryPolicy {
    pub fn new(max_attempts: usize) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            ..Self::default()
        }
    }

    pub fn with_initial_interval(mut self, d: Duration) -> Self {
        self.initial_interval = d;
        self
    }

    pub fn with_backoff_factor(mut self, f: f64) -> Self {
        self.backoff_factor = f;
        self
    }

    pub fn with_max_interval(mut self, d: Duration) -> Self {
        self.max_interval = d;
        self
    }

    pub fn with_jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }

    /// Only retry errors for which `f` returns true (default: every error).
    /// Downcast to inspect, e.g. `e.downcast_ref::<NodeTimeout>().is_some()`.
    pub fn retry_on(mut self, f: impl Fn(&BoxError) -> bool + Send + Sync + 'static) -> Self {
        self.retry_on = Some(Arc::new(f));
        self
    }

    pub(crate) fn should_retry(&self, attempt: usize, err: &BoxError) -> bool {
        if attempt >= self.max_attempts || err.downcast_ref::<GraphInterrupt>().is_some() {
            return false;
        }
        self.retry_on.as_ref().is_none_or(|f| f(err))
    }

    /// Wait after failed attempt `attempt` (1-based).
    pub(crate) fn backoff(&self, attempt: usize) -> Duration {
        let exp = self.backoff_factor.powi(attempt.saturating_sub(1) as i32);
        let base = self.initial_interval.as_secs_f64() * exp;
        let mut secs = base.min(self.max_interval.as_secs_f64());
        if self.jitter {
            secs += secs * 0.5 * pseudo_random();
        }
        Duration::from_secs_f64(secs.max(0.0))
    }
}

/// Cheap jitter source in `[0, 1)` (no `rand` dependency needed).
fn pseudo_random() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let mut x = u64::from(nanos) ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    (x % 10_000) as f64 / 10_000.0
}

/// Caches a node's result keyed by its input. Set it with
/// [`StateGraph::set_node_cache`](crate::StateGraph::set_node_cache).
///
/// The key is a hash of the node name and the JSON-serialized input (or of
/// what `key_fn` returns). Entries live in the compiled graph (shared by its
/// clones) and expire after `ttl` (never when `None`). Failed and
/// interrupted executions are not cached.
#[derive(Clone, Default)]
pub struct CachePolicy {
    pub ttl: Option<Duration>,
    key_fn: Option<CacheKeyFn>,
}

impl fmt::Debug for CachePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CachePolicy")
            .field("ttl", &self.ttl)
            .field("key_fn", &self.key_fn.as_ref().map(|_| "fn"))
            .finish()
    }
}

impl CachePolicy {
    /// Entries never expire.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            ttl: Some(ttl),
            key_fn: None,
        }
    }

    /// Derive the cache key from the serialized input yourself (e.g. only
    /// from the fields the node reads).
    pub fn with_key_fn(mut self, f: impl Fn(&Value) -> String + Send + Sync + 'static) -> Self {
        self.key_fn = Some(Arc::new(f));
        self
    }

    pub(crate) fn key(&self, node: &str, input: &Value) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        node.hash(&mut h);
        match &self.key_fn {
            Some(f) => f(input).hash(&mut h),
            None => input.to_string().hash(&mut h),
        }
        h.finish()
    }
}

/// In-memory result cache shared by clones of a compiled graph.
pub(crate) struct NodeCache<T> {
    entries: Mutex<HashMap<u64, (Option<Instant>, T)>>,
}

impl<T: Clone> NodeCache<T> {
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn get(&self, key: u64) -> Option<T> {
        let mut map = self.entries.lock().unwrap();
        match map.get(&key) {
            Some((Some(exp), _)) if *exp <= Instant::now() => {
                map.remove(&key);
                None
            }
            Some((_, v)) => Some(v.clone()),
            None => None,
        }
    }

    pub(crate) fn put(&self, key: u64, ttl: Option<Duration>, value: T) {
        let exp = ttl.map(|t| Instant::now() + t);
        self.entries.lock().unwrap().insert(key, (exp, value));
    }

    pub(crate) fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_and_caps() {
        let p = RetryPolicy::new(5)
            .with_initial_interval(Duration::from_millis(100))
            .with_max_interval(Duration::from_millis(300))
            .with_jitter(false);
        assert_eq!(p.backoff(1), Duration::from_millis(100));
        assert_eq!(p.backoff(2), Duration::from_millis(200));
        assert_eq!(p.backoff(3), Duration::from_millis(300));
        assert_eq!(p.backoff(4), Duration::from_millis(300));
    }

    #[test]
    fn predicate_and_interrupts_stop_retries() {
        let p = RetryPolicy::new(3).retry_on(|e| e.to_string().contains("flaky"));
        let flaky: BoxError = "flaky".into();
        let fatal: BoxError = "fatal".into();
        let intr: BoxError = Box::new(GraphInterrupt { interrupts: vec![] });
        assert!(p.should_retry(1, &flaky));
        assert!(!p.should_retry(3, &flaky));
        assert!(!p.should_retry(1, &fatal));
        assert!(!p.should_retry(1, &intr));
    }

    #[test]
    fn cache_entries_expire() {
        let c = NodeCache::new();
        c.put(1, Some(Duration::from_millis(0)), 5);
        c.put(2, None, 7);
        assert_eq!(c.get(1), None);
        assert_eq!(c.get(2), Some(7));
        c.clear();
        assert_eq!(c.get(2), None);
    }
}
