//! Response caching for chat models.
//!
//! [`CachedModel`] (built with
//! [`ChatModelExt::with_cache`](crate::resilience::ChatModelExt::with_cache))
//! answers a request it has seen before from a [`ResponseCache`] instead of
//! calling the provider. The key is a 128-bit hash of the model name and the
//! full serialized request (messages, tools, temperature, ...), so any
//! change to the request is a miss. Responses are cached regardless of
//! temperature: enable caching where repeated answers are what you want.
//!
//! Two caches ship with Sinew: [`InMemoryCache`] (LRU with optional TTL) and,
//! behind the `sqlite-cache` feature, `SqliteCache`.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures_util::StreamExt;

use crate::{
    stream::{events_from_response, ChatStream, StreamAccumulator},
    ChatModel, ChatRequest, ChatResponse, Result,
};

/// Storage for cached responses.
#[async_trait]
pub trait ResponseCache: Send + Sync {
    async fn get(&self, key: &str) -> Option<ChatResponse>;
    async fn put(&self, key: &str, response: &ChatResponse);
}

#[async_trait]
impl<C: ResponseCache + ?Sized> ResponseCache for Arc<C> {
    async fn get(&self, key: &str) -> Option<ChatResponse> {
        (**self).get(key).await
    }
    async fn put(&self, key: &str, response: &ChatResponse) {
        (**self).put(key, response).await
    }
}

/// FNV-1a, 128-bit: stable across Rust versions and platforms, so keys stay
/// valid in a persistent cache.
fn fnv1a_128(parts: &[&[u8]]) -> u128 {
    const OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
    const PRIME: u128 = 0x0000000001000000000000000000013B;
    let mut h = OFFSET;
    for p in parts {
        for &b in *p {
            h ^= u128::from(b);
            h = h.wrapping_mul(PRIME);
        }
        // Separator so ("ab","c") and ("a","bc") differ.
        h ^= 0xff;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// The cache key for `request` sent to `model`.
pub fn cache_key(model: &str, request: &ChatRequest) -> String {
    let body = serde_json::to_vec(request).unwrap_or_default();
    format!("{:032x}", fnv1a_128(&[model.as_bytes(), &body]))
}

struct Entry {
    response: ChatResponse,
    tick: u64,
    stored: Instant,
}

struct Lru {
    map: HashMap<String, Entry>,
    order: BTreeMap<u64, String>,
    tick: u64,
}

/// In-process LRU cache. Clones share the same storage.
#[derive(Clone)]
pub struct InMemoryCache {
    inner: Arc<Mutex<Lru>>,
    capacity: usize,
    ttl: Option<Duration>,
}

impl InMemoryCache {
    /// Keeps at most `capacity` responses, evicting the least recently used.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Lru {
                map: HashMap::new(),
                order: BTreeMap::new(),
                tick: 0,
            })),
            capacity: capacity.max(1),
            ttl: None,
        }
    }

    /// Entries older than `ttl` are treated as misses.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    pub fn len(&self) -> usize {
        self.lock().map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        let mut l = self.lock();
        l.map.clear();
        l.order.clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Lru> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[async_trait]
impl ResponseCache for InMemoryCache {
    async fn get(&self, key: &str) -> Option<ChatResponse> {
        let mut l = self.lock();
        l.tick += 1;
        let tick = l.tick;
        let ttl = self.ttl;
        let entry = l.map.get_mut(key)?;
        if ttl.is_some_and(|t| entry.stored.elapsed() > t) {
            let old = entry.tick;
            l.map.remove(key);
            l.order.remove(&old);
            return None;
        }
        let old = std::mem::replace(&mut entry.tick, tick);
        let resp = entry.response.clone();
        l.order.remove(&old);
        l.order.insert(tick, key.to_string());
        Some(resp)
    }

    async fn put(&self, key: &str, response: &ChatResponse) {
        let mut l = self.lock();
        l.tick += 1;
        let tick = l.tick;
        let entry = Entry {
            response: response.clone(),
            tick,
            stored: Instant::now(),
        };
        if let Some(old) = l.map.insert(key.to_string(), entry) {
            l.order.remove(&old.tick);
        }
        l.order.insert(tick, key.to_string());
        while l.map.len() > self.capacity {
            let Some((_, k)) = l.order.pop_first() else {
                break;
            };
            l.map.remove(&k);
        }
    }
}

/// A model behind a [`ResponseCache`].
///
/// `chat` hits are returned directly. `chat_stream` hits are replayed as a
/// stream; misses stream from the provider and are stored once the stream
/// completes successfully.
pub struct CachedModel<M, C> {
    inner: M,
    cache: Arc<C>,
}

impl<M: ChatModel, C: ResponseCache + 'static> CachedModel<M, C> {
    pub fn new(inner: M, cache: C) -> Self {
        Self {
            inner,
            cache: Arc::new(cache),
        }
    }

    pub fn cache(&self) -> &C {
        &self.cache
    }
}

#[async_trait]
impl<M: ChatModel, C: ResponseCache + 'static> ChatModel for CachedModel<M, C> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn capabilities(&self) -> crate::Capabilities {
        self.inner.capabilities()
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let key = cache_key(self.inner.name(), &request);
        if let Some(hit) = self.cache.get(&key).await {
            return Ok(hit);
        }
        let resp = self.inner.chat(request).await?;
        self.cache.put(&key, &resp).await;
        Ok(resp)
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let key = cache_key(self.inner.name(), &request);
        if let Some(hit) = self.cache.get(&key).await {
            let events = events_from_response(hit).into_iter().map(Ok);
            return Ok(Box::pin(futures_util::stream::iter(events)));
        }
        let stream = self.inner.chat_stream(request).await?;
        let cache = self.cache.clone();
        struct St<C> {
            stream: Option<ChatStream>,
            acc: StreamAccumulator,
            failed: bool,
            cache: Arc<C>,
            key: String,
        }
        let st = St {
            stream: Some(stream),
            acc: StreamAccumulator::new(),
            failed: false,
            cache,
            key,
        };
        Ok(futures_util::stream::unfold(st, |mut st| async move {
            let s = st.stream.as_mut()?;
            match s.next().await {
                Some(item) => {
                    match &item {
                        Ok(ev) => st.acc.push(ev),
                        Err(_) => st.failed = true,
                    }
                    Some((item, st))
                }
                None => {
                    st.stream = None;
                    if !st.failed && st.acc.is_done() {
                        let acc = std::mem::take(&mut st.acc);
                        st.cache.put(&st.key, &acc.finish()).await;
                    }
                    None
                }
            }
        })
        .boxed())
    }
}

#[cfg(feature = "sqlite-cache")]
pub use sqlite::SqliteCache;

#[cfg(feature = "sqlite-cache")]
mod sqlite {
    use super::*;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
    use std::str::FromStr;

    /// Persistent cache in a SQLite database (feature `sqlite-cache`).
    #[derive(Clone)]
    pub struct SqliteCache {
        pool: SqlitePool,
        ttl: Option<Duration>,
    }

    fn now_secs() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() as i64)
    }

    impl SqliteCache {
        /// Opens (creating if needed) `url`, e.g. `sqlite://cache.db` or
        /// `sqlite::memory:`.
        pub async fn connect(url: &str) -> Result<Self> {
            let opts = SqliteConnectOptions::from_str(url)
                .map_err(|e| crate::Error::Config(e.to_string()))?
                .create_if_missing(true);
            let pool = SqlitePoolOptions::new()
                .max_connections(4)
                .connect_with(opts)
                .await
                .map_err(|e| crate::Error::Config(e.to_string()))?;
            Self::from_pool(pool).await
        }

        /// Uses an existing pool; creates the `sinew_llm_cache` table.
        pub async fn from_pool(pool: SqlitePool) -> Result<Self> {
            sqlx::query(
                "CREATE TABLE IF NOT EXISTS sinew_llm_cache (
                    key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )",
            )
            .execute(&pool)
            .await
            .map_err(|e| crate::Error::Config(e.to_string()))?;
            Ok(Self { pool, ttl: None })
        }

        /// Entries older than `ttl` are treated as misses.
        pub fn with_ttl(mut self, ttl: Duration) -> Self {
            self.ttl = Some(ttl);
            self
        }
    }

    #[async_trait]
    impl ResponseCache for SqliteCache {
        async fn get(&self, key: &str) -> Option<ChatResponse> {
            let min = self
                .ttl
                .map_or(i64::MIN, |t| now_secs() - t.as_secs() as i64);
            let row: Option<(String,)> = sqlx::query_as(
                "SELECT response FROM sinew_llm_cache WHERE key = ? AND created_at >= ?",
            )
            .bind(key)
            .bind(min)
            .fetch_optional(&self.pool)
            .await
            .ok()?;
            serde_json::from_str(&row?.0).ok()
        }

        async fn put(&self, key: &str, response: &ChatResponse) {
            let Ok(json) = serde_json::to_string(response) else {
                return;
            };
            let _ = sqlx::query(
                "INSERT INTO sinew_llm_cache (key, response, created_at) VALUES (?, ?, ?)
                 ON CONFLICT(key) DO UPDATE SET response = excluded.response, created_at = excluded.created_at",
            )
            .bind(key)
            .bind(json)
            .bind(now_secs())
            .execute(&self.pool)
            .await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mock::ScriptedModel, resilience::ChatModelExt, stream::collect_stream, Message};

    fn req(q: &str) -> ChatRequest {
        ChatRequest::new(vec![Message::user(q)])
    }

    #[test]
    fn keys_depend_on_model_and_request() {
        let a = cache_key("m", &req("hi"));
        assert_eq!(a.len(), 32);
        assert_eq!(a, cache_key("m", &req("hi")));
        assert_ne!(a, cache_key("m2", &req("hi")));
        assert_ne!(a, cache_key("m", &req("hi").with_temperature(0.5)));
    }

    #[tokio::test]
    async fn cached_model_skips_the_provider_on_hits() {
        let inner = Arc::new(ScriptedModel::new([
            Message::assistant("first"),
            Message::assistant("second"),
        ]));
        let m = inner.clone().with_cache(InMemoryCache::new(10));
        assert_eq!(m.chat(req("q")).await.unwrap().message.content, "first");
        assert_eq!(m.chat(req("q")).await.unwrap().message.content, "first");
        assert_eq!(inner.requests().len(), 1);
        // Streams are cached too, and a cached response replays as a stream.
        let s = collect_stream(m.chat_stream(req("other")).await.unwrap())
            .await
            .unwrap();
        assert_eq!(s.message.content, "second");
        let again = collect_stream(m.chat_stream(req("other")).await.unwrap())
            .await
            .unwrap();
        assert_eq!(again.message.content, "second");
        assert_eq!(inner.requests().len(), 2);
    }

    #[tokio::test]
    async fn lru_evicts_and_ttl_expires() {
        let c = InMemoryCache::new(2);
        let r = |t: &str| ChatResponse {
            message: Message::assistant(t),
            usage: Default::default(),
            model: "m".into(),
        };
        c.put("a", &r("a")).await;
        c.put("b", &r("b")).await;
        assert!(c.get("a").await.is_some()); // a is now most recent
        c.put("c", &r("c")).await; // evicts b
        assert!(c.get("b").await.is_none());
        assert!(c.get("a").await.is_some() && c.get("c").await.is_some());
        assert_eq!(c.len(), 2);

        let t = InMemoryCache::new(2).with_ttl(Duration::from_millis(10));
        t.put("x", &r("x")).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(t.get("x").await.is_none());
        assert!(t.is_empty());
    }

    #[cfg(feature = "sqlite-cache")]
    #[tokio::test]
    async fn sqlite_cache_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let url = format!("sqlite://{}", dir.path().join("c.db").display());
        let c = SqliteCache::connect(&url).await.unwrap();
        let r = ChatResponse {
            message: Message::assistant("persisted"),
            usage: crate::Usage::new(3, 4),
            model: "m".into(),
        };
        c.put("k", &r).await;
        let c2 = SqliteCache::connect(&url).await.unwrap();
        let got = c2.get("k").await.unwrap();
        assert_eq!(got.message.content, "persisted");
        assert_eq!(got.usage, crate::Usage::new(3, 4));
        assert!(c2.get("missing").await.is_none());
    }
}
