use async_trait::async_trait;
use serde_json::Value;
use sinew_graph::store::{
    rank, validate_namespace, IndexConfig, Item, PutOptions, SearchItem, SearchQuery, Store,
    StoreError,
};
use sqlx::{Row, SqlitePool};

use crate::{sqlite::open_pool, util::now_millis, CheckpointError};

/// Separates namespace segments in the `namespace` column. Every stored
/// value ends with it, so a prefix match on whole segments is a plain
/// string-prefix match.
const SEP: char = '\u{1f}';

const SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS sinew_store (
        namespace  TEXT    NOT NULL,
        key        TEXT    NOT NULL,
        value      TEXT    NOT NULL,
        embedding  TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        expires_at INTEGER,
        PRIMARY KEY (namespace, key)
    )",
    "CREATE INDEX IF NOT EXISTS sinew_store_updated ON sinew_store (updated_at)",
];

/// Durable long-term [`Store`] on SQLite (LangGraph's `SqliteStore`).
///
/// Items live in the `sinew_store` table. Prefix and filter search run in
/// SQL and Rust; with an [`IndexConfig`], embeddings are stored as JSON next
/// to each item and ranked by cosine similarity in process, which is fine
/// for tens of thousands of items per namespace prefix. Expired items (see
/// [`PutOptions::ttl`]) are skipped on read and removed by
/// [`purge_expired`](Self::purge_expired).
#[derive(Clone)]
pub struct SqliteStore {
    pool: SqlitePool,
    index: Option<IndexConfig>,
}

impl std::fmt::Debug for SqliteStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteStore")
            .field("index", &self.index)
            .finish()
    }
}

impl SqliteStore {
    /// Opens (creating it if needed) the database at `url` and creates the
    /// table. It can share a file with [`SqliteCheckpointer`](crate::SqliteCheckpointer).
    pub async fn connect(url: &str) -> Result<Self, CheckpointError> {
        Self::from_pool(open_pool(url).await?).await
    }

    /// A fresh private in-memory database.
    pub async fn in_memory() -> Result<Self, CheckpointError> {
        Self::connect("sqlite::memory:").await
    }

    pub async fn from_pool(pool: SqlitePool) -> Result<Self, CheckpointError> {
        for ddl in SCHEMA {
            sqlx::query(ddl).execute(&pool).await?;
        }
        Ok(Self { pool, index: None })
    }

    /// Enables semantic search with your embedder.
    pub fn with_index(mut self, index: IndexConfig) -> Self {
        self.index = Some(index);
        self
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Deletes expired items. Returns how many were removed.
    pub async fn purge_expired(&self) -> Result<u64, CheckpointError> {
        let res =
            sqlx::query("DELETE FROM sinew_store WHERE expires_at IS NOT NULL AND expires_at <= ?")
                .bind(now_millis())
                .execute(&self.pool)
                .await?;
        Ok(res.rows_affected())
    }
}

fn encode_ns(ns: &[&str]) -> String {
    let mut s = String::new();
    for seg in ns {
        s.push_str(seg);
        s.push(SEP);
    }
    s
}

fn decode_ns(s: &str) -> Vec<String> {
    s.split(SEP)
        .filter(|p| !p.is_empty())
        .map(String::from)
        .collect()
}

fn db_err(e: sqlx::Error) -> StoreError {
    StoreError::Backend(e.to_string())
}

fn row_item(r: &sqlx::sqlite::SqliteRow) -> Result<(Item, Option<Vec<f32>>), StoreError> {
    let ns: String = r.try_get("namespace").map_err(db_err)?;
    let value: String = r.try_get("value").map_err(db_err)?;
    let emb: Option<String> = r.try_get("embedding").map_err(db_err)?;
    let created: i64 = r.try_get("created_at").map_err(db_err)?;
    let updated: i64 = r.try_get("updated_at").map_err(db_err)?;
    Ok((
        Item {
            namespace: decode_ns(&ns),
            key: r.try_get("key").map_err(db_err)?,
            value: serde_json::from_str(&value)?,
            created_at: created.max(0) as u64,
            updated_at: updated.max(0) as u64,
        },
        emb.map(|e| serde_json::from_str(&e)).transpose()?,
    ))
}

#[async_trait]
impl Store for SqliteStore {
    async fn put_with(
        &self,
        namespace: &[&str],
        key: &str,
        value: Value,
        options: PutOptions,
    ) -> Result<(), StoreError> {
        validate_namespace(namespace)?;
        let embedding = match (&self.index, options.index) {
            (Some(ix), None | Some(true)) => match ix.text_of(&value) {
                Some(text) => ix
                    .embedder
                    .embed(vec![text])
                    .await
                    .map_err(StoreError::Embedding)?
                    .pop()
                    .map(|v| serde_json::to_string(&v))
                    .transpose()?,
                None => None,
            },
            _ => None,
        };
        let now = now_millis();
        let expires = options
            .ttl
            .map(|t| now.saturating_add(i64::try_from(t.as_millis()).unwrap_or(i64::MAX)));
        sqlx::query(
            "INSERT INTO sinew_store
                 (namespace, key, value, embedding, created_at, updated_at, expires_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5, ?6)
             ON CONFLICT (namespace, key) DO UPDATE SET
                 value = excluded.value,
                 embedding = excluded.embedding,
                 updated_at = excluded.updated_at,
                 expires_at = excluded.expires_at",
        )
        .bind(encode_ns(namespace))
        .bind(key)
        .bind(serde_json::to_string(&value)?)
        .bind(embedding)
        .bind(now)
        .bind(expires)
        .execute(&self.pool)
        .await
        .map_err(db_err)?;
        Ok(())
    }

    async fn get(&self, namespace: &[&str], key: &str) -> Result<Option<Item>, StoreError> {
        let row = sqlx::query(
            "SELECT * FROM sinew_store
             WHERE namespace = ? AND key = ? AND (expires_at IS NULL OR expires_at > ?)",
        )
        .bind(encode_ns(namespace))
        .bind(key)
        .bind(now_millis())
        .fetch_optional(&self.pool)
        .await
        .map_err(db_err)?;
        row.map(|r| row_item(&r).map(|(i, _)| i)).transpose()
    }

    async fn delete(&self, namespace: &[&str], key: &str) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM sinew_store WHERE namespace = ? AND key = ?")
            .bind(encode_ns(namespace))
            .bind(key)
            .execute(&self.pool)
            .await
            .map_err(db_err)?;
        Ok(())
    }

    async fn search(
        &self,
        prefix: &[&str],
        query: SearchQuery,
    ) -> Result<Vec<SearchItem>, StoreError> {
        let query_vec = match (&self.index, &query.query) {
            (Some(ix), Some(q)) => ix
                .embedder
                .embed(vec![q.clone()])
                .await
                .map_err(StoreError::Embedding)?
                .pop(),
            _ => None,
        };
        let p = encode_ns(prefix);
        let rows = sqlx::query(
            "SELECT * FROM sinew_store
             WHERE substr(namespace, 1, length(?1)) = ?1
               AND (expires_at IS NULL OR expires_at > ?2)",
        )
        .bind(&p)
        .bind(now_millis())
        .fetch_all(&self.pool)
        .await
        .map_err(db_err)?;
        let candidates = rows.iter().map(row_item).collect::<Result<Vec<_>, _>>()?;
        Ok(rank(candidates, &query, query_vec.as_deref()))
    }

    async fn list_namespaces(
        &self,
        prefix: &[&str],
        max_depth: Option<usize>,
    ) -> Result<Vec<Vec<String>>, StoreError> {
        let rows = sqlx::query(
            "SELECT DISTINCT namespace FROM sinew_store
             WHERE substr(namespace, 1, length(?1)) = ?1
               AND (expires_at IS NULL OR expires_at > ?2)
             ORDER BY namespace",
        )
        .bind(encode_ns(prefix))
        .bind(now_millis())
        .fetch_all(&self.pool)
        .await
        .map_err(db_err)?;
        let mut out: Vec<Vec<String>> = Vec::new();
        for r in rows {
            let ns: String = r.try_get(0).map_err(db_err)?;
            let mut ns = decode_ns(&ns);
            if let Some(d) = max_depth {
                ns.truncate(d);
            }
            if out.last() != Some(&ns) {
                out.push(ns);
            }
        }
        out.dedup();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::json;
    use sinew_graph::store::FnEmbedder;

    use super::*;

    fn letters(text: &str) -> Vec<f32> {
        let mut v = vec![0f32; 26];
        for c in text.to_lowercase().chars() {
            if c.is_ascii_lowercase() {
                v[(c as u8 - b'a') as usize] += 1.0;
            }
        }
        v
    }

    #[tokio::test]
    async fn crud_prefix_filter_and_namespaces() {
        let s = SqliteStore::in_memory().await.unwrap();
        s.put(&["users", "ana"], "lang", json!({"v": "pt"}))
            .await
            .unwrap();
        s.put(&["users", "anabel"], "lang", json!({"v": "en"}))
            .await
            .unwrap();
        s.put(&["docs"], "x", json!({"v": "en"})).await.unwrap();
        let ana = s.get(&["users", "ana"], "lang").await.unwrap().unwrap();
        assert_eq!(ana.value["v"], "pt");
        assert_eq!(ana.namespace, vec!["users", "ana"]);
        // Prefixes match whole segments: "ana" does not match "anabel".
        let hits = s
            .search(&["users", "ana"], SearchQuery::new())
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        let en = s
            .search(&[], SearchQuery::new().filter("v", json!("en")))
            .await
            .unwrap();
        assert_eq!(en.len(), 2);
        assert_eq!(
            s.list_namespaces(&[], Some(1)).await.unwrap(),
            vec![vec!["docs"], vec!["users"]]
        );
        // Upsert keeps created_at, bumps updated_at.
        s.put(&["users", "ana"], "lang", json!({"v": "fr"}))
            .await
            .unwrap();
        let again = s.get(&["users", "ana"], "lang").await.unwrap().unwrap();
        assert_eq!(again.value["v"], "fr");
        assert_eq!(again.created_at, ana.created_at);
        s.delete(&["users", "ana"], "lang").await.unwrap();
        assert!(s.get(&["users", "ana"], "lang").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn semantic_search_and_ttl() {
        let s = SqliteStore::in_memory()
            .await
            .unwrap()
            .with_index(IndexConfig::new(FnEmbedder(letters)).with_fields(["text"]));
        for (k, t) in [("a", "zebra zoo"), ("b", "apple banana"), ("c", "pizza")] {
            s.put(&["m"], k, json!({ "text": t })).await.unwrap();
        }
        let hits = s
            .search(&["m"], SearchQuery::new().query("bananas, apples").limit(2))
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].item.key, "b");
        s.put_with(
            &["m"],
            "tmp",
            json!({"text": "x"}),
            PutOptions {
                ttl: Some(Duration::from_millis(0)),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(2)).await;
        assert!(s.get(&["m"], "tmp").await.unwrap().is_none());
        assert_eq!(s.purge_expired().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let url = format!("sqlite://{}", dir.path().join("s.db").display());
        {
            let s = SqliteStore::connect(&url).await.unwrap();
            s.put(&["a"], "k", json!(1)).await.unwrap();
            s.pool().close().await;
        }
        let s = SqliteStore::connect(&url).await.unwrap();
        assert_eq!(s.get(&["a"], "k").await.unwrap().unwrap().value, json!(1));
    }
}
