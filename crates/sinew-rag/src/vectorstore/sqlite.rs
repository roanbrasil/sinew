use std::{str::FromStr, sync::Arc, time::Duration};

use async_trait::async_trait;
use futures_util::TryStreamExt;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous},
    QueryBuilder, Row, Sqlite, SqlitePool,
};

use super::{check_lengths, normalize, topk::TopK, Filter, ScoredDocument, VectorStore};
use crate::{document::ensure_ids, Document, Embeddings, Error, Metadata, Result};

const SCHEMA: &[&str] = &["CREATE TABLE IF NOT EXISTS sinew_vectors (
        collection TEXT NOT NULL,
        id         TEXT NOT NULL,
        content    TEXT NOT NULL,
        metadata   TEXT NOT NULL,
        embedding  BLOB NOT NULL,
        PRIMARY KEY (collection, id)
    )"];

/// Vector store in a SQLite database (a file, or memory for tests).
///
/// Vectors are normalised and stored as little-endian `f32` blobs; search is
/// a brute-force scan that streams `(id, embedding)` rows, keeps a top-k heap
/// and then fetches only the winning documents. Metadata filters are
/// evaluated in Rust (the metadata column is read only when a filter is
/// given). Good for up to a few hundred thousand chunks on one machine;
/// every collection lives in the same `sinew_vectors` table.
#[derive(Clone)]
pub struct SqliteVectorStore {
    pool: SqlitePool,
    collection: String,
    embeddings: Arc<dyn Embeddings>,
}

impl SqliteVectorStore {
    /// Opens (creating it if needed) the database at `url`, e.g.
    /// `sqlite://rag.db` or `sqlite::memory:`, and creates the schema.
    pub async fn connect(url: &str, embeddings: impl Embeddings + 'static) -> Result<Self> {
        let in_memory = url.contains(":memory:") || url.contains("mode=memory");
        let options = SqliteConnectOptions::from_str(url)?
            .create_if_missing(true)
            .busy_timeout(Duration::from_secs(10));
        let pool = if in_memory {
            SqlitePoolOptions::new()
                .max_connections(1)
                .min_connections(1)
                .idle_timeout(None)
                .max_lifetime(None)
                .connect_with(options)
                .await?
        } else {
            SqlitePoolOptions::new()
                .max_connections(8)
                .connect_with(
                    options
                        .journal_mode(SqliteJournalMode::Wal)
                        .synchronous(SqliteSynchronous::Normal),
                )
                .await?
        };
        Self::from_pool(pool, embeddings).await
    }

    /// A private in-memory database.
    pub async fn in_memory(embeddings: impl Embeddings + 'static) -> Result<Self> {
        Self::connect("sqlite::memory:", embeddings).await
    }

    /// Wraps an existing pool and creates the schema if it is missing.
    pub async fn from_pool(
        pool: SqlitePool,
        embeddings: impl Embeddings + 'static,
    ) -> Result<Self> {
        for stmt in SCHEMA {
            sqlx::query(*stmt).execute(&pool).await?;
        }
        Ok(Self {
            pool,
            collection: "default".into(),
            embeddings: Arc::new(embeddings),
        })
    }

    /// Uses another collection in the same database (default `"default"`).
    pub fn with_collection(mut self, name: impl Into<String>) -> Self {
        self.collection = name.into();
        self
    }

    pub fn collection(&self) -> &str {
        &self.collection
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    async fn stored_dim(&self) -> Result<Option<usize>> {
        let row = sqlx::query(
            "SELECT length(embedding) AS n FROM sinew_vectors WHERE collection = ? LIMIT 1",
        )
        .bind(&self.collection)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| r.get::<i64, _>("n") as usize / 4))
    }
}

fn to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn from_blob(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Dot product against a little-endian `f32` blob without decoding it into
/// a temporary vector.
#[inline]
fn dot_blob(q: &[f32], b: &[u8]) -> f32 {
    let mut acc = [0f32; 4];
    let (qc, qr) = q.as_chunks::<4>();
    let (bc, br) = b.as_chunks::<16>();
    for (x, y) in qc.iter().zip(bc) {
        let (lanes, _) = y.as_chunks::<4>();
        for ((s, xi), yi) in acc.iter_mut().zip(x).zip(lanes) {
            *s += xi * f32::from_le_bytes(*yi);
        }
    }
    let mut sum = (acc[0] + acc[2]) + (acc[1] + acc[3]);
    for (x, y) in qr.iter().zip(br.as_chunks::<4>().0) {
        sum += x * f32::from_le_bytes(*y);
    }
    sum
}

fn parse_metadata(s: &str) -> Metadata {
    serde_json::from_str(s).unwrap_or_default()
}

#[async_trait]
impl VectorStore for SqliteVectorStore {
    fn embeddings(&self) -> &Arc<dyn Embeddings> {
        &self.embeddings
    }

    async fn add_vectors(
        &self,
        mut docs: Vec<Document>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<String>> {
        check_lengths(docs.len(), vectors.len())?;
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let expected = match self.stored_dim().await? {
            Some(d) => d,
            None => vectors[0].len(),
        };
        if let Some(bad) = vectors.iter().find(|v| v.len() != expected || v.is_empty()) {
            return Err(Error::Dimensions {
                expected,
                got: bad.len(),
            });
        }
        let ids = ensure_ids(&mut docs);
        let mut tx = self.pool.begin().await?;
        for (doc, mut v) in docs.iter().zip(vectors) {
            normalize(&mut v);
            sqlx::query(
                "INSERT INTO sinew_vectors (collection, id, content, metadata, embedding)
                 VALUES (?, ?, ?, ?, ?)
                 ON CONFLICT (collection, id) DO UPDATE SET
                   content = excluded.content,
                   metadata = excluded.metadata,
                   embedding = excluded.embedding",
            )
            .bind(&self.collection)
            .bind(doc.id.as_deref())
            .bind(&doc.content)
            .bind(serde_json::to_string(&doc.metadata)?)
            .bind(to_blob(&v))
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(ids)
    }

    async fn delete(&self, ids: &[String]) -> Result<usize> {
        let mut removed = 0;
        for chunk in ids.chunks(500) {
            let mut qb =
                QueryBuilder::<Sqlite>::new("DELETE FROM sinew_vectors WHERE collection = ");
            qb.push_bind(&self.collection).push(" AND id IN (");
            let mut sep = qb.separated(", ");
            for id in chunk {
                sep.push_bind(id);
            }
            qb.push(")");
            removed += qb.build().execute(&self.pool).await?.rows_affected() as usize;
        }
        Ok(removed)
    }

    async fn get(&self, ids: &[String]) -> Result<Vec<Document>> {
        let mut by_id = std::collections::HashMap::new();
        for chunk in ids.chunks(500) {
            let mut qb = QueryBuilder::<Sqlite>::new(
                "SELECT id, content, metadata FROM sinew_vectors WHERE collection = ",
            );
            qb.push_bind(&self.collection).push(" AND id IN (");
            let mut sep = qb.separated(", ");
            for id in chunk {
                sep.push_bind(id);
            }
            qb.push(")");
            for row in qb.build().fetch_all(&self.pool).await? {
                let id: String = row.get("id");
                let doc = Document {
                    id: Some(id.clone()),
                    content: row.get("content"),
                    metadata: parse_metadata(row.get("metadata")),
                };
                by_id.insert(id, doc);
            }
        }
        Ok(ids.iter().filter_map(|id| by_id.remove(id)).collect())
    }

    async fn len(&self) -> Result<usize> {
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM sinew_vectors WHERE collection = ?")
            .bind(&self.collection)
            .fetch_one(&self.pool)
            .await?;
        Ok(n as usize)
    }

    async fn search_by_vector(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        with_vectors: bool,
    ) -> Result<Vec<ScoredDocument>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut q = query.to_vec();
        normalize(&mut q);
        let want_bytes = q.len() * 4;
        let sql = if filter.is_some() {
            "SELECT id, embedding, metadata FROM sinew_vectors WHERE collection = ?"
        } else {
            "SELECT id, embedding FROM sinew_vectors WHERE collection = ?"
        };
        let mut top = TopK::new(k);
        let mut ids: Vec<String> = Vec::new();
        {
            let mut rows = sqlx::query(sql).bind(&self.collection).fetch(&self.pool);
            while let Some(row) = rows.try_next().await? {
                let blob: &[u8] = row.get("embedding");
                if blob.len() != want_bytes {
                    return Err(Error::Dimensions {
                        expected: blob.len() / 4,
                        got: q.len(),
                    });
                }
                if let Some(f) = filter {
                    let meta: &str = row.get("metadata");
                    if !f.matches(&parse_metadata(meta)) {
                        continue;
                    }
                }
                let score = dot_blob(&q, blob);
                // Only allocate the id for rows that enter the top k.
                if top.accepts(score) {
                    ids.push(row.get("id"));
                    top.push(score, ids.len() - 1);
                }
            }
        }
        let hits = top.into_sorted();
        if hits.is_empty() {
            return Ok(Vec::new());
        }
        let wanted: Vec<String> = hits.iter().map(|h| ids[h.idx].clone()).collect();
        let mut qb = QueryBuilder::<Sqlite>::new(if with_vectors {
            "SELECT id, content, metadata, embedding FROM sinew_vectors WHERE collection = "
        } else {
            "SELECT id, content, metadata FROM sinew_vectors WHERE collection = "
        });
        qb.push_bind(&self.collection).push(" AND id IN (");
        let mut sep = qb.separated(", ");
        for id in &wanted {
            sep.push_bind(id);
        }
        qb.push(")");
        let mut found = std::collections::HashMap::new();
        for row in qb.build().fetch_all(&self.pool).await? {
            let id: String = row.get("id");
            let vector = with_vectors.then(|| from_blob(row.get::<&[u8], _>("embedding")));
            let doc = Document {
                id: Some(id.clone()),
                content: row.get("content"),
                metadata: parse_metadata(row.get("metadata")),
            };
            found.insert(id, (doc, vector));
        }
        Ok(hits
            .iter()
            .zip(&wanted)
            .filter_map(|(h, id)| {
                found.remove(id).map(|(document, vector)| ScoredDocument {
                    document,
                    score: h.score,
                    vector,
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_roundtrip_and_dot() {
        let v: Vec<f32> = (0..11).map(|i| i as f32 - 3.5).collect();
        let b = to_blob(&v);
        assert_eq!(from_blob(&b), v);
        let naive: f32 = v.iter().map(|x| x * x).sum();
        assert!((dot_blob(&v, &b) - naive).abs() < 1e-3);
    }
}
