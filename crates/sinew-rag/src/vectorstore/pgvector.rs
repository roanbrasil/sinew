use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use sqlx::{postgres::PgPoolOptions, PgPool, Postgres, QueryBuilder, Row};

use super::{check_lengths, Filter, ScoredDocument, VectorStore};
use crate::{document::ensure_ids, Document, Embeddings, Error, Metadata, Result};

/// Serializes concurrent schema creation.
const SCHEMA_LOCK: i64 = 0x5_1E7E_7EC7;

const SCHEMA: &[&str] = &[
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE TABLE IF NOT EXISTS sinew_pgvector (
        collection TEXT   NOT NULL,
        id         TEXT   NOT NULL,
        content    TEXT   NOT NULL,
        metadata   JSONB  NOT NULL DEFAULT '{}'::jsonb,
        embedding  vector NOT NULL,
        PRIMARY KEY (collection, id)
    )",
];

/// Postgres + [pgvector](https://github.com/pgvector/pgvector) (feature
/// `pgvector`).
///
/// Rows live in one `sinew_pgvector` table keyed by `(collection, id)`;
/// metadata is `jsonb` and filters are translated to SQL (`@>` for
/// equality, numeric comparisons guarded by `jsonb_typeof`). Scores are
/// cosine similarity (`1 - (embedding <=> query)`).
///
/// The `embedding` column is untyped `vector`, so different collections can
/// use different models. Searches are exact scans unless you set
/// [`with_dimensions`](Self::with_dimensions) and call
/// [`create_hnsw_index`](Self::create_hnsw_index), which indexes
/// `embedding::vector(n)` for approximate search.
///
/// Creating the extension needs a role allowed to do so (or create it once
/// as a superuser beforehand).
#[derive(Clone)]
pub struct PgVectorStore {
    pool: PgPool,
    collection: String,
    dimensions: Option<usize>,
    embeddings: Arc<dyn Embeddings>,
}

impl PgVectorStore {
    /// Connects to `url` and creates the extension and table if needed.
    pub async fn connect(url: &str, embeddings: impl Embeddings + 'static) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(url)
            .await?;
        Self::from_pool(pool, embeddings).await
    }

    pub async fn from_pool(pool: PgPool, embeddings: impl Embeddings + 'static) -> Result<Self> {
        let mut tx = pool.begin().await?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)")
            .bind(SCHEMA_LOCK)
            .execute(&mut *tx)
            .await?;
        for stmt in SCHEMA {
            sqlx::query(stmt).execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(Self {
            pool,
            collection: "default".into(),
            dimensions: None,
            embeddings: Arc::new(embeddings),
        })
    }

    /// Uses another collection (default `"default"`).
    pub fn with_collection(mut self, name: impl Into<String>) -> Self {
        self.collection = name.into();
        self
    }

    /// Declares the vector size so searches use `embedding::vector(n)`,
    /// which an index from [`create_hnsw_index`](Self::create_hnsw_index)
    /// can serve.
    pub fn with_dimensions(mut self, n: usize) -> Self {
        self.dimensions = Some(n);
        self
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Creates an HNSW cosine index on `embedding::vector(n)` (requires
    /// [`with_dimensions`](Self::with_dimensions); every row in the table
    /// must have that size).
    pub async fn create_hnsw_index(&self) -> Result<()> {
        let n = self
            .dimensions
            .ok_or_else(|| Error::Config("create_hnsw_index needs with_dimensions(n)".into()))?;
        let sql = format!(
            "CREATE INDEX IF NOT EXISTS sinew_pgvector_hnsw_{n} ON sinew_pgvector \
             USING hnsw ((embedding::vector({n})) vector_cosine_ops)"
        );
        sqlx::query(&sql).execute(&self.pool).await?;
        Ok(())
    }

    /// Removes every row of this collection.
    pub async fn clear(&self) -> Result<u64> {
        Ok(
            sqlx::query("DELETE FROM sinew_pgvector WHERE collection = $1")
                .bind(&self.collection)
                .execute(&self.pool)
                .await?
                .rows_affected(),
        )
    }

    fn vector_expr(&self) -> String {
        match self.dimensions {
            Some(n) => format!("(embedding::vector({n}))"),
            None => "embedding".into(),
        }
    }

    fn cast(&self) -> String {
        match self.dimensions {
            Some(n) => format!("::vector({n})"),
            None => "::vector".into(),
        }
    }
}

fn vector_literal(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 10 + 2);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&x.to_string());
    }
    s.push(']');
    s
}

fn parse_vector(s: &str) -> Vec<f32> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

fn push_filter(qb: &mut QueryBuilder<'_, Postgres>, f: &Filter) {
    let num_cmp = |qb: &mut QueryBuilder<'_, Postgres>, k: &str, op: &str, x: f64| {
        qb.push("(CASE WHEN jsonb_typeof(metadata->")
            .push_bind(k.to_string())
            .push(") = 'number' THEN (metadata->>")
            .push_bind(k.to_string())
            .push(format!(")::float8 {op} "))
            .push_bind(x)
            .push(" ELSE FALSE END)");
    };
    let contains = |qb: &mut QueryBuilder<'_, Postgres>, k: &str, v: &serde_json::Value| {
        qb.push("metadata @> ")
            .push_bind(json!({ k: v }).to_string())
            .push("::jsonb");
    };
    match f {
        Filter::Eq(k, v) => contains(qb, k, v),
        Filter::Ne(k, v) => {
            qb.push("NOT (");
            contains(qb, k, v);
            qb.push(")");
        }
        Filter::In(k, vs) | Filter::Nin(k, vs) => {
            let negate = matches!(f, Filter::Nin(..));
            if vs.is_empty() {
                qb.push(if negate { "TRUE" } else { "FALSE" });
                return;
            }
            qb.push(if negate { "NOT (" } else { "(" });
            for (i, v) in vs.iter().enumerate() {
                if i > 0 {
                    qb.push(" OR ");
                }
                contains(qb, k, v);
            }
            qb.push(")");
        }
        Filter::Gt(k, x) => num_cmp(qb, k, ">", *x),
        Filter::Gte(k, x) => num_cmp(qb, k, ">=", *x),
        Filter::Lt(k, x) => num_cmp(qb, k, "<", *x),
        Filter::Lte(k, x) => num_cmp(qb, k, "<=", *x),
        Filter::Exists(k) => {
            qb.push("COALESCE(jsonb_typeof(metadata->")
                .push_bind(k.clone())
                .push("), 'null') <> 'null'");
        }
        Filter::And(fs) | Filter::Or(fs) => {
            let and = matches!(f, Filter::And(_));
            if fs.is_empty() {
                qb.push(if and { "TRUE" } else { "FALSE" });
                return;
            }
            qb.push("(");
            for (i, sub) in fs.iter().enumerate() {
                if i > 0 {
                    qb.push(if and { " AND " } else { " OR " });
                }
                push_filter(qb, sub);
            }
            qb.push(")");
        }
        Filter::Not(inner) => {
            qb.push("NOT (");
            push_filter(qb, inner);
            qb.push(")");
        }
    }
}

fn row_doc(row: &sqlx::postgres::PgRow) -> Document {
    let meta: String = row.get("metadata");
    Document {
        id: Some(row.get("id")),
        content: row.get("content"),
        metadata: serde_json::from_str::<Metadata>(&meta).unwrap_or_default(),
    }
}

#[async_trait]
impl VectorStore for PgVectorStore {
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
        if let Some(n) = self.dimensions {
            if let Some(bad) = vectors.iter().find(|v| v.len() != n) {
                return Err(Error::Dimensions {
                    expected: n,
                    got: bad.len(),
                });
            }
        }
        let ids = ensure_ids(&mut docs);
        let mut tx = self.pool.begin().await?;
        // Multi-row upserts, 500 rows per statement.
        let rows: Vec<(&Document, &Vec<f32>)> = docs.iter().zip(&vectors).collect();
        for chunk in rows.chunks(500) {
            let mut qb = QueryBuilder::<Postgres>::new(
                "INSERT INTO sinew_pgvector (collection, id, content, metadata, embedding) ",
            );
            qb.push_values(chunk, |mut b, (d, v)| {
                b.push_bind(&self.collection)
                    .push_bind(d.id.clone())
                    .push_bind(&d.content)
                    .push_bind(serde_json::to_string(&d.metadata).unwrap_or_else(|_| "{}".into()))
                    .push_unseparated("::jsonb")
                    .push_bind(vector_literal(v))
                    .push_unseparated("::vector");
            });
            qb.push(
                " ON CONFLICT (collection, id) DO UPDATE SET content = EXCLUDED.content, \
                 metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
            );
            qb.build().execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(ids)
    }

    async fn delete(&self, ids: &[String]) -> Result<usize> {
        let n = sqlx::query("DELETE FROM sinew_pgvector WHERE collection = $1 AND id = ANY($2)")
            .bind(&self.collection)
            .bind(ids)
            .execute(&self.pool)
            .await?
            .rows_affected();
        Ok(n as usize)
    }

    async fn get(&self, ids: &[String]) -> Result<Vec<Document>> {
        let rows = sqlx::query(
            "SELECT id, content, metadata::text AS metadata FROM sinew_pgvector \
             WHERE collection = $1 AND id = ANY($2)",
        )
        .bind(&self.collection)
        .bind(ids)
        .fetch_all(&self.pool)
        .await?;
        let mut by_id: std::collections::HashMap<String, Document> = rows
            .iter()
            .map(|r| {
                let d = row_doc(r);
                (d.id.clone().unwrap_or_default(), d)
            })
            .collect();
        Ok(ids.iter().filter_map(|id| by_id.remove(id)).collect())
    }

    async fn len(&self) -> Result<usize> {
        let n: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM sinew_pgvector WHERE collection = $1")
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
        let expr = self.vector_expr();
        let cast = self.cast();
        let lit = vector_literal(query);
        let mut qb = QueryBuilder::<Postgres>::new(format!(
            "SELECT id, content, metadata::text AS metadata, 1 - ({expr} <=> "
        ));
        qb.push_bind(lit.clone()).push(format!("{cast}) AS score"));
        if with_vectors {
            qb.push(", embedding::text AS vec");
        }
        qb.push(" FROM sinew_pgvector WHERE collection = ")
            .push_bind(&self.collection);
        if let Some(f) = filter {
            qb.push(" AND ");
            push_filter(&mut qb, f);
        }
        qb.push(format!(" ORDER BY {expr} <=> "))
            .push_bind(lit)
            .push(format!("{cast} LIMIT "))
            .push_bind(k as i64);
        let rows = qb.build().fetch_all(&self.pool).await?;
        Ok(rows
            .iter()
            .map(|r| {
                let vector = with_vectors.then(|| {
                    let mut v = parse_vector(&r.get::<String, _>("vec"));
                    super::normalize(&mut v);
                    v
                });
                ScoredDocument {
                    document: row_doc(r),
                    score: r.get::<f64, _>("score") as f32,
                    vector,
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_text_roundtrip() {
        let v = vec![0.5, -1.25, 3.0];
        assert_eq!(vector_literal(&v), "[0.5,-1.25,3]");
        assert_eq!(parse_vector("[0.5,-1.25,3]"), v);
    }

    #[test]
    fn filter_sql() {
        let mut qb = QueryBuilder::<Postgres>::new("");
        push_filter(
            &mut qb,
            &Filter::eq("lang", "en")
                .and(Filter::gte("year", 2020.0))
                .and(Filter::is_in("tag", ["a", "b"]).not()),
        );
        let sql = qb.sql().to_string();
        assert!(sql.starts_with("(metadata @> $1::jsonb AND (CASE WHEN jsonb_typeof(metadata->$2)"));
        assert!(sql.contains("NOT ((metadata @> $5::jsonb OR metadata @> $6::jsonb))"));
    }
}
