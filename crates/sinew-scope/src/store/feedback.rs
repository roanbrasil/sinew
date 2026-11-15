//! Feedback on runs, annotation queues, appending examples to datasets and
//! pairwise experiment storage.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sinew_eval::{Dataset, Example, PairwiseReport};
use sqlx::{sqlite::SqliteRow, Row};

use super::{
    from_json, now_ms, put_dataset_tx, row_to_run, to_json, Result, Store, StoreError, StoredRun,
};

/// Body of `POST /api/feedback`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct FeedbackInput {
    pub run_id: String,
    pub key: String,
    pub score: Option<f64>,
    pub value: Option<Value>,
    pub comment: Option<String>,
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackRow {
    pub id: String,
    pub run_id: String,
    pub trace_id: Option<String>,
    pub key: String,
    pub score: Option<f64>,
    pub value: Option<Value>,
    pub comment: Option<String>,
    pub source: String,
    pub created_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Queue {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Feedback keys reviewers score.
    pub keys: Vec<String>,
    pub created_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueueSummary {
    #[serde(flatten)]
    pub queue: Queue,
    pub pending: i64,
    pub done: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueueItem {
    pub run_id: String,
    pub status: String,
    pub added_ms: i64,
    pub done_ms: Option<i64>,
    /// The run (absent if it was deleted by retention).
    pub run: Option<StoredRun>,
}

fn row_to_feedback(r: &SqliteRow) -> Result<FeedbackRow> {
    Ok(FeedbackRow {
        id: r.try_get("id")?,
        run_id: r.try_get("run_id")?,
        trace_id: r.try_get("trace_id")?,
        key: r.try_get("key")?,
        score: r.try_get("score")?,
        value: r
            .try_get::<Option<String>, _>("value")?
            .map(|v| from_json(&v)),
        comment: r.try_get("comment")?,
        source: r.try_get("source")?,
        created_ms: r.try_get("created_ms")?,
    })
}

fn row_to_queue(r: &SqliteRow) -> Result<Queue> {
    Ok(Queue {
        id: r.try_get("id")?,
        name: r.try_get("name")?,
        description: r.try_get("description")?,
        keys: serde_json::from_str(&r.try_get::<String, _>("keys")?).unwrap_or_default(),
        created_ms: r.try_get("created_ms")?,
    })
}

impl Store {
    /// Stores feedback on a run. The run must exist.
    pub async fn add_feedback(&self, f: &FeedbackInput) -> Result<FeedbackRow> {
        if f.key.trim().is_empty() || f.key.len() > 200 {
            return Err(StoreError::Invalid(
                "feedback key is required (max 200 chars)".into(),
            ));
        }
        if f.score.is_none() && f.value.is_none() && f.comment.is_none() {
            return Err(StoreError::Invalid(
                "feedback needs a score, a value or a comment".into(),
            ));
        }
        if f.score.is_some_and(|s| !s.is_finite()) {
            return Err(StoreError::Invalid("score must be a finite number".into()));
        }
        let trace_id: Option<String> = sqlx::query_scalar("SELECT trace_id FROM runs WHERE id = ?")
            .bind(&f.run_id)
            .fetch_optional(self.pool())
            .await?;
        let Some(trace_id) = trace_id else {
            return Err(StoreError::Invalid(format!("unknown run `{}`", f.run_id)));
        };
        let row = FeedbackRow {
            id: uuid_like(),
            run_id: f.run_id.clone(),
            trace_id: Some(trace_id),
            key: f.key.trim().to_string(),
            score: f.score,
            value: f.value.clone(),
            comment: f.comment.clone().filter(|c| !c.trim().is_empty()),
            source: f
                .source
                .clone()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "api".into()),
            created_ms: now_ms(),
        };
        sqlx::query(
            "INSERT INTO feedback (id, run_id, trace_id, key, score, value, comment, source, created_ms) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&row.id)
        .bind(&row.run_id)
        .bind(&row.trace_id)
        .bind(&row.key)
        .bind(row.score)
        .bind(row.value.as_ref().map(to_json))
        .bind(&row.comment)
        .bind(&row.source)
        .bind(row.created_ms)
        .execute(self.pool())
        .await?;
        Ok(row)
    }

    /// Feedback filtered by run, trace and/or key (oldest first, max 1000).
    pub async fn feedback(
        &self,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        key: Option<&str>,
    ) -> Result<Vec<FeedbackRow>> {
        let mut qb = sqlx::QueryBuilder::new("SELECT * FROM feedback WHERE 1 = 1");
        if let Some(r) = run_id.filter(|s| !s.is_empty()) {
            qb.push(" AND run_id = ").push_bind(r.to_string());
        }
        if let Some(t) = trace_id.filter(|s| !s.is_empty()) {
            qb.push(" AND trace_id = ").push_bind(t.to_string());
        }
        if let Some(k) = key.filter(|s| !s.is_empty()) {
            qb.push(" AND key = ").push_bind(k.to_string());
        }
        qb.push(" ORDER BY created_ms, id LIMIT 1000");
        qb.build()
            .fetch_all(self.pool())
            .await?
            .iter()
            .map(row_to_feedback)
            .collect()
    }

    /// Mean score and count per key, optionally within a project and time
    /// range (by run start).
    pub async fn feedback_stats(
        &self,
        project: Option<&str>,
        from_ms: Option<i64>,
        to_ms: Option<i64>,
    ) -> Result<Vec<Value>> {
        let mut qb = sqlx::QueryBuilder::new(
            "SELECT f.key, COUNT(*) AS n, AVG(f.score) AS mean, COUNT(f.score) AS scored \
             FROM feedback f JOIN runs r ON r.id = f.run_id WHERE 1 = 1",
        );
        if let Some(p) = project.filter(|p| !p.is_empty()) {
            qb.push(" AND r.project = ").push_bind(p.to_string());
        }
        if let Some(f) = from_ms {
            qb.push(" AND r.start_ms >= ").push_bind(f);
        }
        if let Some(t) = to_ms {
            qb.push(" AND r.start_ms <= ").push_bind(t);
        }
        qb.push(" GROUP BY f.key ORDER BY f.key");
        let rows = qb.build().fetch_all(self.pool()).await?;
        rows.iter()
            .map(|r| {
                Ok(json!({
                    "key": r.try_get::<String, _>("key")?,
                    "count": r.try_get::<i64, _>("n")?,
                    "scored": r.try_get::<i64, _>("scored")?,
                    "mean": r.try_get::<Option<f64>, _>("mean")?,
                }))
            })
            .collect()
    }

    pub async fn delete_feedback(&self, id: &str) -> Result<bool> {
        let r = sqlx::query("DELETE FROM feedback WHERE id = ?")
            .bind(id)
            .execute(self.pool())
            .await?;
        Ok(r.rows_affected() > 0)
    }

    /// Appends an example to a dataset, creating the dataset if needed.
    /// Returns the example count.
    pub async fn add_example(&self, dataset: &str, example: &Example) -> Result<i64> {
        if dataset.trim().is_empty() {
            return Err(StoreError::Invalid("dataset name is required".into()));
        }
        let mut tx = self.pool().begin().await?;
        let exists: Option<i64> = sqlx::query_scalar("SELECT 1 FROM datasets WHERE name = ?")
            .bind(dataset)
            .fetch_optional(&mut *tx)
            .await?;
        if exists.is_none() {
            put_dataset_tx(
                &mut tx,
                &Dataset::new(dataset).with_description("created from runs"),
            )
            .await?;
        }
        let next: i64 =
            sqlx::query_scalar("SELECT COALESCE(MAX(idx) + 1, 0) FROM examples WHERE dataset = ?")
                .bind(dataset)
                .fetch_one(&mut *tx)
                .await?;
        sqlx::query(
            "INSERT INTO examples (dataset, idx, id, inputs, reference, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(dataset)
        .bind(next)
        .bind(&example.id)
        .bind(to_json(&example.inputs))
        .bind(example.reference.as_ref().map(to_json))
        .bind(to_json(&example.metadata))
        .execute(&mut *tx)
        .await?;
        sqlx::query("UPDATE datasets SET updated_ms = ? WHERE name = ?")
            .bind(now_ms())
            .bind(dataset)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(next + 1)
    }

    pub async fn create_queue(
        &self,
        name: &str,
        description: &str,
        keys: &[String],
    ) -> Result<Queue> {
        if name.trim().is_empty() {
            return Err(StoreError::Invalid("queue name is required".into()));
        }
        let q = Queue {
            id: uuid_like(),
            name: name.trim().to_string(),
            description: description.to_string(),
            keys: keys
                .iter()
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .collect(),
            created_ms: now_ms(),
        };
        sqlx::query(
            "INSERT INTO annotation_queues (id, name, description, keys, created_ms) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&q.id)
        .bind(&q.name)
        .bind(&q.description)
        .bind(serde_json::to_string(&q.keys).unwrap_or_else(|_| "[]".into()))
        .bind(q.created_ms)
        .execute(self.pool())
        .await?;
        Ok(q)
    }

    pub async fn queues(&self) -> Result<Vec<QueueSummary>> {
        let rows = sqlx::query(
            "SELECT q.*, \
             (SELECT COUNT(*) FROM queue_items i WHERE i.queue_id = q.id AND i.status = 'pending') AS pending, \
             (SELECT COUNT(*) FROM queue_items i WHERE i.queue_id = q.id AND i.status = 'done') AS done \
             FROM annotation_queues q ORDER BY q.created_ms DESC",
        )
        .fetch_all(self.pool())
        .await?;
        rows.iter()
            .map(|r| {
                Ok(QueueSummary {
                    queue: row_to_queue(r)?,
                    pending: r.try_get("pending")?,
                    done: r.try_get("done")?,
                })
            })
            .collect()
    }

    pub async fn queue(&self, id: &str) -> Result<Option<Queue>> {
        sqlx::query("SELECT * FROM annotation_queues WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await?
            .as_ref()
            .map(row_to_queue)
            .transpose()
    }

    pub async fn delete_queue(&self, id: &str) -> Result<bool> {
        let mut tx = self.pool().begin().await?;
        sqlx::query("DELETE FROM queue_items WHERE queue_id = ?")
            .bind(id)
            .execute(&mut *tx)
            .await?;
        let r = sqlx::query("DELETE FROM annotation_queues WHERE id = ?")
            .bind(id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(r.rows_affected() > 0)
    }

    /// Adds runs to a queue (already queued runs are left as they are).
    /// Returns how many were added.
    pub async fn enqueue(&self, queue_id: &str, run_ids: &[String]) -> Result<u64> {
        if self.queue(queue_id).await?.is_none() {
            return Err(StoreError::Invalid(format!("unknown queue `{queue_id}`")));
        }
        let now = now_ms();
        let mut tx = self.pool().begin().await?;
        let mut added = 0;
        for id in run_ids {
            let r = sqlx::query(
                "INSERT INTO queue_items (queue_id, run_id, status, added_ms) \
                 SELECT ?, id, 'pending', ? FROM runs WHERE id = ? \
                 ON CONFLICT(queue_id, run_id) DO NOTHING",
            )
            .bind(queue_id)
            .bind(now)
            .bind(id)
            .execute(&mut *tx)
            .await?;
            added += r.rows_affected();
        }
        tx.commit().await?;
        Ok(added)
    }

    /// Items of a queue, optionally by status (`pending` / `done`), oldest
    /// first, with their runs.
    pub async fn queue_items(
        &self,
        queue_id: &str,
        status: Option<&str>,
        limit: i64,
    ) -> Result<Vec<QueueItem>> {
        let mut qb = sqlx::QueryBuilder::new(
            "SELECT i.run_id AS item_run_id, i.status, i.added_ms, i.done_ms, r.* \
             FROM queue_items i LEFT JOIN runs r ON r.id = i.run_id WHERE i.queue_id = ",
        );
        qb.push_bind(queue_id.to_string());
        if let Some(s) = status.filter(|s| !s.is_empty()) {
            qb.push(" AND i.status = ").push_bind(s.to_string());
        }
        qb.push(" ORDER BY i.added_ms, i.run_id LIMIT ")
            .push_bind(limit.clamp(1, 1000));
        let rows = qb.build().fetch_all(self.pool()).await?;
        rows.iter()
            .map(|r| {
                let has_run = r.try_get::<Option<String>, _>("id")?.is_some();
                Ok(QueueItem {
                    run_id: r.try_get("item_run_id")?,
                    status: r.try_get("status")?,
                    added_ms: r.try_get("added_ms")?,
                    done_ms: r.try_get("done_ms")?,
                    run: if has_run { Some(row_to_run(r)?) } else { None },
                })
            })
            .collect()
    }

    /// Marks an item done, storing the reviewer's feedback (source
    /// `annotation:<queue name>`).
    pub async fn complete_queue_item(
        &self,
        queue_id: &str,
        run_id: &str,
        feedback: &[FeedbackInput],
    ) -> Result<bool> {
        let Some(q) = self.queue(queue_id).await? else {
            return Ok(false);
        };
        for f in feedback {
            let mut f = f.clone();
            f.run_id = run_id.to_string();
            if f.source.is_none() {
                f.source = Some(format!("annotation:{}", q.name));
            }
            if f.score.is_some() || f.value.is_some() || f.comment.is_some() {
                self.add_feedback(&f).await?;
            }
        }
        let r = sqlx::query(
            "UPDATE queue_items SET status = 'done', done_ms = ? WHERE queue_id = ? AND run_id = ?",
        )
        .bind(now_ms())
        .bind(queue_id)
        .bind(run_id)
        .execute(self.pool())
        .await?;
        Ok(r.rows_affected() > 0)
    }

    pub async fn remove_queue_item(&self, queue_id: &str, run_id: &str) -> Result<bool> {
        let r = sqlx::query("DELETE FROM queue_items WHERE queue_id = ? AND run_id = ?")
            .bind(queue_id)
            .bind(run_id)
            .execute(self.pool())
            .await?;
        Ok(r.rows_affected() > 0)
    }

    /// Stores a pairwise comparison (upsert by id).
    pub async fn put_pairwise(&self, r: &PairwiseReport) -> Result<()> {
        if r.id.is_empty() {
            return Err(StoreError::Invalid("pairwise report id is required".into()));
        }
        let report = serde_json::to_string(r).map_err(|e| StoreError::Invalid(e.to_string()))?;
        sqlx::query(
            "INSERT INTO pairwise (id, name, dataset, a_id, b_id, created_ms, summary, report) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name, \
             dataset = excluded.dataset, a_id = excluded.a_id, b_id = excluded.b_id, \
             created_ms = excluded.created_ms, summary = excluded.summary, report = excluded.report",
        )
        .bind(&r.id)
        .bind(&r.name)
        .bind(&r.dataset)
        .bind(&r.experiment_a.id)
        .bind(&r.experiment_b.id)
        .bind(r.created_ms as i64)
        .bind(serde_json::to_string(&r.summary).unwrap_or_else(|_| "{}".into()))
        .bind(report)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Pairwise comparisons (summaries), newest first.
    pub async fn pairwise_list(&self, dataset: Option<&str>) -> Result<Vec<Value>> {
        let mut qb = sqlx::QueryBuilder::new(
            "SELECT id, name, dataset, a_id, b_id, created_ms, summary FROM pairwise",
        );
        if let Some(d) = dataset.filter(|d| !d.is_empty()) {
            qb.push(" WHERE dataset = ").push_bind(d.to_string());
        }
        qb.push(" ORDER BY created_ms DESC");
        let rows = qb.build().fetch_all(self.pool()).await?;
        rows.iter()
            .map(|r| {
                Ok(json!({
                    "id": r.try_get::<String, _>("id")?,
                    "name": r.try_get::<String, _>("name")?,
                    "dataset": r.try_get::<String, _>("dataset")?,
                    "a_id": r.try_get::<String, _>("a_id")?,
                    "b_id": r.try_get::<String, _>("b_id")?,
                    "created_ms": r.try_get::<i64, _>("created_ms")?,
                    "summary": from_json(&r.try_get::<String, _>("summary")?),
                }))
            })
            .collect()
    }

    pub async fn pairwise(&self, id: &str) -> Result<Option<PairwiseReport>> {
        let text: Option<String> = sqlx::query_scalar("SELECT report FROM pairwise WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await?;
        text.map(|t| serde_json::from_str(&t).map_err(|e| StoreError::Invalid(e.to_string())))
            .transpose()
    }
}

/// A random UUID v4 string.
pub(crate) fn uuid_like() -> String {
    uuid::Uuid::new_v4().to_string()
}
