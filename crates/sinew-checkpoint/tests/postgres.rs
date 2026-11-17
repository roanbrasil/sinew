//! Postgres integration tests. They run only when `SINEW_TEST_PG_URL` is set,
//! e.g. `postgres://postgres:sinew@127.0.0.1:55432/postgres`; otherwise each
//! test returns early. Every test works in its own throwaway schema.
#![cfg(feature = "postgres")]

mod common;

use std::sync::Arc;

use sinew_checkpoint::{
    sqlx::{
        postgres::{PgConnectOptions, PgPoolOptions},
        Executor,
    },
    Checkpointer, PostgresCheckpointer,
};

/// A store in a fresh schema, plus a guard that drops the schema.
struct Fixture {
    store: PostgresCheckpointer,
    admin: sqlx::PgPool,
    schema: String,
}

impl Fixture {
    async fn new() -> Option<Self> {
        let url = std::env::var("SINEW_TEST_PG_URL").ok()?;
        let schema = format!("sinew_test_{}", uuid::Uuid::new_v4().simple());
        let admin = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("connect to SINEW_TEST_PG_URL");
        admin
            .execute(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .await
            .unwrap();
        let opts: PgConnectOptions = url.parse().unwrap();
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect_with(opts.options([("search_path", schema.as_str())]))
            .await
            .unwrap();
        let store = PostgresCheckpointer::from_pool(pool).await.unwrap();
        Some(Self {
            store,
            admin,
            schema,
        })
    }

    async fn cleanup(self) {
        self.store.pool().close().await;
        self.admin
            .execute(sqlx::AssertSqlSafe(format!("DROP SCHEMA {} CASCADE", self.schema)))
            .await
            .unwrap();
    }
}

macro_rules! pg_test {
    ($name:ident, |$f:ident| $body:expr) => {
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn $name() {
            let Some($f) = Fixture::new().await else {
                eprintln!("SINEW_TEST_PG_URL not set; skipping");
                return;
            };
            $body;
            $f.cleanup().await;
        }
    };
}

pg_test!(round_trip, |f| common::round_trip(&f.store).await);
pg_test!(ordering, |f| common::ordering(&f.store).await);
pg_test!(multiple_threads, |f| common::multiple_threads(&f.store)
    .await);
pg_test!(pruning, |f| common::pruning(&f.store).await);
pg_test!(retention_on_put, |f| common::retention_on_put(
    &f.store.clone().with_retention(3)
)
.await);
pg_test!(concurrent_puts, |f| common::concurrent_puts(Arc::new(
    f.store.clone()
))
.await);
pg_test!(concurrent_puts_with_retention, |f| {
    common::concurrent_puts_with_retention(Arc::new(f.store.clone().with_retention(5))).await
});

pg_test!(ensure_schema_is_idempotent_and_race_free, |f| {
    let mut handles = Vec::new();
    for _ in 0..8 {
        let store = f.store.clone();
        handles.push(tokio::spawn(async move { store.ensure_schema().await }));
    }
    for h in handles {
        h.await.unwrap().unwrap();
    }
    f.store.put(common::cp("t", 0, "a", 1)).await.unwrap();
    assert_eq!(f.store.history("t").await.unwrap().len(), 1);
});

pg_test!(checkpoints_are_queryable_jsonb, |f| {
    f.store.put(common::cp("t", 3, "review", 41)).await.unwrap();
    let (n,): (i64,) = sqlx::query_as(
        "SELECT (checkpoint->'state'->>'n')::bigint FROM sinew_checkpoints WHERE step = 3",
    )
    .fetch_one(f.store.pool())
    .await
    .unwrap();
    assert_eq!(n, 41);
});

pg_test!(namespaces_and_ids, |f| common::namespaces_and_ids(&f.store)
    .await);

pg_test!(migrates_tables_from_older_versions, |f| {
    // Recreate the table as the first release did and add an old row.
    let pool = f.store.pool().clone();
    pool.execute("DROP TABLE sinew_checkpoints").await.unwrap();
    pool.execute(
        "CREATE TABLE sinew_checkpoints (
            id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            thread_id  TEXT        NOT NULL,
            step       BIGINT      NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            checkpoint JSONB       NOT NULL
        )",
    )
    .await
    .unwrap();
    pool.execute(
        r#"INSERT INTO sinew_checkpoints (thread_id, step, checkpoint)
           VALUES ('old', 2, '{"thread_id":"old","step":2,"next":"b","state":{"n":7}}')"#,
    )
    .await
    .unwrap();
    let store = PostgresCheckpointer::from_pool(pool).await.unwrap();
    let old = store.latest("old").await.unwrap().unwrap();
    assert!(old.checkpoint_id.starts_with("legacy-"));
    assert_eq!(
        store.get("old", &old.checkpoint_id).await.unwrap(),
        Some(old.clone())
    );
    assert_eq!(store.history_in("old", "").await.unwrap(), vec![old]);
    common::namespaces_and_ids(&store).await;
});
