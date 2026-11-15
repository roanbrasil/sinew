#![cfg(feature = "sqlite")]

mod common;

use std::sync::Arc;

use sinew_checkpoint::{CheckpointStore, Checkpointer, SqliteCheckpointer};
use tempfile::TempDir;

async fn memory() -> SqliteCheckpointer {
    SqliteCheckpointer::in_memory().await.unwrap()
}

/// A file database in a temp dir; keep the `TempDir` alive while using it.
async fn file() -> (SqliteCheckpointer, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}", dir.path().join("cp.db").display());
    (SqliteCheckpointer::connect(&url).await.unwrap(), dir)
}

macro_rules! both {
    ($($name:ident => $scenario:ident),* $(,)?) => {
        mod in_memory {
            use super::*;
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
            async fn $name() {
                common::$scenario(&memory().await).await;
            })*
        }
        mod on_file {
            use super::*;
            $(#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
            async fn $name() {
                let (store, _dir) = file().await;
                common::$scenario(&store).await;
            })*
        }
    };
}

both! {
    round_trip => round_trip,
    ordering => ordering,
    multiple_threads => multiple_threads,
    pruning => pruning,
}

#[tokio::test]
async fn retention_in_memory() {
    common::retention_on_put(&memory().await.with_retention(3)).await;
}

#[tokio::test]
async fn retention_on_file() {
    let (store, _dir) = file().await;
    common::retention_on_put(&store.with_retention(3)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_puts_in_memory() {
    common::concurrent_puts(Arc::new(memory().await)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_puts_on_file() {
    let (store, _dir) = file().await;
    common::concurrent_puts(Arc::new(store)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_puts_with_retention_on_file() {
    let (store, _dir) = file().await;
    common::concurrent_puts_with_retention(Arc::new(store.with_retention(5))).await;
}

/// Two independent pools on the same file, as two processes would have.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_pools_share_a_file() {
    let (a, dir) = file().await;
    let url = format!("sqlite://{}", dir.path().join("cp.db").display());
    let b = SqliteCheckpointer::connect(&url).await.unwrap();
    tokio::join!(
        async {
            for i in 0..50 {
                a.put(common::cp("t", i, "n", i as i64)).await.unwrap();
            }
        },
        async {
            for i in 0..50 {
                b.put(common::cp("t", i, "n", 100 + i as i64))
                    .await
                    .unwrap();
            }
        }
    );
    assert_eq!(a.history("t").await.unwrap().len(), 100);
    assert_eq!(b.history("t").await.unwrap().len(), 100);
}

#[tokio::test]
async fn data_survives_reopen_and_schema_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}", dir.path().join("cp.db").display());
    {
        let store = SqliteCheckpointer::connect(&url).await.unwrap();
        store.put(common::cp("t", 0, "a", 1)).await.unwrap();
        store.put(common::cp("t", 1, "b", 2)).await.unwrap();
        store.ensure_schema().await.unwrap();
        store.pool().close().await;
    }
    let store = SqliteCheckpointer::connect(&url).await.unwrap();
    assert_eq!(store.history("t").await.unwrap().len(), 2);
    assert_eq!(store.latest("t").await.unwrap().unwrap().next, "b");
    assert_eq!(store.list_threads().await.unwrap(), vec!["t"]);
}

#[tokio::test]
async fn from_existing_pool() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}?mode=rwc", dir.path().join("cp.db").display());
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let store = SqliteCheckpointer::from_pool(pool).await.unwrap();
    store.put(common::cp("t", 0, "a", 1)).await.unwrap();
    assert_eq!(store.latest("t").await.unwrap().unwrap().step, 0);
}

#[tokio::test]
async fn in_memory_databases_are_private() {
    let a = memory().await;
    let b = memory().await;
    a.put(common::cp("t", 0, "a", 1)).await.unwrap();
    assert_eq!(b.latest("t").await.unwrap(), None);
    // Clones share the same database.
    let a2 = a.clone();
    assert!(a2.latest("t").await.unwrap().is_some());
}
