//! `sinew-scope`: the LangSmith-style side of Sinew.
//!
//! - [`server`]: axum HTTP server with run ingest (`POST /api/runs`), OTLP/HTTP
//!   JSON ingest (`POST /v1/traces`), a query API, datasets and experiments,
//!   and an embedded web UI at `/`.
//! - [`store`]: SQLite storage (sqlx).
//! - [`otlp`]: OpenTelemetry span → [`sinew_trace::Run`] mapping (GenAI
//!   semantic conventions).
//! - [`view`]: the terminal tree viewer.
//!
//! Start a server in-process (tests, examples):
//!
//! ```no_run
//! # async fn demo() -> std::io::Result<()> {
//! let store = sinew_scope::Store::memory().await.unwrap();
//! let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
//! let addr = listener.local_addr()?;
//! tokio::spawn(sinew_scope::server::serve_with_shutdown(
//!     listener,
//!     sinew_scope::router(store, None),
//!     std::future::pending(),
//! ));
//! println!("UI at http://{addr}/");
//! # Ok(()) }
//! ```

pub mod otlp;
pub mod server;
pub mod store;
pub mod view;

pub use server::router;
pub use store::{Store, StoreError, TraceQuery};
