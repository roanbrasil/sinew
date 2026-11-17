//! Local axum server that replays recorded responses and records requests.

use std::sync::{Arc, Mutex};

use axum::{
    body::Bytes,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
    Router,
};

/// Loads `tests/fixtures/<name>`.
macro_rules! fixture {
    ($name:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/",
            $name
        ))
    };
}
pub(crate) use fixture;

/// One scripted reply, chosen by request path.
#[derive(Clone)]
pub(crate) struct Reply {
    pub path: &'static str,
    pub status: u16,
    pub headers: Vec<(&'static str, String)>,
    pub body: String,
}

impl Reply {
    pub fn json(path: &'static str, body: &str) -> Self {
        Self::new(path, 200, "application/json", body)
    }

    pub fn new(path: &'static str, status: u16, content_type: &str, body: &str) -> Self {
        Self {
            path,
            status,
            headers: vec![("content-type", content_type.to_string())],
            body: body.to_string(),
        }
    }

    pub fn header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }
}

/// A request the mock received.
#[derive(Debug, Clone)]
pub(crate) struct Recorded {
    pub method: Method,
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: String,
}

pub(crate) struct Mock {
    /// `http://127.0.0.1:<port>`.
    pub base: String,
    pub requests: Arc<Mutex<Vec<Recorded>>>,
}

impl Mock {
    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }
}

pub(crate) async fn serve(replies: Vec<Reply>) -> Mock {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let log = requests.clone();
    let replies = Arc::new(replies);
    let app = Router::new().fallback(
        move |method: Method, uri: Uri, headers: HeaderMap, body: Bytes| {
            let log = log.clone();
            let replies = replies.clone();
            async move {
                log.lock().unwrap().push(Recorded {
                    method,
                    uri: uri.clone(),
                    headers,
                    body: String::from_utf8_lossy(&body).into_owned(),
                });
                match replies.iter().find(|r| r.path == uri.path()) {
                    Some(r) => {
                        let mut resp: Response = r.body.clone().into_response();
                        *resp.status_mut() = StatusCode::from_u16(r.status).unwrap();
                        for (k, v) in &r.headers {
                            resp.headers_mut().insert(*k, v.parse().unwrap());
                        }
                        resp
                    }
                    None => (StatusCode::NOT_FOUND, "no reply scripted").into_response(),
                }
            }
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Mock {
        base: format!("http://{addr}"),
        requests,
    }
}
