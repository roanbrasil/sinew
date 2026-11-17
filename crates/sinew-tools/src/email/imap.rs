//! `email_list` and `email_read` over IMAP, read-only.

use std::{
    io,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use async_imap::{types::Flag, Session};
use futures_util::StreamExt;
use serde::Serialize;
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::TcpStream,
};
use tokio_rustls::{
    client::TlsStream,
    rustls::{self, pki_types::ServerName},
};

use crate::{strip_tags, truncate_chars, ToolError};

/// Largest number of messages one `email_list` call returns.
pub const MAX_LIST: usize = 50;

/// Connection settings for an IMAP mailbox. Every call opens a connection,
/// logs in, examines the mailbox read-only and logs out. Cheap to clone.
///
/// ```no_run
/// use sinew_core::ToolBox;
/// use sinew_tools::email::{email_list, email_read, ImapReader};
///
/// let imap = ImapReader::new("imap.example.com", "bot@example.com", "app-password");
/// let tools = ToolBox::new()
///     .with(email_list::tool(imap.clone()))
///     .with(email_read::tool(imap));
/// # let _ = tools;
/// ```
#[derive(Clone)]
pub struct ImapReader {
    host: String,
    port: u16,
    tls: bool,
    username: String,
    password: Arc<str>,
    mailbox: String,
    max_body_chars: usize,
    timeout: Duration,
}

impl std::fmt::Debug for ImapReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImapReader")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("tls", &self.tls)
            .field("username", &self.username)
            .field("mailbox", &self.mailbox)
            .finish_non_exhaustive()
    }
}

/// One line of [`email_list`](fn@email_list).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmailSummary {
    /// Pass it to `email_read`.
    pub uid: u32,
    pub from: String,
    pub to: String,
    pub subject: String,
    pub date: String,
    pub seen: bool,
}

/// A message from [`email_read`](fn@email_read).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmailMessage {
    pub uid: u32,
    pub from: String,
    pub to: String,
    pub cc: String,
    pub subject: String,
    pub date: String,
    /// Plain-text body (HTML-only messages are converted to text).
    pub text: String,
    /// File names of attachments (not downloaded).
    pub attachments: Vec<String>,
    pub truncated: bool,
}

#[derive(Debug)]
enum Conn {
    Plain(TcpStream),
    Tls(Box<TlsStream<TcpStream>>),
}

impl AsyncRead for Conn {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Conn::Plain(s) => Pin::new(s).poll_read(cx, buf),
            Conn::Tls(s) => Pin::new(s.as_mut()).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for Conn {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Conn::Plain(s) => Pin::new(s).poll_write(cx, buf),
            Conn::Tls(s) => Pin::new(s.as_mut()).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Conn::Plain(s) => Pin::new(s).poll_flush(cx),
            Conn::Tls(s) => Pin::new(s.as_mut()).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Conn::Plain(s) => Pin::new(s).poll_shutdown(cx),
            Conn::Tls(s) => Pin::new(s.as_mut()).poll_shutdown(cx),
        }
    }
}

fn imap_err(e: impl std::fmt::Display) -> ToolError {
    ToolError::Request(format!("IMAP: {e}"))
}

fn env(k: &str) -> Result<String, ToolError> {
    std::env::var(k)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| ToolError::Config(format!("{k} is not set")))
}

impl ImapReader {
    /// IMAP over TLS on port 993, mailbox `INBOX`.
    pub fn new(
        host: impl Into<String>,
        username: impl Into<String>,
        password: impl AsRef<str>,
    ) -> Self {
        Self {
            host: host.into(),
            port: 993,
            tls: true,
            username: username.into(),
            password: Arc::from(password.as_ref()),
            mailbox: "INBOX".into(),
            max_body_chars: 20_000,
            timeout: Duration::from_secs(30),
        }
    }

    /// `IMAP_HOST`, `IMAP_USERNAME`, `IMAP_PASSWORD` and optionally
    /// `IMAP_PORT` from the environment.
    pub fn from_env() -> Result<Self, ToolError> {
        let mut r = Self::new(
            env("IMAP_HOST")?,
            env("IMAP_USERNAME")?,
            env("IMAP_PASSWORD")?,
        );
        if let Ok(p) = env("IMAP_PORT") {
            r.port = p
                .trim()
                .parse()
                .map_err(|_| ToolError::Config(format!("IMAP_PORT `{p}` is not a port")))?;
        }
        Ok(r)
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Plain TCP without TLS. Only for a server on the same machine (a
    /// local bridge such as Proton Mail Bridge) or tests: the password is
    /// sent in clear text.
    pub fn insecure_plaintext(mut self) -> Self {
        self.tls = false;
        self
    }

    /// Mailbox used when the model does not name one (default `INBOX`).
    pub fn mailbox(mut self, name: impl Into<String>) -> Self {
        self.mailbox = name.into();
        self
    }

    /// Body characters returned by `email_read` at most (default 20 000).
    pub fn max_body_chars(mut self, n: usize) -> Self {
        self.max_body_chars = n;
        self
    }

    /// Timeout for a whole call (default 30 s).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    async fn connect(&self) -> Result<Session<Conn>, ToolError> {
        let tcp = TcpStream::connect((self.host.as_str(), self.port))
            .await
            .map_err(|e| {
                ToolError::Request(format!(
                    "cannot connect to {}:{}: {e}",
                    self.host, self.port
                ))
            })?;
        let conn = if self.tls {
            let mut roots = rustls::RootCertStore::empty();
            roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            let config = rustls::ClientConfig::builder_with_provider(Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .map_err(imap_err)?
            .with_root_certificates(roots)
            .with_no_client_auth();
            let name = ServerName::try_from(self.host.clone())
                .map_err(|e| ToolError::Config(format!("bad IMAP host name: {e}")))?;
            let tls = tokio_rustls::TlsConnector::from(Arc::new(config))
                .connect(name, tcp)
                .await
                .map_err(|e| {
                    ToolError::Request(format!("TLS handshake with {} failed: {e}", self.host))
                })?;
            Conn::Tls(Box::new(tls))
        } else {
            Conn::Plain(tcp)
        };
        let mut client = async_imap::Client::new(conn);
        client
            .read_response()
            .await
            .map_err(imap_err)?
            .ok_or_else(|| imap_err("the server closed the connection before its greeting"))?;
        client
            .login(&self.username, self.password.as_ref())
            .await
            .map_err(|(e, _)| ToolError::Request(format!("IMAP login failed: {e}")))
    }

    async fn with_session<T, F>(&self, f: F) -> Result<T, ToolError>
    where
        F: for<'a> FnOnce(
            &'a mut Session<Conn>,
        ) -> Pin<
            Box<dyn std::future::Future<Output = Result<T, ToolError>> + Send + 'a>,
        >,
    {
        let work = async {
            let mut session = self.connect().await?;
            let out = f(&mut session).await;
            let _ = session.logout().await;
            out
        };
        tokio::time::timeout(self.timeout, work)
            .await
            .map_err(|_| {
                ToolError::Request(format!("IMAP call timed out after {:?}", self.timeout))
            })?
    }

    /// Newest messages first.
    pub async fn list(
        &self,
        mailbox: Option<&str>,
        unread_only: bool,
        limit: usize,
    ) -> Result<Vec<EmailSummary>, ToolError> {
        let mailbox = mailbox
            .filter(|m| !m.trim().is_empty())
            .unwrap_or(&self.mailbox)
            .to_string();
        let limit = limit.clamp(1, MAX_LIST);
        self.with_session(move |s| {
            Box::pin(async move {
                s.examine(&mailbox).await.map_err(imap_err)?;
                let mut uids: Vec<u32> = s
                    .uid_search(if unread_only { "UNSEEN" } else { "ALL" })
                    .await
                    .map_err(imap_err)?
                    .into_iter()
                    .collect();
                uids.sort_unstable_by_key(|u| std::cmp::Reverse(*u));
                uids.truncate(limit);
                if uids.is_empty() {
                    return Ok(Vec::new());
                }
                let set = uids
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                let mut out = Vec::new();
                let mut stream = s
                    .uid_fetch(&set, "(UID FLAGS BODY.PEEK[HEADER])")
                    .await
                    .map_err(imap_err)?;
                while let Some(f) = stream.next().await {
                    let f = f.map_err(imap_err)?;
                    let Some(uid) = f.uid else { continue };
                    let (headers, _) = mailparse::parse_headers(f.header().unwrap_or_default())
                        .map_err(imap_err)?;
                    let h = |k: &str| headers_value(&headers, k);
                    out.push(EmailSummary {
                        uid,
                        from: h("From"),
                        to: h("To"),
                        subject: h("Subject"),
                        date: h("Date"),
                        seen: f.flags().any(|fl| fl == Flag::Seen),
                    });
                }
                out.sort_by_key(|e| std::cmp::Reverse(e.uid));
                Ok(out)
            })
        })
        .await
    }

    /// One message by UID.
    pub async fn read(&self, uid: u32, mailbox: Option<&str>) -> Result<EmailMessage, ToolError> {
        let mailbox = mailbox
            .filter(|m| !m.trim().is_empty())
            .unwrap_or(&self.mailbox)
            .to_string();
        let max_chars = self.max_body_chars;
        self.with_session(move |s| {
            Box::pin(async move {
                s.examine(&mailbox).await.map_err(imap_err)?;
                let mut stream = s
                    .uid_fetch(uid.to_string(), "(UID FLAGS BODY.PEEK[])")
                    .await
                    .map_err(imap_err)?;
                let mut raw = None;
                while let Some(f) = stream.next().await {
                    let f = f.map_err(imap_err)?;
                    if f.uid == Some(uid) {
                        raw = f.body().map(<[u8]>::to_vec);
                    }
                }
                let raw = raw.ok_or_else(|| {
                    ToolError::Invalid(format!(
                        "no message with uid {uid} in `{mailbox}`; list the mailbox first"
                    ))
                })?;
                parse_message(uid, &raw, max_chars)
            })
        })
        .await
    }
}

fn headers_value(headers: &[mailparse::MailHeader<'_>], key: &str) -> String {
    use mailparse::MailHeaderMap;
    headers
        .get_first_value(key)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn parse_message(uid: u32, raw: &[u8], max_chars: usize) -> Result<EmailMessage, ToolError> {
    let mail = mailparse::parse_mail(raw).map_err(imap_err)?;
    let h = |k: &str| headers_value(&mail.headers, k);
    let mut plain = None;
    let mut html = None;
    let mut attachments = Vec::new();
    collect_parts(&mail, &mut plain, &mut html, &mut attachments);
    let text = plain
        .or_else(|| html.map(|h| strip_tags(&h)))
        .unwrap_or_default();
    let (text, truncated) = truncate_chars(text.trim(), max_chars);
    Ok(EmailMessage {
        uid,
        from: h("From"),
        to: h("To"),
        cc: h("Cc"),
        subject: h("Subject"),
        date: h("Date"),
        text,
        attachments,
        truncated,
    })
}

fn collect_parts(
    part: &mailparse::ParsedMail<'_>,
    plain: &mut Option<String>,
    html: &mut Option<String>,
    attachments: &mut Vec<String>,
) {
    let disposition = part.get_content_disposition();
    if disposition.disposition == mailparse::DispositionType::Attachment {
        let name = disposition
            .params
            .get("filename")
            .or_else(|| part.ctype.params.get("name"))
            .cloned()
            .unwrap_or_else(|| "(unnamed)".into());
        attachments.push(name);
        return;
    }
    if part.subparts.is_empty() {
        let mime = part.ctype.mimetype.to_ascii_lowercase();
        if mime == "text/plain" && plain.is_none() {
            *plain = part.get_body().ok();
        } else if mime == "text/html" && html.is_none() {
            *html = part.get_body().ok();
        }
    }
    for sub in &part.subparts {
        collect_parts(sub, plain, html, attachments);
    }
}

/// List the most recent e-mails in a mailbox (newest first) with sender,
/// subject, date and whether they were read. Read-only: nothing is marked as
/// read. Use the returned uid with `email_read` to open a message.
#[sinew_core::tool]
pub async fn email_list(
    #[state] imap: ImapReader,
    /// Mailbox (folder) name; defaults to the application's mailbox, usually INBOX.
    mailbox: Option<String>,
    /// Only messages not yet read.
    unread_only: Option<bool>,
    /// Number of messages, 1 to 50 (default 10).
    limit: Option<usize>,
) -> Result<Vec<EmailSummary>, ToolError> {
    imap.list(
        mailbox.as_deref(),
        unread_only.unwrap_or(false),
        limit.unwrap_or(10),
    )
    .await
}

/// Read one e-mail by uid (from `email_list`): headers, plain-text body and
/// attachment names. Read-only: the message is not marked as read.
#[sinew_core::tool]
pub async fn email_read(
    #[state] imap: ImapReader,
    /// Message uid from email_list.
    uid: u32,
    /// Mailbox the uid belongs to; defaults to the application's mailbox.
    mailbox: Option<String>,
) -> Result<EmailMessage, ToolError> {
    imap.read(uid, mailbox.as_deref()).await
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;
    use sinew_core::Tool;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    use super::*;

    const MSG_101: &str = "From: Ana <ana@example.com>\r\nTo: bot@example.com\r\nSubject: =?UTF-8?Q?Relat=C3=B3rio?=\r\nDate: Mon, 16 Nov 2026 09:00:00 -0500\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nOl\u{e1}, segue o relat\u{f3}rio.\r\n";
    const MSG_102: &str = "From: ops@example.com\r\nTo: bot@example.com\r\nSubject: Alert\r\nDate: Tue, 17 Nov 2026 08:00:00 -0500\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=\"b1\"\r\n\r\n--b1\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<p>Disk <b>full</b> on db-1</p>\r\n--b1\r\nContent-Type: text/csv; name=\"usage.csv\"\r\nContent-Disposition: attachment; filename=\"usage.csv\"\r\n\r\na,b\r\n--b1--\r\n";

    fn headers_of(msg: &str) -> &str {
        &msg[..msg.find("\r\n\r\n").unwrap() + 4]
    }

    /// A scripted IMAP server with two messages: 101 (seen) and 102 (unseen).
    async fn imap_server() -> (u16, Arc<Mutex<Vec<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let log = Arc::new(Mutex::new(Vec::new()));
        let log2 = log.clone();
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                let log = log2.clone();
                tokio::spawn(async move {
                    let (r, mut w) = sock.into_split();
                    let mut r = BufReader::new(r);
                    w.write_all(b"* OK [CAPABILITY IMAP4rev1] mock ready\r\n")
                        .await
                        .unwrap();
                    let mut line = String::new();
                    loop {
                        line.clear();
                        if r.read_line(&mut line).await.unwrap_or(0) == 0 {
                            break;
                        }
                        log.lock().unwrap().push(line.trim_end().to_string());
                        let (tag, cmd) = line.trim_end().split_once(' ').unwrap();
                        let upper = cmd.to_ascii_uppercase();
                        let mut out = String::new();
                        if upper.starts_with("LOGIN") {
                            if !cmd.contains("\"secret\"") {
                                out = format!("{tag} NO [AUTHENTICATIONFAILED] bad password\r\n");
                            } else {
                                out = format!("{tag} OK LOGIN completed\r\n");
                            }
                        } else if upper.starts_with("EXAMINE") {
                            out.push_str("* 2 EXISTS\r\n* 0 RECENT\r\n* FLAGS (\\Seen)\r\n* OK [UIDVALIDITY 7] ok\r\n");
                            out.push_str(&format!("{tag} OK [READ-ONLY] EXAMINE completed\r\n"));
                        } else if upper.starts_with("UID SEARCH") {
                            let hits = if upper.contains("UNSEEN") {
                                "102"
                            } else {
                                "101 102"
                            };
                            out = format!("* SEARCH {hits}\r\n{tag} OK SEARCH completed\r\n");
                        } else if upper.starts_with("UID FETCH") {
                            let full = upper.contains("BODY.PEEK[]");
                            for (seq, uid, msg, flags) in
                                [(1, 101, MSG_101, "\\Seen"), (2, 102, MSG_102, "")]
                            {
                                let set = upper.split(' ').nth(2).unwrap();
                                if !set.split(',').any(|u| u == uid.to_string()) {
                                    continue;
                                }
                                let (section, data) = if full {
                                    ("BODY[]", msg)
                                } else {
                                    ("BODY[HEADER]", headers_of(msg))
                                };
                                out.push_str(&format!(
                                    "* {seq} FETCH (UID {uid} FLAGS ({flags}) {section} {{{}}}\r\n{data})\r\n",
                                    data.len()
                                ));
                            }
                            out.push_str(&format!("{tag} OK FETCH completed\r\n"));
                        } else if upper.starts_with("LOGOUT") {
                            out = format!("* BYE\r\n{tag} OK LOGOUT completed\r\n");
                        } else {
                            out = format!("{tag} BAD unknown command\r\n");
                        }
                        w.write_all(out.as_bytes()).await.unwrap();
                        if upper.starts_with("LOGOUT") {
                            break;
                        }
                    }
                });
            }
        });
        (port, log)
    }

    fn reader(port: u16, password: &str) -> ImapReader {
        ImapReader::new("127.0.0.1", "bot@example.com", password)
            .port(port)
            .insecure_plaintext()
    }

    #[tokio::test]
    async fn lists_newest_first_without_marking_seen() {
        let (port, log) = imap_server().await;
        let t = email_list::tool(reader(port, "secret"));
        assert!(!t.side_effects());
        let out = t.call(json!({})).await.unwrap();
        assert_eq!(out[0]["uid"], 102);
        assert_eq!(out[0]["subject"], "Alert");
        assert_eq!(out[0]["seen"], false);
        assert_eq!(out[1]["subject"], "Relatório");
        assert_eq!(out[1]["from"], "Ana <ana@example.com>");
        assert_eq!(out[1]["seen"], true);
        let unread = t
            .call(json!({"unread_only": true, "limit": 5}))
            .await
            .unwrap();
        assert_eq!(unread.as_array().unwrap().len(), 1);
        let cmds = log.lock().unwrap().join("\n");
        assert!(cmds.contains("EXAMINE \"INBOX\""), "{cmds}");
        assert!(!cmds.contains("SELECT"), "{cmds}");
        assert!(cmds.contains("BODY.PEEK[HEADER]"), "{cmds}");
        assert!(cmds.contains("LOGOUT"), "{cmds}");
    }

    #[tokio::test]
    async fn reads_a_message() {
        let (port, _log) = imap_server().await;
        let t = email_read::tool(reader(port, "secret"));
        let out = t.call(json!({"uid": 102})).await.unwrap();
        assert_eq!(out["subject"], "Alert");
        assert_eq!(out["text"], "Disk full on db-1");
        assert_eq!(out["attachments"], json!(["usage.csv"]));
        let out = t.call(json!({"uid": 101})).await.unwrap();
        assert_eq!(out["text"], "Olá, segue o relatório.");
        let err = t.call(json!({"uid": 999})).await.unwrap_err();
        assert!(err.to_string().contains("no message with uid 999"), "{err}");
    }

    #[tokio::test]
    async fn reports_login_failures() {
        let (port, _log) = imap_server().await;
        let err = reader(port, "wrong")
            .list(None, false, 10)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("login failed"), "{err}");
        assert!(!format!("{:?}", reader(port, "wrong")).contains("wrong"));
    }
}
