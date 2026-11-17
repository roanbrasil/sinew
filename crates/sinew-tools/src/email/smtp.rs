//! `send_email` over SMTP with lettre.

use std::sync::Arc;

use lettre::{
    message::{header::ContentType, Mailbox},
    AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor,
};
use serde::Serialize;

use crate::ToolError;

/// An SMTP connection plus the sender identity and the recipient
/// allow-list. Cheap to clone.
///
/// ```no_run
/// use sinew_core::ToolBox;
/// use sinew_tools::email::{send_email, SmtpMailer};
///
/// // smtps:// (implicit TLS, port 465) or smtp://...?tls=required (STARTTLS, 587).
/// let mailer = SmtpMailer::from_url(
///     "smtps://bot%40example.com:app-password@smtp.example.com",
///     "Support Bot <bot@example.com>",
/// )?
/// .allow_recipient("@example.com")
/// .allow_recipient("partner@vendor.com");
/// let tools = ToolBox::new().with(send_email::tool(mailer));
/// # let _ = tools;
/// # Ok::<(), sinew_tools::ToolError>(())
/// ```
#[derive(Clone)]
pub struct SmtpMailer {
    transport: AsyncSmtpTransport<Tokio1Executor>,
    from: Mailbox,
    allowed: Arc<Vec<String>>,
    allow_any: bool,
    max_recipients: usize,
}

impl std::fmt::Debug for SmtpMailer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmtpMailer")
            .field("from", &self.from.to_string())
            .field("allowed", &self.allowed)
            .field("allow_any", &self.allow_any)
            .field("max_recipients", &self.max_recipients)
            .finish_non_exhaustive()
    }
}

fn mailbox(s: &str) -> Result<Mailbox, ToolError> {
    s.trim()
        .parse()
        .map_err(|e| ToolError::Invalid(format!("bad e-mail address `{s}`: {e}")))
}

impl SmtpMailer {
    /// Wraps a configured lettre transport. `from` is an address with an
    /// optional display name: `Bot <bot@example.com>`.
    pub fn new(
        transport: AsyncSmtpTransport<Tokio1Executor>,
        from: &str,
    ) -> Result<Self, ToolError> {
        Ok(Self {
            transport,
            from: mailbox(from).map_err(|e| ToolError::Config(e.to_string()))?,
            allowed: Arc::new(Vec::new()),
            allow_any: false,
            max_recipients: 10,
        })
    }

    /// Connects from a URL as accepted by lettre's
    /// `AsyncSmtpTransport::from_url`: `smtps://user:pass@host[:port]`
    /// (implicit TLS) or `smtp://user:pass@host:587?tls=required`
    /// (STARTTLS). Percent-encode `@` in the user name.
    pub fn from_url(url: &str, from: &str) -> Result<Self, ToolError> {
        let transport = AsyncSmtpTransport::<Tokio1Executor>::from_url(url)
            .map_err(|e| ToolError::Config(format!("bad SMTP URL: {e}")))?
            .build();
        Self::new(transport, from)
    }

    /// [`from_url`](Self::from_url) with `SMTP_URL` and `SMTP_FROM` from the
    /// environment.
    pub fn from_env() -> Result<Self, ToolError> {
        let var = |k: &str| {
            std::env::var(k)
                .ok()
                .filter(|v| !v.trim().is_empty())
                .ok_or_else(|| ToolError::Config(format!("{k} is not set")))
        };
        Self::from_url(&var("SMTP_URL")?, &var("SMTP_FROM")?)
    }

    /// Allows a recipient: a full address (`ana@example.com`) or a domain
    /// (`@example.com` or `example.com`, subdomains included). Until one is
    /// allowed (or [`allow_any_recipient`](Self::allow_any_recipient) is
    /// called) every recipient is refused.
    pub fn allow_recipient(mut self, pattern: impl AsRef<str>) -> Self {
        Arc::make_mut(&mut self.allowed).push(pattern.as_ref().trim().to_ascii_lowercase());
        self
    }

    /// Allows any recipient. Only for mailers whose use is otherwise
    /// controlled (human approval, a sandbox account).
    pub fn allow_any_recipient(mut self) -> Self {
        self.allow_any = true;
        self
    }

    /// Recipients per message at most, `to` and `cc` together (default 10).
    pub fn max_recipients(mut self, n: usize) -> Self {
        self.max_recipients = n.max(1);
        self
    }

    fn check_recipient(&self, mbox: &Mailbox) -> Result<(), ToolError> {
        if self.allow_any {
            return Ok(());
        }
        let addr = mbox.email.to_string().to_ascii_lowercase();
        let domain = mbox.email.domain().to_ascii_lowercase();
        let ok = self.allowed.iter().any(|p| {
            if p.contains('@') && !p.starts_with('@') {
                *p == addr
            } else {
                let d = p.trim_start_matches('@');
                domain == d || domain.ends_with(&format!(".{d}"))
            }
        });
        if ok {
            Ok(())
        } else {
            Err(ToolError::Blocked(format!(
                "recipient `{addr}` is not on the allow-list"
            )))
        }
    }

    /// Sends a plain-text message.
    pub async fn send(
        &self,
        to: &[String],
        cc: &[String],
        subject: &str,
        body: &str,
        reply_to: Option<&str>,
    ) -> Result<SentEmail, ToolError> {
        if to.is_empty() {
            return Err(ToolError::Invalid(
                "at least one `to` recipient is required".into(),
            ));
        }
        if to.len() + cc.len() > self.max_recipients {
            return Err(ToolError::Blocked(format!(
                "at most {} recipients per message",
                self.max_recipients
            )));
        }
        if subject.trim().is_empty() {
            return Err(ToolError::Invalid("the subject is empty".into()));
        }
        let to: Vec<Mailbox> = to.iter().map(|s| mailbox(s)).collect::<Result<_, _>>()?;
        let cc: Vec<Mailbox> = cc.iter().map(|s| mailbox(s)).collect::<Result<_, _>>()?;
        for m in to.iter().chain(&cc) {
            self.check_recipient(m)?;
        }
        let message_id = sinew_core::tool::idempotency_key().map(|key| {
            format!(
                "<sinew.{:016x}@{}>",
                fnv1a(key.as_bytes()),
                self.from.email.domain()
            )
        });
        let mut builder = Message::builder()
            .from(self.from.clone())
            .subject(subject.trim())
            .message_id(message_id)
            .header(ContentType::TEXT_PLAIN);
        for m in &to {
            builder = builder.to(m.clone());
        }
        for m in &cc {
            builder = builder.cc(m.clone());
        }
        if let Some(r) = reply_to.filter(|r| !r.trim().is_empty()) {
            builder = builder.reply_to(mailbox(r)?);
        }
        let message = builder
            .body(body.to_string())
            .map_err(|e| ToolError::Invalid(format!("cannot build the message: {e}")))?;
        let message_id = message
            .headers()
            .get_raw("Message-ID")
            .unwrap_or_default()
            .to_string();
        self.transport
            .send(message)
            .await
            .map_err(|e| ToolError::Request(format!("SMTP send failed: {e}")))?;
        Ok(SentEmail {
            message_id,
            recipients: to.iter().chain(&cc).map(|m| m.email.to_string()).collect(),
        })
    }
}

/// 64-bit FNV-1a: a short, stable digest for Message-IDs.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Confirmation from [`send_email`](fn@send_email).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SentEmail {
    pub message_id: String,
    pub recipients: Vec<String>,
}

/// Send a plain-text e-mail. This delivers a real message and cannot be
/// undone: send only when the user asked for it, and confirm the recipients
/// and content first if anything is unclear. Recipients outside the
/// application's allow-list are refused.
#[sinew_core::tool(side_effects)]
pub async fn send_email(
    #[state] mailer: SmtpMailer,
    /// Recipient addresses, e.g. ["ana@example.com"].
    to: Vec<String>,
    /// Subject line.
    subject: String,
    /// Plain-text body.
    body: String,
    /// Carbon-copy addresses.
    cc: Option<Vec<String>>,
    /// Address replies should go to, if different from the sender.
    reply_to: Option<String>,
) -> Result<SentEmail, ToolError> {
    mailer
        .send(
            &to,
            &cc.unwrap_or_default(),
            &subject,
            &body,
            reply_to.as_deref(),
        )
        .await
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::json;
    use sinew_core::{tool::with_idempotency_key, Tool};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    use super::*;

    /// A minimal SMTP server that accepts one message per connection and
    /// records the transcript.
    async fn smtp_server() -> (u16, Arc<Mutex<Vec<String>>>) {
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
                    w.write_all(b"220 mock ESMTP\r\n").await.unwrap();
                    let mut line = String::new();
                    let mut in_data = false;
                    loop {
                        line.clear();
                        if r.read_line(&mut line).await.unwrap_or(0) == 0 {
                            break;
                        }
                        log.lock().unwrap().push(line.clone());
                        if in_data {
                            if line == ".\r\n" {
                                in_data = false;
                                w.write_all(b"250 queued\r\n").await.unwrap();
                            }
                            continue;
                        }
                        let cmd = line.to_ascii_uppercase();
                        let reply: &[u8] = if cmd.starts_with("EHLO") {
                            b"250-mock\r\n250 8BITMIME\r\n"
                        } else if cmd.starts_with("DATA") {
                            in_data = true;
                            b"354 go ahead\r\n"
                        } else if cmd.starts_with("QUIT") {
                            w.write_all(b"221 bye\r\n").await.unwrap();
                            break;
                        } else {
                            b"250 ok\r\n"
                        };
                        w.write_all(reply).await.unwrap();
                    }
                });
            }
        });
        (port, log)
    }

    fn mailer(port: u16) -> SmtpMailer {
        let transport = AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous("127.0.0.1")
            .port(port)
            .build();
        SmtpMailer::new(transport, "Bot <bot@example.com>").unwrap()
    }

    #[tokio::test]
    async fn sends_to_allowed_recipients() {
        let (port, log) = smtp_server().await;
        let t = send_email::tool(mailer(port).allow_recipient("@example.com"));
        assert!(t.side_effects());
        let out = with_idempotency_key(
            "thread-1/tools/2/call_a".into(),
            t.call(json!({
                "to": ["Ana <ana@example.com>"],
                "cc": ["ops@team.example.com"],
                "subject": "Weekly report",
                "body": "All green.\nSee you Monday."
            })),
        )
        .await
        .unwrap();
        assert_eq!(
            out["recipients"],
            json!(["ana@example.com", "ops@team.example.com"])
        );
        let id = out["message_id"].as_str().unwrap();
        assert!(
            id.starts_with("<sinew.") && id.ends_with("@example.com>"),
            "{id}"
        );
        let transcript = log.lock().unwrap().concat();
        assert!(
            transcript.contains("MAIL FROM:<bot@example.com>"),
            "{transcript}"
        );
        assert!(transcript.contains("RCPT TO:<ana@example.com>"));
        assert!(transcript.contains("RCPT TO:<ops@team.example.com>"));
        assert!(transcript.contains("Subject: Weekly report"));
        assert!(transcript.contains("All green."));
        assert!(transcript.contains(&format!("Message-ID: {id}")));

        // Same key, same Message-ID: a retried call is recognizable.
        let again = with_idempotency_key(
            "thread-1/tools/2/call_a".into(),
            t.call(json!({"to": ["ana@example.com"], "subject": "Weekly report", "body": "x"})),
        )
        .await
        .unwrap();
        assert_eq!(again["message_id"], id);
    }

    #[tokio::test]
    async fn refuses_recipients_off_the_allow_list() {
        let (port, log) = smtp_server().await;
        let m = mailer(port)
            .allow_recipient("boss@corp.com")
            .allow_recipient("example.com")
            .max_recipients(2);
        let t = send_email::tool(m.clone());
        for args in [
            json!({"to": ["eve@evil.com"], "subject": "s", "body": "b"}),
            json!({"to": ["ana@example.com"], "cc": ["x@corp.com"], "subject": "s", "body": "b"}),
            json!({"to": ["ana@notexample.com"], "subject": "s", "body": "b"}),
            json!({"to": ["a@example.com", "b@example.com", "c@example.com"], "subject": "s", "body": "b"}),
        ] {
            let err = t.call(args.clone()).await.unwrap_err().to_string();
            assert!(err.contains("blocked"), "{args}: {err}");
        }
        for args in [
            json!({"to": [], "subject": "s", "body": "b"}),
            json!({"to": ["not an address"], "subject": "s", "body": "b"}),
            json!({"to": ["ana@example.com"], "subject": " ", "body": "b"}),
        ] {
            let err = t.call(args.clone()).await.unwrap_err().to_string();
            assert!(err.contains("invalid input"), "{args}: {err}");
        }
        assert!(log.lock().unwrap().is_empty(), "nothing reached the server");
        // An empty allow-list refuses everyone.
        let err = mailer(port)
            .send(&["ana@example.com".into()], &[], "s", "b", None)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Blocked(_)));
        assert!(m
            .send(&["boss@corp.com".into()], &[], "s", "b", None)
            .await
            .is_ok());
    }
}
