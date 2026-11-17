//! E-mail tools: `send_email` over SMTP (feature `email-smtp`), and
//! `email_list` / `email_read` over IMAP (feature `email-imap`).
//!
//! Sending is a side effect: [`send_email`](fn@send_email) reports
//! `side_effects() == true`, only writes to recipients on the mailer's
//! allow-list, and derives the `Message-ID` from the call's idempotency key
//! when the runtime provides one, so a duplicate caused by a retry can be
//! recognized by the receiving side.
//!
//! Reading opens the mailbox with `EXAMINE` (read-only) and fetches with
//! `BODY.PEEK`, so listing or reading never marks a message as seen.

#[cfg(feature = "email-imap")]
mod imap;
#[cfg(feature = "email-smtp")]
mod smtp;

#[cfg(feature = "email-imap")]
pub use imap::{email_list, email_read, EmailMessage, EmailSummary, ImapReader};
#[cfg(feature = "email-smtp")]
pub use smtp::{send_email, SentEmail, SmtpMailer};
