//! Human-in-the-loop: the graph drafts an email, then pauses before `send`.
//! A person reads the draft, edits it, and the thread resumes from the
//! checkpoint. Swap `MemoryCheckpointer` for a database-backed one to survive
//! restarts.
//!
//! ```bash
//! cargo run -p sample-05-human-in-the-loop
//! ```

use std::io::{self, BufRead, IsTerminal, Write};

use sample_common::{pick_model, BoxError};
use serde::{Deserialize, Serialize};
use sinew::prelude::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Email {
    to: String,
    request: String,
    body: String,
    sent: bool,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = pick_model(vec![Message::assistant(
        "Hi Ana, the release moves to Friday because the security review needs two more days. Sorry for the short notice.",
    )])?;

    let mut g = StateGraph::<Email>::new();
    g.add_node("draft", move |mut s: Email| {
        let model = model.clone();
        async move {
            let prompt = format!("Write a short email to {}: {}", s.to, s.request);
            s.body = model
                .chat(ChatRequest::new(vec![Message::user(prompt)]))
                .await?
                .message
                .content;
            Ok(s)
        }
    });
    g.add_node("send", |mut s: Email| async move {
        println!("\n📤 sending to {}:\n{}\n", s.to, s.body);
        s.sent = true;
        Ok(s)
    });
    g.add_edge("draft", "send");
    g.add_edge("send", END);
    g.set_entry_point("draft");

    let graph = g
        .compile()?
        .with_checkpointer(MemoryCheckpointer::new())
        .interrupt_before(["send"]);
    let thread = "email-42";

    let first = graph
        .run(
            thread,
            Some(Email {
                to: "ana@example.com".into(),
                request: "the release slips to Friday".into(),
                ..Default::default()
            }),
        )
        .await?;
    let RunOutcome::Interrupted { state, next } = first else {
        unreachable!("graph pauses before send")
    };
    println!("⏸  paused before `{next}`. Draft:\n{}\n", state.body);

    // The human step. Interactive when run in a terminal; scripted otherwise.
    let edited = if io::stdin().is_terminal() {
        print!("Press Enter to approve, or type a replacement body: ");
        io::stdout().flush()?;
        let mut line = String::new();
        io::stdin().lock().read_line(&mut line)?;
        line.trim().to_string()
    } else {
        format!("{} I'll confirm the exact time tomorrow.", state.body)
    };

    if !edited.is_empty() {
        let (mut s, _) = graph.get_state(thread).await?;
        s.body = edited;
        graph.update_state(thread, s).await?;
        println!("✏️  draft updated by the reviewer");
    }

    let done = graph.run(thread, None).await?.into_state();
    println!("sent = {}", done.sent);
    Ok(())
}
