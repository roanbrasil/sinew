//! The smallest Sinew program: build a request, call a model, read the answer.
//!
//! ```bash
//! cargo run -p sample-01-hello-chat              # offline
//! cargo run -p sample-01-hello-chat -- --ollama  # local model
//! ```

use sample_common::{pick_model, BoxError};
use sinew::prelude::*;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = pick_model(vec![Message::assistant(
        "Rust gives you memory safety without a garbage collector.",
    )])?;

    let request = ChatRequest::new(vec![
        Message::system("You answer in one sentence."),
        Message::user("Why would I write LLM infrastructure in Rust?"),
    ])
    .with_temperature(0.2)
    .with_max_tokens(200);

    let response = model.chat(request).await?;

    println!("model : {}", response.model);
    println!("answer: {}", response.message.content);
    println!(
        "tokens: {} in, {} out",
        response.usage.input_tokens, response.usage.output_tokens
    );
    Ok(())
}
