//! A two-step chain: the first call extracts key facts, the second writes a
//! headline from them. Templates fail loudly on missing variables, so a typo in
//! a placeholder is caught before a model call is wasted.
//!
//! ```bash
//! cargo run -p sample-02-prompt-chain
//! cargo run -p sample-02-prompt-chain -- --ollama
//! ```

use std::collections::HashMap;

use sample_common::{pick_model, BoxError};
use sinew::prelude::*;

const ARTICLE: &str = "The city council approved a plan on Tuesday to add 40 km of protected bike \
lanes by 2028. The budget is 120 million dollars, funded half by the province.";

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let model = pick_model(vec![
        Message::assistant(
            "- 40 km of protected bike lanes\n- done by 2028\n- 120 M$ budget, half provincial",
        ),
        Message::assistant("Council Backs 40 km of Protected Bike Lanes by 2028"),
    ])?;

    let extract =
        PromptTemplate::new("List the 3 most important facts in this text as bullets:\n\n{text}")?;
    let headline = PromptTemplate::new(
        "Write a {style} headline (max {words} words) from these facts:\n{facts}",
    )?;
    println!("headline template needs: {:?}\n", headline.variables());

    // Step 1: facts.
    let prompt = extract.render(&HashMap::from([("text", ARTICLE.to_string())]))?;
    let facts = model
        .chat(ChatRequest::new(vec![Message::user(prompt)]))
        .await?
        .message
        .content;
    println!("facts:\n{facts}\n");

    // Step 2: headline from the facts.
    let vars = HashMap::from([
        ("style", "newspaper".to_string()),
        ("words", "12".to_string()),
        ("facts", facts),
    ]);
    let prompt = headline.render(&vars)?;
    let title = model
        .chat(ChatRequest::new(vec![Message::user(prompt)]))
        .await?
        .message
        .content;
    println!("headline: {title}");

    // A missing variable is an error, not a silent "{facts}" in the prompt.
    let err = headline
        .render(&HashMap::from([("style", "x".to_string())]))
        .unwrap_err();
    println!("\nexpected error: {err}");
    Ok(())
}
