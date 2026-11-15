//! Structured output: describe the shape you want as a JSON Schema and get a
//! typed Rust value back. By default the schema is offered as a forced tool
//! call (works with any tool-calling model); `StructuredMode::Json` asks for a
//! JSON reply instead. Invalid output is sent back to the model once with the
//! validation error, then reported as `Error::Structured`.
//!
//! ```bash
//! cargo run -p sample-10-structured-output
//! cargo run -p sample-10-structured-output -- --ollama
//! ```

use sample_common::{pick_model, BoxError};
use serde::Deserialize;
use serde_json::json;
use sinew::core::StructuredMode;
use sinew::prelude::*;

#[derive(Debug, Deserialize)]
struct Invoice {
    vendor: String,
    total: f64,
    currency: String,
    due_date: String,
    items: Vec<LineItem>,
}

#[derive(Debug, Deserialize)]
struct LineItem {
    description: String,
    quantity: u32,
    unit_price: f64,
}

const EMAIL: &str = "Hi! Attached is invoice #8812 from Northwind Supplies. 3 ergonomic chairs at $240 \
each and 2 standing desks at $510 each, total $1,740.00 USD, payment due by 2026-12-01. Thanks, Marta";

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let invoice_json = json!({
        "vendor": "Northwind Supplies", "total": 1740.0, "currency": "USD", "due_date": "2026-12-01",
        "items": [
            {"description": "ergonomic chair", "quantity": 3, "unit_price": 240.0},
            {"description": "standing desk", "quantity": 2, "unit_price": 510.0}
        ]
    });
    let model = pick_model(vec![
        // Tool mode: the model "calls" the schema tool with the data.
        Message::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "1".into(),
                name: "invoice".into(),
                arguments: invoice_json.clone(),
            }],
        ),
        // JSON mode, first reply is missing a required field → retried.
        Message::assistant(r#"{"label": "billing"}"#),
        Message::assistant(r#"{"label": "billing", "urgent": false}"#),
    ])?;

    let schema = json!({
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "total": {"type": "number"},
            "currency": {"type": "string", "description": "ISO 4217 code"},
            "due_date": {"type": "string", "description": "YYYY-MM-DD"},
            "items": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "unit_price": {"type": "number"}
                },
                "required": ["description", "quantity", "unit_price"]
            }}
        },
        "required": ["vendor", "total", "currency", "due_date", "items"]
    });

    let extractor =
        StructuredOutput::new("invoice", schema).with_description("Record the invoice fields");
    let req = ChatRequest::new(vec![Message::user(format!(
        "Extract the invoice from this email:\n\n{EMAIL}"
    ))]);
    let invoice: Invoice = extractor.generate(model.as_ref(), req).await?;

    println!(
        "{} — {:.2} {} due {}",
        invoice.vendor, invoice.total, invoice.currency, invoice.due_date
    );
    let mut sum = 0.0;
    for item in &invoice.items {
        println!(
            "  {:>2} × {:<16} {:>8.2}",
            item.quantity, item.description, item.unit_price
        );
        sum += item.quantity as f64 * item.unit_price;
    }
    // Typed data means you can check it with ordinary code.
    println!("line items add up: {}", (sum - invoice.total).abs() < 0.01);

    // JSON mode, for a small classification.
    #[derive(Debug, Deserialize)]
    struct Triage {
        label: String,
        urgent: bool,
    }
    let triage_schema = json!({
        "type": "object",
        "properties": {"label": {"type": "string", "enum": ["billing", "support", "sales"]}, "urgent": {"type": "boolean"}},
        "required": ["label", "urgent"]
    });
    let triage: Triage = StructuredOutput::new("triage", triage_schema)
        .with_mode(StructuredMode::Json)
        .generate(
            model.as_ref(),
            ChatRequest::new(vec![Message::user(format!(
                "Classify this email:\n\n{EMAIL}"
            ))]),
        )
        .await?;
    println!("\ntriage: {} (urgent: {})", triage.label, triage.urgent);
    Ok(())
}
