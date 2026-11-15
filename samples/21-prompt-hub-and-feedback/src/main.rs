//! The LangSmith-style workflow around a prompt, end to end:
//!
//! 1. push two versions of a support prompt to the scope **prompt hub** and
//!    tag them `prod` and `staging`;
//! 2. answer customer questions with `support-answer@prod` pulled through
//!    `PromptHub` (cached locally), tracing every call with **cost** and
//!    **time to first token**;
//! 3. record **user feedback** (thumbs) and send the other runs to an
//!    **annotation queue** for human review;
//! 4. an **online evaluation rule** scores every answer in the background;
//! 5. a **pairwise A/B experiment** compares `prod` with `staging` using an
//!    LLM judge (asked in both orders to cancel position bias).
//!
//! Everything shows up in the web UI: Monitor, Traces, Prompts, Annotate,
//! Rules & alerts, Experiments.
//!
//! ```bash
//! cargo run -p sample-21-prompt-hub-and-feedback -- --keep        # offline
//! cargo run -p sample-21-prompt-hub-and-feedback -- --ollama --keep
//! cargo run -p sample-21-prompt-hub-and-feedback -- --model openai:gpt-4.1-mini
//! cargo run -p sample-21-prompt-hub-and-feedback -- --url http://127.0.0.1:7878
//! ```

use std::{collections::HashMap, time::Duration};

use sample_common::{is_live, pick_model, BoxError};
use serde_json::{json, Value};
use sinew::core::{ChatRequest, Message};
use sinew::eval::{
    evaluate_pairwise, Contains, Dataset, Experiment, Feedback, LlmPairwiseJudge,
    PairwiseEvaluator, PromptHub, PromptPush, ScopeClient,
};
use sinew::trace::{traced_chat_stream, HttpSink, HttpSinkConfig, RunKind, Tracer};

const PROJECT: &str = "support-bot";

const QUESTIONS: [(&str, &str); 4] = [
    ("u1", "Where is my order #1042?"),
    ("u2", "Can I change the delivery address of order #2077?"),
    ("u3", "My order arrived damaged. What now?"),
    ("u1", "How long do refunds take?"),
];

/// Offline answers, in call order.
fn scripted_answers() -> Vec<Message> {
    [
        "Your order #1042 left our warehouse yesterday and should arrive tomorrow.",
        "Yes. You can change the address of order #2077 until it ships, from the Orders page.",
        "Sorry about that! Reply with a photo and we will send a replacement order right away.",
        "Refunds reach your card 3 to 5 business days after we receive the item.",
    ]
    .into_iter()
    .map(Message::assistant)
    .collect()
}

fn render(template: &sinew::core::PromptTemplate, question: &str) -> Result<String, BoxError> {
    Ok(template.render(&HashMap::from([
        ("question", question.to_string()),
        ("company", "Acme Shop".to_string()),
    ]))?)
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let keep = args.iter().any(|a| a == "--keep");
    let url = match args
        .iter()
        .position(|a| a == "--url")
        .and_then(|i| args.get(i + 1))
    {
        Some(u) => u.trim_end_matches('/').to_string(),
        None => {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let url = format!("http://{}", listener.local_addr()?);
            let app = sinew_scope::router(sinew_scope::Store::memory().await?, None);
            tokio::spawn(sinew_scope::server::serve_with_shutdown(
                listener,
                app,
                std::future::pending(),
            ));
            url
        }
    };
    println!("scope UI: {url}\n");
    let client = ScopeClient::new(&url);
    let hub = PromptHub::new(&url).with_ttl(Duration::from_secs(30));

    // 1. Prompt hub: two versions, tagged.
    let v1 = hub
        .push(
            PromptPush::new(
                "support-answer",
                "You are the support assistant of {company}. Answer in one short sentence.\n\
                 Customer: {question}",
            )
            .description("Customer support answer")
            .model_config(json!({"temperature": 0.2}))
            .message("first version")
            .tag("prod"),
        )
        .await?;
    let v2 = hub
        .push(
            PromptPush::new(
                "support-answer",
                "You are the friendly support assistant of {company}.\n\
                 Answer in one short sentence and mention the order number when there is one.\n\
                 Customer: {question}",
            )
            .model_config(json!({"temperature": 0.2}))
            .message("mention the order number")
            .tag("staging"),
        )
        .await?;
    println!(
        "prompt hub: support-answer v{} ({}) = prod, v{} ({}) = staging",
        v1.version, v1.commit, v2.version, v2.commit
    );
    let diff: Value = client
        .get_json("/api/prompts/support-answer/diff?from=prod&to=staging")
        .await?;
    for line in diff["template"].as_array().into_iter().flatten() {
        let mark = match line["op"].as_str() {
            Some("add") => "+",
            Some("del") => "-",
            _ => " ",
        };
        println!("  {mark} {}", line["text"].as_str().unwrap_or(""));
    }

    // Offline the "model" is a script named `scripted`: give it a demo price
    // so costs show up in the UI. Live models use the built-in price table.
    if !is_live() {
        let mut pricing: Value = client.get_json("/api/pricing").await?;
        if let Some(models) = pricing["models"].as_array_mut() {
            models
                .push(json!({"model": "scripted", "input_per_mtok": 1.0, "output_per_mtok": 4.0}));
        }
        client.put_json::<Value>("/api/pricing", &pricing).await?;
    }

    // 4 (set up first). An online evaluation rule and an annotation queue.
    client
        .post_json::<Value>(
            "/api/rules",
            &json!({
                "name": "answer checks",
                "filter": {"project": PROJECT, "kind": "chain", "root_only": true},
                "sample_rate": 1.0,
                "evaluators": [
                    {"type": "not_empty"},
                    {"type": "contains", "needle": "order", "ignore_case": true, "key": "mentions_order"}
                ]
            }),
        )
        .await?;
    let queue = client
        .create_queue(
            "support review",
            "Is the answer correct and kind?",
            &["correctness", "tone"],
        )
        .await?;

    // 2. Answer questions with the prod prompt, traced.
    let sink = HttpSink::new(
        HttpSinkConfig::new(&url)
            .project(PROJECT)
            .flush_interval(Duration::from_millis(200)),
    );
    let tracer = Tracer::new().with_sink(sink);
    let model = pick_model(scripted_answers())?;
    let mut run_ids = Vec::new();
    for (user, question) in QUESTIONS {
        let prompt = hub.pull_version("support-answer@prod").await?; // cached after the first pull
        let text = render(&prompt.prompt_template()?, question)?;
        let mut root = tracer.start(
            "support-bot",
            RunKind::Chain,
            None,
            json!({ "question": question }),
        );
        root.set_metadata("tags", json!(["prod"]));
        root.set_metadata("user", json!(user));
        root.set_metadata("prompt", json!(format!("support-answer@{}", prompt.commit)));
        let parent = root.reference();
        let resp = traced_chat_stream(
            model.as_ref(),
            ChatRequest::new(vec![Message::user(text)]).with_temperature(0.2),
            Some(&tracer),
            Some(&parent),
            |_| {},
        )
        .await?;
        println!("  {user}: {question}\n      → {}", resp.message.content);
        run_ids.push(parent.run_id.clone());
        root.finish(json!(resp.message.content));
    }
    let t = tracer.clone();
    tokio::task::spawn_blocking(move || t.flush()).await?;

    // 3. User feedback on two answers, the rest go to human review.
    client
        .create_feedback(&Feedback::score(&run_ids[0], "thumbs", 1.0).with_source("user"))
        .await?;
    client
        .create_feedback(
            &Feedback::score(&run_ids[2], "thumbs", 0.0)
                .with_comment("I wanted a refund, not a replacement")
                .with_source("user"),
        )
        .await?;
    let rest: Vec<&str> = run_ids[1..].iter().map(String::as_str).collect();
    client.add_to_queue(&queue, &rest).await?;
    println!(
        "\nfeedback: 2 thumbs recorded, {} runs queued for review",
        rest.len()
    );

    // 5. Pairwise A/B: prod vs staging on a small dataset, judged by an LLM.
    let mut dataset = Dataset::new("support-questions");
    for (q, needle) in [
        ("Where is my order #1042?", "1042"),
        ("Can I return order #3310?", "3310"),
        ("Do you ship to Canada?", "Canada"),
    ] {
        dataset.push(json!({"question": q}), Some(json!(needle)));
    }
    let mut reports = Vec::new();
    for (label, reference, scripted) in [
        (
            "prod",
            "support-answer@prod",
            [
                "It is on its way.",
                "Yes, within 30 days.",
                "Yes, we ship to Canada.",
            ],
        ),
        (
            "staging",
            "support-answer@staging",
            [
                "Order #1042 is on its way and arrives tomorrow.",
                "Yes, order #3310 can be returned within 30 days.",
                "Yes, we ship to Canada in 5 to 7 days.",
            ],
        ),
    ] {
        let template = hub.pull(reference).await?;
        let model = pick_model(scripted.into_iter().map(Message::assistant).collect())?;
        let report = Experiment::new(&dataset)
            .evaluator(Contains::reference().ignore_case())
            .concurrency(1)
            .name(format!("support-answer {label}"))
            .metadata(json!({"prompt": reference}))
            .run(|inputs: Value| {
                let (template, model) = (template.clone(), model.clone());
                async move {
                    let text = render(&template, inputs["question"].as_str().unwrap_or(""))?;
                    let r = model
                        .chat(ChatRequest::new(vec![Message::user(text)]))
                        .await?;
                    Ok::<Value, BoxError>(json!(r.message.content))
                }
            })
            .await;
        println!("{}", report.summary());
        client.upload_experiment(&dataset, &report).await?;
        reports.push(report);
    }
    // Offline judge: prefers B in both orders (A, B, A, B... after the swap).
    let judge = pick_model(
        (0..6)
            .map(|i| {
                Message::assistant(if i % 2 == 0 {
                    "{\"preference\": \"B\", \"reason\": \"B is specific to the question\"}"
                } else {
                    "{\"preference\": \"A\", \"reason\": \"A is specific to the question\"}"
                })
            })
            .collect(),
    )?;
    let evaluators: Vec<Box<dyn PairwiseEvaluator>> = vec![Box::new(
        LlmPairwiseJudge::new(
            judge,
            "Which answer is more helpful and specific to the customer's question?",
        )
        .named("helpfulness"),
    )];
    let pairwise = evaluate_pairwise(&reports[0], &reports[1], &evaluators, 1).await;
    println!("\n{}", pairwise.summary_text());
    client.upload_pairwise(&pairwise).await?;

    // Give the background rule a moment, then show what it stored.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let fb: Vec<Value> = client.feedback_for_run(&run_ids[0]).await?;
    println!("\nfeedback on the first answer:");
    for f in &fb {
        println!(
            "  {:<16} {}  ({})",
            f["key"].as_str().unwrap_or(""),
            f["score"],
            f["source"].as_str().unwrap_or("")
        );
    }
    let m: Value = client
        .get_json(&format!(
            "/api/metrics?project={PROJECT}&level=llm&group_by=model"
        ))
        .await?;
    let tot = &m["totals"];
    println!(
        "\nLLM calls: {} · p50 {}ms · TTFT p50 {}ms · tokens {} → {} · cost ${:.6}",
        tot["count"],
        tot["p50_latency_ms"],
        tot["ttft_p50_ms"],
        tot["input_tokens"],
        tot["output_tokens"],
        tot["cost_usd"].as_f64().unwrap_or(0.0)
    );

    println!("\nmonitor:   {url}/#/monitor");
    println!("prompt:    {url}/#/prompt/support-answer");
    println!("review:    {url}/#/queue/{queue}");
    println!("pairwise:  {url}/#/pairwise/{}", pairwise.id);
    if keep {
        println!("\nserver running — open the links above; Ctrl+C to stop");
        tokio::signal::ctrl_c().await?;
    }
    Ok(())
}
