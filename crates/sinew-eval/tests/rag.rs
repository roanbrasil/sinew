//! RAG metrics with scripted judges and deterministic fake embeddings:
//! known inputs, known scores.
#![cfg(feature = "rag")]

use std::sync::Arc;

use serde_json::{json, Value};
use sinew_core::{mock::ScriptedModel, ChatModel, Message};
use sinew_eval::{
    rag::{
        rag_output, AnswerRelevance, AnswerSimilarity, CitationAccuracy, ContextPrecision,
        ContextRecall, Faithfulness, QuestionType, RagMetric, RagSample, RetrievalMetric,
        TestsetGenerator,
    },
    Dataset, Experiment,
};
use sinew_rag::{Bm25Retriever, Document, FakeEmbeddings, RagChain, Retriever};

fn scripted(replies: &[&str]) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new(
        replies.iter().map(|r| Message::assistant(*r)),
    ))
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn paris_sample() -> RagSample {
    RagSample::new(
        "What is the capital of France and how many people live there?",
        "Paris is the capital of France [1]. About 2.1 million people live in Paris [2].",
    )
    .with_context("fr", "Paris is the capital and largest city of France.")
    .with_context(
        "pop",
        "The city of Paris has about 2.1 million inhabitants.",
    )
    .with_context("it", "Rome is the capital of Italy.")
    .with_reference("Paris is the capital. It has about 2.1 million inhabitants.")
    .with_relevant_ids(["fr", "pop"])
}

#[tokio::test]
async fn retrieval_metrics_on_known_rankings() {
    // Relevant: fr, pop. Retrieved: it, fr, pop.
    let s = RagSample::new("q", "a")
        .with_context("it", "...")
        .with_context("fr", "...")
        .with_context("pop", "...")
        .with_relevant_ids(["fr", "pop"]);
    let score = |m: RetrievalMetric| {
        let s = s.clone();
        async move { m.measure(&s).await.score }
    };
    assert_eq!(score(RetrievalMetric::hit_rate(1)).await, 0.0);
    assert_eq!(score(RetrievalMetric::hit_rate(2)).await, 1.0);
    assert_eq!(score(RetrievalMetric::recall(2)).await, 0.5);
    assert_eq!(score(RetrievalMetric::recall(3)).await, 1.0);
    assert!(close(score(RetrievalMetric::precision(3)).await, 2.0 / 3.0));
    assert_eq!(score(RetrievalMetric::mrr()).await, 0.5);
    // DCG = 1/log2(3) + 1/log2(4); IDCG = 1 + 1/log2(3).
    let dcg = 1.0 / 3f64.log2() + 0.5;
    let idcg = 1.0 + 1.0 / 3f64.log2();
    assert!(close(score(RetrievalMetric::ndcg(3)).await, dcg / idcg));

    let r = RetrievalMetric::mrr().measure(&s).await;
    assert_eq!(
        r.reason,
        "relevant ids at ranks [2, 3] of 3 results (2 relevant)"
    );
    assert_eq!(r.details["ranks"], json!([2, 3]));
    assert_eq!(r.usage.calls, 0);

    // Repeated ids count once.
    let dup = RagSample::new("q", "a")
        .with_context("fr", "chunk 1")
        .with_context("fr", "chunk 2")
        .with_context("pop", "...")
        .with_relevant_ids(["fr", "pop"]);
    assert_eq!(RetrievalMetric::recall(2).measure(&dup).await.score, 1.0);

    let none = RagSample::new("q", "a").with_context("x", "...");
    let r = RetrievalMetric::hit_rate(3).measure(&none).await;
    assert_eq!(
        (r.score, r.reason.as_str()),
        (0.0, "example has no relevant ids")
    );
}

/// Retrieval metrics validated on a small hand-labeled set with a real
/// BM25 retriever.
#[tokio::test]
async fn retrieval_metrics_on_a_labeled_set() {
    let corpus = vec![
        Document::new("SqliteCheckpointer saves graph threads to a SQLite file.").with_id("sqlite"),
        Document::new("PostgresCheckpointer saves graph threads to Postgres for many hosts.")
            .with_id("postgres"),
        Document::new("chat_stream yields tokens as the provider produces them.").with_id("stream"),
        Document::new("The MCP client lists tools exposed by a server.").with_id("mcp"),
    ];
    let retriever = Bm25Retriever::from_documents(corpus).with_k(2);
    // (question, relevant ids, expected hit@2, recall@2, mrr)
    let labeled: [(&str, &[&str], f64, f64, f64); 3] = [
        (
            "Which checkpointer uses Postgres?",
            &["postgres"],
            1.0,
            1.0,
            1.0,
        ),
        ("save graph threads", &["sqlite", "postgres"], 1.0, 1.0, 1.0),
        ("How do I stream tokens?", &["stream"], 1.0, 1.0, 1.0),
    ];
    for (q, relevant, hit, recall, mrr) in labeled {
        let docs = retriever.retrieve(q).await.unwrap();
        let s = RagSample::from_documents(q, &docs, "").with_relevant_ids(relevant.iter().copied());
        assert_eq!(
            RetrievalMetric::hit_rate(2).measure(&s).await.score,
            hit,
            "{q}"
        );
        assert_eq!(
            RetrievalMetric::recall(2).measure(&s).await.score,
            recall,
            "{q}"
        );
        assert_eq!(RetrievalMetric::mrr().measure(&s).await.score, mrr, "{q}");
    }
    // A question whose relevant document is not retrieved at all.
    let docs = retriever
        .retrieve("Which checkpointer uses Postgres?")
        .await
        .unwrap();
    let s = RagSample::from_documents("q", &docs, "").with_relevant_ids(["mcp"]);
    assert_eq!(RetrievalMetric::hit_rate(2).measure(&s).await.score, 0.0);
    assert_eq!(RetrievalMetric::ndcg(2).measure(&s).await.score, 0.0);
}

#[tokio::test]
async fn faithfulness_counts_supported_claims() {
    let judge = scripted(&[
        r#"{"claims": ["Paris is the capital of France.", "About 2.1 million people live in Paris.", "Paris is in Germany."]}"#,
        r#"{"verdicts": [
            {"index": 1, "supported": true, "reason": "context 1"},
            {"index": 2, "supported": true, "reason": "context 2"},
            {"index": 3, "supported": false, "reason": "not stated"}]}"#,
    ]);
    let r = Faithfulness::new(judge.clone())
        .measure(&paris_sample())
        .await;
    assert!(close(r.score, 2.0 / 3.0));
    assert_eq!(
        r.reason,
        "2 of 3 claims supported; unsupported: Paris is in Germany."
    );
    assert_eq!(r.details["claims"][2]["supported"], false);
    assert_eq!(r.usage.calls, 2);
    assert!(r.usage.input_tokens > 0 && r.usage.output_tokens > 0);
    assert_eq!(r.usage.cost_usd, None, "scripted model has no price");
    let reqs = judge.requests();
    assert!(reqs[0]
        .messages
        .iter()
        .any(|m| m.content.contains("ANSWER:\nParis is the capital")));
    let verify = &reqs[1].messages.last().unwrap().content;
    assert!(verify.contains("[2] The city of Paris has about 2.1 million inhabitants."));
    assert!(verify.contains("3. Paris is in Germany."));
    assert_eq!(reqs[1].temperature, Some(0.0));

    // No claims → 1 after one call.
    let judge = scripted(&[r#"{"claims": []}"#]);
    let r = Faithfulness::new(judge)
        .measure(&RagSample::new("q", "I don't know."))
        .await;
    assert_eq!((r.score, r.usage.calls), (1.0, 1));

    // Claims but no contexts → 0 after one call.
    let judge = scripted(&[r#"{"claims": ["Paris is the capital."]}"#]);
    let r = Faithfulness::new(judge)
        .measure(&RagSample::new("q", "Paris is the capital."))
        .await;
    assert_eq!((r.score, r.usage.calls), (0.0, 1));

    // A judge that fails scores 0 with the reason.
    let r = Faithfulness::new(scripted(&[]))
        .measure(&paris_sample())
        .await;
    assert_eq!(r.score, 0.0);
    assert!(
        r.reason.starts_with("judge failed (claims)"),
        "{}",
        r.reason
    );

    // Nested arrays sent as JSON strings are decoded without a retry.
    let judge = scripted(&[
        r#"{"claims": "[\"a\", \"b\"]"}"#,
        r#"{"verdicts": "[{\"index\": 1, \"supported\": true}, {\"index\": 2, \"supported\": true}]"}"#,
    ]);
    let r = Faithfulness::new(judge).measure(&paris_sample()).await;
    assert_eq!((r.score, r.usage.calls), (1.0, 2));

    // An invalid reply is retried once.
    let judge = scripted(&[
        "no JSON here",
        r#"{"claims": ["a"]}"#,
        r#"{"verdicts": [{"index": 1, "supported": true}]}"#,
    ]);
    let r = Faithfulness::new(judge).measure(&paris_sample()).await;
    assert_eq!((r.score, r.usage.calls), (1.0, 3));

    // Skipped verdicts count as unsupported.
    let judge = scripted(&[
        r#"{"claims": ["a", "b"]}"#,
        r#"{"verdicts": [{"index": 2, "supported": true}]}"#,
    ]);
    let r = Faithfulness::new(judge).measure(&paris_sample()).await;
    assert_eq!(r.score, 0.5);
}

#[tokio::test]
async fn answer_relevance_judge_and_generated_questions() {
    let judge = scripted(&[r#"{"score": 8, "reasoning": "addresses both parts"}"#]);
    let r = AnswerRelevance::new(judge).measure(&paris_sample()).await;
    assert!(close(r.score, 0.8));
    assert_eq!(r.reason, "addresses both parts");

    // Generated questions identical to the original → similarity 1.
    let q = "What is the capital of France?";
    let reply = json!({"questions": [q, q], "noncommittal": false}).to_string();
    let emb = Arc::new(FakeEmbeddings::new(256));
    let r = AnswerRelevance::from_generated_questions(scripted(&[&reply]), emb.clone(), 2)
        .measure(&RagSample::new(q, "Paris."))
        .await;
    assert!((r.score - 1.0).abs() < 1e-6, "{}", r.score);
    assert_eq!(r.details["questions"].as_array().unwrap().len(), 2);

    // One identical and one generated question: the mean of the two cosines.
    let other = "Which river flows through Paris?";
    let reply = json!({"questions": [q, other], "noncommittal": false}).to_string();
    let r = AnswerRelevance::from_generated_questions(scripted(&[&reply]), emb.clone(), 2)
        .measure(&RagSample::new(q, "Paris."))
        .await;
    let a = emb.embed(q);
    let b = emb.embed(other);
    let cos: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    assert!((r.score - (1.0 + cos.max(0.0)) / 2.0).abs() < 1e-5);

    let reply = r#"{"questions": ["What?"], "noncommittal": true}"#;
    let r = AnswerRelevance::from_generated_questions(scripted(&[reply]), emb, 1)
        .measure(&RagSample::new(q, "I am not sure."))
        .await;
    assert_eq!(r.score, 0.0);
    assert_eq!(r.details["noncommittal"], true);
}

#[tokio::test]
async fn context_precision_is_average_precision() {
    let judge = scripted(&[r#"{"verdicts": [
        {"index": 1, "useful": true}, {"index": 2, "useful": false}, {"index": 3, "useful": true}]}"#]);
    let r = ContextPrecision::new(judge.clone())
        .measure(&paris_sample())
        .await;
    assert!(close(r.score, (1.0 + 2.0 / 3.0) / 2.0));
    assert_eq!(r.details["against"], "reference");
    assert_eq!(r.details["contexts"][0]["id"], "fr");
    assert!(judge.requests()[0]
        .messages
        .last()
        .unwrap()
        .content
        .contains("ANSWER:\nParis is the capital. It has about"));

    // Without a reference the answer is used.
    let judge = scripted(&[r#"{"verdicts": [{"index": 1, "useful": true}]}"#]);
    let mut s = paris_sample();
    s.reference = None;
    let r = ContextPrecision::new(judge).measure(&s).await;
    assert_eq!(r.score, 1.0);
    assert_eq!(r.details["against"], "answer");

    let r = ContextPrecision::new(scripted(&[]))
        .measure(&RagSample::new("q", "a"))
        .await;
    assert_eq!((r.score, r.usage.calls), (0.0, 0));
}

#[tokio::test]
async fn context_recall_counts_attributed_statements() {
    let judge = scripted(&[r#"{"statements": [
        {"statement": "Paris is the capital.", "attributed": true},
        {"statement": "It has about 2.1 million inhabitants.", "attributed": true},
        {"statement": "It hosted the 2024 Olympics.", "attributed": false}]}"#]);
    let r = ContextRecall::new(judge).measure(&paris_sample()).await;
    assert!(close(r.score, 2.0 / 3.0));
    assert!(r.reason.ends_with("missing: It hosted the 2024 Olympics."));

    let r = ContextRecall::new(scripted(&[]))
        .measure(&RagSample::new("q", "a").with_contexts(["c"]))
        .await;
    assert_eq!(r.score, 0.0);
    assert_eq!(r.reason, "example has no reference answer");
}

#[tokio::test]
async fn citation_accuracy_checks_each_citation() {
    let s = RagSample::new(
        "q",
        "Paris is the capital [1]. It has 2.1 million people [2][3]. Rome too [9]. Nice city.",
    )
    .with_contexts([
        "Paris is the capital of France.",
        "Paris has 2.1 million inhabitants.",
        "Rome is in Italy.",
    ]);
    // Valid citations, in order: (s1,[1]) (s2,[2]) (s2,[3]); [9] is invalid.
    let judge = scripted(&[r#"{"verdicts": [
        {"index": 1, "supported": true}, {"index": 2, "supported": true}, {"index": 3, "supported": false}]}"#]);
    let r = CitationAccuracy::new(judge.clone()).measure(&s).await;
    assert!(close(r.score, 2.0 / 4.0));
    assert_eq!(
        r.reason,
        "2 of 4 citations supported; 1 sentences cite nothing"
    );
    assert_eq!(r.details["citations"][3]["passage"], 9);
    assert_eq!(r.details["uncited_sentences"], json!(["Nice city."]));
    let prompt = judge.requests()[0].messages.last().unwrap().content.clone();
    assert!(prompt.contains("3. STATEMENT: It has 2.1 million people. CITES: [3]"));
    assert!(prompt.contains("[3] Rome is in Italy."));

    let r = CitationAccuracy::new(scripted(&[]))
        .measure(&RagSample::new("q", "No citations here.").with_contexts(["c"]))
        .await;
    assert_eq!((r.score, r.usage.calls), (0.0, 0));
    assert_eq!(r.reason, "the answer cites no passages");

    // Only invalid citations: no judge call.
    let r = CitationAccuracy::new(scripted(&[]))
        .measure(&RagSample::new("q", "Paris [4].").with_contexts(["c"]))
        .await;
    assert_eq!((r.score, r.usage.calls), (0.0, 0));
}

#[tokio::test]
async fn answer_similarity_is_cosine() {
    let emb = Arc::new(FakeEmbeddings::new(256));
    let m = AnswerSimilarity::new(emb.clone());
    let a = "Paris is the capital of France";
    let b = "The capital of France is Paris, a large city";
    let r = m.measure(&RagSample::new("q", a).with_reference(b)).await;
    let (va, vb) = (emb.embed(a), emb.embed(b));
    let cos: f64 = va
        .iter()
        .zip(&vb)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    assert!((r.score - cos.max(0.0)).abs() < 1e-6);
    assert!(r.score > 0.5 && r.score < 1.0);

    let pass = AnswerSimilarity::new(emb.clone()).threshold(0.5);
    assert_eq!(
        pass.measure(&RagSample::new("q", a).with_reference(b))
            .await
            .score,
        1.0
    );
    let strict = AnswerSimilarity::new(emb).threshold(0.99);
    assert_eq!(
        strict
            .measure(&RagSample::new("q", a).with_reference(b))
            .await
            .score,
        0.0
    );
    let r = m.measure(&RagSample::new("q", a)).await;
    assert_eq!(r.reason, "example has no reference answer");
}

#[tokio::test]
async fn generator_writes_deduplicates_and_exports() {
    let docs = vec![
        Document::new(
            "SqliteCheckpointer saves graph threads to a SQLite file so they survive restarts.",
        )
        .with_id("sqlite"),
        Document::new(
            "PostgresCheckpointer saves graph threads to Postgres, shared by many hosts.",
        )
        .with_metadata("source", "docs/postgres.md"),
        Document::new("Too short."),
    ];
    let model = scripted(&[
        r#"{"question": "Where does SqliteCheckpointer save threads?", "answer": "In a SQLite file."}"#,
        r#"{"question": "Which checkpointers save graph threads, and where?", "answer": "SqliteCheckpointer to SQLite, PostgresCheckpointer to Postgres."}"#,
        r#"{"question": "Where does the SqliteCheckpointer save threads?", "answer": "SQLite."}"#,
        r#"{"question": "", "answer": ""}"#,
        r#"{"question": "What lets several hosts share threads?", "answer": "PostgresCheckpointer."}"#,
    ]);
    let set = TestsetGenerator::new(model.clone())
        .count(3)
        .mode(sinew_core::StructuredMode::Json)
        .generate(&docs)
        .await;
    assert_eq!(set.len(), 3);
    assert_eq!(set.duplicates, 1);
    assert_eq!(set.failures.len(), 1);
    assert!(set.failures[0].contains("no question"));
    assert_eq!(set.usage.calls, 5);
    let types: Vec<QuestionType> = set.examples.iter().map(|e| e.question_type).collect();
    assert_eq!(
        types,
        vec![
            QuestionType::Factual,
            QuestionType::MultiHop,
            QuestionType::Factual
        ]
    );
    // The multi-hop question pairs the two long documents; the short one is never used.
    let mut ids = set.examples[1].source_ids.clone();
    ids.sort();
    assert_eq!(ids, vec!["docs/postgres.md", "sqlite"]);
    assert!(set
        .examples
        .iter()
        .all(|e| !e.source_ids.iter().any(|i| i == "doc-2")));
    let reqs = model.requests();
    assert!(reqs[1]
        .messages
        .last()
        .unwrap()
        .content
        .contains("DOCUMENT 2:"));
    assert!(reqs[2]
        .messages
        .last()
        .unwrap()
        .content
        .contains("QUESTIONS ALREADY WRITTEN:\n- Which checkpointers"));

    let ds = set.to_dataset("synthetic");
    assert_eq!(ds.examples[1].metadata["question_type"], "multi_hop");
    let back = Dataset::from_jsonl_str("synthetic", &ds.to_jsonl_string()).unwrap();
    assert_eq!(back.examples, ds.examples);

    // Same seed, same order.
    let replay = |m| {
        TestsetGenerator::new(m)
            .count(1)
            .question_types([QuestionType::Factual])
            .seed(9)
    };
    let a = replay(scripted(&[r#"{"question": "a?", "answer": "a"}"#]))
        .generate(&docs)
        .await;
    let b = replay(scripted(&[r#"{"question": "b?", "answer": "b"}"#]))
        .generate(&docs)
        .await;
    assert_eq!(a.examples[0].source_ids, b.examples[0].source_ids);
}

/// Synthetic set → RAG chain → every metric in one experiment.
#[tokio::test]
async fn rag_experiment_end_to_end() {
    let corpus = vec![
        Document::new(
            "SqliteCheckpointer saves graph threads to a SQLite file so they survive restarts.",
        )
        .with_id("sqlite"),
        Document::new("chat_stream yields tokens as the provider produces them.").with_id("stream"),
    ];
    let generator = scripted(&[
        r#"{"question": "Which checkpointer saves graph threads to SQLite?", "answer": "SqliteCheckpointer."}"#,
    ]);
    let set = TestsetGenerator::new(generator)
        .count(1)
        .question_types([QuestionType::Factual])
        .min_chars(10)
        .generate(&corpus[..1])
        .await;
    let ds = set.to_dataset("rag");
    let relevant = ds.examples[0].reference.as_ref().unwrap()["source_ids"][0]
        .as_str()
        .unwrap()
        .to_string();

    let answerer: Arc<dyn ChatModel> =
        scripted(&["Use SqliteCheckpointer, which saves threads to a SQLite file [1]."]);
    let retriever: Arc<dyn Retriever> = Arc::new(Bm25Retriever::from_documents(corpus).with_k(2));
    let rag = RagChain::new(answerer, retriever);

    // Judge replies in evaluator order.
    let judge: Arc<dyn ChatModel> = scripted(&[
        r#"{"claims": ["SqliteCheckpointer saves threads to a SQLite file."]}"#,
        r#"{"verdicts": [{"index": 1, "supported": true}]}"#,
        r#"{"score": 1, "reasoning": "direct"}"#,
        r#"{"verdicts": [{"index": 1, "useful": true}, {"index": 2, "useful": false}]}"#,
        r#"{"statements": [{"statement": "SqliteCheckpointer.", "attributed": true}]}"#,
        r#"{"verdicts": [{"index": 1, "supported": true}]}"#,
    ]);
    let report = Experiment::new(&ds)
        .evaluator(RetrievalMetric::hit_rate(1))
        .evaluator(RetrievalMetric::mrr())
        .evaluator(Faithfulness::new(judge.clone()))
        .evaluator(AnswerRelevance::new(judge.clone()))
        .evaluator(ContextPrecision::new(judge.clone()))
        .evaluator(ContextRecall::new(judge.clone()))
        .evaluator(CitationAccuracy::new(judge))
        .evaluator(AnswerSimilarity::new(Arc::new(FakeEmbeddings::new(256))))
        .concurrency(1)
        .run(|inputs: Value| {
            let rag = rag.clone();
            async move {
                let q = inputs["question"].as_str().unwrap_or_default().to_string();
                rag.invoke(&q).await.map(|a| rag_output(&a))
            }
        })
        .await;
    let r = &report.results[0];
    assert_eq!(r.error, None);
    assert_eq!(relevant, "sqlite");
    assert_eq!(r.output.as_ref().unwrap()["contexts"][0]["id"], "sqlite");
    assert_eq!(r.score("hit_rate@1"), Some(1.0));
    assert_eq!(r.score("mrr"), Some(1.0));
    assert_eq!(r.score("faithfulness"), Some(1.0));
    assert_eq!(r.score("answer_relevance"), Some(1.0));
    assert_eq!(r.score("context_precision"), Some(1.0));
    assert_eq!(r.score("context_recall"), Some(1.0));
    assert_eq!(r.score("citation_accuracy"), Some(1.0));
    let sim = r.score("answer_similarity").unwrap();
    assert!(sim > 0.0 && sim < 1.0);
    let faith = r.scores.iter().find(|s| s.key == "faithfulness").unwrap();
    assert!(
        faith
            .reason
            .as_deref()
            .unwrap()
            .contains("[judge: 2 calls,"),
        "{:?}",
        faith.reason
    );
    assert!(report.summary().contains("context_precision"));
}
