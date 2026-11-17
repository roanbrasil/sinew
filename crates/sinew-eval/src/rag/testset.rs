//! Synthetic test sets: question / reference answer / source ids triples
//! written by a chat model from a corpus.

use std::{collections::BTreeSet, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::json;
use sinew_core::{ChatModel, StructuredMode};
use sinew_rag::{english_tokenize, Document};

use super::{judge::ask as judge_ask, JudgeUsage, Meter};
use crate::{Dataset, Example};

/// The kind of question to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionType {
    /// Answered by one document.
    Factual,
    /// Needs information from two documents.
    MultiHop,
}

impl QuestionType {
    /// `"factual"` or `"multi_hop"`.
    ///
    /// ```
    /// assert_eq!(sinew_eval::rag::QuestionType::MultiHop.as_str(), "multi_hop");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            QuestionType::Factual => "factual",
            QuestionType::MultiHop => "multi_hop",
        }
    }
}

/// One generated test case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticExample {
    pub question: String,
    pub reference_answer: String,
    /// Ids of the documents the question was written from: the document id,
    /// else its `source` metadata, else `doc-<index>` (its position in the
    /// corpus slice).
    pub source_ids: Vec<String>,
    pub question_type: QuestionType,
}

/// The output of [`TestsetGenerator::generate`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Testset {
    pub examples: Vec<SyntheticExample>,
    /// Questions dropped as duplicates of earlier ones.
    pub duplicates: usize,
    /// Attempts that failed (model errors, invalid or empty replies).
    pub failures: Vec<String>,
    /// Model calls made while generating.
    pub usage: JudgeUsage,
}

impl Testset {
    /// Number of examples.
    ///
    /// ```
    /// assert_eq!(sinew_eval::rag::Testset::default().len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// True when no example was generated.
    ///
    /// ```
    /// assert!(sinew_eval::rag::Testset::default().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// A [`Dataset`]: inputs `{"question": ...}`, reference
    /// `{"answer": ..., "source_ids": [...]}` (what
    /// [`RagSample::from_eval`](super::RagSample::from_eval) reads) and
    /// metadata `{"question_type": ..., "synthetic": true}`. Example ids are
    /// `synthetic-<n>`. Save it with [`Dataset::save_jsonl`].
    ///
    /// ```
    /// use sinew_eval::rag::{QuestionType, SyntheticExample, Testset};
    ///
    /// let set = Testset {
    ///     examples: vec![SyntheticExample {
    ///         question: "Where is Paris?".into(),
    ///         reference_answer: "In France.".into(),
    ///         source_ids: vec!["fr".into()],
    ///         question_type: QuestionType::Factual,
    ///     }],
    ///     ..Default::default()
    /// };
    /// let ds = set.to_dataset("geo");
    /// assert_eq!(ds.examples[0].inputs["question"], "Where is Paris?");
    /// assert_eq!(ds.examples[0].reference.as_ref().unwrap()["source_ids"][0], "fr");
    /// ```
    pub fn to_dataset(&self, name: impl Into<String>) -> Dataset {
        let mut ds = Dataset::new(name).with_description("synthetic RAG test set");
        for (i, e) in self.examples.iter().enumerate() {
            let mut ex = Example::new(json!({"question": e.question}))
                .with_id(format!("synthetic-{}", i + 1))
                .with_reference(json!({"answer": e.reference_answer, "source_ids": e.source_ids}));
            ex.metadata = json!({"question_type": e.question_type.as_str(), "synthetic": true});
            ds = ds.with_example(ex);
        }
        ds
    }
}

/// Writes evaluation questions from a corpus with any chat model.
///
/// Attempts alternate between the configured question types. Each attempt
/// takes the next document of a shuffled order (fixed by
/// [`seed`](Self::seed)); a multi-hop attempt pairs it with the other
/// document that shares the most words with it. The model returns
/// `{"question", "answer"}` through structured output, sees the questions
/// written so far, and may return empty strings when the text has nothing
/// to ask about. A question whose word set overlaps an earlier one by at
/// least the [`dedup_threshold`](Self::dedup_threshold) (Jaccard) is a
/// duplicate. Generation stops at [`count`](Self::count) examples or after
/// [`max_attempts`](Self::max_attempts).
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use std::sync::Arc;
/// use sinew_core::{mock::ScriptedModel, Message};
/// use sinew_eval::rag::{QuestionType, TestsetGenerator};
/// use sinew_rag::Document;
///
/// let model = Arc::new(ScriptedModel::new([Message::assistant(
///     r#"{"question": "Which checkpointer survives restarts?", "answer": "SqliteCheckpointer."}"#,
/// )]));
/// let docs = [Document::new("SqliteCheckpointer stores graph threads on disk, so they survive restarts.")
///     .with_id("checkpointing")];
/// let set = TestsetGenerator::new(model)
///     .count(1)
///     .question_types([QuestionType::Factual])
///     .generate(&docs)
///     .await;
/// assert_eq!(set.examples[0].source_ids, vec!["checkpointing"]);
/// let dataset = set.to_dataset("rag-synthetic");
/// assert_eq!(dataset.len(), 1);
/// # }
/// ```
#[derive(Clone)]
pub struct TestsetGenerator {
    model: Arc<dyn ChatModel>,
    count: usize,
    types: Vec<QuestionType>,
    mode: StructuredMode,
    seed: u64,
    min_chars: usize,
    max_chars: usize,
    max_attempts: Option<usize>,
    dedup_threshold: f64,
}

impl TestsetGenerator {
    /// Defaults: 10 examples, factual and multi-hop alternating, seed 0,
    /// documents of at least 50 characters, each cut to 2000 in the prompt,
    /// at most `3 × count` attempts, duplicate threshold 0.8.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let g = sinew_eval::rag::TestsetGenerator::new(Arc::new(sinew_core::mock::ScriptedModel::new([])));
    /// ```
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            model,
            count: 10,
            types: vec![QuestionType::Factual, QuestionType::MultiHop],
            mode: StructuredMode::Tool,
            seed: 0,
            min_chars: 50,
            max_chars: 2000,
            max_attempts: None,
            dedup_threshold: 0.8,
        }
    }

    /// Number of examples to write.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let g = sinew_eval::rag::TestsetGenerator::new(Arc::new(sinew_core::mock::ScriptedModel::new([]))).count(25);
    /// ```
    pub fn count(mut self, n: usize) -> Self {
        self.count = n;
        self
    }

    /// Question types, used in turn (an empty list keeps the default).
    ///
    /// ```
    /// # use std::sync::Arc;
    /// use sinew_eval::rag::{QuestionType, TestsetGenerator};
    /// let g = TestsetGenerator::new(Arc::new(sinew_core::mock::ScriptedModel::new([])))
    ///     .question_types([QuestionType::Factual, QuestionType::Factual, QuestionType::MultiHop]);
    /// ```
    pub fn question_types(mut self, types: impl IntoIterator<Item = QuestionType>) -> Self {
        let t: Vec<QuestionType> = types.into_iter().collect();
        if !t.is_empty() {
            self.types = t;
        }
        self
    }

    /// How the structured reply is requested (default
    /// [`StructuredMode::Tool`]).
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let g = sinew_eval::rag::TestsetGenerator::new(Arc::new(sinew_core::mock::ScriptedModel::new([])))
    ///     .mode(sinew_core::StructuredMode::Json);
    /// ```
    pub fn mode(mut self, mode: StructuredMode) -> Self {
        self.mode = mode;
        self
    }

    /// Seed of the document order.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let g = sinew_eval::rag::TestsetGenerator::new(Arc::new(sinew_core::mock::ScriptedModel::new([]))).seed(42);
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Documents with fewer characters (after trimming) are skipped.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let g = sinew_eval::rag::TestsetGenerator::new(Arc::new(sinew_core::mock::ScriptedModel::new([]))).min_chars(200);
    /// ```
    pub fn min_chars(mut self, n: usize) -> Self {
        self.min_chars = n;
        self
    }

    /// Documents are cut to `n` characters in the prompt.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let g = sinew_eval::rag::TestsetGenerator::new(Arc::new(sinew_core::mock::ScriptedModel::new([]))).max_chars(1000);
    /// ```
    pub fn max_chars(mut self, n: usize) -> Self {
        self.max_chars = n.max(1);
        self
    }

    /// Maximum model calls (default `3 × count`).
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let g = sinew_eval::rag::TestsetGenerator::new(Arc::new(sinew_core::mock::ScriptedModel::new([]))).max_attempts(5);
    /// ```
    pub fn max_attempts(mut self, n: usize) -> Self {
        self.max_attempts = Some(n);
        self
    }

    /// Jaccard word overlap at which a question is a duplicate (default
    /// 0.8; 1.0 drops only questions with the same words).
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let g = sinew_eval::rag::TestsetGenerator::new(Arc::new(sinew_core::mock::ScriptedModel::new([]))).dedup_threshold(0.9);
    /// ```
    pub fn dedup_threshold(mut self, t: f64) -> Self {
        self.dedup_threshold = t;
        self
    }

    /// Writes the test set. Never fails: model errors and unusable replies
    /// are listed in [`Testset::failures`]; an empty corpus gives an empty
    /// set.
    ///
    /// ```
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// # use std::sync::Arc;
    /// let g = sinew_eval::rag::TestsetGenerator::new(Arc::new(sinew_core::mock::ScriptedModel::new([])));
    /// assert!(g.generate(&[]).await.is_empty());
    /// # }
    /// ```
    pub async fn generate(&self, documents: &[Document]) -> Testset {
        let mut set = Testset::default();
        let eligible: Vec<usize> = (0..documents.len())
            .filter(|&i| documents[i].content.trim().chars().count() >= self.min_chars.max(1))
            .collect();
        if eligible.is_empty() || self.count == 0 {
            return set;
        }
        let order = shuffled(&eligible, self.seed);
        let words: Vec<BTreeSet<String>> = documents
            .iter()
            .map(|d| english_tokenize(&d.content).into_iter().collect())
            .collect();
        let meter = Meter::new(self.model.as_ref());
        let max_attempts = self.max_attempts.unwrap_or(self.count.saturating_mul(3));
        let mut seen: Vec<BTreeSet<String>> = Vec::new();
        let schema = json!({
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"}
            },
            "required": ["question", "answer"]
        });
        for attempt in 0..max_attempts {
            if set.examples.len() >= self.count {
                break;
            }
            let anchor = order[attempt % order.len()];
            let mut kind = self.types[attempt % self.types.len()];
            let partner = if kind == QuestionType::MultiHop {
                let p = order
                    .iter()
                    .copied()
                    .filter(|&j| j != anchor)
                    .max_by(|&a, &b| {
                        jaccard(&words[anchor], &words[a])
                            .total_cmp(&jaccard(&words[anchor], &words[b]))
                            // Ties: the earlier one in `order` wins.
                            .then_with(|| pos(&order, b).cmp(&pos(&order, a)))
                    });
                if p.is_none() {
                    kind = QuestionType::Factual;
                }
                p
            } else {
                None
            };
            let mut docs = vec![anchor];
            docs.extend(partner);
            let (system, user) = self.prompt(kind, documents, &docs, &set.examples);
            let reply: QuestionAnswer = match judge_ask(
                &meter,
                self.mode,
                "test_question",
                schema.clone(),
                system,
                user,
            )
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    set.failures.push(format!("attempt {}: {e}", attempt + 1));
                    continue;
                }
            };
            let (q, a) = (reply.question.trim(), reply.answer.trim());
            if q.is_empty() || a.is_empty() {
                set.failures.push(format!(
                    "attempt {}: the model wrote no question for {}",
                    attempt + 1,
                    docs.iter()
                        .map(|&i| doc_id(documents, i))
                        .collect::<Vec<_>>()
                        .join(" + ")
                ));
                continue;
            }
            let qw = question_words(q);
            if seen.iter().any(|s| jaccard(s, &qw) >= self.dedup_threshold) {
                set.duplicates += 1;
                continue;
            }
            seen.push(qw);
            set.examples.push(SyntheticExample {
                question: q.to_string(),
                reference_answer: a.to_string(),
                source_ids: docs.iter().map(|&i| doc_id(documents, i)).collect(),
                question_type: kind,
            });
        }
        set.usage = meter.usage();
        set
    }

    fn prompt(
        &self,
        kind: QuestionType,
        documents: &[Document],
        docs: &[usize],
        written: &[SyntheticExample],
    ) -> (&'static str, String) {
        let system = match kind {
            QuestionType::Factual => {
                "You write evaluation data for a question-answering system over a document \
                 collection. From the DOCUMENT, write one question a user could ask that the \
                 document answers, and the correct answer, using only the document. The question \
                 must make sense on its own: do not mention \"the document\" or \"the text\". \
                 Keep the answer short. Do not repeat a question already written. If the \
                 document has nothing to ask about, return empty strings."
            }
            QuestionType::MultiHop => {
                "You write evaluation data for a question-answering system over a document \
                 collection. From DOCUMENT 1 and DOCUMENT 2, write one question whose answer needs \
                 information from both documents (for example, it combines a fact from each), and \
                 the correct answer, using only these documents. The question must make sense on \
                 its own: do not mention \"the documents\". Keep the answer short. Do not repeat a \
                 question already written. If the documents cannot be combined, return empty \
                 strings."
            }
        };
        let mut user = String::new();
        for (n, &i) in docs.iter().enumerate() {
            let label = if docs.len() == 1 {
                "DOCUMENT".to_string()
            } else {
                format!("DOCUMENT {}", n + 1)
            };
            let text: String = documents[i]
                .content
                .trim()
                .chars()
                .take(self.max_chars)
                .collect();
            user.push_str(&format!("{label}:\n{text}\n\n"));
        }
        if !written.is_empty() {
            user.push_str("QUESTIONS ALREADY WRITTEN:\n");
            for e in written.iter().rev().take(20) {
                user.push_str(&format!("- {}\n", e.question));
            }
        }
        (system, user.trim_end().to_string())
    }
}

#[derive(Deserialize)]
struct QuestionAnswer {
    question: String,
    answer: String,
}

fn doc_id(documents: &[Document], i: usize) -> String {
    let d = &documents[i];
    d.id.clone()
        .or_else(|| d.source().map(str::to_string))
        .unwrap_or_else(|| format!("doc-{i}"))
}

fn pos(order: &[usize], x: usize) -> usize {
    order.iter().position(|&o| o == x).unwrap_or(usize::MAX)
}

/// Content words of a question; falls back to lower-cased words when every
/// word is a stop word.
fn question_words(q: &str) -> BTreeSet<String> {
    let w: BTreeSet<String> = english_tokenize(q).into_iter().collect();
    if w.is_empty() {
        q.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(str::to_lowercase)
            .collect()
    } else {
        w
    }
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 1.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

/// Fisher-Yates with a xorshift64* generator (deterministic per seed).
fn shuffled(items: &[usize], seed: u64) -> Vec<usize> {
    let mut v = items.to_vec();
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    for i in (1..v.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_is_a_deterministic_permutation() {
        let items: Vec<usize> = (0..10).collect();
        let a = shuffled(&items, 7);
        assert_eq!(a, shuffled(&items, 7));
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, items);
    }

    #[test]
    fn near_duplicate_questions_overlap() {
        let a = question_words("How do I persist graph threads?");
        let b = question_words("How can I persist graph threads?");
        assert_eq!(jaccard(&a, &b), 1.0);
        let c = question_words("Which embeddings does Ollama serve?");
        assert!(jaccard(&a, &c) < 0.2);
    }
}
