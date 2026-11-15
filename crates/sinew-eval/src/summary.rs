//! Summary evaluators: one score for a whole experiment (pass rate, F1, ...),
//! computed from every [`ExampleResult`] after the run.

use crate::{as_text, ExampleResult, Score};

/// Scores a whole experiment. Results include every repetition.
pub trait SummaryEvaluator: Send + Sync {
    /// Key of the summary score.
    fn name(&self) -> &str;
    fn evaluate(&self, results: &[ExampleResult]) -> Score;
}

/// Fraction of examples whose `key` score is at least `threshold`. Examples
/// that errored or lack the score count as failures.
pub struct PassRate {
    key: String,
    threshold: f64,
    name: String,
}

impl PassRate {
    pub fn new(key: impl Into<String>, threshold: f64) -> Self {
        let key = key.into();
        Self {
            name: format!("pass_rate:{key}"),
            key,
            threshold,
        }
    }

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl SummaryEvaluator for PassRate {
    fn name(&self) -> &str {
        &self.name
    }
    fn evaluate(&self, results: &[ExampleResult]) -> Score {
        if results.is_empty() {
            return Score::new(self.name(), 0.0).with_reason("no results");
        }
        let pass = results
            .iter()
            .filter(|r| r.score(&self.key).is_some_and(|s| s >= self.threshold))
            .count();
        Score::new(self.name(), pass as f64 / results.len() as f64).with_reason(format!(
            "{pass}/{} with {} >= {}",
            results.len(),
            self.key,
            self.threshold
        ))
    }
}

/// Binary-classification F1: the output and the reference are labels (text,
/// trimmed, case-insensitive) and `positive` is the positive class.
pub struct ClassificationF1 {
    positive: String,
}

impl ClassificationF1 {
    pub fn new(positive: impl Into<String>) -> Self {
        Self {
            positive: positive.into().trim().to_lowercase(),
        }
    }
}

impl SummaryEvaluator for ClassificationF1 {
    fn name(&self) -> &str {
        "f1"
    }
    fn evaluate(&self, results: &[ExampleResult]) -> Score {
        let (mut tp, mut fp, mut fneg) = (0usize, 0usize, 0usize);
        for r in results {
            let Some(reference) = &r.reference else {
                continue;
            };
            let truth = as_text(reference).trim().to_lowercase() == self.positive;
            let pred = r
                .output
                .as_ref()
                .is_some_and(|o| as_text(o).trim().to_lowercase() == self.positive);
            match (pred, truth) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fneg += 1,
                _ => {}
            }
        }
        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fneg == 0 {
            0.0
        } else {
            tp as f64 / (tp + fneg) as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Score::new(self.name(), f1).with_reason(format!(
            "precision {precision:.3}, recall {recall:.3} (tp {tp}, fp {fp}, fn {fneg})"
        ))
    }
}

/// A summary evaluator from a closure.
pub struct FnSummary<F> {
    name: String,
    f: F,
}

impl<F> FnSummary<F>
where
    F: Fn(&[ExampleResult]) -> Score + Send + Sync,
{
    pub fn new(name: impl Into<String>, f: F) -> Self {
        Self {
            name: name.into(),
            f,
        }
    }
}

impl<F> SummaryEvaluator for FnSummary<F>
where
    F: Fn(&[ExampleResult]) -> Score + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }
    fn evaluate(&self, results: &[ExampleResult]) -> Score {
        let mut s = (self.f)(results);
        s.key = self.name.clone();
        s
    }
}
