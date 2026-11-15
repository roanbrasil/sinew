//! Pairwise (comparative) evaluation: given two experiments on the same
//! dataset, a [`PairwiseEvaluator`] sees both outputs for each example and
//! states a preference. [`LlmPairwiseJudge`] asks a model which response is
//! better, optionally in both orders to cancel position bias.

use std::{collections::BTreeMap, collections::HashMap, sync::Arc};

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinew_core::{ChatModel, ChatRequest, Message};

use crate::{as_text, now_ms, ExampleResult, ExperimentReport};

/// Which side won.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Preference {
    A,
    B,
    Tie,
}

/// The verdict of one pairwise evaluator on one example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseScore {
    pub key: String,
    /// `None` when the evaluator could not decide (judge error, bad reply).
    pub preference: Option<Preference>,
    /// 1 for a win, 0.5 for a tie, 0 for a loss (0 and 0 when undecided).
    pub score_a: f64,
    pub score_b: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl PairwiseScore {
    pub fn new(key: impl Into<String>, preference: Option<Preference>) -> Self {
        let (score_a, score_b) = match preference {
            Some(Preference::A) => (1.0, 0.0),
            Some(Preference::B) => (0.0, 1.0),
            Some(Preference::Tie) => (0.5, 0.5),
            None => (0.0, 0.0),
        };
        Self {
            key: key.into(),
            preference,
            score_a,
            score_b,
            reason: None,
        }
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// What a pairwise evaluator sees for one example.
#[derive(Debug, Clone, Copy)]
pub struct PairwiseInput<'a> {
    pub inputs: &'a Value,
    pub reference: Option<&'a Value>,
    pub a: &'a Value,
    pub b: &'a Value,
}

/// Compares two outputs for the same inputs. Must not panic; return an
/// undecided score with a reason instead.
#[async_trait]
pub trait PairwiseEvaluator: Send + Sync {
    fn name(&self) -> &str;
    async fn compare(&self, input: PairwiseInput<'_>) -> PairwiseScore;
}

/// A pairwise evaluator from a synchronous closure.
pub struct FnPairwise<F> {
    name: String,
    f: F,
}

impl<F> FnPairwise<F>
where
    F: Fn(PairwiseInput<'_>) -> Option<Preference> + Send + Sync,
{
    pub fn new(name: impl Into<String>, f: F) -> Self {
        Self {
            name: name.into(),
            f,
        }
    }
}

#[async_trait]
impl<F> PairwiseEvaluator for FnPairwise<F>
where
    F: Fn(PairwiseInput<'_>) -> Option<Preference> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }
    async fn compare(&self, input: PairwiseInput<'_>) -> PairwiseScore {
        PairwiseScore::new(self.name.clone(), (self.f)(input))
    }
}

/// Parses a pairwise judge reply: JSON `{"preference": "A" | "B" | "tie",
/// "reason": "..."}` (also `winner`), possibly wrapped in prose or a code
/// fence; a bare `A`, `B` or `tie` also works.
pub fn parse_preference(text: &str) -> Option<(Preference, Option<String>)> {
    let pick = |s: &str| match s.trim().trim_matches('"').to_ascii_lowercase().as_str() {
        "a" | "response a" | "1" => Some(Preference::A),
        "b" | "response b" | "2" => Some(Preference::B),
        "tie" | "equal" | "draw" | "both" | "none" => Some(Preference::Tie),
        _ => None,
    };
    if let (Some(start), Some(end)) = (text.find('{'), text.rfind('}')) {
        if let Some(v) = text
            .get(start..=end)
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
        {
            let p = ["preference", "winner", "choice"].iter().find_map(|k| {
                v[*k].as_str().and_then(pick).or_else(|| match &v[*k] {
                    Value::Number(n) => pick(&n.to_string()),
                    _ => None,
                })
            });
            if let Some(p) = p {
                return Some((p, v["reason"].as_str().map(str::to_string)));
            }
        }
    }
    pick(text).map(|p| (p, None))
}

/// LLM-as-judge for pairs: asks the model which response better satisfies
/// the rubric. With [`LlmPairwiseJudge::swap_positions`] (on by default) it
/// asks twice with the responses swapped and reports a tie when the two
/// verdicts disagree, which cancels position bias.
pub struct LlmPairwiseJudge {
    model: Arc<dyn ChatModel>,
    rubric: String,
    key: String,
    swap: bool,
}

impl LlmPairwiseJudge {
    pub fn new(model: Arc<dyn ChatModel>, rubric: impl Into<String>) -> Self {
        Self {
            model,
            rubric: rubric.into(),
            key: "preference".into(),
            swap: true,
        }
    }

    pub fn named(mut self, key: impl Into<String>) -> Self {
        self.key = key.into();
        self
    }

    /// Judge both orders (default true).
    pub fn swap_positions(mut self, on: bool) -> Self {
        self.swap = on;
        self
    }

    async fn ask(
        &self,
        i: &PairwiseInput<'_>,
        first: &Value,
        second: &Value,
    ) -> Result<(Preference, Option<String>), String> {
        let reference = i.reference.map(as_text).unwrap_or_else(|| "(none)".into());
        let messages = vec![
            Message::system(
                "You are an impartial judge comparing two responses to the same input. \
                 Use the RUBRIC. Do not let response order or length sway you. Reply with \
                 only a JSON object: {\"preference\": \"A\" | \"B\" | \"tie\", \
                 \"reason\": \"<one sentence>\"}.",
            ),
            Message::user(format!(
                "RUBRIC:\n{}\n\nINPUT:\n{}\n\nREFERENCE:\n{}\n\nRESPONSE A:\n{}\n\nRESPONSE B:\n{}",
                self.rubric,
                as_text(i.inputs),
                reference,
                as_text(first),
                as_text(second)
            )),
        ];
        let resp = self
            .model
            .chat(ChatRequest::new(messages).with_temperature(0.0))
            .await
            .map_err(|e| format!("judge failed: {e}"))?;
        parse_preference(&resp.message.content).ok_or_else(|| {
            format!(
                "unparseable judge reply: {}",
                resp.message.content.chars().take(200).collect::<String>()
            )
        })
    }
}

#[async_trait]
impl PairwiseEvaluator for LlmPairwiseJudge {
    fn name(&self) -> &str {
        &self.key
    }
    async fn compare(&self, i: PairwiseInput<'_>) -> PairwiseScore {
        let first = match self.ask(&i, i.a, i.b).await {
            Ok(v) => v,
            Err(e) => return PairwiseScore::new(self.key.clone(), None).with_reason(e),
        };
        if !self.swap {
            let s = PairwiseScore::new(self.key.clone(), Some(first.0));
            return match first.1 {
                Some(r) => s.with_reason(r),
                None => s,
            };
        }
        let second = match self.ask(&i, i.b, i.a).await {
            Ok((p, r)) => (
                match p {
                    Preference::A => Preference::B,
                    Preference::B => Preference::A,
                    Preference::Tie => Preference::Tie,
                },
                r,
            ),
            Err(e) => return PairwiseScore::new(self.key.clone(), None).with_reason(e),
        };
        if first.0 == second.0 {
            let s = PairwiseScore::new(self.key.clone(), Some(first.0));
            match first.1 {
                Some(r) => s.with_reason(r),
                None => s,
            }
        } else {
            PairwiseScore::new(self.key.clone(), Some(Preference::Tie)).with_reason(format!(
                "verdict changed with the order ({:?} then {:?}); counted as a tie",
                first.0, second.0
            ))
        }
    }
}

/// Both outputs and every pairwise verdict for one example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseResult {
    pub example_id: String,
    #[serde(default)]
    pub repetition: usize,
    pub inputs: Value,
    #[serde(default)]
    pub reference: Option<Value>,
    #[serde(default)]
    pub output_a: Option<Value>,
    #[serde(default)]
    pub output_b: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub scores: Vec<PairwiseScore>,
}

/// Win/tie counts for one evaluator key.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PairwiseStats {
    pub wins_a: usize,
    pub wins_b: usize,
    pub ties: usize,
    pub undecided: usize,
    /// (wins + ties / 2) / decided, per side.
    pub win_rate_a: f64,
    pub win_rate_b: f64,
}

/// Reference to one side of a comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentRef {
    pub id: String,
    pub name: String,
}

/// The outcome of [`evaluate_pairwise`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseReport {
    pub id: String,
    pub name: String,
    pub dataset: String,
    pub created_ms: u64,
    pub experiment_a: ExperimentRef,
    pub experiment_b: ExperimentRef,
    #[serde(default)]
    pub metadata: Value,
    pub results: Vec<PairwiseResult>,
    /// Per evaluator key.
    pub summary: BTreeMap<String, PairwiseStats>,
}

impl PairwiseReport {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Short text summary.
    pub fn summary_text(&self) -> String {
        let mut out = format!(
            "pairwise {}: A = {}, B = {}, {} examples",
            self.name,
            self.experiment_a.name,
            self.experiment_b.name,
            self.results.len()
        );
        for (k, s) in &self.summary {
            out.push_str(&format!(
                "\n  {k:<20} A wins {}  B wins {}  ties {}  undecided {}  win rate A {:.2} B {:.2}",
                s.wins_a, s.wins_b, s.ties, s.undecided, s.win_rate_a, s.win_rate_b
            ));
        }
        out
    }
}

fn stats(results: &[PairwiseResult]) -> BTreeMap<String, PairwiseStats> {
    let mut out: BTreeMap<String, PairwiseStats> = BTreeMap::new();
    for s in results.iter().flat_map(|r| &r.scores) {
        let e = out.entry(s.key.clone()).or_default();
        match s.preference {
            Some(Preference::A) => e.wins_a += 1,
            Some(Preference::B) => e.wins_b += 1,
            Some(Preference::Tie) => e.ties += 1,
            None => e.undecided += 1,
        }
    }
    for e in out.values_mut() {
        let decided = (e.wins_a + e.wins_b + e.ties) as f64;
        if decided > 0.0 {
            e.win_rate_a = (e.wins_a as f64 + e.ties as f64 / 2.0) / decided;
            e.win_rate_b = (e.wins_b as f64 + e.ties as f64 / 2.0) / decided;
        }
    }
    out
}

/// Compares two experiments example by example (aligned by example id and
/// repetition). At most `concurrency` examples are judged at once. Examples
/// missing on one side or failed on either side are kept with an `error` and
/// no verdicts.
pub async fn evaluate_pairwise(
    a: &ExperimentReport,
    b: &ExperimentReport,
    evaluators: &[Box<dyn PairwiseEvaluator>],
    concurrency: usize,
) -> PairwiseReport {
    let b_index: HashMap<(&str, usize), &ExampleResult> = b
        .results
        .iter()
        .map(|r| ((r.example_id.as_str(), r.repetition), r))
        .collect();
    let mut results: Vec<(usize, PairwiseResult)> =
        futures::stream::iter(a.results.iter().enumerate())
            .map(|(i, ra)| {
                let rb = b_index
                    .get(&(ra.example_id.as_str(), ra.repetition))
                    .copied();
                async move {
                    let mut res = PairwiseResult {
                        example_id: ra.example_id.clone(),
                        repetition: ra.repetition,
                        inputs: ra.inputs.clone(),
                        reference: ra.reference.clone(),
                        output_a: ra.output.clone(),
                        output_b: rb.and_then(|r| r.output.clone()),
                        error: None,
                        scores: Vec::new(),
                    };
                    res.error = match (rb, &ra.error) {
                        (None, _) => Some("example missing in experiment B".into()),
                        (_, Some(e)) => Some(format!("A failed: {e}")),
                        (Some(rb), None) => rb.error.as_ref().map(|e| format!("B failed: {e}")),
                    };
                    if res.error.is_none() {
                        if let (Some(oa), Some(ob)) = (&res.output_a, &res.output_b) {
                            for e in evaluators {
                                res.scores.push(
                                    e.compare(PairwiseInput {
                                        inputs: &ra.inputs,
                                        reference: ra.reference.as_ref(),
                                        a: oa,
                                        b: ob,
                                    })
                                    .await,
                                );
                            }
                        }
                    }
                    (i, res)
                }
            })
            .buffer_unordered(concurrency.max(1))
            .collect()
            .await;
    results.sort_by_key(|(i, _)| *i);
    let results: Vec<PairwiseResult> = results.into_iter().map(|(_, r)| r).collect();
    let created_ms = now_ms();
    PairwiseReport {
        id: uuid::Uuid::new_v4().to_string(),
        name: format!("{} vs {}", a.name, b.name),
        dataset: a.dataset.clone(),
        created_ms,
        experiment_a: ExperimentRef {
            id: a.id.clone(),
            name: a.name.clone(),
        },
        experiment_b: ExperimentRef {
            id: b.id.clone(),
            name: b.name.clone(),
        },
        metadata: Value::Null,
        summary: stats(&results),
        results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sinew_core::mock::ScriptedModel;

    #[test]
    fn parses_preferences() {
        assert_eq!(
            parse_preference("```json\n{\"preference\": \"B\", \"reason\": \"shorter\"}\n```"),
            Some((Preference::B, Some("shorter".into())))
        );
        assert_eq!(
            parse_preference("{\"winner\": \"tie\"}").unwrap().0,
            Preference::Tie
        );
        assert_eq!(parse_preference(" A ").unwrap().0, Preference::A);
        assert_eq!(parse_preference("no idea"), None);
    }

    #[tokio::test]
    async fn swapped_judge_cancels_position_bias() {
        // Always says "A": first order prefers a, swapped order prefers b → tie.
        let biased = Arc::new(ScriptedModel::new([
            Message::assistant("{\"preference\": \"A\"}"),
            Message::assistant("{\"preference\": \"A\"}"),
        ]));
        let judge = LlmPairwiseJudge::new(biased.clone(), "better?");
        let s = judge
            .compare(PairwiseInput {
                inputs: &json!("q"),
                reference: None,
                a: &json!("x"),
                b: &json!("y"),
            })
            .await;
        assert_eq!(s.preference, Some(Preference::Tie));
        let reqs = biased.requests();
        assert!(reqs[0].messages[1].content.contains("RESPONSE A:\nx"));
        assert!(reqs[1].messages[1].content.contains("RESPONSE A:\ny"));

        // Consistent judge: prefers y in both orders → B.
        let fair = Arc::new(ScriptedModel::new([
            Message::assistant("{\"preference\": \"B\", \"reason\": \"y is right\"}"),
            Message::assistant("{\"preference\": \"A\"}"),
        ]));
        let s = LlmPairwiseJudge::new(fair, "better?")
            .compare(PairwiseInput {
                inputs: &json!("q"),
                reference: None,
                a: &json!("x"),
                b: &json!("y"),
            })
            .await;
        assert_eq!(s.preference, Some(Preference::B));
        assert_eq!((s.score_a, s.score_b), (0.0, 1.0));
    }
}
