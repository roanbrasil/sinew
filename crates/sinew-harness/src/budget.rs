//! Per-turn budgets: steps, tokens, cost and wall-clock time.

use std::time::Duration;

use sinew_core::Usage;

/// Limits for one turn (one user message, including resumes after
/// approvals). When a limit is hit the turn ends with a
/// [`StopReason`](crate::StopReason) instead of an error.
#[derive(Debug, Clone, PartialEq)]
pub struct Budget {
    /// Maximum model calls per turn (default 50).
    pub max_steps: u32,
    /// Maximum input + output tokens per turn.
    pub max_total_tokens: Option<u64>,
    /// Maximum estimated cost per turn in USD. Needs [`Pricing`]; without it
    /// only the token budget applies.
    pub max_cost_usd: Option<f64>,
    /// Wall-clock limit per turn. No new model call starts after it, and a
    /// call still running when it expires is cancelled.
    pub timeout: Option<Duration>,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            max_steps: 50,
            max_total_tokens: None,
            max_cost_usd: None,
            timeout: None,
        }
    }
}

/// Price per million tokens, used to estimate cost from [`Usage`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pricing {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
}

impl Pricing {
    pub fn new(input_per_mtok: f64, output_per_mtok: f64) -> Self {
        Self {
            input_per_mtok,
            output_per_mtok,
        }
    }

    pub fn cost(&self, usage: Usage) -> f64 {
        (f64::from(usage.input_tokens) * self.input_per_mtok
            + f64::from(usage.output_tokens) * self.output_per_mtok)
            / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_per_million() {
        let p = Pricing::new(3.0, 15.0);
        let c = p.cost(Usage {
            input_tokens: 1_000_000,
            output_tokens: 100_000,
            ..Default::default()
        });
        assert!((c - 4.5).abs() < 1e-9);
    }
}
