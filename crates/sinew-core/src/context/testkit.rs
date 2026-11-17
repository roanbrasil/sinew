//! Test helpers: seeded random transcripts with many tool calls.

use serde_json::json;

use crate::{Message, ToolCall};

pub(crate) fn call(id: &str, name: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: json!({ "q": id }),
    }
}

/// Small deterministic PRNG (SplitMix64).
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A valid history of about `n` messages: a system prompt, then user turns,
/// each followed by zero to three tool rounds (one to three parallel calls
/// with results of random size) and a final assistant answer.
pub(crate) fn transcript(seed: u64, n: usize) -> Vec<Message> {
    let mut r = Rng::new(seed);
    let mut out = vec![Message::system("You are a careful research agent.")];
    let mut id = 0;
    let mut turn = 0;
    while out.len() < n {
        turn += 1;
        out.push(Message::user(format!(
            "turn {turn}: please look up topic {} and report",
            r.below(1000)
        )));
        for _ in 0..r.below(4) {
            let k = 1 + r.below(3) as usize;
            let calls: Vec<ToolCall> = (0..k)
                .map(|_| {
                    id += 1;
                    let name = ["search", "read_file", "grep"][r.below(3) as usize];
                    call(&format!("c{id}"), name)
                })
                .collect();
            let results: Vec<Message> = calls
                .iter()
                .map(|c| {
                    let size = 20 + r.below(3000) as usize;
                    Message::tool_result(c.id.clone(), "result data ".repeat(size / 12 + 1))
                })
                .collect();
            out.push(Message::assistant_with_tools("let me check", calls));
            out.extend(results);
        }
        out.push(Message::assistant(format!(
            "turn {turn} answer: {}",
            "detail ".repeat(r.below(40) as usize)
        )));
    }
    out
}
