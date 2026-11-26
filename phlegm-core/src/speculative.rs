//! Drafter-independent sample-and-match policy (SERVER-SPECULATIVE).
//! Drafts never advance the sampler or reach the output callback.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeculativeMode {
    #[default]
    Off,
    Auto,
    Lookup,
    Mtp,
}

impl std::str::FromStr for SpeculativeMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "lookup" => Ok(Self::Lookup),
            "mtp" => Ok(Self::Mtp),
            _ => Err("speculative must be off, auto, lookup or mtp".into()),
        }
    }
}

pub enum Draft {
    Tokens(Vec<u32>),
    /// Ask protocol 2 to append at most this many drafts after the emitted token.
    Bridge(usize),
}

pub trait Drafter {
    /// History includes the emitted token that has not yet been run.
    fn draft(&mut self, history: &[u32], max: usize) -> Draft;
    fn policy(&self, width: usize) -> SpecPolicy {
        SpecPolicy::new(width)
    }
}

pub struct NoneDrafter;
impl Drafter for NoneDrafter {
    fn draft(&mut self, _: &[u32], _: usize) -> Draft {
        Draft::Tokens(Vec::new())
    }
}

pub struct Bridge {
    pub max: usize,
}
impl Drafter for Bridge {
    fn draft(&mut self, _: &[u32], max: usize) -> Draft {
        Draft::Bridge(max.min(self.max))
    }
    // A model draft's miss says little about the next draft, unlike a lookup miss.
    fn policy(&self, width: usize) -> SpecPolicy {
        SpecPolicy::for_model_drafts(width)
    }
}

pub struct Lookup {
    pub nmax: usize,
    pub nmin: usize,
    pub extend: bool,
}
impl Default for Lookup {
    fn default() -> Self {
        Self {
            nmax: 4,
            nmin: 2,
            extend: true,
        }
    }
}
impl Drafter for Lookup {
    fn draft(&mut self, history: &[u32], max: usize) -> Draft {
        Draft::Tokens(self.lookup(history, max).1)
    }
}
impl Lookup {
    /// Port of stage 0's replay.py, checked against OPEN-SPEC-LOOKUP's 13 cases.
    pub fn lookup(&self, history: &[u32], k: usize) -> (usize, Vec<u32>) {
        let mut work = history.to_vec();
        let mut out = Vec::new();
        let mut first_n = 0;
        while out.len() < k {
            let mut found = None;
            for n in (self.nmin.max(1)..=self.nmax.min(work.len().saturating_sub(1))).rev() {
                let suffix = &work[work.len() - n..];
                if let Some(start) = (0..work.len() - n)
                    .rev()
                    .find(|&s| &work[s..s + n] == suffix)
                {
                    found = Some((n, start + n));
                    break;
                }
            }
            let Some((n, start)) = found else { break };
            if first_n == 0 {
                first_n = n;
            }
            let chunk = work[start..work.len().min(start + k - out.len())].to_vec();
            out.extend_from_slice(&chunk);
            work.extend(chunk);
            if !self.extend {
                break;
            }
        }
        if self.extend && !out.is_empty() {
            out.resize(k, *out.last().unwrap());
        }
        (first_n, out)
    }
}

/// Bounded per-request backoff: halve the budget after a rejection; after a
/// completely rejected draft, run two width-one cycles before probing again.
/// Model drafts skip those cycles while the running acceptance stays at or above `floor`.
pub struct SpecPolicy {
    ceiling: usize,
    budget: usize,
    cooldown: usize,
    rate: f32,
    floor: f32,
    // SERVER-SPEC-COST: the measured price of a cycle. Drafting pays only while a drafted cycle's tokens per
    // millisecond beat a plain step's; below that the policy runs plain steps and probes again later.
    draft_ms: Option<f64>,
    draft_tokens: f64,
    plain_ms: Option<f64>,
    drafted_cycles: usize,
}
impl SpecPolicy {
    pub fn new(width: usize) -> Self {
        let ceiling = width.saturating_sub(1);
        Self {
            ceiling,
            budget: ceiling,
            cooldown: 0,
            rate: 1.0,
            floor: f32::INFINITY,
            draft_ms: None,
            draft_tokens: 0.0,
            plain_ms: None,
            drafted_cycles: 0,
        }
    }
    /// Plain steps to run before drafting is tried again once it measured slower than plain decode.
    const PROBE_AFTER: usize = 16;
    /// A cycle's wall time: `drafted` draft rows, `emitted` tokens it produced (SERVER-SPEC-COST).
    pub fn observe_time(&mut self, drafted: usize, emitted: usize, ms: f64) {
        let ema = |old: Option<f64>, x: f64| Some(old.map_or(x, |o| 0.8 * o + 0.2 * x));
        if drafted == 0 {
            self.plain_ms = ema(self.plain_ms, ms);
            return;
        }
        self.drafted_cycles += 1;
        self.draft_ms = ema(self.draft_ms, ms);
        self.draft_tokens = if self.drafted_cycles == 1 { emitted as f64 } else { 0.8 * self.draft_tokens + 0.2 * emitted as f64 };
        let Some(plain) = self.plain_ms else {
            // nothing to compare with yet: one plain step after a few drafted cycles measures it
            if self.drafted_cycles == 4 {
                self.cooldown = self.cooldown.max(1);
            }
            return;
        };
        let draft = self.draft_ms.unwrap();
        if self.drafted_cycles >= 4 && self.draft_tokens / draft < 1.0 / plain {
            self.cooldown = self.cooldown.max(Self::PROBE_AFTER);
            self.drafted_cycles = 2;     // a probe gets two cycles to show it pays before backing off again
        }
    }
    /// A drafted cycle costs about 1.5 plain steps (Qwen3.6-35B-A3B MTP, XDNA2), so it loses below half accepted.
    pub fn for_model_drafts(width: usize) -> Self {
        Self {
            floor: 0.5,
            ..Self::new(width)
        }
    }
    pub fn max_draft(&mut self, remaining: usize, room: usize) -> usize {
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return 0;
        }
        self.budget
            .min(remaining.saturating_sub(1))
            .min(room.saturating_sub(1))
    }
    pub fn observe(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        self.rate = 0.875 * self.rate + 0.125 * accepted as f32 / drafted as f32;
        if accepted < drafted {
            self.budget = (self.budget / 2).max(1).min(self.ceiling);
            if accepted == 0 && self.rate < self.floor {
                self.cooldown = 2;
            }
        } else {
            self.budget = (self.budget + 1).min(self.ceiling);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lookup_matches_all_replay_fixtures() {
        let cases: serde_json::Value = serde_json::from_str(include_str!(
            "../../open-kernels/specs/open-engine/tests/fixtures/spec_lookup_cases.json"
        ))
        .unwrap();
        assert_eq!(cases["cases"].as_array().unwrap().len(), 13);
        for c in cases["cases"].as_array().unwrap() {
            let ids = |key: &str| {
                c[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect::<Vec<_>>()
            };
            let lookup = Lookup {
                nmax: c["nmax"].as_u64().unwrap() as usize,
                nmin: c["nmin"].as_u64().unwrap() as usize,
                extend: c["extend"].as_bool().unwrap(),
            };
            assert_eq!(
                lookup.lookup(&ids("hist"), c["k"].as_u64().unwrap() as usize),
                (c["n"].as_u64().unwrap() as usize, ids("drafts")),
                "{}",
                c["what"]
            );
        }
        assert_eq!(Lookup::default().lookup(&[], 3), (0, vec![]));
        assert_eq!(Lookup::default().lookup(&[1, 2, 1, 2], 0), (0, vec![]));
    }
    #[test]
    fn policy_bounds_and_backs_off() {
        let mut p = SpecPolicy::new(4);
        assert_eq!(p.max_draft(20, 20), 3);
        assert_eq!(p.max_draft(2, 20), 1);
        assert_eq!(p.max_draft(20, 1), 0);
        p.observe(3, 0);
        assert_eq!(p.max_draft(20, 20), 0);
        assert_eq!(p.max_draft(20, 20), 0);
        assert_eq!(p.max_draft(20, 20), 1);
        p.observe(1, 1);
        assert_eq!(p.max_draft(20, 20), 2);
        assert_eq!(SpecPolicy::new(16).max_draft(100, 100), 15);
    }
    #[test]
    fn drafting_backs_off_when_a_drafted_cycle_costs_more_than_its_tokens() {
        // plain step 73 ms; a drafted cycle 138 ms for 1.8 tokens is 76.7 ms a token: slower, so plain steps follow
        let mut p = SpecPolicy::for_model_drafts(2);
        for _ in 0..4 {
            assert_eq!(p.max_draft(100, 100), 1);
            p.observe_time(1, 2, 138.0);
        }
        assert_eq!(p.max_draft(100, 100), 0, "one plain step to measure plain decode");
        p.observe_time(0, 1, 73.0);
        assert_eq!(p.max_draft(100, 100), 1);
        p.observe_time(1, 1, 138.0);
        assert_eq!(p.max_draft(100, 100), 0, "slower than plain: back off");
        for _ in 0..SpecPolicy::PROBE_AFTER - 1 {
            assert_eq!(p.max_draft(100, 100), 0);
            p.observe_time(0, 1, 73.0);
        }
        assert_eq!(p.max_draft(100, 100), 1, "probe again");
        // a cheap cycle that pays keeps drafting
        let mut q = SpecPolicy::for_model_drafts(2);
        q.observe_time(0, 1, 73.0);
        for _ in 0..10 {
            assert_eq!(q.max_draft(100, 100), 1);
            q.observe_time(1, 2, 120.0);
        }
    }

    #[test]
    fn model_drafts_keep_drafting_until_acceptance_falls_below_half() {
        let mut p = SpecPolicy::for_model_drafts(2);
        p.observe(1, 0);
        assert_eq!(p.max_draft(20, 20), 1);
        for _ in 0..4 {
            p.observe(1, 0);
        }
        // rate 0.875^5 = 0.51: still drafting; the sixth miss takes it to 0.45
        assert_eq!(p.max_draft(20, 20), 1);
        p.observe(1, 0);
        assert_eq!(p.max_draft(20, 20), 0);
        assert_eq!(p.max_draft(20, 20), 0);
        assert_eq!(p.max_draft(20, 20), 1);
    }
}
