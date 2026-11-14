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
pub struct SpecPolicy {
    ceiling: usize,
    budget: usize,
    cooldown: usize,
}
impl SpecPolicy {
    pub fn new(width: usize) -> Self {
        let ceiling = width.saturating_sub(1);
        Self {
            ceiling,
            budget: ceiling,
            cooldown: 0,
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
        if accepted < drafted {
            self.budget = (self.budget / 2).max(1).min(self.ceiling);
            if accepted == 0 {
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
            "../../specs/open-engine/tests/fixtures/spec_lookup_cases.json"
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
    }
}
