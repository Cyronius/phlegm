//! Logit sampler for the open NPU engine, ported from
//! `tools/kernel-interp/sampler.py`.
//!
//! Consumes the FULL logits vector the generate loop computes on CPU from the
//! NPU's final hidden state (`forward::Model::logits`, shape `[VOCAB]` f32) and
//! returns the next token id. Supports greedy, temperature, top-k, top-p
//! (nucleus), and repetition penalty. Seedable and deterministic given a seed
//! — NOT bit-compatible with the Python version's draws (numpy's PCG64 isn't
//! reproduced here), only internally deterministic.
//!
//! Transform order matches the common HF `generate` pipeline:
//!   1. repetition penalty (on already-produced tokens)
//!   2. temperature scale
//!   3. top-k filter
//!   4. top-p (nucleus) filter
//!   5. softmax -> multinomial draw   (temperature == 0 -> argmax, no draw)

const NEG_INF: f32 = -1e30;

/// splitmix64, used both to seed and (re-mixed each call) to draw uniforms.
/// Small, dependency-free, and reproducible given a seed — the property the
/// Python sampler's self-test actually cares about, not cross-language parity.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// A logit bias that grows with the reply: 0 through `start` reply tokens, then rising linearly to
/// `strength` over `window` more (Clover's `dllm_length_penalty` on the stop tokens and
/// `dllm_think_end_bias` on `</think>`, `sampling_clover_moe.py`).
#[derive(Clone, Debug, PartialEq)]
pub struct Ramp {
    pub ids: Vec<u32>,
    pub strength: f32,
    pub start: usize,
    pub window: usize,
    /// Off once the reply holds one of `ids` (the reasoning block closed).
    pub until_seen: bool,
}

/// Penalties on the reply so far (not the prompt), as OpenAI's API and vLLM define presence and
/// frequency: each token already in the reply loses `presence`, plus `frequency` per occurrence.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Penalties {
    pub presence: f32,
    pub frequency: f32,
    /// The repetition penalty counts only the reply's tokens (Clover), not the whole context.
    pub repetition_reply_only: bool,
    pub ramps: Vec<Ramp>,
}

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub penalties: Penalties,
    /// The reply so far: every token this sampler returned, by id, and their number.
    counts: std::collections::HashMap<u32, u32>,
    produced: usize,
    rng: Rng,
    constraint: Option<Box<crate::structured_output::Constraint>>,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, seed: Option<u64>) -> Sampler {
        assert!(temperature >= 0.0);
        assert!(top_p > 0.0 && top_p <= 1.0);
        assert!(repetition_penalty > 0.0);
        // No seed given: a fresh random one, so repeated requests differ (OpenAI's and llama.cpp's
        // servers behave the same way; numpy's default_rng(None) too). A seed reproduces a reply.
        let seed = seed.unwrap_or_else(random_seed);
        Sampler {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            penalties: Penalties::default(),
            counts: Default::default(),
            produced: 0,
            rng: Rng::new(seed),
            constraint: None,
        }
    }

    pub fn with_penalties(mut self, penalties: Penalties) -> Sampler {
        self.penalties = penalties;
        self
    }

    pub fn with_constraint(mut self, constraint: crate::structured_output::Constraint) -> Self {
        self.constraint = Some(Box::new(constraint));
        self
    }

    pub fn is_constrained(&self) -> bool {
        self.constraint.is_some()
    }

    pub fn greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// Presence, frequency and the length ramps, in Clover's order (before the repetition penalty).
    fn apply_reply_penalties(&self, logits: &mut [f32]) {
        let p = &self.penalties;
        if p.presence != 0.0 || p.frequency != 0.0 {
            for (&id, &n) in &self.counts {
                if let Some(v) = logits.get_mut(id as usize) {
                    *v -= n as f32 * p.frequency;
                    *v -= p.presence;
                }
            }
        }
    }

    fn apply_ramps(&self, logits: &mut [f32]) {
        // The token being chosen is reply position produced + 1.
        let at = (self.produced + 1) as f32;
        for r in &self.penalties.ramps {
            if r.strength == 0.0 || (r.until_seen && r.ids.iter().any(|id| self.counts.contains_key(id))) {
                continue;
            }
            let bias = ((at - r.start as f32) / r.window.max(1) as f32).clamp(0.0, 1.0) * r.strength;
            if bias == 0.0 {
                continue;
            }
            for &id in &r.ids {
                if let Some(v) = logits.get_mut(id as usize) {
                    *v += bias;
                }
            }
        }
    }

    fn apply_repetition_penalty(&self, logits: &mut [f32], history: &[i64]) {
        if self.repetition_penalty == 1.0 {
            return;
        }
        let mut seen: Vec<i64> = if self.penalties.repetition_reply_only {
            self.counts.keys().map(|&id| i64::from(id)).collect()
        } else {
            history.to_vec()
        };
        if seen.is_empty() {
            return;
        }
        seen.sort_unstable();
        seen.dedup();
        for id in seen {
            if id < 0 || id as usize >= logits.len() {
                continue;
            }
            let v = logits[id as usize];
            // HF CTRL convention: positive logits divided, negative multiplied.
            logits[id as usize] = if v > 0.0 { v / self.repetition_penalty } else { v * self.repetition_penalty };
        }
    }

    fn apply_top_p(&self, logits: &mut [f32]) {
        if self.top_p >= 1.0 {
            return;
        }
        // Only the tokens still in play: top-k's masked ones carry no probability, so leaving them
        // out of the sort changes nothing but its length.
        let mut order: Vec<usize> = (0..logits.len()).filter(|&i| logits[i] > NEG_INF).collect();
        if order.is_empty() {
            return;
        }
        order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap()); // descending
        let sorted: Vec<f32> = order.iter().map(|&i| logits[i]).collect();
        let probs = softmax(&sorted);
        let mut cumsum = 0f32;
        let mut keep = vec![false; order.len()];
        for (rank, &p) in probs.iter().enumerate() {
            keep[rank] = cumsum < self.top_p;
            cumsum += p;
        }
        keep[0] = true; // always keep the top token
        for (rank, &idx) in order.iter().enumerate() {
            if !keep[rank] {
                logits[idx] = NEG_INF;
            }
        }
    }

    /// `logits`: `[VOCAB]` f32. `history`: prior token ids (for repetition
    /// penalty). Returns the next token id, which the sampler counts as the reply's next token.
    pub fn sample(&mut self, logits: &[f32], history: &[i64]) -> usize {
        if self.constraint.is_none() {
            let id = self.choose::<false>(logits, history, &[]);
            *self.counts.entry(id as u32).or_default() += 1;
            self.produced += 1;
            return id;
        }
        self.try_sample(logits, history).expect("sampling failed")
    }

    pub fn try_sample(&mut self, logits: &[f32], history: &[i64]) -> Result<usize, String> {
        if self.constraint.is_none() {
            return Ok(self.sample(logits, history));
        }
        let allowed = self.constraint.as_mut().unwrap().allowed(logits.len());
        if !allowed.iter().any(|v| *v) {
            return Err("structured output has no valid token in the backend vocabulary".into());
        }
        let id = self.choose::<true>(logits, history, &allowed);
        if let Some(c) = &mut self.constraint {
            if !c.advance(id) {
                return Err("structured output rejected sampled token".into());
            }
        }
        *self.counts.entry(id as u32).or_default() += 1;
        self.produced += 1;
        Ok(id)
    }

    fn choose<const MASKED: bool>(
        &mut self,
        logits: &[f32],
        history: &[i64],
        allowed: &[bool],
    ) -> usize {
        // With no transforms, greedy decoding can read the original logits. In particular,
        // adding a dormant constraint must not add a full-vocabulary allocation/copy.
        if !MASKED
            && self.greedy()
            && self.repetition_penalty == 1.0
            && self.penalties.presence == 0.0
            && self.penalties.frequency == 0.0
            && self.penalties.ramps.iter().all(|r| r.strength == 0.0)
        {
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "non-finite logits (interval-3 blowup?)"
            );
            return argmax(logits);
        }
        let mut lg = logits.to_vec();
        assert!(lg.iter().all(|v| v.is_finite()), "non-finite logits (interval-3 blowup?)");

        self.apply_reply_penalties(&mut lg);
        self.apply_repetition_penalty(&mut lg, history);
        self.apply_ramps(&mut lg);
        if MASKED {
            for (v, keep) in lg.iter_mut().zip(allowed) {
                if !keep {
                    *v = f32::NEG_INFINITY;
                }
            }
        }

        if self.greedy() {
            return argmax(&lg);
        }

        for v in lg.iter_mut() {
            *v /= self.temperature;
        }
        // Top-k as the candidate list (in vocabulary order; ties at the k-th kept), so top-p, the
        // softmax and the draw see k tokens instead of the vocabulary (262k vocabulary, top-k 64,
        // top-p 0.95: 1.1 ms a token instead of 4.2, release build, loaded host, 2026-11-04). The
        // draws are those of masking the full vector: masked tokens have probability exactly 0
        // and add nothing to the running sum. Top-p alone still sorts the whole vocabulary.
        let mut idx: Vec<usize> = match self.top_k {
            k if k > 0 && k < lg.len() => {
                let mut scratch = lg.clone();
                let kth = *scratch.select_nth_unstable_by(k - 1, |a, b| b.partial_cmp(a).unwrap()).1;
                (0..lg.len()).filter(|&i| lg[i] >= kth).collect()
            }
            _ => (0..lg.len()).collect(),
        };
        if MASKED {
            idx.retain(|&i| lg[i].is_finite());
        }
        let mut cand: Vec<f32> = idx.iter().map(|&i| lg[i]).collect();
        self.apply_top_p(&mut cand);
        let probs = softmax(&cand);
        let draw = self.rng.next_f64() as f32;
        let mut cumsum = 0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if draw < cumsum {
                return idx[i];
            }
        }
        idx[probs.len() - 1]
    }
}

/// 64 random bits without a dependency: std's RandomState keys come from the OS's random source
/// (once per thread, then stepped for every new state), hashed with the clock for good measure.
fn random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH);
    h.write_u128(now.map_or(0, |d| d.as_nanos()));
    h.finish()
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}

/// Convenience: argmax with no state.
pub fn greedy(logits: &[f32]) -> usize {
    argmax(logits)
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let mx = x.iter().cloned().fold(f32::MIN, f32::max);
    let mut e: Vec<f32> = x.iter().map(|v| (v - mx).exp()).collect();
    let sum: f32 = e.iter().sum();
    for v in e.iter_mut() {
        *v /= sum;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constraints_win_over_logits_penalties_and_sampled_candidate_lists() {
        use crate::structured_output::{Constraint, Vocabulary};
        for (t, k, p) in [(0., 0, 1.), (1., 64, 0.95), (1., 0, 0.95)] {
            let vocab = std::sync::Arc::new(Vocabulary::new(
                (0..256)
                    .map(|b| Some(vec![b as u8]))
                    .chain([None])
                    .collect(),
                [256].into_iter().collect(),
            ));
            let c = Constraint::new(&serde_json::json!({"const":"ok"}), vocab).unwrap();
            let mut sampler = Sampler::new(t, k, p, 1.1, Some(7))
                .with_constraint(c)
                .with_penalties(Penalties {
                    presence: 2.,
                    frequency: 2.,
                    ramps: vec![Ramp {
                        ids: vec![b'x' as u32, 256],
                        strength: 20.,
                        start: 0,
                        window: 1,
                        until_seen: false,
                    }],
                    ..Default::default()
                });
            let mut history = vec![];
            for expected in [
                b'"' as usize,
                b'o' as usize,
                b'k' as usize,
                b'"' as usize,
                256,
            ] {
                let mut logits = vec![-100.; 257];
                logits[b'x' as usize] = 1000.;
                logits[256] = 999.;
                logits[expected] = 900.;
                let got = sampler.try_sample(&logits, &history).unwrap();
                assert_eq!(got, expected);
                history.push(got as i64);
            }
        }
    }

    // Mirrors sampler.py's __main__ self-test.
    fn synthetic_logits(seed: u64, v: usize, peak: usize, peak_val: f32) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut lg: Vec<f32> = (0..v).map(|_| (rng.next_f64() as f32 - 0.5) * 4.0).collect();
        lg[peak] = peak_val;
        lg
    }

    #[test]
    fn greedy_picks_argmax() {
        let lg = synthetic_logits(1, 1000, 42, 100.0);
        assert_eq!(Sampler::new(0.0, 0, 1.0, 1.0, None).sample(&lg, &[]), 42);
        assert_eq!(greedy(&lg), 42);
    }

    #[test]
    fn low_temp_concentrates_on_peak() {
        let lg = synthetic_logits(1, 1000, 42, 100.0);
        let mut s = Sampler::new(0.5, 0, 1.0, 1.0, Some(1));
        for _ in 0..20 {
            assert_eq!(s.sample(&lg, &[]), 42);
        }
    }

    #[test]
    fn seed_reproducible() {
        let flat = synthetic_logits(2, 1000, 0, -999.0); // no dominant peak (peak below the floor)
        let a = Sampler::new(1.0, 0, 1.0, 1.0, Some(7)).sample(&flat, &[]);
        let b = Sampler::new(1.0, 0, 1.0, 1.0, Some(7)).sample(&flat, &[]);
        assert_eq!(a, b);
    }

    #[test]
    fn no_seed_draws_differently_each_time() {
        // 1000 near-flat tokens: two of eight unseeded samplers matching on all four draws by
        // chance has probability about 28 x 1000^-4.
        let flat = synthetic_logits(2, 1000, 0, -999.0);
        let runs: std::collections::HashSet<Vec<usize>> = (0..8)
            .map(|_| {
                let mut s = Sampler::new(1.0, 0, 1.0, 1.0, None);
                (0..4).map(|_| s.sample(&flat, &[])).collect()
            })
            .collect();
        assert_eq!(runs.len(), 8);
    }

    #[test]
    fn top_k_restricts_support() {
        let mut small = vec![-10.0f32; 100];
        let top_ids = [3usize, 17, 50, 88, 91];
        for (i, &t) in top_ids.iter().enumerate() {
            small[t] = 5.0 + i as f32;
        }
        let mut s = Sampler::new(1.0, 3, 1.0, 1.0, Some(2));
        let mut got = std::collections::HashSet::new();
        for _ in 0..500 {
            got.insert(s.sample(&small, &[]));
        }
        assert_eq!(got, [50usize, 88, 91].into_iter().collect());
    }

    #[test]
    fn top_p_keeps_nucleus() {
        let mut peaked = vec![-30.0f32; 100];
        peaked[10] = 10.0; // ~99% of mass
        peaked[11] = 5.0;
        peaked[12] = 4.0;
        let mut s = Sampler::new(1.0, 0, 0.9, 1.0, Some(3));
        for _ in 0..500 {
            assert_eq!(s.sample(&peaked, &[]), 10);
        }
    }

    /// The sampler as it was before the candidate list: every transform on the full vector.
    fn full_vector_sample(s: &Sampler, rng: &mut Rng, logits: &[f32]) -> usize {
        let mut lg: Vec<f32> = logits.iter().map(|v| v / s.temperature).collect();
        if s.top_k > 0 && s.top_k < lg.len() {
            let mut sorted = lg.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let kth = sorted[s.top_k - 1];
            lg.iter_mut().filter(|v| **v < kth).for_each(|v| *v = NEG_INF);
        }
        if s.top_p < 1.0 {
            let mut order: Vec<usize> = (0..lg.len()).collect();
            order.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
            let probs = softmax(&order.iter().map(|&i| lg[i]).collect::<Vec<_>>());
            let mut cum = 0f32;
            for (rank, &i) in order.iter().enumerate() {
                if rank > 0 && cum >= s.top_p {
                    lg[i] = NEG_INF;
                }
                cum += probs[rank];
            }
        }
        let probs = softmax(&lg);
        let draw = rng.next_f64() as f32;
        let mut cum = 0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if draw < cum {
                return i;
            }
        }
        probs.len() - 1
    }

    #[test]
    fn the_candidate_list_draws_what_the_full_vector_draws() {
        for (seed, t, k, p) in [(1u64, 1.0f32, 20usize, 0.95f32), (2, 0.7, 64, 0.8), (3, 1.0, 0, 0.9),
                                (4, 0.1, 50, 1.0), (5, 2.0, 5, 1.0)] {
            let lg = synthetic_logits(seed, 4096, 7, 3.0);
            let mut s = Sampler::new(t, k, p, 1.0, Some(seed));
            let mut rng = Rng::new(seed);
            for _ in 0..300 {
                assert_eq!(s.sample(&lg, &[]), full_vector_sample(&s, &mut rng, &lg), "seed {seed}");
            }
        }
    }

    #[test]
    fn repetition_penalty_demotes_repeated_token() {
        let mut two = vec![0f32; 100];
        two[5] = 2.0;
        two[6] = 2.0; // tie between 5 and 6
        assert_eq!(Sampler::new(0.0, 0, 1.0, 1.5, None).sample(&two, &[5]), 6);
    }

    #[test]
    fn presence_and_frequency_count_the_reply_only() {
        // Token 5 is in the prompt (history) but not the reply: no presence penalty for it.
        let mut lg = vec![0f32; 10];
        lg[5] = 1.0;
        lg[6] = 0.95;
        let mut s = Sampler::new(0.0, 0, 1.0, 1.0, None)
            .with_penalties(Penalties { presence: 0.9, ..Default::default() });
        assert_eq!(s.sample(&lg, &[5, 5, 5]), 5);
        // Now 5 is in the reply: 1.0 - 0.9 < 0.95.
        assert_eq!(s.sample(&lg, &[5, 5, 5, 5]), 6);
        // Frequency: 6 once (-0.2), 5 once (-0.2); 7 at 0.85 wins over 0.8 / 0.75.
        lg[7] = 0.85;
        let mut f = Sampler::new(0.0, 0, 1.0, 1.0, None)
            .with_penalties(Penalties { frequency: 0.2, ..Default::default() });
        f.sample(&[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], &[]);
        f.sample(&[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], &[]);
        assert_eq!(f.sample(&lg, &[]), 7);
    }

    #[test]
    fn repetition_can_count_the_reply_only() {
        let mut two = vec![0f32; 100];
        two[5] = 2.0;
        two[6] = 1.9;
        let mut s = Sampler::new(0.0, 0, 1.0, 1.5, None)
            .with_penalties(Penalties { repetition_reply_only: true, ..Default::default() });
        assert_eq!(s.sample(&two, &[5]), 5); // the prompt's 5 does not count
        assert_eq!(s.sample(&two, &[5, 5]), 6); // the reply's does
    }

    #[test]
    fn ramps_grow_with_the_reply_and_stop_once_seen() {
        // Bias on token 9 from reply position 2, full (+1) at position 4; off once 9 is produced.
        let ramp = Ramp { ids: vec![9], strength: 1.0, start: 2, window: 2, until_seen: true };
        let mut s = Sampler::new(0.0, 0, 1.0, 1.0, None)
            .with_penalties(Penalties { ramps: vec![ramp], ..Default::default() });
        let mut lg = vec![0f32; 10];
        lg[1] = 0.6;
        let picks: Vec<usize> = (0..5).map(|_| s.sample(&lg, &[])).collect();
        // positions 1, 2: bias 0, 0; position 3: +0.5 < 0.6; position 4: +1 > 0.6; then 9 is seen.
        assert_eq!(picks, vec![1, 1, 1, 9, 1]);
    }
}
