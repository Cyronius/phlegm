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

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, seed: Option<u64>) -> Sampler {
        assert!(temperature >= 0.0);
        assert!(top_p > 0.0 && top_p <= 1.0);
        assert!(repetition_penalty > 0.0);
        // No seed given: mix in a nondeterministic-ish default (matches numpy's
        // default_rng(None) in spirit, not in value).
        let seed = seed.unwrap_or(0x2545F4914F6CDD1D);
        Sampler { temperature, top_k, top_p, repetition_penalty, rng: Rng::new(seed) }
    }

    pub fn greedy(&self) -> bool {
        self.temperature == 0.0
    }

    fn apply_repetition_penalty(&self, logits: &mut [f32], history: &[i64]) {
        if self.repetition_penalty == 1.0 || history.is_empty() {
            return;
        }
        let mut seen: Vec<i64> = history.to_vec();
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

    fn apply_top_k(&self, logits: &mut [f32]) {
        let k = self.top_k;
        if k == 0 || k >= logits.len() {
            return;
        }
        let mut sorted = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let kth = sorted[k - 1];
        for v in logits.iter_mut() {
            if *v < kth {
                *v = NEG_INF;
            }
        }
    }

    fn apply_top_p(&self, logits: &mut [f32]) {
        if self.top_p >= 1.0 {
            return;
        }
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap()); // descending
        let sorted: Vec<f32> = order.iter().map(|&i| logits[i]).collect();
        let probs = softmax(&sorted);
        let mut cumsum = 0f32;
        let mut keep = vec![false; logits.len()];
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
    /// penalty). Returns the next token id.
    pub fn sample(&mut self, logits: &[f32], history: &[i64]) -> usize {
        let mut lg = logits.to_vec();
        assert!(lg.iter().all(|v| v.is_finite()), "non-finite logits (interval-3 blowup?)");

        self.apply_repetition_penalty(&mut lg, history);

        if self.greedy() {
            return argmax(&lg);
        }

        for v in lg.iter_mut() {
            *v /= self.temperature;
        }
        self.apply_top_k(&mut lg);
        self.apply_top_p(&mut lg);
        let probs = softmax(&lg);
        let draw = self.rng.next_f64() as f32;
        let mut cumsum = 0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if draw < cumsum {
                return i;
            }
        }
        probs.len() - 1
    }
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

    #[test]
    fn repetition_penalty_demotes_repeated_token() {
        let mut two = vec![0f32; 100];
        two[5] = 2.0;
        two[6] = 2.0; // tie between 5 and 6
        assert_eq!(Sampler::new(0.0, 0, 1.0, 1.5, None).sample(&two, &[5]), 6);
    }
}
