"""Logit sampler for the open NPU engine.

Consumes the FULL logits vector the generate loop computes on CPU from the
NPU's final hidden state (shape [vocab] f32, vocab = 248320 for Qwen3.6) and
returns the next token id. Supports greedy, temperature, top-k, top-p
(nucleus), and repetition penalty. Seedable and deterministic given a seed.

The transform order matches the common HF `generate` pipeline:
  1. repetition penalty (on already-produced tokens)
  2. temperature scale
  3. top-k filter
  4. top-p (nucleus) filter
  5. softmax -> multinomial draw   (temperature == 0 -> argmax, no draw)

    from sampler import Sampler
    s = Sampler(temperature=0.7, top_k=20, top_p=0.8,
                repetition_penalty=1.05, seed=0)
    nxt = s.sample(logits, history)   # history = ids seen so far (for rep pen)
"""
import numpy as np

NEG_INF = -1e30


class Sampler:
    def __init__(self, temperature=1.0, top_k=0, top_p=1.0,
                 repetition_penalty=1.0, seed=None):
        assert temperature >= 0.0
        assert top_k >= 0
        assert 0.0 < top_p <= 1.0
        assert repetition_penalty > 0.0
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.rng = np.random.default_rng(seed)

    @property
    def greedy(self):
        return self.temperature == 0.0

    # ---- individual transforms (operate on a f32 copy) -----------------
    def _apply_repetition_penalty(self, logits, history):
        if self.repetition_penalty == 1.0 or not len(history):
            return logits
        idx = np.unique(np.asarray(list(history), dtype=np.int64))
        idx = idx[(idx >= 0) & (idx < logits.shape[0])]
        vals = logits[idx]
        # HF CTRL convention: positive logits divided, negative multiplied.
        pos = vals > 0
        vals[pos] /= self.repetition_penalty
        vals[~pos] *= self.repetition_penalty
        logits[idx] = vals
        return logits

    def _apply_top_k(self, logits):
        k = self.top_k
        if k <= 0 or k >= logits.shape[0]:
            return logits
        # Keep the k largest; mask the rest.
        kth = np.partition(logits, -k)[-k]
        logits[logits < kth] = NEG_INF
        return logits

    def _apply_top_p(self, logits):
        if self.top_p >= 1.0:
            return logits
        order = np.argsort(logits)[::-1]          # descending
        sorted_logits = logits[order]
        probs = _softmax(sorted_logits)
        cumsum = np.cumsum(probs)
        # Keep the smallest prefix whose cumulative prob >= top_p (>=1 token).
        keep = cumsum < self.top_p
        keep[0] = True                            # always keep the top token
        remove = order[~keep]
        logits[remove] = NEG_INF
        return logits

    # ---- public API -----------------------------------------------------
    def sample(self, logits, history=()):
        """logits: 1-D array [vocab] f32. history: iterable of prior token ids
        (for repetition penalty). Returns an int token id."""
        logits = np.asarray(logits, dtype=np.float32).copy()
        assert logits.ndim == 1, logits.shape
        assert np.isfinite(logits).all(), "non-finite logits (interval-3 blowup?)"

        logits = self._apply_repetition_penalty(logits, history)

        if self.greedy:
            return int(np.argmax(logits))

        logits = logits / self.temperature
        logits = self._apply_top_k(logits)
        logits = self._apply_top_p(logits)
        probs = _softmax(logits)
        return int(self.rng.choice(probs.shape[0], p=probs))


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def greedy(logits):
    """Convenience: argmax with no state."""
    return int(np.argmax(np.asarray(logits)))


if __name__ == "__main__":
    # Self-test: NO NPU needed. Synthetic logits with a known structure.
    rng = np.random.default_rng(0)
    V = 248320

    # A logit vector with an obvious winner at index 42.
    lg = rng.standard_normal(V).astype(np.float32)
    lg[42] = 100.0

    # 1) greedy picks the max.
    assert Sampler(temperature=0.0).sample(lg) == 42
    assert greedy(lg) == 42
    print("[OK ] greedy picks argmax")

    # 2) temperature sampling with a dominant logit still lands on it.
    s = Sampler(temperature=0.5, seed=1)
    picks = [s.sample(lg) for _ in range(20)]
    assert all(p == 42 for p in picks), set(picks)
    print("[OK ] low-temp sampling concentrates on the peak")

    # 3) seedable / deterministic: same seed -> same draw sequence.
    flat = rng.standard_normal(V).astype(np.float32)      # near-uniform
    a = [Sampler(temperature=1.0, seed=7).sample(flat) for _ in range(1)]
    b = [Sampler(temperature=1.0, seed=7).sample(flat) for _ in range(1)]
    assert a == b, (a, b)
    c = Sampler(temperature=1.0, seed=8).sample(flat)
    print(f"[OK ] seed reproducible (seed7 -> {a[0]}, seed8 -> {c})")

    # 4) top-k restricts the support to k tokens.
    small = np.full(100, -10.0, dtype=np.float32)
    top_ids = [3, 17, 50, 88, 91]
    for i, t in enumerate(top_ids):
        small[t] = 5.0 + i               # all clearly above the floor
    s = Sampler(temperature=1.0, top_k=3, seed=2)
    got = {s.sample(small) for _ in range(500)}
    # top-3 by logit are 91(9), 88(8), 50(7)
    assert got <= {50, 88, 91}, got
    assert got == {50, 88, 91}, got
    print("[OK ] top-k=3 restricts support to the 3 highest logits:", sorted(got))

    # 5) top-p keeps only the nucleus.
    peaked = np.full(100, -30.0, dtype=np.float32)
    peaked[10] = 10.0     # ~99% of mass
    peaked[11] = 5.0
    peaked[12] = 4.0
    s = Sampler(temperature=1.0, top_p=0.9, seed=3)
    got = {s.sample(peaked) for _ in range(500)}
    assert got == {10}, got
    print("[OK ] top-p=0.9 keeps only the dominant token:", got)

    # 6) repetition penalty pushes probability off already-seen tokens.
    two = np.zeros(100, dtype=np.float32)
    two[5] = 2.0
    two[6] = 2.0          # tie between 5 and 6
    # penalize 5 -> 6 should win greedily
    assert Sampler(temperature=0.0, repetition_penalty=1.5).sample(two, history=[5]) == 6
    print("[OK ] repetition penalty demotes a repeated token")

    print("sampler.py self-test PASSED")
