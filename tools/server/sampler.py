"""Sampler interface + a greedy default with a real temperature/top-p hook.

The backend calls `sampler.sample(logits)` where logits is a 1-D float array over
the vocab and gets back a token id. Greedy (temperature <= 0) needs no numpy; the
temperature/top-p path uses numpy if present and otherwise falls back to greedy so
the mock backend has zero hard deps.
"""
from __future__ import annotations
from typing import Sequence
import math
import random


class Sampler:
    def __init__(self, temperature: float = 0.0, top_p: float = 1.0, seed: int | None = None):
        self.temperature = max(0.0, float(temperature))
        self.top_p = float(top_p)
        self._rng = random.Random(seed)

    def sample(self, logits: Sequence[float]) -> int:
        if self.temperature <= 0.0 or self.top_p <= 0.0:
            return self._greedy(logits)
        return self._temp_top_p(logits)

    @staticmethod
    def _greedy(logits: Sequence[float]) -> int:
        best_i, best_v = 0, float("-inf")
        for i, v in enumerate(logits):
            if v > best_v:
                best_v, best_i = v, i
        return best_i

    def _temp_top_p(self, logits: Sequence[float]) -> int:
        # softmax(logits / T), then nucleus filter, then sample.
        try:
            import numpy as np
            lg = np.asarray(logits, dtype=np.float64) / self.temperature
            lg -= lg.max()
            p = np.exp(lg)
            p /= p.sum()
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            keep = cum <= self.top_p
            keep[0] = True  # always keep the top token
            idx = order[keep]
            probs = p[idx]
            probs /= probs.sum()
            r = self._rng.random()
            c = 0.0
            for tok, pr in zip(idx.tolist(), probs.tolist()):
                c += pr
                if r <= c:
                    return int(tok)
            return int(idx[-1])
        except ImportError:
            # pure-python softmax sampling fallback
            m = max(logits)
            exps = [math.exp((v - m) / self.temperature) for v in logits]
            s = sum(exps)
            r = self._rng.random() * s
            c = 0.0
            for i, e in enumerate(exps):
                c += e
                if r <= c:
                    return i
            return len(logits) - 1


def sampler_from_request(temperature: float | None, top_p: float | None, seed: int | None) -> Sampler:
    return Sampler(
        temperature=0.0 if temperature is None else temperature,
        top_p=1.0 if top_p is None else top_p,
        seed=seed,
    )
