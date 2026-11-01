"""MOSS Image's host-side V2 nearest-grid sampling schedule.

The DiT, encoder, and VAE remain NPU operations. This module computes the
small FP32 sigma/timestep arrays and combines FP32 predictions between runs.
It follows MOSSImageScheduler in the pinned pipeline_moss_image.py.
"""

from __future__ import annotations

import math

import numpy as np


def sigmas(steps: int = 50, *, shift: float = 5.0,
           training_shift: float = 1.0, training_steps: int = 1000) -> np.ndarray:
    """Descending snapped sigmas, including the terminal zero, as float32."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if training_steps < 2 or not all(math.isfinite(v) and v > 0
                                     for v in (shift, training_shift)):
        raise ValueError("training_steps and shifts must be positive")

    def grid(count: int, amount: float) -> np.ndarray:
        lower = np.float32(amount / (training_steps + amount - 1))
        values = np.linspace(np.float32(1), lower, count, dtype=np.float32)
        amount32 = np.float32(amount)
        return (amount32 * values /
                (np.float32(1) + (amount32 - np.float32(1)) * values)).astype(np.float32)

    requested = grid(steps, shift)
    training = grid(training_steps, training_shift)[::-1].copy()
    upper = np.searchsorted(training, requested).clip(max=training_steps - 1)
    lower = np.maximum(upper - 1, 0)
    nearest = np.where(abs(requested - training[lower]) <=
                       abs(training[upper] - requested), lower, upper)
    snapped = training[nearest]
    keep = np.concatenate(([True], snapped[1:] != snapped[:-1]))
    return np.concatenate((snapped[keep], np.zeros(1, dtype=np.float32)))


def timestep_features(sigma: float) -> np.ndarray:
    """The transformer's 256-wide cosine/sine input for one sigma."""
    timestep = np.float32(np.float32(sigma) * np.float32(1000))
    frequency = 10000.0 ** (-np.arange(128, dtype=np.float64) / 128)
    phase = float(timestep) * frequency
    return np.concatenate((np.cos(phase), np.sin(phase))).astype(np.float32)


def cfg_euler_step(latents: np.ndarray, conditional: np.ndarray,
                   unconditional: np.ndarray, guidance: float,
                   sigma: float, next_sigma: float) -> np.ndarray:
    """One FP32 CFG and Euler update; branch skipping is in the DiT schedule."""
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance must be finite and nonnegative")
    x = np.asarray(latents, dtype=np.float32)
    c = np.asarray(conditional, dtype=np.float32)
    u = np.asarray(unconditional, dtype=np.float32)
    if x.shape != c.shape or x.shape != u.shape:
        raise ValueError("latent and prediction shapes differ")
    prediction = u + np.float32(guidance) * (c - u)
    return x + np.float32(next_sigma - sigma) * prediction
