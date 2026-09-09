"""Noisy/missing sensors and bounded causal temperature state estimation."""

from __future__ import annotations

import numpy as np


def observe(temperatures_c: np.ndarray, rng: np.random.Generator, noise_std_c: float, missing_probability: float) -> tuple[np.ndarray, np.ndarray]:
    """Return noisy readings and a validity mask; missing values are NaN, never zero."""
    values = temperatures_c + rng.normal(0.0, noise_std_c, len(temperatures_c))
    valid = rng.random(len(temperatures_c)) >= missing_probability
    values[~valid] = np.nan
    return values, valid


def correct_estimate(predicted_c: np.ndarray, observed_c: np.ndarray, valid: np.ndarray, gain: float) -> np.ndarray:
    """Blend current valid readings into a physics prediction with bounded correction."""
    result = predicted_c.copy()
    innovation = np.clip(observed_c[valid] - result[valid], -3.0, 3.0)
    result[valid] += np.clip(gain, 0.0, 1.0) * innovation
    return result

