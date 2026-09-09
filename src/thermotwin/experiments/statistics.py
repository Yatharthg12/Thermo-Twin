"""Replication-level paired bootstrap intervals that preserve temporal dependence."""

from __future__ import annotations

import numpy as np


def paired_bootstrap_interval(baseline: np.ndarray, candidate: np.ndarray, seed: int = 0, draws: int = 2000) -> dict:
    """Bootstrap independent paired run differences; refuse inference from fewer than three runs."""
    baseline, candidate = np.asarray(baseline, float), np.asarray(candidate, float)
    if baseline.shape != candidate.shape:
        raise ValueError("paired outcomes must have equal shape")
    if len(baseline) < 3:
        return {"mean_difference": float(np.mean(candidate - baseline)) if len(baseline) else None,
                "ci95": None, "reason": "fewer than three independent paired replications"}
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(differences, size=(draws, len(differences)), replace=True), axis=1)
    return {"mean_difference": float(np.mean(differences)), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))], "reason": None}

