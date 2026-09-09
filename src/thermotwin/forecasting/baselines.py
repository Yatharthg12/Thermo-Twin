"""Persistence and nominal physics forecast baselines."""

from __future__ import annotations

import numpy as np


def persistence_predict(X: np.ndarray, n_horizons: int) -> np.ndarray:
    """Repeat the causal origin temperature at every forecast horizon."""
    return np.repeat(X[:, [0]], n_horizons, axis=1)


def physics_predict(nominal: np.ndarray) -> np.ndarray:
    """Return the precomputed transparent nominal thermal forecast."""
    return np.asarray(nominal).copy()

