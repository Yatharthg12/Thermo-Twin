"""Causal synthetic offered-load scenarios and persistent workload allocation."""

from __future__ import annotations

import numpy as np


SCENARIOS = ("normal", "burst", "skewed", "hot_ambient", "cooling_degradation")


def scenario_inputs(step: int, n_racks: int, scenario: str, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """Generate current offered utilization, ambient temperature, and contextual humidity."""
    phase = 2 * np.pi * (step % 1440) / 1440.0
    base = 0.48 + 0.18 * np.sin(phase - 0.8)
    rack_phase = np.linspace(0, 2 * np.pi, n_racks, endpoint=False)
    load = base + 0.08 * np.sin(phase * 3 + rack_phase) + rng.normal(0, 0.018, n_racks)
    if scenario == "burst" and 5 <= step % 240 < 55:
        load += 0.32
    if scenario == "skewed":
        load[: max(1, n_racks // 3)] += 0.28
        load[max(1, n_racks // 3):] -= 0.08
    ambient = 27.0 + 4.0 * np.sin(phase - 1.2)
    if scenario == "hot_ambient":
        ambient += 6.0
    humidity = 48.0 + 10.0 * np.sin(phase + 0.5)
    return np.clip(load, 0.05, 1.18), float(ambient), float(np.clip(humidity, 20, 85))


def allocate(offered: np.ndarray, placement_offset: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Apply persistent zero-sum placement offsets and expose rather than hide unserved demand."""
    shifted = offered.copy() if placement_offset is None else offered + placement_offset
    served = np.clip(shifted, 0.0, 1.0)
    target = min(float(np.sum(np.maximum(offered, 0.0))), float(len(offered)))
    delta = target - float(served.sum())
    if delta > 1e-12:
        headroom = 1.0 - served
        total = float(headroom.sum())
        if total > 0:
            served += min(delta, total) * headroom / total
    elif delta < -1e-12:
        total = float(served.sum())
        if total > 0:
            served -= min(-delta, total) * served / total
    unserved = max(float(np.sum(np.maximum(offered, 0.0))) - float(served.sum()), 0.0)
    return served, unserved
