"""Dimensionless normalized trajectory objective with transparent raw components."""

from __future__ import annotations

import numpy as np

from thermotwin.types import Action, RolloutResult


def score_rollout(result: RolloutResult, action: Action, config: dict) -> tuple[float, dict[str, float]]:
    """Score energy, squared exceedance, hard limit, imbalance, movement, and migration."""
    thresholds, weights, norm = config["thresholds"], config["control"]["weights"], config["control"]["normalization"]
    temp = result.temperatures_c
    hotspot_degree2 = float(np.mean(np.maximum(temp - thresholds["hotspot_c"], 0.0) ** 2)) if temp.size else float("inf")
    hard_degree2 = float(np.mean(np.maximum(temp - thresholds["hard_limit_c"], 0.0) ** 2)) if temp.size else float("inf")
    imbalance_c = float(np.mean(np.std(temp, axis=1))) if temp.size else float("inf")
    movement = float(np.sum(np.abs(action.supply_delta_c)) / 2.0 + np.sum(np.abs(action.airflow_delta)) / 0.2)
    migration = float(action.migration[2]) if action.migration else 0.0
    terms = {
        "energy": result.energy_kwh / norm["energy_kwh"],
        "hotspot": hotspot_degree2 / norm["temperature_c"] ** 2,
        "hard_limit": hard_degree2 / norm["temperature_c"] ** 2,
        "imbalance": imbalance_c / norm["temperature_c"],
        "movement": movement / norm["movement"],
        "migration": migration / max(config["workload"]["migration_limit"], 1e-9),
    }
    return float(sum(weights[key] * terms[key] for key in terms)), terms

