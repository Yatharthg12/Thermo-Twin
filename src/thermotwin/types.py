"""Shared immutable state, action, rollout, and decision records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _array(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Snapshot:
    """Policy-visible state; simulated truth is deliberately absent from this record."""

    time_s: float
    estimated_temperatures_c: np.ndarray
    observed_temperatures_c: np.ndarray
    sensor_valid: np.ndarray
    utilization: np.ndarray
    offered_utilization: np.ndarray
    it_power_w: np.ndarray
    inlet_temperatures_c: np.ndarray
    supply_actual_c: np.ndarray
    supply_command_c: np.ndarray
    airflow_actual: np.ndarray
    airflow_command: np.ndarray
    ambient_c: float
    humidity_pct: float
    cooling_power_w: float
    fan_power_w: float
    unserved_utilization: float = 0.0
    migration_cooldowns: np.ndarray = field(default_factory=lambda: np.empty(0))
    recent_estimated_history_c: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def __post_init__(self) -> None:
        for name in (
            "estimated_temperatures_c", "observed_temperatures_c",
            "sensor_valid", "utilization", "offered_utilization", "it_power_w",
            "inlet_temperatures_c", "supply_actual_c", "supply_command_c",
            "airflow_actual", "airflow_command",
            "migration_cooldowns",
            "recent_estimated_history_c",
        ):
            object.__setattr__(self, name, _array(getattr(self, name)))

    def as_dict(self, include_truth: bool = False) -> dict[str, Any]:
        """Serialize bounded public state; ``include_truth`` is retained for API compatibility but ignored."""
        data = {
            "time_s": self.time_s,
            "estimated_temperatures_c": self.estimated_temperatures_c.tolist(),
            "observed_temperatures_c": [None if not ok else float(v) for v, ok in zip(self.observed_temperatures_c, self.sensor_valid)],
            "sensor_valid": self.sensor_valid.astype(bool).tolist(),
            "utilization": self.utilization.tolist(),
            "offered_utilization": self.offered_utilization.tolist(),
            "it_power_w": self.it_power_w.tolist(),
            "inlet_temperatures_c": self.inlet_temperatures_c.tolist(),
            "supply_actual_c": self.supply_actual_c.tolist(),
            "supply_command_c": self.supply_command_c.tolist(),
            "airflow_actual": self.airflow_actual.tolist(),
            "airflow_command": self.airflow_command.tolist(),
            "ambient_c": self.ambient_c,
            "humidity_pct": self.humidity_pct,
            "cooling_power_w": self.cooling_power_w,
            "fan_power_w": self.fan_power_w,
            "unserved_utilization": self.unserved_utilization,
            "migration_cooldowns": self.migration_cooldowns.astype(int).tolist(),
            "recent_estimated_history_c": self.recent_estimated_history_c.tolist(),
        }
        return data


@dataclass(frozen=True)
class Action:
    """Bounded controller command deltas and one optional workload transfer."""

    action_id: str
    supply_delta_c: tuple[float, ...]
    airflow_delta: tuple[float, ...]
    migration: tuple[int, int, float] | None = None
    label: str = ""

    @classmethod
    def noop(cls, zones: int) -> "Action":
        return cls("noop", (0.0,) * zones, (0.0,) * zones, None, "Hold settings")


@dataclass
class RolloutResult:
    """Pure projected trajectory and constraint/energy summaries."""

    times_s: np.ndarray
    temperatures_c: np.ndarray
    inlet_temperatures_c: np.ndarray
    total_power_w: np.ndarray
    energy_kwh: float
    final_snapshot: Snapshot
    feasible: bool
    max_temperature_c: float
    hotspot_rack_minutes: float
    hard_limit_rack_minutes: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRecord:
    """One controller choice with candidate rationale and measured latency."""

    time_s: float
    controller: str
    chosen_action: Action
    candidates: list[dict[str, Any]]
    fallback: bool
    explanation: str
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "controller": self.controller,
            "chosen_action": {
                "action_id": self.chosen_action.action_id,
                "label": self.chosen_action.label,
                "supply_delta_c": list(self.chosen_action.supply_delta_c),
                "airflow_delta": list(self.chosen_action.airflow_delta),
                "migration": self.chosen_action.migration,
            },
            "candidates": self.candidates,
            "fallback": self.fallback,
            "explanation": self.explanation,
            "latency_ms": self.latency_ms,
        }
