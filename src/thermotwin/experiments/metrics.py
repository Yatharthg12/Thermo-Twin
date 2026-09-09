"""Energy-integrated and duration-aware thermal/control outcome metrics."""

from __future__ import annotations

from typing import Any
import numpy as np

from thermotwin.types import Snapshot


def integrate_energy_kwh(times_s: np.ndarray, power_w: np.ndarray) -> float:
    """Integrate interval-ending power samples over elapsed simulated seconds."""
    times, power = np.asarray(times_s, float), np.asarray(power_w, float)
    if len(times) < 2 or len(power) != len(times):
        return 0.0
    dt = np.diff(times)
    if np.any(dt < 0) or not np.all(np.isfinite(power)):
        raise ValueError("energy integration requires sorted times and finite power")
    return float(np.sum(power[1:] * dt) / 3_600_000.0)


def hotspot_events(times_s: np.ndarray, temperatures_c: np.ndarray, threshold_c: float) -> tuple[int, int]:
    """Count contiguous per-rack exceedance intervals and those active at recording start."""
    above = np.asarray(temperatures_c) > threshold_c
    if above.ndim == 1:
        above = above[:, None]
    starts = above.copy()
    starts[1:] &= ~above[:-1]
    boundary_active = int(np.sum(above[0])) if len(above) else 0
    return int(np.sum(starts)), boundary_active


def duration_rack_minutes(times_s: np.ndarray, temperatures_c: np.ndarray, threshold_c: float) -> float:
    """Use interval-ending rectangular weighting for irregularly sampled rack temperatures."""
    times, temps = np.asarray(times_s, float), np.asarray(temperatures_c, float)
    if len(times) < 2:
        return 0.0
    return float(np.sum((temps[1:] > threshold_c) * np.diff(times)[:, None]) / 60.0)


def pue_approximation(it_energy_kwh: float, cooling_energy_kwh: float) -> dict[str, Any]:
    """Return cooling-only-overhead PUE or an explicit unavailable reason."""
    if it_energy_kwh <= 0:
        return {"value": None, "reason": "IT energy denominator is zero"}
    return {"value": float((it_energy_kwh + cooling_energy_kwh) / it_energy_kwh), "reason": None}


def summarize_snapshots(history: list[Snapshot], hotspot_c: float, hard_limit_c: float, migrations: int = 0, decision_latencies_ms: list[float] | None = None, truth_history: list[np.ndarray] | None = None) -> dict[str, Any]:
    """Summarize thermal, energy, service, and controller outcomes from a closed-loop trace."""
    times = np.asarray([item.time_s for item in history])
    temps = np.asarray(truth_history if truth_history is not None else [item.estimated_temperatures_c for item in history])
    it_power = np.asarray([np.sum(item.it_power_w) for item in history])
    cooling_power = np.asarray([item.cooling_power_w + item.fan_power_w for item in history])
    it_energy = integrate_energy_kwh(times, it_power)
    cooling_energy = integrate_energy_kwh(times, cooling_power)
    events, boundary = hotspot_events(times, temps, hotspot_c)
    total_rack_minutes = max(float((times[-1] - times[0]) / 60 * temps.shape[1]), 0.0) if len(times) else 0.0
    hard_minutes = duration_rack_minutes(times, temps, hard_limit_c)
    degree_minutes = float(np.sum(np.maximum(temps[1:] - hotspot_c, 0.0) * np.diff(times)[:, None]) / 60) if len(times) > 1 else 0.0
    offered = sum(float(np.sum(item.offered_utilization)) for item in history[1:])
    served = sum(float(np.sum(item.utilization)) for item in history[1:])
    pue = pue_approximation(it_energy, cooling_energy)
    return {
        "duration_min": float((times[-1] - times[0]) / 60) if len(times) else 0.0,
        "it_energy_kwh": it_energy, "cooling_energy_kwh": cooling_energy,
        "total_energy_kwh": it_energy + cooling_energy,
        "pue_cooling_only": pue["value"], "pue_unavailable_reason": pue["reason"],
        "hotspot_events": events, "boundary_active_hotspot_events": boundary,
        "hotspot_rack_minutes": duration_rack_minutes(times, temps, hotspot_c),
        "hard_limit_rack_minutes": hard_minutes,
        "hard_limit_time_fraction": hard_minutes / total_rack_minutes if total_rack_minutes else None,
        "hotspot_degree_minutes": degree_minutes, "peak_temperature_c": float(np.max(temps)) if temps.size else None,
        "mean_thermal_imbalance_c": float(np.mean(np.std(temps, axis=1))) if temps.size else None,
        "offered_utilization_steps": offered, "served_utilization_steps": served,
        "service_fraction": served / offered if offered else None,
        "migration_count": int(migrations),
        "mean_decision_latency_ms": float(np.mean(decision_latencies_ms)) if decision_latencies_ms else 0.0,
    }


def relative_savings(value: float, baseline: float, baseline_name: str) -> dict[str, Any]:
    """Compute guarded relative reduction against an explicitly named baseline."""
    if baseline == 0:
        return {"value": None, "baseline": baseline_name, "reason": "baseline denominator is zero"}
    return {"value": float((baseline - value) / baseline), "baseline": baseline_name, "reason": None}
