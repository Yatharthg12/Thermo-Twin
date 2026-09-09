"""Cooling actuator dynamics, finite extraction capacity, COP, and fan power."""

from __future__ import annotations

import numpy as np


def advance_actuator(actual: np.ndarray, command: np.ndarray, tau_s: float, slew_per_s: float, dt_s: float) -> np.ndarray:
    """Apply first-order lag and a hard slew-rate limit without teleporting actuators."""
    desired_rate = (command - actual) / max(tau_s, 1e-9)
    rate = np.clip(desired_rate, -slew_per_s, slew_per_s)
    return actual + rate * dt_s


def inlet_temperatures(
    rack_temperatures_c: np.ndarray,
    supply_c: np.ndarray,
    rack_zone: np.ndarray,
    recirculation_fraction: float,
) -> np.ndarray:
    """Mix zone supply and current zone return using bounded nonnegative weights summing to one."""
    zone_return = np.array([rack_temperatures_c[rack_zone == z].mean() for z in range(len(supply_c))])
    return (1.0 - recirculation_fraction) * supply_c[rack_zone] + recirculation_fraction * zone_return[rack_zone]


def capacity_limited_conductance(
    temperatures_c: np.ndarray,
    inlet_c: np.ndarray,
    base_conductance_w_per_k: float,
    airflow: np.ndarray,
    rack_zone: np.ndarray,
    capacity_w_per_zone: float | np.ndarray,
    airflow_exponent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale conductance when requested sensible heat extraction exceeds zone capacity."""
    conductance = base_conductance_w_per_k * np.power(airflow[rack_zone], airflow_exponent)
    requested = np.maximum(conductance * (temperatures_c - inlet_c), 0.0)
    delivered = np.zeros(len(airflow))
    capacities = np.broadcast_to(np.asarray(capacity_w_per_zone, dtype=float), (len(airflow),))
    for zone in range(len(airflow)):
        indices = rack_zone == zone
        zone_requested = float(requested[indices].sum())
        scale = min(1.0, capacities[zone] / max(zone_requested, 1e-12))
        conductance[indices] *= scale
        delivered[zone] = zone_requested * scale
    return conductance, delivered


def cooling_electrical_power(
    extracted_w_by_zone: np.ndarray,
    airflow: np.ndarray,
    supply_c: np.ndarray,
    ambient_c: float,
    config: dict,
    efficiency_factor: float | np.ndarray = 1.0,
) -> tuple[float, float, np.ndarray]:
    """Return compressor and cubic fan electrical powers using a simplified bounded COP."""
    lift = np.maximum(ambient_c - supply_c, 0.0)
    factor = np.broadcast_to(np.asarray(efficiency_factor, dtype=float), supply_c.shape)
    cop = np.clip((5.6 - 0.075 * lift + 0.055 * (supply_c - 18.0)) * factor, config["cop_min"], config["cop_max"])
    compressor_w = float(np.sum(extracted_w_by_zone / cop))
    normalized = airflow / config["airflow_max"]
    fan_w = float(np.sum(config["fan_power_w_per_zone"] * normalized**3))
    return compressor_w, fan_w, cop
