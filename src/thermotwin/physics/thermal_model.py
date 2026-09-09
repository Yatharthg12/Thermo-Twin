"""Lumped-capacitance rack thermal balance and RK4 integration."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np


def chain_coupling(n_racks: int, rack_zone: np.ndarray, coefficient_w_per_k: float) -> np.ndarray:
    """Build symmetric within-zone nearest-neighbour conductance matrix in W/K."""
    matrix = np.zeros((n_racks, n_racks), dtype=float)
    for idx in range(n_racks - 1):
        if rack_zone[idx] == rack_zone[idx + 1]:
            matrix[idx, idx + 1] = matrix[idx + 1, idx] = coefficient_w_per_k
    return matrix


def thermal_derivative(
    temperatures_c: np.ndarray,
    it_power_w: np.ndarray,
    disturbance_w: np.ndarray,
    inlet_temperatures_c: np.ndarray,
    conductance_w_per_k: np.ndarray,
    coupling_w_per_k: np.ndarray,
    capacitance_j_per_k: np.ndarray,
) -> np.ndarray:
    """Compute dT/dt in K/s; coupling is conservative when its matrix is symmetric."""
    delta = temperatures_c[None, :] - temperatures_c[:, None]
    coupled_w = np.sum(coupling_w_per_k * delta, axis=1)
    extraction_w = conductance_w_per_k * (temperatures_c - inlet_temperatures_c)
    return (it_power_w + disturbance_w - extraction_w + coupled_w) / capacitance_j_per_k


def rk4_step(state: np.ndarray, dt_s: float, derivative: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Advance one stable fixed-step fourth-order Runge-Kutta update."""
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt_s * k1)
    k3 = derivative(state + 0.5 * dt_s * k2)
    k4 = derivative(state + dt_s * k3)
    result = state + dt_s * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    if not np.all(np.isfinite(result)) or np.max(np.abs(result)) > 200:
        raise FloatingPointError("thermal integration produced a non-finite or unstable state")
    return result

