"""Stateful hidden simulated plant with mismatched physics and independent RNG streams."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import numpy as np

from thermotwin.physics.cooling_model import advance_actuator, capacity_limited_conductance, cooling_electrical_power, inlet_temperatures
from thermotwin.physics.observation import correct_estimate, observe
from thermotwin.physics.thermal_model import chain_coupling, rk4_step, thermal_derivative
from thermotwin.physics.workload import allocate
from thermotwin.types import Action, Snapshot


@dataclass
class HiddenPlantState:
    """Evaluation-only truth and mismatch values that policy code never receives."""

    temperatures_c: np.ndarray
    degradation: np.ndarray
    disturbance_w: np.ndarray


class DataCenterPlant:
    """Advance simulated truth while exposing only noisy policy-visible snapshots."""

    def __init__(self, config: dict[str, Any], seed: int | None = None, matched: bool = False):
        self.config = copy.deepcopy(config)
        topo, cool, phys = config["topology"], config["cooling"], config["physics"]
        self.n_racks, self.n_zones = int(topo["racks"]), int(topo["zones"])
        self.rack_zone = np.asarray(topo["rack_zone"], dtype=int)
        seed = int(seed if seed is not None else config["project"]["seed"])
        sequence = np.random.SeedSequence(seed)
        process_seed, observation_seed = sequence.spawn(2)
        self.process_rng = np.random.default_rng(process_seed)
        self.observation_rng = np.random.default_rng(observation_seed)
        mismatch = 0.0 if matched else float(phys["mismatch_fraction"])
        degradation = np.clip(1.0 + self.process_rng.normal(0, mismatch / 2, self.n_zones), 0.68, 1.15)
        self.hidden = HiddenPlantState(
            temperatures_c=23.5 + self.process_rng.normal(0, 0.25, self.n_racks),
            degradation=degradation,
            disturbance_w=np.zeros(self.n_racks),
        )
        self.time_s = 0.0
        self.supply_actual_c = np.full(self.n_zones, cool["supply_initial_c"], dtype=float)
        self.supply_command_c = self.supply_actual_c.copy()
        self.airflow_actual = np.full(self.n_zones, cool["airflow_initial"], dtype=float)
        self.airflow_command = self.airflow_actual.copy()
        self.offered_utilization = np.full(self.n_racks, 0.45)
        self.placement_offset = np.zeros(self.n_racks)
        self.utilization, self.unserved = allocate(self.offered_utilization, self.placement_offset)
        self.migration_cooldowns = np.zeros(self.n_racks, dtype=int)
        self.estimated_c = self.hidden.temperatures_c.copy()
        self._last_inlet = inlet_temperatures(self.hidden.temperatures_c, self.supply_actual_c, self.rack_zone, phys["recirculation_fraction"])
        self._last_it_power = self.it_power(self.utilization)
        self._last_cooling_w = 0.0
        self._last_fan_w = 0.0
        self._last_observed, self._last_valid = observe(self.hidden.temperatures_c, self.observation_rng, config["sensors"]["noise_std_c"], config["sensors"]["missing_probability"])
        self.history: list[Snapshot] = [self.snapshot()]
        self.truth_history: list[np.ndarray] = [self.hidden.temperatures_c.copy()]

    def it_power(self, utilization: np.ndarray) -> np.ndarray:
        """Map served capacity fraction to rack IT electrical power in watts."""
        phys = self.config["physics"]
        return phys["idle_power_w"] + phys["dynamic_power_w"] * np.clip(utilization, 0.0, 1.0) ** 1.12

    def apply_action(self, action: Action) -> tuple[bool, str]:
        """Apply validated commands and persistent conservative migration to the plant."""
        cool, work = self.config["cooling"], self.config["workload"]
        if len(action.supply_delta_c) != self.n_zones or len(action.airflow_delta) != self.n_zones:
            return False, "dimension_mismatch"
        supply = self.supply_command_c + np.asarray(action.supply_delta_c)
        airflow = self.airflow_command + np.asarray(action.airflow_delta)
        if not np.all(np.isfinite(supply)) or not np.all(np.isfinite(airflow)):
            return False, "nonfinite_command"
        if np.any(supply < cool["supply_min_c"]) or np.any(supply > cool["supply_max_c"]):
            return False, "supply_bounds"
        if np.any(airflow < cool["airflow_min"]) or np.any(airflow > cool["airflow_max"]):
            return False, "airflow_bounds"
        if action.migration is not None:
            source, destination, amount = action.migration
            if source == destination or not (0 <= source < self.n_racks and 0 <= destination < self.n_racks):
                return False, "migration_rack"
            if not np.isfinite(amount) or amount <= 0 or amount > work["migration_limit"]:
                return False, "migration_rate"
            if self.migration_cooldowns[source] > 0 or self.migration_cooldowns[destination] > 0:
                return False, "migration_cooldown"
            current, _ = allocate(self.offered_utilization, self.placement_offset)
            if current[source] < amount or current[destination] + amount > work["rack_capacity"]:
                return False, "migration_capacity"
            self.placement_offset[source] -= amount
            self.placement_offset[destination] += amount
            cooldown = int(work["migration_cooldown_steps"])
            self.migration_cooldowns[[source, destination]] = cooldown
        self.supply_command_c = supply
        self.airflow_command = airflow
        return True, "ok"

    def advance(
        self,
        offered_utilization: np.ndarray,
        ambient_c: float,
        humidity_pct: float,
        action: Action | None = None,
        degradation: float | None = None,
    ) -> Snapshot:
        """Commit one observation interval using current exogenous inputs and optional action."""
        if action is not None:
            accepted, reason = self.apply_action(action)
            if not accepted:
                raise ValueError(f"infeasible action: {reason}")
        self.offered_utilization = np.asarray(offered_utilization, dtype=float).copy()
        if self.offered_utilization.shape != (self.n_racks,) or not np.all(np.isfinite(self.offered_utilization)):
            raise ValueError("offered utilization must be a finite rack vector")
        self.utilization, self.unserved = allocate(self.offered_utilization, self.placement_offset)
        self.migration_cooldowns = np.maximum(self.migration_cooldowns - 1, 0)
        cfg, cool, phys = self.config, self.config["cooling"], self.config["physics"]
        total_dt = float(phys["observation_interval_s"])
        internal_dt = float(phys["integration_step_s"])
        steps = int(np.ceil(total_dt / internal_dt))
        dt = total_dt / steps
        self.hidden.disturbance_w = 0.82 * self.hidden.disturbance_w + self.process_rng.normal(0, phys["process_noise_w"], self.n_racks)
        if degradation is not None:
            self.hidden.degradation[:] = float(degradation)
        it_power = self.it_power(self.utilization)
        migration_w = cfg["workload"]["migration_overhead_w"] if action and action.migration else 0.0
        if migration_w:
            it_power = it_power + migration_w / self.n_racks
        extracted = np.zeros(self.n_zones)
        estimate_state = self.estimated_c.copy()
        estimate_inlet = self._last_inlet.copy()
        for _ in range(steps):
            self.supply_actual_c = advance_actuator(self.supply_actual_c, self.supply_command_c, cool["supply_tau_s"], cool["supply_slew_c_per_min"] / 60.0, dt)
            self.airflow_actual = advance_actuator(self.airflow_actual, self.airflow_command, cool["airflow_tau_s"], cool["airflow_slew_per_min"] / 60.0, dt)
            inlet = inlet_temperatures(self.hidden.temperatures_c, self.supply_actual_c, self.rack_zone, phys["recirculation_fraction"])
            conductance, extracted = capacity_limited_conductance(
                self.hidden.temperatures_c, inlet, phys["conductance_w_per_k"],
                self.airflow_actual, self.rack_zone,
                cool["capacity_w_per_zone"] * self.hidden.degradation,
                phys["airflow_exponent"],
            )
            capacitance = np.full(self.n_racks, phys["capacitance_j_per_k"] * (1.0 + (self.hidden.degradation[self.rack_zone] - 1.0) * 0.4))
            coupling = chain_coupling(self.n_racks, self.rack_zone, phys["coupling_w_per_k"])
            derivative = lambda temp: thermal_derivative(temp, it_power, self.hidden.disturbance_w, inlet, conductance, coupling, capacitance)
            self.hidden.temperatures_c = rk4_step(self.hidden.temperatures_c, dt, derivative)
            estimate_inlet = inlet_temperatures(estimate_state, self.supply_actual_c, self.rack_zone, phys["recirculation_fraction"])
            estimate_conductance, _ = capacity_limited_conductance(
                estimate_state, estimate_inlet, phys["conductance_w_per_k"], self.airflow_actual,
                self.rack_zone, cool["capacity_w_per_zone"], phys["airflow_exponent"],
            )
            nominal_capacitance = np.full(self.n_racks, phys["capacitance_j_per_k"])
            estimate_derivative = lambda temp: thermal_derivative(temp, it_power, np.zeros(self.n_racks), estimate_inlet, estimate_conductance, coupling, nominal_capacitance)
            estimate_state = rk4_step(estimate_state, dt, estimate_derivative)
        compressor, fan, _ = cooling_electrical_power(extracted, self.airflow_actual, self.supply_actual_c, ambient_c, cool, self.hidden.degradation)
        observed, valid = observe(self.hidden.temperatures_c, self.observation_rng, cfg["sensors"]["noise_std_c"], cfg["sensors"]["missing_probability"])
        self.estimated_c = correct_estimate(estimate_state, observed, valid, cfg["sensors"]["estimator_gain"])
        self.time_s += total_dt
        self._last_inlet, self._last_it_power = estimate_inlet.copy(), it_power.copy()
        self._last_cooling_w, self._last_fan_w = compressor, fan
        self._last_observed, self._last_valid = observed.copy(), valid.copy()
        snap = self.snapshot(ambient_c, humidity_pct)
        self.history.append(snap)
        self.truth_history.append(self.hidden.temperatures_c.copy())
        return snap

    def snapshot(self, ambient_c: float = 27.0, humidity_pct: float = 48.0) -> Snapshot:
        """Create a defensive-copy snapshot separating hidden truth from policy-visible estimates."""
        previous = getattr(self, "history", [])
        recent = [item.estimated_temperatures_c for item in previous[-11:]] + [self.estimated_c]
        return Snapshot(
            time_s=self.time_s,
            estimated_temperatures_c=self.estimated_c,
            observed_temperatures_c=self._last_observed,
            sensor_valid=self._last_valid,
            utilization=self.utilization,
            offered_utilization=self.offered_utilization,
            it_power_w=self._last_it_power,
            inlet_temperatures_c=self._last_inlet,
            supply_actual_c=self.supply_actual_c,
            supply_command_c=self.supply_command_c,
            airflow_actual=self.airflow_actual,
            airflow_command=self.airflow_command,
            ambient_c=float(ambient_c),
            humidity_pct=float(humidity_pct),
            cooling_power_w=self._last_cooling_w,
            fan_power_w=self._last_fan_w,
            unserved_utilization=self.unserved,
            migration_cooldowns=self.migration_cooldowns,
            recent_estimated_history_c=np.asarray(recent),
        )

    def clone(self) -> "DataCenterPlant":
        """Deep-copy state and RNGs for deterministic isolated diagnostics."""
        return copy.deepcopy(self)
