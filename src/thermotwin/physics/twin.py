"""Pure nominal counterfactual rollouts used by predictive controllers."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from thermotwin.physics.plant import DataCenterPlant
from thermotwin.types import Action, RolloutResult, Snapshot


class DigitalTwin:
    """Build deterministic matched-model projections without mutating a live plant."""

    def __init__(self, config: dict[str, Any], residual_predictor: Callable[..., np.ndarray] | None = None):
        self.config = copy.deepcopy(config)
        self.residual_predictor = residual_predictor

    def _from_snapshot(self, snapshot: Snapshot) -> DataCenterPlant:
        plant = DataCenterPlant(self.config, seed=0, matched=True)
        plant.time_s = snapshot.time_s
        plant.hidden.temperatures_c = snapshot.estimated_temperatures_c.copy()
        plant.estimated_c = snapshot.estimated_temperatures_c.copy()
        plant.supply_actual_c = snapshot.supply_actual_c.copy()
        plant.supply_command_c = snapshot.supply_command_c.copy()
        plant.airflow_actual = snapshot.airflow_actual.copy()
        plant.airflow_command = snapshot.airflow_command.copy()
        plant.offered_utilization = snapshot.offered_utilization.copy()
        plant.utilization = snapshot.utilization.copy()
        plant.placement_offset = snapshot.utilization - snapshot.offered_utilization
        if len(snapshot.migration_cooldowns) == plant.n_racks:
            plant.migration_cooldowns = snapshot.migration_cooldowns.astype(int).copy()
        plant.hidden.disturbance_w[:] = 0.0
        plant.history = []
        return plant

    def simulate_action(
        self,
        snapshot: Snapshot,
        action: Action,
        horizon_seconds: float,
        forecast_inputs: dict[str, np.ndarray | float],
        use_residual: bool = False,
    ) -> RolloutResult:
        """Evaluate one hold-first-action rollout with causal caller-supplied exogenous forecasts."""
        step_s = float(self.config["control"]["rollout_step_s"])
        n_steps = max(1, int(np.ceil(horizon_seconds / step_s)))
        load = np.asarray(forecast_inputs.get("utilization", snapshot.offered_utilization), dtype=float)
        if load.ndim == 1:
            load = np.repeat(load[None, :], n_steps, axis=0)
        ambient = np.asarray(forecast_inputs.get("ambient_c", snapshot.ambient_c), dtype=float)
        if ambient.ndim == 0:
            ambient = np.repeat(ambient[None], n_steps)
        plant = self._from_snapshot(snapshot)
        accepted, reason = plant.apply_action(action)
        if not accepted:
            return RolloutResult(np.array([]), np.empty((0, plant.n_racks)), np.empty((0, plant.n_racks)), np.array([]), float("inf"), snapshot, False, float("inf"), float("inf"), float("inf"), {"rejection_reason": reason})
        times, temps, inlets, powers = [], [], [], []
        hot, hard = self.config["thresholds"]["hotspot_c"], self.config["thresholds"]["hard_limit_c"]
        for idx in range(n_steps):
            snap = plant.advance(load[min(idx, len(load) - 1)], float(ambient[min(idx, len(ambient) - 1)]), snapshot.humidity_pct)
            if use_residual and self.residual_predictor is not None:
                correction = np.asarray(self.residual_predictor(snap, action), dtype=float)
                if correction.shape == (plant.n_racks,):
                    plant.hidden.temperatures_c += correction
                    plant.estimated_c += correction
                    snap = plant.snapshot(float(ambient[min(idx, len(ambient) - 1)]), snapshot.humidity_pct)
            times.append(snap.time_s)
            temps.append(snap.estimated_temperatures_c.copy())
            inlets.append(snap.inlet_temperatures_c.copy())
            powers.append(float(snap.it_power_w.sum() + snap.cooling_power_w + snap.fan_power_w))
        temp_array = np.asarray(temps)
        dt_min = step_s / 60.0
        energy = float(np.sum(powers) * step_s / 3_600_000.0)
        max_temp = float(temp_array.max())
        return RolloutResult(
            np.asarray(times), temp_array, np.asarray(inlets), np.asarray(powers), energy, snap,
            bool(max_temp <= hard), max_temp,
            float(np.sum(temp_array > hot) * dt_min), float(np.sum(temp_array > hard) * dt_min),
            {"rejection_reason": None, "model": "hybrid" if use_residual else "physics", "steps": n_steps},
        )


class LearnedOnlyTwin(DigitalTwin):
    """Action-conditioned learned transition rollout that intentionally omits physics for ablation."""

    def __init__(self, config: dict[str, Any], transition_model: Any):
        super().__init__(config)
        self.transition_model = transition_model

    def simulate_action(
        self,
        snapshot: Snapshot,
        action: Action,
        horizon_seconds: float,
        forecast_inputs: dict[str, np.ndarray | float],
        use_residual: bool = True,
    ) -> RolloutResult:
        from thermotwin.control.constraints import validate_action

        valid, reason = validate_action(snapshot, action, self.config)
        if not valid:
            return RolloutResult(np.array([]), np.empty((0, len(snapshot.utilization))), np.empty((0, len(snapshot.utilization))), np.array([]), float("inf"), snapshot, False, float("inf"), float("inf"), float("inf"), {"rejection_reason": reason})
        step_s = float(self.config["control"]["rollout_step_s"])
        steps = max(1, int(np.ceil(horizon_seconds / step_s)))
        current = snapshot
        temperatures, inlets, powers, times = [], [], [], []
        for index in range(steps):
            applied = action if index == 0 else Action.noop(len(snapshot.supply_actual_c))
            predicted = self.transition_model.predict_step(current, applied)
            supply_command = current.supply_command_c + np.asarray(applied.supply_delta_c)
            airflow_command = current.airflow_command + np.asarray(applied.airflow_delta)
            current = replace(
                current, time_s=current.time_s + step_s, estimated_temperatures_c=predicted,
                observed_temperatures_c=predicted,
                supply_command_c=supply_command, airflow_command=airflow_command,
            )
            temperatures.append(predicted.copy())
            inlets.append(current.inlet_temperatures_c.copy())
            powers.append(float(np.sum(current.it_power_w) + current.cooling_power_w + current.fan_power_w))
            times.append(current.time_s)
        temp = np.asarray(temperatures)
        hot, hard = self.config["thresholds"]["hotspot_c"], self.config["thresholds"]["hard_limit_c"]
        return RolloutResult(
            np.asarray(times), temp, np.asarray(inlets), np.asarray(powers),
            float(np.sum(powers) * step_s / 3_600_000.0), current, bool(np.max(temp) <= hard), float(np.max(temp)),
            float(np.sum(temp > hot) * step_s / 60), float(np.sum(temp > hard) * step_s / 60),
            {"rejection_reason": None, "model": "learned_only", "steps": steps},
        )
