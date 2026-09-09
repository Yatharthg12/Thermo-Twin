"""Feasibility-first bounded MPC-style hold-first-action counterfactual search."""

from __future__ import annotations

import time
import numpy as np

from thermotwin.control.constraints import validate_action
from thermotwin.control.objectives import score_rollout
from thermotwin.physics.twin import DigitalTwin
from thermotwin.types import Action, DecisionRecord, Snapshot


def optimize(
    snapshot: Snapshot,
    candidates: list[Action],
    twin: DigitalTwin,
    config: dict,
    controller_name: str,
    use_residual: bool,
) -> DecisionRecord:
    """Evaluate every statically feasible candidate, prefer safety, and log decomposed costs."""
    started = time.perf_counter()
    minutes = int(config["control"]["horizon_min"])
    steps = max(1, int(minutes * 60 / config["control"]["rollout_step_s"]))
    trend = np.zeros_like(snapshot.utilization)
    forecast_util = np.clip(snapshot.utilization[None, :] + np.arange(steps)[:, None] * trend, 0, 1)
    forecast_inputs = {"utilization": forecast_util, "ambient_c": np.full(steps, snapshot.ambient_c)}
    records, evaluated = [], []
    for action in candidates:
        valid, reason = validate_action(snapshot, action, config)
        if not valid:
            records.append({"action_id": action.action_id, "label": action.label, "evaluated": False, "rejection_reason": reason})
            continue
        rollout = twin.simulate_action(snapshot, action, minutes * 60, forecast_inputs, use_residual=use_residual)
        if not rollout.temperatures_c.size:
            records.append({"action_id": action.action_id, "label": action.label, "evaluated": False, "rejection_reason": rollout.metadata.get("reason", "rollout_rejected")})
            continue
        score, terms = score_rollout(rollout, action, config)
        record = {
            "action_id": action.action_id, "label": action.label, "evaluated": True,
            "predicted_feasible": rollout.feasible, "predicted_peak_c": rollout.max_temperature_c,
            "predicted_energy_kwh": rollout.energy_kwh, "objective": score, "objective_terms": terms,
            "rejection_reason": None,
        }
        records.append(record)
        evaluated.append((action, rollout, score, record))
    if not evaluated:
        chosen, fallback, explanation = Action.noop(len(snapshot.supply_actual_c)), True, "No valid candidates; deterministic no-op fallback"
    else:
        feasible = [item for item in evaluated if item[1].feasible]
        if feasible:
            chosen = min(feasible, key=lambda item: (item[2], item[0].action_id))[0]
            fallback = False
            explanation = "Lowest normalized objective among predicted-feasible candidates"
        else:
            chosen = min(evaluated, key=lambda item: (item[1].hard_limit_rack_minutes, item[1].max_temperature_c, item[2], item[0].action_id))[0]
            fallback = True
            explanation = "All candidates predicted unsafe; least-violation emergency fallback"
    return DecisionRecord(snapshot.time_s, controller_name, chosen, records, fallback, explanation, (time.perf_counter() - started) * 1000)

