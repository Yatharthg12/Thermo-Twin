"""Closed-loop scenario replay shared by controller comparisons and robustness tests."""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from thermotwin.experiments.metrics import summarize_snapshots
from thermotwin.physics.plant import DataCenterPlant
from thermotwin.physics.workload import scenario_inputs
from thermotwin.types import Action


def run_closed_loop(
    config: dict[str, Any],
    scenario: str,
    controller_name: str,
    policy_factory: Callable[[], Any],
    seed: int,
    steps: int,
    cancel: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Any]]:
    """Replay one causal trace; identical seed/config pairs share exogenous inputs and hidden mismatch."""
    plant = DataCenterPlant(config, seed=seed)
    workload_rng = np.random.default_rng(seed ^ 0xC0FFEE)
    policy = policy_factory()
    decisions, migrations, latencies = [], 0, []
    held_action = Action.noop(plant.n_zones)
    interval = int(config["control"]["decision_interval_steps"])
    for step in range(steps):
        if cancel and cancel():
            raise InterruptedError("experiment cancelled")
        scenario_step = step + (90 if scenario == "burst" else 0)
        offered, ambient, humidity = scenario_inputs(scenario_step, plant.n_racks, scenario, workload_rng)
        if step % interval == 0:
            decision = policy.decide(plant.snapshot(ambient, humidity))
            held_action = decision.chosen_action
            decisions.append(decision.as_dict())
            latencies.append(decision.latency_ms)
            migrations += int(held_action.migration is not None)
        else:
            held_action = Action.noop(plant.n_zones)
        degradation = 0.72 if scenario == "cooling_degradation" and step >= steps // 2 else None
        accepted, _ = plant.apply_action(held_action)
        if not accepted:
            held_action = Action.noop(plant.n_zones)
        plant.advance(offered, ambient, humidity, degradation=degradation)
    metrics = summarize_snapshots(
        plant.history, config["thresholds"]["hotspot_c"], config["thresholds"]["hard_limit_c"],
        migrations=migrations, decision_latencies_ms=latencies, truth_history=plant.truth_history,
    )
    metrics.update({"scenario": scenario, "controller": controller_name, "seed": seed, "steps": steps})
    return metrics, decisions, plant.history


def topology_config(config: dict[str, Any], racks: int) -> dict[str, Any]:
    """Return a valid three-zone (or smaller) topology without mutating the caller's config."""
    result = copy.deepcopy(config)
    zones = min(3, racks)
    mapping = np.minimum(np.arange(racks) * zones // racks, zones - 1).astype(int).tolist()
    result["topology"].update({"racks": racks, "zones": zones, "rack_zone": mapping})
    return result
