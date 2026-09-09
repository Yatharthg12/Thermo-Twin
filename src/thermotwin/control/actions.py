"""Deterministic bounded controller candidate generation."""

from __future__ import annotations

import numpy as np

from thermotwin.types import Action, Snapshot


def generate_candidates(snapshot: Snapshot, config: dict, allow_migration: bool = True) -> list[Action]:
    """Generate no-op, per-zone actuator changes, migration, combined, and maximum-cooling actions."""
    zones = len(snapshot.supply_command_c)
    cooling, work = config["cooling"], config["workload"]
    actions: list[Action] = [Action.noop(zones)]
    for zone in range(zones):
        for amount in (-0.5, 0.5):
            delta = [0.0] * zones
            delta[zone] = amount
            candidate = snapshot.supply_command_c[zone] + amount
            if cooling["supply_min_c"] <= candidate <= cooling["supply_max_c"]:
                actions.append(Action(f"supply-z{zone}-{amount:+.1f}", tuple(delta), (0.0,) * zones, None, f"Zone {zone + 1} supply {amount:+.1f} °C"))
        for amount in (-0.08, 0.08):
            delta = [0.0] * zones
            delta[zone] = amount
            candidate = snapshot.airflow_command[zone] + amount
            if cooling["airflow_min"] <= candidate <= cooling["airflow_max"]:
                actions.append(Action(f"air-z{zone}-{amount:+.2f}", (0.0,) * zones, tuple(delta), None, f"Zone {zone + 1} airflow {amount:+.2f}"))
    temperatures = snapshot.estimated_temperatures_c
    source = int(np.argmax(temperatures + 2.0 * snapshot.utilization))
    destination = int(np.argmin(temperatures + 2.0 * snapshot.utilization))
    migration_amount = min(float(work["migration_limit"]), max(0.0, float(snapshot.utilization[source] - snapshot.utilization[destination]) / 2))
    cooldowns_clear = len(snapshot.migration_cooldowns) == 0 or (snapshot.migration_cooldowns[source] == 0 and snapshot.migration_cooldowns[destination] == 0)
    if allow_migration and migration_amount > 0.02 and snapshot.utilization[destination] + migration_amount <= work["rack_capacity"] and cooldowns_clear:
        migration = (source, destination, migration_amount)
        actions.append(Action(f"migrate-r{source}-r{destination}", (0.0,) * zones, (0.0,) * zones, migration, f"Move {migration_amount:.2f} capacity R{source + 1} → R{destination + 1}"))
        hot_zone = min(source * zones // max(len(temperatures), 1), zones - 1)
        airflow = [0.0] * zones
        if snapshot.airflow_command[hot_zone] + 0.08 <= cooling["airflow_max"]:
            airflow[hot_zone] = 0.08
        actions.append(Action(f"combined-r{source}-z{hot_zone}", (0.0,) * zones, tuple(airflow), migration, "Migrate load and boost hot-zone airflow"))
    supply_max = tuple(float(cooling["supply_min_c"] - value) for value in snapshot.supply_command_c)
    airflow_max = tuple(float(cooling["airflow_max"] - value) for value in snapshot.airflow_command)
    actions.append(Action("maximum-safe-cooling", supply_max, airflow_max, None, "Maximum configured cooling"))
    unique: dict[str, Action] = {action.action_id: action for action in actions}
    return list(unique.values())[: int(config["control"]["max_candidates"])]
