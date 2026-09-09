"""Static action validation independent of counterfactual thermal feasibility."""

from __future__ import annotations

import math
import numpy as np

from thermotwin.types import Action, Snapshot


def validate_action(snapshot: Snapshot, action: Action, config: dict) -> tuple[bool, str]:
    """Return a machine-readable reason before expensive rollout evaluation."""
    zones, racks = len(snapshot.supply_command_c), len(snapshot.utilization)
    if len(action.supply_delta_c) != zones or len(action.airflow_delta) != zones:
        return False, "dimension_mismatch"
    values = [*action.supply_delta_c, *action.airflow_delta]
    if not all(math.isfinite(float(value)) for value in values):
        return False, "nonfinite_command"
    cooling = config["cooling"]
    supply = snapshot.supply_command_c + np.asarray(action.supply_delta_c)
    airflow = snapshot.airflow_command + np.asarray(action.airflow_delta)
    if np.any(supply < cooling["supply_min_c"]) or np.any(supply > cooling["supply_max_c"]):
        return False, "supply_bounds"
    if np.any(airflow < cooling["airflow_min"]) or np.any(airflow > cooling["airflow_max"]):
        return False, "airflow_bounds"
    if action.migration:
        source, destination, amount = action.migration
        work = config["workload"]
        if source == destination or source not in range(racks) or destination not in range(racks):
            return False, "migration_rack"
        if not math.isfinite(amount) or not 0 < amount <= work["migration_limit"]:
            return False, "migration_rate"
        if len(snapshot.migration_cooldowns) and (snapshot.migration_cooldowns[source] or snapshot.migration_cooldowns[destination]):
            return False, "migration_cooldown"
        if snapshot.utilization[source] < amount or snapshot.utilization[destination] + amount > work["rack_capacity"]:
            return False, "migration_capacity"
    return True, "ok"

