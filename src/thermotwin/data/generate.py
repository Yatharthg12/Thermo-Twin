"""Generate reproducible action-varied simulated trajectories and seed manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from thermotwin.data.schemas import SCHEMA_VERSION, validate_frame
from thermotwin.physics.plant import DataCenterPlant
from thermotwin.physics.workload import scenario_inputs
from thermotwin.types import Action
from thermotwin.services.artifacts import atomic_json


def _exploration_action(plant: DataCenterPlant, step: int, rng: np.random.Generator) -> Action:
    zones = plant.n_zones
    if step % 8:
        return Action.noop(zones)
    supply = np.zeros(zones)
    airflow = np.zeros(zones)
    zone = int(rng.integers(0, zones))
    if rng.random() < 0.55:
        change = float(rng.choice([-0.5, 0.5]))
        proposed = plant.supply_command_c[zone] + change
        cfg = plant.config["cooling"]
        if cfg["supply_min_c"] <= proposed <= cfg["supply_max_c"]:
            supply[zone] = change
    else:
        change = float(rng.choice([-0.08, 0.08]))
        proposed = plant.airflow_command[zone] + change
        cfg = plant.config["cooling"]
        if cfg["airflow_min"] <= proposed <= cfg["airflow_max"]:
            airflow[zone] = change
    return Action(f"explore-{step}", tuple(supply), tuple(airflow), None, "Bounded exploration")


def generate_dataset(
    config: dict[str, Any],
    output_path: str | Path,
    scenarios: list[str] | None = None,
    steps: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Generate long-form observations with controller-visible and evaluation-only truth columns."""
    scenarios = scenarios or list(config["experiments"]["scenarios"])
    steps = int(steps or config["experiments"]["scenario_steps"])
    seed = int(config["project"]["seed"])
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(scenarios) * 2)
    rows: list[dict[str, Any]] = []
    seed_map: dict[str, Any] = {"root": seed, "streams": {}}
    for index, scenario in enumerate(scenarios):
        plant_seed = int(child_seeds[index * 2].generate_state(1)[0])
        workload_seed = int(child_seeds[index * 2 + 1].generate_state(1)[0])
        seed_map["streams"][scenario] = {
            "plant_root_process_and_observation": plant_seed, "workload": workload_seed,
            "exploration_actions": workload_seed ^ 0xA5A5A5A5,
        }
        workload_rng = np.random.default_rng(workload_seed)
        action_rng = np.random.default_rng(workload_seed ^ 0xA5A5A5A5)
        plant = DataCenterPlant(config, seed=plant_seed)
        run_id = f"dataset-{scenario}-{plant_seed:08x}"
        if progress:
            progress(f"Generating {scenario} trajectory ({steps} minutes)")
        for step in range(steps):
            offered, ambient, humidity = scenario_inputs(step, plant.n_racks, scenario, workload_rng)
            action = _exploration_action(plant, step, action_rng)
            degradation = 0.72 if scenario == "cooling_degradation" and step > steps // 2 else None
            snap = plant.advance(offered, ambient, humidity, action, degradation)
            for rack in range(plant.n_racks):
                zone = int(plant.rack_zone[rack])
                rows.append({
                    "run_id": run_id, "scenario": scenario, "step": step, "time_s": snap.time_s,
                    "rack_id": rack, "zone_id": zone,
                    "observed_temp_c": float(snap.observed_temperatures_c[rack]) if snap.sensor_valid[rack] else np.nan,
                    "sensor_valid": bool(snap.sensor_valid[rack]),
                    "estimated_temp_c": float(snap.estimated_temperatures_c[rack]),
                    "utilization": float(snap.utilization[rack]),
                    "offered_utilization": float(snap.offered_utilization[rack]),
                    "ambient_c": snap.ambient_c, "humidity_pct": snap.humidity_pct,
                    "supply_actual_c": float(snap.supply_actual_c[zone]),
                    "supply_command_c": float(snap.supply_command_c[zone]),
                    "airflow_actual": float(snap.airflow_actual[zone]),
                    "airflow_command": float(snap.airflow_command[zone]),
                    "it_power_w": float(snap.it_power_w[rack]),
                    "cooling_power_w": snap.cooling_power_w / plant.n_racks,
                    "fan_power_w": snap.fan_power_w / plant.n_racks,
                    "unserved_utilization": snap.unserved_utilization / plant.n_racks,
                    "true_temp_c": float(plant.hidden.temperatures_c[rack]),
                    "degradation_factor": float(plant.hidden.degradation[zone]),
                    "schema_version": SCHEMA_VERSION,
                })
    frame = pd.DataFrame.from_records(rows)
    validate_frame(frame)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(output)
    seed_path = output.with_name(output.stem + "_seeds.json")
    atomic_json(seed_path, seed_map)
    return frame


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a trusted generated CSV."""
    frame = pd.read_csv(path)
    validate_frame(frame)
    return frame
