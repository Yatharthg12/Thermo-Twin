"""Causal tabular and recurrent windows built only inside raw split boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from thermotwin.data.splitting import SplitRange, assert_partition_local, chronological_ranges


FEATURE_NAMES = [
    "temp_now", "temp_lag1", "temp_lag3", "temp_lag6", "temp_lag_window",
    "util_now", "util_lag1", "util_mean", "supply_c", "airflow", "ambient_c",
    "rack_fraction", "zone_fraction", "time_sin", "time_cos",
]
SEQUENCE_FEATURE_NAMES = ["temperature_c", "utilization", "supply_c", "airflow", "ambient_c", "rack_fraction", "zone_fraction"]


@dataclass
class SupervisedPartition:
    """Aligned tabular, recurrent, target, identity, and nominal forecast arrays."""

    X: np.ndarray
    X_sequence: np.ndarray
    y: np.ndarray
    nominal: np.ndarray
    hotspot_events: np.ndarray
    rack_ids: np.ndarray
    run_ids: np.ndarray
    scenario_ids: np.ndarray
    origins: np.ndarray


def _nominal_forecast(temp: float, util: float, supply: float, airflow: float, horizons: np.ndarray) -> np.ndarray:
    equilibrium = supply + (2500.0 + 6500.0 * np.clip(util, 0, 1) ** 1.12) / (620.0 * max(airflow, 0.4) ** 0.72)
    tau_minutes = 900000.0 / (620.0 * max(airflow, 0.4) ** 0.72) / 60.0
    return equilibrium + (temp - equilibrium) * np.exp(-horizons / tau_minutes)


def build_supervised(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, SupervisedPartition], list[dict[str, Any]]]:
    """Build direct multi-horizon samples; no window or target can cross a raw partition."""
    window = int(config["forecasting"]["window_steps"])
    horizons = np.asarray(config["forecasting"]["horizons_min"], dtype=int)
    collected: dict[str, dict[str, list[Any]]] = {
        name: {key: [] for key in ("X", "X_sequence", "y", "nominal", "hotspot_events", "rack_ids", "run_ids", "scenario_ids", "origins")}
        for name in ("train", "validation", "calibration", "test")
    }
    split_manifest: list[dict[str, Any]] = []
    for run_id, run in frame.groupby("run_id", sort=True):
        scenario = str(run["scenario"].iloc[0]) if "scenario" in run else "unknown"
        pivot = {column: run.pivot(index="step", columns="rack_id", values=column).sort_index() for column in (
            "estimated_temp_c", "true_temp_c", "utilization", "supply_actual_c", "airflow_actual", "ambient_c", "zone_id"
        )}
        n_steps, n_racks = pivot["true_temp_c"].shape
        heldout = scenario in set(config["forecasting"].get("heldout_scenarios", []))
        ranges = [SplitRange("test", 0, n_steps)] if heldout else chronological_ranges(n_steps, window, int(horizons.max()))
        split_manifest.append({"run_id": run_id, "scenario": scenario, "heldout_scenario": heldout, "ranges": [r.__dict__ for r in ranges], "purge_steps": 0 if heldout else window + int(horizons.max())})
        temp = pivot["estimated_temp_c"].ffill().bfill().to_numpy(float)
        truth = pivot["true_temp_c"].to_numpy(float)
        util = pivot["utilization"].to_numpy(float)
        supply = pivot["supply_actual_c"].to_numpy(float)
        airflow = pivot["airflow_actual"].to_numpy(float)
        ambient = pivot["ambient_c"].to_numpy(float)
        zones = pivot["zone_id"].to_numpy(float)
        max_zone = max(float(np.nanmax(zones)), 1.0)
        for split in ranges:
            for origin in range(split.start + window - 1, split.end - int(horizons.max())):
                input_start, target_end = origin - window + 1, origin + int(horizons.max())
                assert_partition_local(input_start, origin, target_end, split)
                angle = 2 * np.pi * ((origin % 1440) / 1440.0)
                for rack in range(n_racks):
                    seq = np.column_stack([
                        temp[input_start:origin + 1, rack], util[input_start:origin + 1, rack],
                        supply[input_start:origin + 1, rack], airflow[input_start:origin + 1, rack],
                        ambient[input_start:origin + 1, rack], np.full(window, rack / max(n_racks - 1, 1)),
                        np.full(window, zones[origin, rack] / max_zone),
                    ])
                    lag_index = lambda n: max(input_start, origin - n)
                    x = [
                        temp[origin, rack], temp[lag_index(1), rack], temp[lag_index(3), rack],
                        temp[lag_index(6), rack], temp[input_start, rack], util[origin, rack],
                        util[lag_index(1), rack], util[input_start:origin + 1, rack].mean(),
                        supply[origin, rack], airflow[origin, rack], ambient[origin, rack],
                        rack / max(n_racks - 1, 1), zones[origin, rack] / max_zone, np.sin(angle), np.cos(angle),
                    ]
                    target = truth[origin + horizons, rack]
                    hotspot_events = np.asarray([
                        np.max(truth[origin + 1: origin + horizon + 1, rack]) > config["thresholds"]["hotspot_c"]
                        for horizon in horizons
                    ], dtype=np.float32)
                    bucket = collected[split.name]
                    bucket["X"].append(x)
                    bucket["X_sequence"].append(seq)
                    bucket["y"].append(target)
                    bucket["nominal"].append(_nominal_forecast(temp[origin, rack], util[origin, rack], supply[origin, rack], airflow[origin, rack], horizons))
                    bucket["hotspot_events"].append(hotspot_events)
                    bucket["rack_ids"].append(rack)
                    bucket["run_ids"].append(run_id)
                    bucket["scenario_ids"].append(scenario)
                    bucket["origins"].append(origin)
    partitions = {
        name: SupervisedPartition(
            X=np.asarray(values["X"], dtype=np.float32),
            X_sequence=np.asarray(values["X_sequence"], dtype=np.float32),
            y=np.asarray(values["y"], dtype=np.float32),
            nominal=np.asarray(values["nominal"], dtype=np.float32),
            hotspot_events=np.asarray(values["hotspot_events"], dtype=np.float32),
            rack_ids=np.asarray(values["rack_ids"], dtype=int),
            run_ids=np.asarray(values["run_ids"], dtype=str),
            scenario_ids=np.asarray(values["scenario_ids"], dtype=str),
            origins=np.asarray(values["origins"], dtype=int),
        ) for name, values in collected.items()
    }
    return partitions, split_manifest
