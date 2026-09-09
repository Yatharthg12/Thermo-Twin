"""Versioned long-form dataset schema and validation."""

from __future__ import annotations

import pandas as pd


SCHEMA_VERSION = "1.0"
VISIBLE_COLUMNS = [
    "run_id", "scenario", "step", "time_s", "rack_id", "zone_id", "observed_temp_c",
    "sensor_valid", "estimated_temp_c", "utilization", "offered_utilization", "ambient_c",
    "humidity_pct", "supply_actual_c", "supply_command_c", "airflow_actual", "airflow_command",
    "it_power_w", "cooling_power_w", "fan_power_w", "unserved_utilization",
]
TRUTH_COLUMNS = ["true_temp_c", "degradation_factor"]
REQUIRED_COLUMNS = VISIBLE_COLUMNS + TRUTH_COLUMNS + ["schema_version"]


def validate_frame(frame: pd.DataFrame) -> None:
    """Validate required columns, schema identity, uniqueness, and finite core fields."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    if set(frame["schema_version"].astype(str).unique()) != {SCHEMA_VERSION}:
        raise ValueError("unsupported dataset schema version")
    if frame.duplicated(["run_id", "step", "rack_id"]).any():
        raise ValueError("duplicate run/step/rack observations")
    core = ["time_s", "estimated_temp_c", "utilization", "ambient_c", "it_power_w"]
    if frame[core].isna().any().any():
        raise ValueError("core controller-visible fields cannot be missing")

