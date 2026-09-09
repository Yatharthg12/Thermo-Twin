"""Load, merge, validate, and fingerprint project configuration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None, profile: str | None = None) -> dict[str, Any]:
    """Return a validated configuration, applying a profile and optional YAML override."""
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    selected = profile or ("default" if path is None else None)
    if selected and selected != "default":
        profile_path = PROJECT_ROOT / "configs" / f"{selected}.yaml"
        if not profile_path.is_file():
            raise ValueError(f"Unknown profile: {selected}")
        with profile_path.open("r", encoding="utf-8") as handle:
            config = _merge(config, yaml.safe_load(handle) or {})
    if path is not None:
        custom_path = Path(path).resolve()
        if custom_path != DEFAULT_CONFIG.resolve() and custom_path.is_file():
            with custom_path.open("r", encoding="utf-8") as handle:
                config = _merge(config, yaml.safe_load(handle) or {})
        elif not custom_path.is_file():
            raise FileNotFoundError(custom_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Reject inconsistent dimensions, bounds, intervals, and unsafe numeric values."""
    import math

    racks = int(config["topology"]["racks"])
    seed = int(config["project"]["seed"])
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("project seed must be between 0 and 2^32-1")
    zones = int(config["topology"]["zones"])
    mapping = config["topology"]["rack_zone"]
    if racks < 2 or zones < 1 or len(mapping) != racks:
        raise ValueError("topology racks/zones/rack_zone are inconsistent")
    if set(mapping) - set(range(zones)) or set(mapping) != set(range(zones)):
        raise ValueError("rack_zone must assign every declared zone")
    hot = float(config["thresholds"]["hotspot_c"])
    hard = float(config["thresholds"]["hard_limit_c"])
    if not hot < hard:
        raise ValueError("hotspot_c must be below hard_limit_c")
    obs = float(config["physics"]["observation_interval_s"])
    integ = float(config["physics"]["integration_step_s"])
    if obs <= 0 or integ <= 0 or integ > obs:
        raise ValueError("integration step must be positive and no larger than observation interval")
    horizons = [int(h) for h in config["forecasting"]["horizons_min"]]
    if not horizons or any(h <= 0 for h in horizons) or horizons != sorted(set(horizons)):
        raise ValueError("forecast horizons must be unique increasing positive minutes")
    if int(config["forecasting"]["window_steps"]) < 2:
        raise ValueError("feature window must contain at least two samples")
    if int(config["experiments"]["scenario_steps"]) <= max(horizons) + int(config["forecasting"]["window_steps"]):
        raise ValueError("scenario is too short for the requested window and horizon")
    c = config["cooling"]
    if not c["supply_min_c"] < c["supply_max_c"] or not c["airflow_min"] < c["airflow_max"]:
        raise ValueError("cooling actuator limits are invalid")
    if c["capacity_w_per_zone"] <= 0 or c["cop_min"] <= 0 or c["cop_min"] > c["cop_max"]:
        raise ValueError("cooling capacity/COP bounds are invalid")
    if not 0 <= config["sensors"]["missing_probability"] < 1:
        raise ValueError("sensor missing_probability must be in [0, 1)")
    if not 0 <= config["physics"]["recirculation_fraction"] < 1:
        raise ValueError("recirculation_fraction must be in [0, 1)")
    for section in config.values():
        if isinstance(section, dict):
            for value in section.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("configuration cannot contain NaN or infinity")


def config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable short digest for cache and provenance checks."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def project_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a configured output directory relative to the project root."""
    path = Path(config["project"][key])
    return path if path.is_absolute() else PROJECT_ROOT / path
