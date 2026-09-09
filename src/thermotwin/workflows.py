"""High-level idempotent preparation workflow shared by CLI and background jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable

from thermotwin.config import config_fingerprint, load_config, project_path, validate_config
from thermotwin.data.generate import generate_dataset
from thermotwin.experiments.runner import run_experiments
from thermotwin.forecasting.registry import load_bundle
from thermotwin.forecasting.train import train_all
from thermotwin.services.artifacts import atomic_json, source_fingerprint


def prepared_paths(config: dict) -> tuple[Path, Path, Path]:
    """Return profile-scoped dataset, model directory, and preparation manifest paths."""
    root = project_path(config, "artifacts_dir") / config["project"]["profile"]
    return root / "dataset.csv", root / "models", root / "preparation.json"


def prepare_project(
    profile: str = "smoke", force: bool = False, progress: Callable[[str], None] | None = None,
    cancel: Callable[[], bool] | None = None, config_path: str | Path | None = None, seed: int | None = None,
) -> tuple[Path, dict]:
    """Generate, train, run all six experiments, and cache only matching source/config artifacts."""
    config = load_config(config_path, profile=profile)
    if seed is not None:
        config["project"]["seed"] = int(seed)
        validate_config(config)
    dataset, models, preparation = prepared_paths(config)
    expected = {"source_fingerprint": source_fingerprint(), "config_fingerprint": config_fingerprint(config), "profile": config["project"]["profile"]}
    cached = False
    if not force and preparation.is_file() and dataset.is_file() and (models / "manifest.json").is_file():
        previous = json.loads(preparation.read_text(encoding="utf-8"))
        cached = previous.get("status") == "complete" and all(previous.get(key) == value for key, value in expected.items())
    started = time.perf_counter()
    if cancel and cancel():
        raise InterruptedError("preparation cancelled")
    if cached:
        if progress:
            progress("Using fingerprint-matched generated data and trained models")
        try:
            bundle, metadata = load_bundle(models)
        except (ValueError, OSError, KeyError):
            cached = False
            if progress:
                progress("Cached models are incompatible with this interpreter; rebuilding")
    if not cached:
        if progress:
            progress("Generating reproducible action-varied trajectories")
        generate_dataset(config, dataset, progress=progress)
        if cancel and cancel():
            raise InterruptedError("preparation cancelled after data generation")
        bundle, metadata = train_all(config, dataset, models, progress)
    if cancel and cancel():
        raise InterruptedError("preparation cancelled before experiments")
    run_dir, summary = run_experiments(config, dataset, bundle, metadata, selected=[1, 2, 3, 4, 5, 6], resume=not force, progress=progress, cancel=cancel)
    record = {
        **expected, "status": summary["status"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": time.perf_counter() - started, "dataset": str(dataset), "models": str(models), "latest_run_id": run_dir.name,
    }
    atomic_json(preparation, record)
    return run_dir, record


def load_prepared(config: dict) -> tuple[dict | None, dict | None]:
    """Load compatible trusted model artifacts or return an explicit simulation-only state."""
    _, models, preparation = prepared_paths(config)
    if not preparation.is_file() or not (models / "manifest.json").is_file():
        return None, None
    try:
        bundle, metadata = load_bundle(models)
        if metadata.get("config_fingerprint") != config_fingerprint(config):
            return None, None
        return bundle, metadata
    except (OSError, ValueError, KeyError):
        return None, None
