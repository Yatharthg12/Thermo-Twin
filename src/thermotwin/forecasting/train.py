"""Train all required forecast families and select/calibrate the operational hybrid."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sklearn
import torch

from thermotwin.config import config_fingerprint
from thermotwin.data.features import FEATURE_NAMES, build_supervised
from thermotwin.data.generate import load_dataset
from thermotwin.forecasting.baselines import persistence_predict
from thermotwin.forecasting.classical import make_classical_models
from thermotwin.forecasting.hybrid import fit_action_models, fit_operational_hybrid
from thermotwin.forecasting.recurrent import train_gru
from thermotwin.forecasting.registry import file_fingerprint, save_bundle
from thermotwin.forecasting.uncertainty import HotspotCalibrator, IntervalCalibrator


def train_all(config: dict[str, Any], dataset_path: str | Path, output_dir: str | Path, progress: Callable[[str], None] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit classical models, a real GRU, residual hybrid, and calibration on disjoint temporal data."""
    started = time.perf_counter()
    frame = load_dataset(dataset_path)
    parts, split_manifest = build_supervised(frame, config)
    train, validation, calibration = parts["train"], parts["validation"], parts["calibration"]
    seed = int(config["project"]["seed"])
    models = make_classical_models(config, seed)
    training_times: dict[str, float] = {}
    for name, model in models.items():
        if progress:
            progress(f"Training {name}")
        tick = time.perf_counter()
        model.fit(train.X, train.y)
        training_times[name] = time.perf_counter() - tick
    if progress:
        progress("Training compact GRU")
    tick = time.perf_counter()
    recurrent = train_gru(train.X_sequence, train.y, validation.X_sequence, validation.y, config, seed)
    training_times["recurrent"] = time.perf_counter() - tick
    if progress:
        progress("Training action-aware operational residual model")
    tick = time.perf_counter()
    hybrid = fit_operational_hybrid(train.X, train.y, train.nominal)
    training_times["hybrid"] = time.perf_counter() - tick
    action_residual, learned_transition, transition_samples = fit_action_models(frame, config)
    validation_candidates = {
        "ridge": models["ridge"].predict(validation.X),
        "random_forest": models["random_forest"].predict(validation.X),
        "gradient_boosting": models["gradient_boosting"].predict(validation.X),
        "recurrent": recurrent.predict(validation.X_sequence),
        "hybrid": validation.nominal + hybrid.predict(validation.X),
        "persistence": persistence_predict(validation.X, validation.y.shape[1]),
        "physics": validation.nominal,
    }
    validation_mae = {name: float(np.mean(np.abs(pred - validation.y))) for name, pred in validation_candidates.items()}
    selection = min(validation_mae, key=validation_mae.get)
    calibration_prediction = calibration.nominal + hybrid.predict(calibration.X)
    interval = IntervalCalibrator.fit(calibration.y, calibration_prediction, float(config["forecasting"]["nominal_interval"]))
    lower, upper = interval.interval(calibration_prediction)
    hotspot = HotspotCalibrator(float(config["thresholds"]["hotspot_c"])).fit(calibration_prediction, calibration.hotspot_events, upper - lower)
    bundle = {"classical": models, "recurrent": recurrent, "hybrid": hybrid, "interval": interval, "hotspot": hotspot,
              "action_residual": action_residual, "learned_transition": learned_transition}
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_fingerprint": config_fingerprint(config),
        "data_fingerprint": file_fingerprint(dataset_path), "feature_names": FEATURE_NAMES,
        "horizons_min": config["forecasting"]["horizons_min"], "split_ranges": split_manifest,
        "seed": seed, "training_times_s": training_times, "validation_mae_c": validation_mae,
        "operational_selection": "hybrid", "selection_rationale": (
            f"Operational hybrid retained for action-conditioned control; validation-best benchmark was {selection}. "
            "The residual model has bounded CPU inference and explicitly corrects nominal action rollouts."
        ),
        "packages": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__},
        "wall_time_s": time.perf_counter() - started, "sample_counts": {name: len(part.X) for name, part in parts.items()},
        "gru_history": recurrent.history, "action_transition_samples": transition_samples,
        "hyperparameters": config["forecasting"],
    }
    metadata["model_id"] = hashlib.sha256(
        f"{metadata['data_fingerprint']}:{metadata['config_fingerprint']}:{seed}:1.0".encode()
    ).hexdigest()[:16]
    save_bundle(output_dir, bundle, metadata)
    return bundle, metadata
