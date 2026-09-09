"""Trusted project-generated model bundle persistence with schema fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import torch

from thermotwin.data.features import FEATURE_NAMES, SEQUENCE_FEATURE_NAMES
from thermotwin.forecasting.recurrent import GRUForecaster, TrainedGRU


MODEL_SCHEMA = "1.0"


def file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def save_bundle(path: str | Path, bundle: dict[str, Any], metadata: dict[str, Any]) -> Path:
    """Atomically save only models trained inside this project plus a JSON manifest."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in bundle["classical"].items():
        joblib.dump(model, directory / f"{name}.joblib")
    joblib.dump(bundle["hybrid"], directory / "hybrid.joblib")
    joblib.dump(bundle["interval"], directory / "interval.joblib")
    joblib.dump(bundle["hotspot"], directory / "hotspot.joblib")
    joblib.dump(bundle["action_residual"], directory / "action_residual.joblib")
    joblib.dump(bundle["learned_transition"], directory / "learned_transition.joblib")
    recurrent: TrainedGRU = bundle["recurrent"]
    torch.save({
        "state_dict": recurrent.model.state_dict(), "mean": recurrent.mean, "scale": recurrent.scale,
        "history": recurrent.history, "trained_epochs": recurrent.trained_epochs,
        "n_features": recurrent.model.gru.input_size, "hidden": recurrent.model.gru.hidden_size,
        "n_horizons": recurrent.model.head.out_features,
    }, directory / "recurrent.pt")
    manifest = {"schema": MODEL_SCHEMA, "feature_names": FEATURE_NAMES, "sequence_feature_names": SEQUENCE_FEATURE_NAMES, **metadata}
    temp = directory / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    temp.replace(directory / "manifest.json")
    return directory


def load_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a bundle only from a local directory carrying the expected project schema."""
    directory = Path(path).resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model artifacts unavailable; run 'python -m thermotwin prepare --profile smoke': {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MODEL_SCHEMA or manifest.get("feature_names") != FEATURE_NAMES:
        raise ValueError("incompatible model/config schema")
    recorded = manifest.get("packages", {})
    current = {"numpy": np.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__}
    mismatches = [f"{name} {recorded[name]} != {current[name]}" for name in current if recorded.get(name) and recorded[name] != current[name]]
    if mismatches:
        raise ValueError("incompatible model package versions; rebuild artifacts: " + "; ".join(mismatches))
    classical = {name: joblib.load(directory / f"{name}.joblib") for name in ("ridge", "random_forest", "gradient_boosting")}
    payload = torch.load(directory / "recurrent.pt", map_location="cpu", weights_only=False)
    model = GRUForecaster(payload["n_features"], payload["hidden"], payload["n_horizons"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    recurrent = TrainedGRU(model, payload["mean"], payload["scale"], payload["history"], payload["trained_epochs"])
    bundle = {
        "classical": classical, "hybrid": joblib.load(directory / "hybrid.joblib"),
        "interval": joblib.load(directory / "interval.joblib"), "hotspot": joblib.load(directory / "hotspot.joblib"),
        "action_residual": joblib.load(directory / "action_residual.joblib"),
        "learned_transition": joblib.load(directory / "learned_transition.joblib"),
        "recurrent": recurrent,
    }
    return bundle, manifest
