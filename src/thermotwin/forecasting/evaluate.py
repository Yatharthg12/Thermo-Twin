"""Forecast accuracy, timing, uncertainty coverage, and event classification metrics."""

from __future__ import annotations

import time
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support


def regression_rows(model_name: str, actual: np.ndarray, predicted: np.ndarray, horizons: list[int], elapsed_s: float = 0.0) -> list[dict]:
    """Return aggregate MAE/RMSE rows with counts for aligned horizons."""
    rows = []
    for index, horizon in enumerate(horizons):
        error = predicted[:, index] - actual[:, index]
        rows.append({
            "model": model_name, "horizon_min": horizon, "mae_c": float(np.mean(np.abs(error))),
            "rmse_c": float(np.sqrt(np.mean(error**2))), "count": int(len(error)),
            "inference_s": float(elapsed_s),
        })
    return rows


def timed_predict(predictor, *args) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    prediction = np.asarray(predictor(*args))
    return prediction, time.perf_counter() - start


def uncertainty_rows(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, horizons: list[int], nominal_level: float) -> list[dict]:
    return [{
        "horizon_min": horizon, "nominal_level": nominal_level,
        "coverage": float(np.mean((actual[:, i] >= lower[:, i]) & (actual[:, i] <= upper[:, i]))),
        "mean_width_c": float(np.mean(upper[:, i] - lower[:, i])), "count": int(len(actual)),
    } for i, horizon in enumerate(horizons)]


def hotspot_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Score probabilities for the explicitly supplied any-window binary event."""
    labels = np.asarray(labels, dtype=int)
    predicted = probabilities >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    result = {
        "brier": float(brier_score_loss(labels, probabilities)), "precision": float(precision),
        "recall": float(recall), "f1": float(f1), "count": int(len(labels)),
    }
    if len(np.unique(labels)) < 2:
        result.update({"pr_auc": None, "pr_auc_reason": "labels contain only one class"})
    else:
        result["pr_auc"] = float(average_precision_score(labels, probabilities))
    return result
