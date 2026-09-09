"""Split-conformal-style empirical intervals and hotspot event probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class IntervalCalibrator:
    """Absolute-error quantiles calibrated on a held-out temporal block."""

    level: float
    radii: np.ndarray

    @classmethod
    def fit(cls, actual: np.ndarray, predicted: np.ndarray, level: float = 0.90) -> "IntervalCalibrator":
        return cls(level, np.quantile(np.abs(actual - predicted), level, axis=0, method="higher"))

    def interval(self, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return predicted - self.radii, predicted + self.radii


class HotspotCalibrator:
    """Per-horizon logistic estimates of any exceedance within each complete future window."""

    def __init__(self, threshold_c: float):
        self.threshold_c = threshold_c
        self.models: list[LogisticRegression | None] = []
        self.constants: list[float | None] = []

    def fit(self, predicted: np.ndarray, event_labels: np.ndarray, widths: np.ndarray) -> "HotspotCalibrator":
        self.models, self.constants = [], []
        for index in range(predicted.shape[1]):
            labels = np.asarray(event_labels[:, index], dtype=int)
            X = np.column_stack([predicted[:, : index + 1].max(axis=1), widths[:, : index + 1].max(axis=1)])
            if len(np.unique(labels)) < 2:
                self.models.append(None)
                self.constants.append(float(labels.mean()))
            else:
                self.models.append(LogisticRegression(random_state=0).fit(X, labels))
                self.constants.append(None)
        return self

    def predict_proba(self, predicted: np.ndarray, widths: np.ndarray) -> np.ndarray:
        columns = []
        for index, model in enumerate(self.models):
            if model is None:
                columns.append(np.full(len(predicted), self.constants[index] if self.constants[index] is not None else 0.0))
            else:
                X = np.column_stack([predicted[:, : index + 1].max(axis=1), widths[:, : index + 1].max(axis=1)])
                columns.append(model.predict_proba(X)[:, 1])
        return np.column_stack(columns)
