"""Action-conditioned learned residual and learned-only transition models for control."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

from thermotwin.types import Action, Snapshot


@dataclass
class ResidualCorrector:
    """Shared one-step residual regressor evaluated per rack with candidate-action features."""

    model: Pipeline | None = None
    fallback_gain: float = 0.035

    def fit(self, X: np.ndarray, truth: np.ndarray, nominal: np.ndarray) -> "ResidualCorrector":
        self.model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=2.0))])
        self.model.fit(X, truth - nominal)
        return self

    def __call__(self, snapshot: Snapshot, action: Action) -> np.ndarray:
        n_racks, zones = len(snapshot.estimated_temperatures_c), len(snapshot.supply_actual_c)
        per_zone = max(1, n_racks // zones)
        rows = []
        for rack in range(n_racks):
            zone = min(rack // per_zone, zones - 1)
            rows.append([
                snapshot.estimated_temperatures_c[rack], snapshot.utilization[rack],
                snapshot.supply_actual_c[zone], snapshot.airflow_actual[zone], snapshot.ambient_c,
                action.supply_delta_c[zone], action.airflow_delta[zone], rack / max(n_racks - 1, 1),
            ])
        if self.model is not None and getattr(self.model, "n_features_in_", 8) == 8:
            return np.asarray(self.model.predict(np.asarray(rows)), dtype=float)
        supply_delta = np.asarray(action.supply_delta_c)
        airflow_delta = np.asarray(action.airflow_delta)
        return self.fallback_gain * (supply_delta[np.minimum(np.arange(n_racks) // per_zone, zones - 1)] - 3 * airflow_delta[np.minimum(np.arange(n_racks) // per_zone, zones - 1)])


@dataclass
class ActionTransitionModel:
    """Shared learned one-step rack transition used by the no-physics ablation predictor."""

    model: Pipeline

    def rows(self, snapshot: Snapshot, action: Action) -> np.ndarray:
        racks, zones = len(snapshot.utilization), len(snapshot.supply_actual_c)
        rack_zone = np.minimum(np.arange(racks) * zones // max(racks, 1), zones - 1)
        return np.asarray([[
            snapshot.estimated_temperatures_c[r], snapshot.utilization[r],
            snapshot.supply_actual_c[rack_zone[r]], snapshot.airflow_actual[rack_zone[r]], snapshot.ambient_c,
            action.supply_delta_c[rack_zone[r]], action.airflow_delta[rack_zone[r]],
            r / max(racks - 1, 1), rack_zone[r] / max(zones - 1, 1),
        ] for r in range(racks)])

    def predict_step(self, snapshot: Snapshot, action: Action) -> np.ndarray:
        return np.asarray(self.model.predict(self.rows(snapshot, action)), dtype=float)


def fit_action_models(frame: pd.DataFrame, config: dict) -> tuple[ResidualCorrector, ActionTransitionModel, int]:
    """Fit action-conditioned one-step residual and transition models on training-time raw blocks only."""
    from thermotwin.data.splitting import chronological_ranges

    rows, targets, residual_targets = [], [], []
    racks = int(config["topology"]["racks"])
    window = int(config["forecasting"]["window_steps"])
    max_horizon = max(config["forecasting"]["horizons_min"])
    for _, run in frame.groupby("run_id", sort=True):
        if "scenario" in run and str(run["scenario"].iloc[0]) in set(config["forecasting"].get("heldout_scenarios", [])):
            continue
        n_steps = int(run["step"].max()) + 1
        train_range = chronological_ranges(n_steps, window, max_horizon)[0]
        ordered = run.sort_values(["rack_id", "step"])
        for rack, rack_frame in ordered.groupby("rack_id", sort=True):
            rack_frame = rack_frame.set_index("step")
            zone = int(rack_frame["zone_id"].iloc[0])
            for step in range(train_range.start, train_range.end - 1):
                current, nxt = rack_frame.loc[step], rack_frame.loc[step + 1]
                supply_delta = float(nxt["supply_command_c"] - current["supply_command_c"])
                airflow_delta = float(nxt["airflow_command"] - current["airflow_command"])
                feature = [
                    float(current["estimated_temp_c"]), float(current["utilization"]),
                    float(current["supply_actual_c"]), float(current["airflow_actual"]), float(current["ambient_c"]),
                    supply_delta, airflow_delta, rack / max(racks - 1, 1), zone / max(config["topology"]["zones"] - 1, 1),
                ]
                target = float(nxt["true_temp_c"])
                equilibrium = feature[2] + (2500 + 6500 * np.clip(feature[1], 0, 1) ** 1.12) / (620 * max(feature[3], .4) ** .72)
                tau_min = 900000 / (620 * max(feature[3], .4) ** .72) / 60
                nominal = equilibrium + (feature[0] - equilibrium) * np.exp(-1 / tau_min)
                rows.append(feature)
                targets.append(target)
                residual_targets.append(target - nominal)
    X = np.asarray(rows)
    transition_pipeline = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=2.0))]).fit(X, np.asarray(targets))
    residual_pipeline = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=2.0))]).fit(X[:, :8], np.asarray(residual_targets))
    return ResidualCorrector(residual_pipeline), ActionTransitionModel(transition_pipeline), len(X)


def fit_operational_hybrid(X_train: np.ndarray, y_train: np.ndarray, nominal_train: np.ndarray) -> Pipeline:
    """Fit direct residual corrections on development data only."""
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=2.0))])
    model.fit(X_train, y_train - nominal_train)
    return model
