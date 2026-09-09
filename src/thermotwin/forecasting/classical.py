"""Leakage-safe scaled ridge, random-forest, and gradient-boosted forecast models."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def make_classical_models(config: dict, seed: int) -> dict[str, object]:
    """Construct deterministic direct multi-horizon regressors; callers fit train data only."""
    fc = config["forecasting"]
    return {
        "ridge": Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline([("imputer", SimpleImputer()), ("model", RandomForestRegressor(
            n_estimators=int(fc["forest_trees"]), max_depth=12, min_samples_leaf=3,
            random_state=seed, n_jobs=1,
        ))]),
        "gradient_boosting": Pipeline([("imputer", SimpleImputer()), ("model", MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=int(fc["boosting_estimators"]), max_depth=3, random_state=seed), n_jobs=1,
        ))]),
    }

