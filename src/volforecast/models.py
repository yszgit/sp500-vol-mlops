"""Model zoo and evaluation metrics.

Every model maps the same feature DataFrame to a forecast of next-week annualized
realized volatility. Linear models work in log space (log RV is close to Gaussian),
and every model predicts through exp(), so forecasts are always positive.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from volforecast import config

VOL_FLOOR = 1e-4  # floor before taking logs (a zero-return day has rv_d == 0)

# MLflow saves sklearn models with skops (safe deserialization), which only loads types it
# is told to trust: our own classes/functions, plus sklearn's gradient-boosting tree nodes.
# We only ever load models we trained ourselves, so trusting these is safe.
SKOPS_TRUSTED_TYPES = [
    "volforecast.models.NaiveVolForecaster",
    "volforecast.models.LogTargetRegressor",
    "volforecast.models.log_vol",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def log_vol(x):
    """log of a volatility, floored so zero-move days don't produce -inf."""
    return np.log(np.clip(x, VOL_FLOOR, None))


class NaiveVolForecaster(RegressorMixin, BaseEstimator):
    """Random-walk baseline: next week's vol equals the trailing 22-day realized vol."""

    def __init__(self, column: str = "rv_m"):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X[self.column], dtype=float)


class LogTargetRegressor(RegressorMixin, BaseEstimator):
    """Fit the regressor on log(vol); predict exp(f(x)) times Duan's smearing factor.

    exp() of a log-space prediction estimates the median, which sits below the mean and
    gets punished by QLIKE. The smearing factor mean(exp(residual)) corrects that bias.
    """

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        log_y = log_vol(np.asarray(y, dtype=float))
        self.regressor_ = clone(self.regressor).fit(X, log_y)
        residuals = log_y - self.regressor_.predict(X)
        self.smearing_ = float(np.mean(np.exp(residuals)))
        return self

    def predict(self, X) -> np.ndarray:
        return np.exp(self.regressor_.predict(X)) * self.smearing_


def _log_target(regressor) -> LogTargetRegressor:
    return LogTargetRegressor(regressor)


def build_model(kind: str, params: dict[str, Any] | None = None):
    """Return an unfitted estimator for one of: naive, har_ols, harx_ridge, hgb."""
    params = params or {}
    log_tf = FunctionTransformer(log_vol, feature_names_out="one-to-one")

    if kind == "naive":
        return NaiveVolForecaster(**params)
    if kind == "har_ols":
        pre = ColumnTransformer([("log_vol", log_tf, config.HAR_FEATURES)])
        return _log_target(Pipeline([("pre", pre), ("reg", LinearRegression())]))
    if kind == "harx_ridge":
        pre = ColumnTransformer(
            [
                (
                    "log_vol",
                    Pipeline([("log", log_tf), ("scale", StandardScaler())]),
                    config.VOL_FEATURES,
                ),
                ("other", StandardScaler(), config.OTHER_FEATURES),
            ]
        )
        return _log_target(Pipeline([("pre", pre), ("reg", Ridge(**params))]))
    if kind == "hgb":
        pre = ColumnTransformer(
            [
                ("log_vol", log_tf, config.VOL_FEATURES),
                ("other", "passthrough", config.OTHER_FEATURES),
            ]
        )
        reg = HistGradientBoostingRegressor(random_state=42, **params)
        return _log_target(Pipeline([("pre", pre), ("reg", reg)]))
    raise ValueError(f"Unknown model kind: {kind!r}")


def features_for(kind: str) -> list[str]:
    """Input columns a model kind needs (the API uses the same list)."""
    return config.HAR_FEATURES if kind in {"naive", "har_ols"} else config.ALL_FEATURES


# Candidate runs logged to MLflow on every training run: (run name, kind, params)
CANDIDATES: list[tuple[str, str, dict[str, Any]]] = [
    ("naive_rv_m", "naive", {}),
    ("har_ols", "har_ols", {}),
    ("harx_ridge_a0.1", "harx_ridge", {"alpha": 0.1}),
    ("harx_ridge_a10", "harx_ridge", {"alpha": 10.0}),
    ("hgb_depth3", "hgb", {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300}),
    ("hgb_depth6", "hgb", {"max_depth": 6, "learning_rate": 0.05, "max_iter": 300}),
]


def evaluate(y_true, y_pred) -> dict[str, float]:
    """RMSE / MAE / R2 on vol, and QLIKE on variance (standard loss for vol forecasts)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), VOL_FLOOR, None)
    err = y_true - y_pred
    ratio = np.clip(y_true, VOL_FLOOR, None) ** 2 / y_pred**2
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "qlike": float(np.mean(ratio - np.log(ratio) - 1)),
    }
