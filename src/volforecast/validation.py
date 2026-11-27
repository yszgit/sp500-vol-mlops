"""Data validation checks run at the start of every training pipeline run.

Fail fast with a clear message instead of training on broken data.
"""

from __future__ import annotations

import pandas as pd

from volforecast import config


class DataValidationError(ValueError):
    """Raised when an input frame fails validation."""


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Check the raw daily panel. Returns the panel unchanged if every check passes."""
    errors: list[str] = []

    missing = [c for c in config.REQUIRED_PANEL_COLUMNS if c not in panel.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    if panel.empty:
        errors.append("panel is empty")
    if panel["permno"].isna().any() or panel["date"].isna().any():
        errors.append("null permno or date")
    n_dup = int(panel.duplicated(["permno", "date"]).sum())
    if n_dup:
        errors.append(f"{n_dup} duplicate (permno, date) rows")
    ret = panel["return"].dropna()
    if (ret <= -1).any():
        errors.append("simple returns <= -100% found")
    if (ret.abs() > 5).any():
        errors.append("implausible daily returns (> 500%) found")
    if (panel["volume"].dropna() < 0).any():
        errors.append("negative volume found")
    if not panel["isin_sp500"].any():
        errors.append("no S&P 500 member rows")
    missing_ret_share = panel["return"].isna().mean()
    if missing_ret_share > 0.05:
        errors.append(f"{missing_ret_share:.1%} of returns are missing (limit 5%)")

    if errors:
        raise DataValidationError("; ".join(errors))
    return panel


def validate_training_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Check a model-ready frame: features and target present, finite and non-missing."""
    needed = [*features, config.TARGET]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing model columns: {missing}")
    if frame.empty:
        raise DataValidationError("training frame is empty")
    if frame[config.TARGET].isna().any():
        raise DataValidationError("missing target values")
    bad = frame[needed].isna().any()
    if bad.any():
        raise DataValidationError(f"missing values in: {bad[bad].index.tolist()}")
    if (frame[config.TARGET] < 0).any():
        raise DataValidationError("negative realized volatility in target")
    return frame
