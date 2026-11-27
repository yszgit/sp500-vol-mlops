"""Feature engineering: HAR-style realized-volatility features and the forward target.

All features at date t use information up to and including the close of t.
The target is the realized volatility over the next HORIZON trading days (t+1 ... t+H).
Volatilities are annualized decimals (0.25 == 25% annualized vol).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast import config

ANNUALIZE = np.sqrt(config.TRADING_DAYS)


def _rolling(s: pd.Series, by: pd.Series, window: int, how: str = "mean") -> pd.Series:
    """Per-stock trailing rolling statistic (requires a full window)."""
    return s.groupby(by).transform(lambda x: getattr(x.rolling(window, min_periods=window), how)())


def build_features(panel: pd.DataFrame, horizon: int = config.HORIZON) -> pd.DataFrame:
    """Add feature columns and the forward realized-vol target to the daily panel.

    Features are computed on each stock's full history (members and non-members) so the
    rolling windows are complete; filtering to index members happens in ``make_splits``.
    """
    df = panel.sort_values(["permno", "date"]).reset_index(drop=True).copy()
    if "log_return" not in df.columns:
        df["log_return"] = np.log1p(df["return"])
    by = df["permno"]
    r = df["log_return"]
    r2 = r**2

    df["rv_d"] = r.abs() * ANNUALIZE
    df["rv_w"] = np.sqrt(_rolling(r2, by, config.WEEK)) * ANNUALIZE
    df["rv_m"] = np.sqrt(_rolling(r2, by, config.MONTH)) * ANNUALIZE
    df["ret_m"] = _rolling(r, by, config.MONTH, how="sum")  # leverage effect
    avg_volume = _rolling(df["volume"], by, config.MONTH)
    df["log_volume_ratio"] = np.log((df["volume"] + 1.0) / (avg_volume + 1.0))
    # Market-wide vol: cross-sectional mean of weekly RV across index members that day
    df["mkt_rv_w"] = df["rv_w"].where(df["isin_sp500"]).groupby(df["date"]).transform("mean")

    # Target: realized vol over t+1..t+horizon (rolling mean ending at t+h, shifted back)
    fwd_var = r2.groupby(by).transform(
        lambda x: x.rolling(horizon, min_periods=horizon).mean().shift(-horizon)
    )
    df[config.TARGET] = np.sqrt(fwd_var) * ANNUALIZE
    df["target_end_date"] = df.groupby("permno")["date"].shift(-horizon)
    return df


def assign_split(dates: pd.Series) -> pd.Series:
    """warmup (< TRAIN_START), train, val (>= VAL_START), test (>= TEST_START)."""
    d = pd.to_datetime(dates)
    split = np.select(
        [
            d < pd.Timestamp(config.TRAIN_START),
            d < pd.Timestamp(config.VAL_START),
            d < pd.Timestamp(config.TEST_START),
        ],
        ["warmup", "train", "val"],
        default="test",
    )
    out = pd.Series(split, index=dates.index, dtype=object)
    return out.where(d.notna())


def make_splits(
    features: pd.DataFrame, feature_cols: list[str] = config.ALL_FEATURES
) -> dict[str, pd.DataFrame]:
    """Model-ready train / val / test frames.

    Keeps rows where the stock was an index member on date t (no survivorship bias),
    features and target are complete, and the target window does not cross into the
    next split (purging, so no label leaks across the train/val/test boundary).
    """
    df = features.copy()
    df["split"] = assign_split(df["date"])
    same_split = assign_split(df["target_end_date"]) == df["split"]
    keep = (
        df["isin_sp500"]
        & same_split
        & df[[*feature_cols, config.TARGET]].notna().all(axis=1)
        & np.isfinite(df[feature_cols]).all(axis=1)
    )
    df = df.loc[keep]
    return {
        name: df.loc[df["split"] == name].reset_index(drop=True)
        for name in ["train", "val", "test"]
    }
