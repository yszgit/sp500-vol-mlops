"""Synthetic daily panel with the same schema as the CRSP panel.

CRSP data is licensed and can't live in a public repo or in CI, so tests, CI and quick
demos use this generator instead. Returns follow a one-factor GARCH(1,1) model, which
gives volatility clustering (the effect the real model exploits).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast import config


def _garch_path(
    rng: np.random.Generator,
    n_steps: int,
    n_series: int,
    ann_vol: np.ndarray,
    alpha: float = 0.12,
    beta: float = 0.83,
) -> np.ndarray:
    """Simulate GARCH(1,1) innovations; returns an (n_steps, n_series) array of daily returns.

    alpha + beta = 0.95 gives a volatility half-life of about two weeks.
    """
    long_run_var = (ann_vol**2) / config.TRADING_DAYS
    omega = (1 - alpha - beta) * long_run_var
    var = long_run_var.copy()
    eps_prev = np.zeros(n_series)
    out = np.empty((n_steps, n_series))
    for t in range(n_steps):
        var = omega + alpha * eps_prev**2 + beta * var
        eps_prev = np.sqrt(var) * rng.standard_normal(n_series)
        out[t] = eps_prev
    return out


def make_synthetic_panel(
    n_stocks: int = 40,
    start: str = config.START,
    end: str = config.END,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a long panel: permno, date, return, price, volume, isin_sp500, log_return."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n_days = len(dates)

    market = _garch_path(rng, n_days, 1, np.array([0.18]))[:, 0]
    betas = rng.uniform(0.6, 1.4, n_stocks)
    idio_vol = rng.uniform(0.15, 0.45, n_stocks)
    idio = _garch_path(rng, n_days, n_stocks, idio_vol)
    log_ret = market[:, None] * betas[None, :] + idio  # (days, stocks)

    # Volume rises on big-move days (|return| relative to typical size)
    base_volume = rng.lognormal(mean=15, sigma=0.7, size=n_stocks)
    typical = np.abs(log_ret).mean(axis=0)
    volume = base_volume * (0.5 + np.abs(log_ret) / typical) * rng.lognormal(0, 0.2, log_ret.shape)

    price = rng.uniform(20, 300, n_stocks) * np.exp(np.cumsum(log_ret, axis=0))

    # Most stocks are members throughout; ~20% join or leave the index mid-sample
    member = np.ones_like(log_ret, dtype=bool)
    for j in rng.choice(n_stocks, size=max(1, n_stocks // 5), replace=False):
        cut = rng.integers(n_days // 4, 3 * n_days // 4)
        if rng.random() < 0.5:
            member[:cut, j] = False  # added later
        else:
            member[cut:, j] = False  # removed later

    panel = pd.DataFrame(
        {
            "permno": np.tile(10001 + np.arange(n_stocks), n_days),
            "date": np.repeat(dates.values, n_stocks),
            "log_return": log_ret.ravel(),
            "price": price.ravel(),
            "volume": np.round(volume.ravel()),
            "isin_sp500": member.ravel(),
        }
    )
    panel["return"] = np.expm1(panel["log_return"])
    cols = ["permno", "date", "return", "price", "volume", "isin_sp500", "log_return"]
    return panel[cols].sort_values(["permno", "date"]).reset_index(drop=True)
