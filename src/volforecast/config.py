"""Central configuration: study period, splits, feature definitions and MLflow names.

Everything that more than one module needs lives here so the training flow, the API
and the tests all agree on the same definitions.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # read a local .env if present (never committed)

# --- Study period ------------------------------------------------------------------
# 2019 is a warm-up year: it only feeds the rolling windows of early-2020 features.
START = "2019-01-01"
TRAIN_START = "2020-01-01"
VAL_START = "2024-01-01"
TEST_START = "2025-01-01"
END = "2025-12-31"

# --- Target / feature definitions ------------------------------------------------------
TRADING_DAYS = 252  # annualization factor
HORIZON = 5  # forecast horizon in trading days (next week)
WEEK = 5
MONTH = 22

# Volatility-type features (annualized, decimal: 0.25 == 25%). Log-transformed in models.
VOL_FEATURES = ["rv_d", "rv_w", "rv_m", "mkt_rv_w"]
# Other features (already roughly symmetric, used as-is).
OTHER_FEATURES = ["ret_m", "log_volume_ratio"]
# Classic HAR-RV specification (Corsi, 2009): daily, weekly and monthly realized vol.
HAR_FEATURES = ["rv_d", "rv_w", "rv_m"]
ALL_FEATURES = VOL_FEATURES + OTHER_FEATURES
TARGET = "target_rv"

# Columns the raw daily panel must contain (see validation.py)
REQUIRED_PANEL_COLUMNS = ["permno", "date", "return", "volume", "isin_sp500"]

# --- MLflow ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sp500-volatility")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sp500-vol-forecaster")
CHAMPION_ALIAS = "champion"

# --- Data -----------------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_PATH", "data/raw/sp500_panel.parquet")
