"""Preprocessing: known inputs produce the expected features, with no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from volforecast import config
from volforecast.features import ANNUALIZE, build_features, make_splits
from volforecast.ingest import flag_membership


def _single_stock(log_returns, start="2023-01-02"):
    n = len(log_returns)
    return pd.DataFrame(
        {
            "permno": 1,
            "date": pd.bdate_range(start, periods=n),
            "return": np.expm1(log_returns),
            "volume": 1_000.0,
            "isin_sp500": True,
        }
    )


def test_constant_returns_give_known_volatility():
    feats = build_features(_single_stock(np.full(40, 0.01)))
    expected = 0.01 * ANNUALIZE  # sqrt(mean(0.01^2)) annualized
    row = feats.iloc[30]
    for col in ["rv_d", "rv_w", "rv_m", "mkt_rv_w", config.TARGET]:
        assert row[col] == pytest.approx(expected)
    assert row["ret_m"] == pytest.approx(0.01 * config.MONTH)
    assert row["log_volume_ratio"] == pytest.approx(0.0)


def test_rolling_windows_need_full_history():
    feats = build_features(_single_stock(np.full(40, 0.01)))
    assert feats["rv_w"].iloc[: config.WEEK - 1].isna().all()
    assert feats["rv_m"].iloc[: config.MONTH - 1].isna().all()
    assert feats["rv_m"].iloc[config.MONTH - 1 :].notna().all()
    # The last HORIZON days have no complete forward window
    assert feats[config.TARGET].iloc[-config.HORIZON :].isna().all()


def test_target_looks_forward_and_features_look_back():
    r = np.zeros(40)
    spike = 30
    r[spike] = 0.10  # one big move on day 30
    feats = build_features(_single_stock(r))
    # Target on day t covers t+1..t+H: days 25-29 see the spike, day 30 does not
    assert feats[config.TARGET].iloc[spike - config.HORIZON : spike].gt(0).all()
    assert feats[config.TARGET].iloc[spike] == pytest.approx(0.0)
    # Features before the spike must not know about it (no look-ahead)
    assert feats["rv_w"].iloc[spike - 1] == pytest.approx(0.0)
    assert feats["rv_d"].iloc[spike] == pytest.approx(0.10 * ANNUALIZE)


def test_membership_flag_is_point_in_time_and_inclusive():
    panel = pd.DataFrame(
        {
            "permno": [1, 1, 1, 1, 2],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-02"]
            ),
        }
    )
    members = pd.DataFrame(
        {
            "permno": [1, 1],
            "start": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "ending": pd.to_datetime(["2024-01-02", "2024-12-31"]),
        }
    )
    flag = flag_membership(panel, members)
    # Start and end dates count as membership; stock 2 was never a member
    assert flag.tolist() == [False, True, False, True, False]


def test_splits_only_keep_members_and_purge_boundaries(panel):
    splits = make_splits(build_features(panel))
    assert all(frame["isin_sp500"].all() for frame in splits.values())
    assert splits["train"]["date"].min() >= pd.Timestamp(config.TRAIN_START)
    # No training label may use returns from the validation year (and so on)
    assert splits["train"]["target_end_date"].max() < pd.Timestamp(config.VAL_START)
    assert splits["val"]["target_end_date"].max() < pd.Timestamp(config.TEST_START)
    assert splits["test"]["date"].min() >= pd.Timestamp(config.TEST_START)
