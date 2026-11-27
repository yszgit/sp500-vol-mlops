"""Model vs. baseline and sensible-prediction checks."""

import numpy as np
import pandas as pd
import pytest

from volforecast import config
from volforecast.models import CANDIDATES, build_model, evaluate, features_for


def _fit(kind, splits, params=None):
    cols = features_for(kind)
    model = build_model(kind, params).fit(splits["train"][cols], splits["train"][config.TARGET])
    return model, cols


def _val_metrics(kind, splits, params=None):
    model, cols = _fit(kind, splits, params)
    return evaluate(splits["val"][config.TARGET], model.predict(splits["val"][cols]))


@pytest.mark.parametrize("kind", ["har_ols", "harx_ridge"])
def test_model_beats_naive_baseline(splits, kind):
    naive = _val_metrics("naive", splits)
    model = _val_metrics(kind, splits, {"alpha": 1.0} if kind == "harx_ridge" else None)
    assert model["qlike"] < naive["qlike"]
    assert model["rmse"] < naive["rmse"]


@pytest.mark.parametrize("name,kind,params", CANDIDATES)
def test_predictions_are_positive_and_finite(splits, name, kind, params):
    model, cols = _fit(kind, splits, params)
    pred = model.predict(splits["test"][cols])
    assert np.isfinite(pred).all()
    assert (pred > 0).all()
    assert pred.max() < 5.0  # 500% annualized vol would be absurd for S&P 500 stocks


@pytest.mark.parametrize("feature", config.HAR_FEATURES)
def test_higher_recent_volatility_raises_the_forecast(splits, feature):
    model, cols = _fit("har_ols", splits)
    base = pd.DataFrame([{c: 0.2 for c in cols}])
    shocked = base.assign(**{feature: 0.6})
    assert model.predict(shocked)[0] > model.predict(base)[0]


def test_zero_return_day_does_not_break_prediction(splits):
    model, cols = _fit("har_ols", splits)
    row = pd.DataFrame([{"rv_d": 0.0, "rv_w": 0.15, "rv_m": 0.2}])[cols]
    assert np.isfinite(model.predict(row)).all()


def test_metrics_on_perfect_forecast():
    y = np.array([0.1, 0.2, 0.3])
    m = evaluate(y, y)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["qlike"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_unknown_model_kind_raises():
    with pytest.raises(ValueError, match="Unknown model kind"):
        build_model("transformer")
