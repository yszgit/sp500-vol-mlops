"""End-to-end training run against a throwaway local MLflow store."""

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from volforecast import config
from volforecast.models import CANDIDATES, SKOPS_TRUSTED_TYPES, build_model, features_for
from volforecast.train import logged_model_name, run_training


def test_logged_model_names_are_valid_for_every_candidate():
    assert logged_model_name("harx_ridge_a0.1") == "harx_ridge_a0_1"
    for name, _, _ in CANDIDATES:
        assert not set("/:.%\"'") & set(logged_model_name(name))


@pytest.mark.parametrize("name,kind,params", CANDIDATES)
def test_every_candidate_can_be_saved_and_reloaded(splits, tmp_path, name, kind, params):
    cols = features_for(kind)
    train = splits["train"].sample(5_000, random_state=0)
    model = build_model(kind, params).fit(train[cols], train[config.TARGET])
    path = tmp_path / name
    mlflow.sklearn.save_model(model, path, skops_trusted_types=SKOPS_TRUSTED_TYPES)
    reloaded = mlflow.sklearn.load_model(str(path))
    x = splits["val"][cols].head(50)
    np.testing.assert_allclose(reloaded.predict(x), model.predict(x))


def test_training_logs_runs_and_registers_champion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow writes run artifacts under ./mlruns
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    out = run_training(
        synthetic=True,
        tracking_uri=uri,
        experiment="pytest",
        candidates=[("naive_rv_m", "naive", {}), ("har_ols", "har_ols", {})],
    )

    assert len(out["results"]) == 2
    assert out["champion"] is not None and out["champion"]["name"] == "har_ols"

    client = MlflowClient(tracking_uri=uri)
    version = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
    run = client.get_run(version.run_id)
    assert "val_qlike" in run.data.metrics and "test_qlike" in run.data.metrics

    # Each logged model is named after its run, so they are distinguishable in the UI
    mlflow.set_tracking_uri(uri)
    for result in out["results"]:
        model_id = result["model_uri"].removeprefix("models:/")
        assert mlflow.get_logged_model(model_id).name == logged_model_name(result["name"])

    # The registered model loads through the generic pyfunc API (what the model API uses)
    model = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}")
    example = model.input_example
    pred = np.asarray(model.predict(example))
    assert (pred > 0).all()
