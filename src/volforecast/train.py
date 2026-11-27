"""Training pipeline: load -> validate -> features -> train candidates -> MLflow -> registry.

Each candidate model is one MLflow run (params, train/val metrics, the fitted model).
The best non-baseline model on validation QLIKE is registered as ``MODEL_NAME`` with the
``champion`` alias, but only if it beats the naive baseline (a simple quality gate).

    python -m volforecast.train --synthetic                       # demo data, no WRDS needed
    python -m volforecast.train --data-path data/raw/sp500_panel.parquet

The step functions are kept small and side-effect free so they can be wrapped as
Prefect tasks in the orchestrated training flow.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from volforecast import config
from volforecast.features import build_features, make_splits
from volforecast.models import (
    CANDIDATES,
    SKOPS_TRUSTED_TYPES,
    build_model,
    evaluate,
    features_for,
)
from volforecast.synthetic import make_synthetic_panel
from volforecast.validation import validate_panel, validate_training_frame

log = logging.getLogger(__name__)
PACKAGE_DIR = Path(__file__).resolve().parent


# --- Pipeline steps -------------------------------------------------------------------
def load_panel(data_path: str | None = None, synthetic: bool = False) -> pd.DataFrame:
    if synthetic:
        return make_synthetic_panel()
    panel = pd.read_parquet(data_path or config.DATA_PATH)
    # The original notebook export spelled this column "volumn"
    panel = panel.rename(columns={"volumn": "volume"})
    # WRDS NUMERIC columns can arrive as Decimal objects; models need plain floats
    for col in ["return", "price", "volume", "log_return"]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce").astype(float)
    return panel


def prepare_data(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validate_panel(panel)
    splits = make_splits(build_features(panel))
    for frame in splits.values():
        validate_training_frame(frame, config.ALL_FEATURES)
    return splits


def train_candidate(
    name: str,
    kind: str,
    params: dict[str, Any],
    splits: dict[str, pd.DataFrame],
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fit one candidate and log it as an MLflow run. Returns run id, model URI, metrics."""
    cols = features_for(kind)
    x_train, y_train = splits["train"][cols], splits["train"][config.TARGET]
    x_val, y_val = splits["val"][cols], splits["val"][config.TARGET]

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_kind": kind, **(tags or {})})
        mlflow.log_params(
            {
                "model_kind": kind,
                **params,
                "features": ",".join(cols),
                "horizon_days": config.HORIZON,
                "train_period": f"{config.TRAIN_START}..{config.VAL_START}",
                "val_period": f"{config.VAL_START}..{config.TEST_START}",
                "n_train_rows": len(x_train),
                "n_val_rows": len(x_val),
            }
        )
        model = build_model(kind, params).fit(x_train, y_train)
        train_metrics = evaluate(y_train, model.predict(x_train))
        val_pred = model.predict(x_val)
        val_metrics = evaluate(y_val, val_pred)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(x_val.head(200), val_pred[:200]),
            input_example=x_val.head(5),
            code_paths=[str(PACKAGE_DIR)],  # bundle volforecast so the API can load it
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            pip_requirements=[
                f"scikit-learn=={sklearn.__version__}",
                f"numpy=={np.__version__}",
                f"pandas=={pd.__version__}",
            ],
        )
    log.info("%-18s val_qlike=%.4f val_rmse=%.4f", name, val_metrics["qlike"], val_metrics["rmse"])
    return {
        "name": name,
        "kind": kind,
        "run_id": run.info.run_id,
        "model_uri": info.model_uri,
        "val": val_metrics,
    }


def register_champion(
    results: list[dict[str, Any]], splits: dict[str, pd.DataFrame]
) -> dict[str, Any] | None:
    """Register the best non-baseline model if it beats the naive baseline on val QLIKE."""
    baseline = next(r for r in results if r["kind"] == "naive")
    best = min((r for r in results if r["kind"] != "naive"), key=lambda r: r["val"]["qlike"])
    if best["val"]["qlike"] >= baseline["val"]["qlike"]:
        log.warning(
            "Best model %s does not beat the naive baseline; not registering.", best["name"]
        )
        return None

    # One-time look at the held-out test year, for the champion only
    cols = features_for(best["kind"])
    model = mlflow.sklearn.load_model(best["model_uri"])
    test_metrics = evaluate(splits["test"][config.TARGET], model.predict(splits["test"][cols]))
    improvement = 1 - best["val"]["qlike"] / baseline["val"]["qlike"]
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("val_qlike_improvement_vs_naive", improvement)

    version = mlflow.register_model(
        best["model_uri"],
        config.MODEL_NAME,
        tags={"model_kind": best["kind"], "run_name": best["name"]},
    )
    client = MlflowClient()
    client.update_model_version(
        config.MODEL_NAME,
        version.version,
        description=(
            f"{best['name']}: val QLIKE {best['val']['qlike']:.4f} "
            f"({improvement:.1%} better than naive), test QLIKE {test_metrics['qlike']:.4f}"
        ),
    )
    client.set_registered_model_alias(config.MODEL_NAME, config.CHAMPION_ALIAS, version.version)
    log.info("Registered %s v%s as @%s", config.MODEL_NAME, version.version, config.CHAMPION_ALIAS)
    return {**best, "version": version.version, "test": test_metrics}


# --- Entry point ----------------------------------------------------------------------
def run_training(
    data_path: str | None = None,
    synthetic: bool = False,
    tracking_uri: str | None = None,
    experiment: str = config.EXPERIMENT_NAME,
    register: bool = True,
    candidates: list[tuple[str, str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)

    splits = prepare_data(load_panel(data_path, synthetic))
    log.info("Rows: %s", {k: len(v) for k, v in splits.items()})
    tags = {"data_source": "synthetic" if synthetic else "crsp"}
    results = [
        train_candidate(name, kind, params, splits, tags)
        for name, kind, params in (candidates or CANDIDATES)
    ]
    champion = register_champion(results, splits) if register else None
    return {"results": results, "champion": champion}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register the volatility model")
    parser.add_argument(
        "--data-path", default=None, help=f"parquet panel (default {config.DATA_PATH})"
    )
    parser.add_argument("--synthetic", action="store_true", help="use synthetic demo data")
    parser.add_argument("--tracking-uri", default=None, help="MLflow URI (env MLFLOW_TRACKING_URI)")
    parser.add_argument("--experiment", default=config.EXPERIMENT_NAME)
    parser.add_argument("--no-register", action="store_true", help="skip model registration")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = run_training(
        data_path=args.data_path,
        synthetic=args.synthetic,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        register=not args.no_register,
    )
    print("\nValidation leaderboard (lower QLIKE is better):")
    for r in sorted(out["results"], key=lambda r: r["val"]["qlike"]):
        print(f"  {r['name']:<18} qlike={r['val']['qlike']:.4f}  rmse={r['val']['rmse']:.4f}")
    if out["champion"]:
        c = out["champion"]
        print(f"\nChampion: {c['name']} -> {config.MODEL_NAME} v{c['version']} @champion")


if __name__ == "__main__":
    main()
