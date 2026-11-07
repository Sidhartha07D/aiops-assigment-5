"""Register and evaluate SwiftParcel ETA models with MLflow.

Set MLFLOW_TRACKING_URI to a SQLite URI, then run:
  python eta_mlflow.py register
  python eta_mlflow.py gate
  python eta_mlflow.py gate --min-rel-gain 0.60
"""

import argparse
import json
import os
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature


MODEL_NAME = "eta-model"

# The submission keeps this file in partB/ and the supplied starter in the root.
for _root in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
    if (_root / "eta_data.py").exists():
        sys.path.insert(0, str(_root))
        break


def _configure_mlflow():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow_eta.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return uri


class EtaPyfunc(mlflow.pyfunc.PythonModel):
    """Load one supplied bundle and return an inference-friendly DataFrame."""

    def load_context(self, context):
        bundle = joblib.load(context.artifacts["bundle"])
        self.model = bundle["model"]
        self.features = bundle["feature_columns"]
        self.version = bundle["version"]

    def predict(self, context, model_input, params=None):
        frame = pd.DataFrame(model_input)[self.features]
        values = self.model.predict(frame)
        return pd.DataFrame(
            {
                "eta_minutes": np.round(np.asarray(values, dtype=float), 1),
                "model_version": [self.version] * len(frame),
            }
        )


def _pinned_requirements():
    packages = ("mlflow", "scikit-learn", "pandas", "numpy", "joblib", "cloudpickle")
    return [f"{name}=={package_version(name)}" for name in packages]


def register():
    """Log both bundles and point prod/candidate at their registry versions."""
    _configure_mlflow()
    from eta_data import FEATURES, make_data

    mlflow.set_experiment("SwiftParcel ETA")
    client = MlflowClient()
    example = make_data(3, seed=7)[FEATURES]
    rows = []
    for semantic_version, alias in (("v1", "prod"), ("v2", "candidate")):
        path = Path("data") / f"eta_{semantic_version}.joblib"
        bundle = joblib.load(path)
        expected = pd.DataFrame(
            {
                "eta_minutes": np.round(bundle["model"].predict(example), 1),
                "model_version": [semantic_version] * len(example),
            }
        )
        with mlflow.start_run(run_name=f"eta-{semantic_version}"):
            info = mlflow.pyfunc.log_model(
                name=f"eta-{semantic_version}",
                python_model=EtaPyfunc(),
                artifacts={"bundle": str(path)},
                code_paths=[str(Path(__file__).resolve())],
                signature=infer_signature(example, expected),
                input_example=example,
                pip_requirements=_pinned_requirements(),
            )
            registered = mlflow.register_model(info.model_uri, MODEL_NAME)
        client.set_model_version_tag(
            MODEL_NAME, registered.version, "bundle_version", semantic_version
        )
        client.set_registered_model_alias(MODEL_NAME, alias, registered.version)
        rows.append(
            {
                "registry_version": str(registered.version),
                "bundle_version": semantic_version,
                "alias": alias,
            }
        )
    return rows


def decide(
    prod,
    cand,
    min_rel_gain=0.03,
    max_p95_ms=10.0,
    max_slice_regression=0.05,
):
    """Pure all-criteria promotion decision; threshold equality is allowed."""
    criteria = {
        "mae": cand["mae"] <= prod["mae"] * (1 - min_rel_gain),
        "p95_ms": cand["p95_ms"] <= max_p95_ms,
        "mae_rush": cand["mae_rush"]
        <= prod["mae_rush"] * (1 + max_slice_regression),
        "mae_rain": cand["mae_rain"]
        <= prod["mae_rain"] * (1 + max_slice_regression),
    }
    return {"criteria": criteria, "promote": all(criteria.values())}


def _evaluate(model, fresh, features):
    x = fresh[features]
    predicted = model.predict(x)["eta_minutes"].to_numpy(dtype=float)
    true = fresh["eta_minutes"].to_numpy(dtype=float)
    rush = fresh["hour"].isin((7, 8, 9, 16, 17, 18, 19)).to_numpy()
    rain = fresh["rain"].eq(1).to_numpy()

    # Warm up separately from the 200 measured, single-row calls.
    model.predict(x.iloc[:1])
    single_row_ms = []
    for index in range(200):
        start = time.perf_counter()
        model.predict(x.iloc[index : index + 1])
        single_row_ms.append((time.perf_counter() - start) * 1000)

    return {
        "mae": float(np.mean(np.abs(true - predicted))),
        "mae_rush": float(np.mean(np.abs(true[rush] - predicted[rush]))),
        "mae_rain": float(np.mean(np.abs(true[rain] - predicted[rain]))),
        "p95_ms": float(np.percentile(single_row_ms, 95)),
    }


def promotion_gate(**thresholds):
    """Evaluate the aliases on the specified fresh set, then apply decide()."""
    _configure_mlflow()
    from eta_data import FEATURES, make_data

    fresh = make_data(2000, seed=99)
    prod_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@prod")
    cand_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@candidate")
    prod_metrics = _evaluate(prod_model, fresh, FEATURES)
    cand_metrics = _evaluate(cand_model, fresh, FEATURES)
    result = decide(prod_metrics, cand_metrics, **thresholds)
    return {"prod": prod_metrics, "candidate": cand_metrics, **result}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("register")
    gate = commands.add_parser("gate")
    gate.add_argument("--min-rel-gain", type=float, default=0.03)
    gate.add_argument("--max-p95-ms", type=float, default=10.0)
    gate.add_argument("--max-slice-regression", type=float, default=0.05)
    args = parser.parse_args()
    if args.command == "register":
        print(json.dumps(register(), indent=2))
    else:
        print(
            json.dumps(
                promotion_gate(
                    min_rel_gain=args.min_rel_gain,
                    max_p95_ms=args.max_p95_ms,
                    max_slice_regression=args.max_slice_regression,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
