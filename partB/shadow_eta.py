"""Compare the same 200 fresh ETA requests against both MLflow servers."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

for _root in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
    if (_root / "eta_data.py").exists():
        sys.path.insert(0, str(_root))
        break

from eta_data import FEATURES


def call(session, url, row):
    start = time.perf_counter()
    response = session.post(url, json={"dataframe_records": [row]}, timeout=30)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    body = response.json()
    predictions = body.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise ValueError(f"Unexpected response from {url}: {body}")
    prediction = predictions[0]
    return float(prediction["eta_minutes"]), prediction["model_version"], elapsed_ms


def compare(prod_url, candidate_url):
    rows = pd.read_csv("data/eta_fresh.csv").head(200)[FEATURES].to_dict("records")
    if len(rows) != 200:
        raise ValueError("Expected at least 200 fresh rows")

    prod_values, candidate_values = [], []
    prod_latencies, candidate_latencies = [], []
    with requests.Session() as session:
        for row in rows:
            prod, prod_version, prod_ms = call(session, prod_url, row)
            candidate, cand_version, cand_ms = call(session, candidate_url, row)
            if prod_version != "v1" or cand_version != "v2":
                raise ValueError(
                    f"Unexpected served versions: prod={prod_version}, candidate={cand_version}"
                )
            prod_values.append(prod)
            candidate_values.append(candidate)
            prod_latencies.append(prod_ms)
            candidate_latencies.append(cand_ms)

    differences = np.abs(np.asarray(prod_values) - np.asarray(candidate_values))
    return {
        "requests_per_server": len(rows),
        "prod_version": "v1",
        "candidate_version": "v2",
        "mean_absolute_prediction_difference_minutes": round(float(np.mean(differences)), 3),
        "percent_difference_over_10_minutes": round(float(np.mean(differences > 10) * 100), 2),
        "prod_p95_latency_ms": round(float(np.percentile(prod_latencies, 95)), 3),
        "candidate_p95_latency_ms": round(float(np.percentile(candidate_latencies, 95)), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod-url", default="http://127.0.0.1:5101/invocations")
    parser.add_argument("--candidate-url", default="http://127.0.0.1:5102/invocations")
    args = parser.parse_args()
    print(json.dumps(compare(args.prod_url, args.candidate_url), indent=2))


if __name__ == "__main__":
    main()
