"""Release smoke test for the SwiftParcel REST ETA service.

Exit 0 only when all checks pass; exit 1 on any failed check.
"""

import argparse
import math
import sys
import time

import numpy as np
import requests


ROW = {
    "distance_km": 12.0,
    "n_stops": 3,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.4,
    "rain": 0,
    "weight_kg": 5.0,
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}", flush=True)


def endpoint_from_url(url):
    url = url.rstrip("/")
    suffix = "/v1/models/eta:predict"
    if not url.endswith(suffix):
        url += suffix
    return url, url[: -len(suffix)]


def predict(session, url, rows, headers=None):
    start = time.perf_counter()
    response = session.post(url, json={"instances": rows}, headers=headers, timeout=10)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return response, elapsed_ms


def run(url, expect_version, max_p95_ms):
    endpoint, base = endpoint_from_url(url)
    with requests.Session() as session:
        health = session.get(base + "/healthz", timeout=10)
        check(
            health.status_code == 200
            and health.json().get("status") == "ok"
            and health.json().get("model_version") == expect_version,
            f"healthz reports {expect_version}",
        )

        baseline, _ = predict(session, endpoint, [ROW])
        body = baseline.json()
        value = body.get("predictions", [None])[0]
        check(
            baseline.status_code == 200
            and body.get("model_version") == expect_version
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0
            and value == round(value, 1),
            "valid prediction, version, and rounded positive ETA",
        )

        farther, _ = predict(session, endpoint, [{**ROW, "distance_km": 40.0}])
        check(
            farther.status_code == 200 and farther.json()["predictions"][0] > value,
            "sanity: longer distance increases ETA",
        )

        heavier_traffic, _ = predict(session, endpoint, [{**ROW, "traffic_index": 0.9}])
        check(
            heavier_traffic.status_code == 200
            and heavier_traffic.json()["predictions"][0] > value,
            "sanity: heavier traffic increases ETA",
        )

        forward, _ = predict(session, endpoint, [ROW, {**ROW, "distance_km": 40.0}])
        reverse, _ = predict(session, endpoint, [{**ROW, "distance_km": 40.0}, ROW])
        check(
            forward.status_code == reverse.status_code == 200
            and forward.json()["predictions"] == reverse.json()["predictions"][::-1],
            "batch predictions preserve input order",
        )

        invalid, _ = predict(session, endpoint, [{**ROW, "distance_km": 0.0}])
        check(
            invalid.status_code == 422
            and invalid.json().get("error") == "validation_failed"
            and invalid.headers.get("X-Request-ID") == invalid.json().get("request_id"),
            "invalid input returns 422 with matching request ID",
        )

        echoed, _ = predict(
            session, endpoint, [ROW], headers={"X-Request-ID": "smoke-echo-001"}
        )
        check(
            echoed.status_code == 200
            and echoed.headers.get("X-Request-ID") == "smoke-echo-001"
            and echoed.json().get("request_id") == "smoke-echo-001",
            "client request ID is echoed in header and body",
        )

        latencies = []
        for _ in range(30):
            response, elapsed_ms = predict(session, endpoint, [ROW])
            if response.status_code != 200:
                raise AssertionError(f"latency probe returned HTTP {response.status_code}")
            latencies.append(elapsed_ms)
        p95 = float(np.percentile(latencies, 95))
        check(p95 <= max_p95_ms, f"30-request p95 {p95:.2f} ms <= {max_p95_ms:.2f} ms")

    print("Smoke gate passed: release may proceed.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Server base URL or full prediction URL")
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-p95-ms", type=float, default=80.0)
    args = parser.parse_args()
    try:
        run(args.url, args.expect_version, args.max_p95_ms)
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
