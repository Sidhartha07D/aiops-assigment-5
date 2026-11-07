"""Deploy on two Ray CPUs and prove all five canary properties."""

import argparse
import json

import ray
import requests
from ray import serve

from eta_ray import build


URL = "http://127.0.0.1:8000/v1/models/eta:predict"
ROW = {
    "distance_km": 12.0,
    "n_stops": 3,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.4,
    "rain": 0,
    "weight_kg": 5.0,
}


def route(session, customer_id, beta=False):
    headers = {"X-Beta": "1"} if beta else None
    response = session.post(
        URL,
        json={"customer_id": customer_id, "instances": [ROW]},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    assert body["arm"] in ("v1", "v2")
    assert body["model_version"] == body["arm"]
    assert len(body["predictions"]) == 1
    return body["arm"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-running", action="store_true", help="Pause for a localhost screenshot before shutdown"
    )
    args = parser.parse_args()

    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=128 * 1024 * 1024)
    try:
        cpus = ray.cluster_resources()["CPU"]
        assert cpus == 2.0, f"Expected two Ray CPUs, got {cpus}"
        print(f"Ray CPUs: {cpus}", flush=True)

        serve.run(build(weight=0.2), name="swiftparcel", route_prefix="/")
        ids = [f"customer-{number:04d}" for number in range(1000)]
        with requests.Session() as session:
            first = {customer_id: route(session, customer_id) for customer_id in ids}
            candidate_count = sum(arm == "v2" for arm in first.values())
            share = candidate_count / len(ids)
            assert abs(share - 0.2) <= 0.04, f"Candidate share {share:.3f} outside ±0.04"
            print(
                f"PASS share: {candidate_count}/1000 = {share:.1%} candidate at weight 0.2",
                flush=True,
            )

            second_ids = ids[:200]
            assert all(route(session, customer_id) == first[customer_id] for customer_id in second_ids)
            print("PASS stickiness: 200/200 repeat calls stayed on the same arm", flush=True)

            serve.run(build(weight=0.2), name="swiftparcel", route_prefix="/")
            assert all(route(session, customer_id) == first[customer_id] for customer_id in second_ids)
            print("PASS identical redeploy: 200/200 stayed on the same arm", flush=True)

            serve.run(build(weight=0.5), name="swiftparcel", route_prefix="/")
            after = {customer_id: route(session, customer_id) for customer_id in ids}
            moved_back = sum(
                first[customer_id] == "v2" and after[customer_id] == "v1"
                for customer_id in ids
            )
            assert moved_back == 0
            new_share = sum(arm == "v2" for arm in after.values()) / len(ids)
            print(
                f"PASS monotonic increase: 0/1000 candidate customers moved back; "
                f"new candidate share {new_share:.1%}",
                flush=True,
            )

            prod_ids = [customer_id for customer_id in ids if first[customer_id] == "v1"][:50]
            assert len(prod_ids) == 50
            assert all(route(session, customer_id, beta=True) == "v2" for customer_id in prod_ids)
            print("PASS X-Beta override: 50/50 prod customers received v2", flush=True)

        print("All five Ray canary checks passed.", flush=True)
        if args.keep_running:
            input("Open http://127.0.0.1:8000/docs, then press Enter here to stop Ray Serve. ")
    finally:
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
