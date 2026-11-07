"""Contract and failure-path tests for the SwiftParcel REST service."""

import uuid

import pytest
from fastapi.testclient import TestClient

import eta_app


client = TestClient(eta_app.app)
URL = "/v1/models/eta:predict"
ROW = {
    "distance_km": 12.0,
    "n_stops": 3,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.4,
    "rain": 0,
    "weight_kg": 5.0,
}


def post(rows, headers=None):
    return client.post(URL, json={"instances": rows}, headers=headers)


def assert_validation(response):
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_failed"
    assert body["details"]
    assert all("field" in item and "msg" in item for item in body["details"])
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_valid_request_returns_rounded_minutes_and_version():
    response = post([ROW])
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "v1"
    assert len(body["predictions"]) == 1
    value = body["predictions"][0]
    assert isinstance(value, (float, int))
    assert value == round(value, 1)
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_batch_order_is_preserved():
    distant = {**ROW, "distance_km": 80.0, "n_stops": 10}
    forward = post([ROW, distant])
    reverse = post([distant, ROW])
    assert forward.status_code == reverse.status_code == 200
    predictions = forward.json()["predictions"]
    assert predictions[0] != predictions[1]
    assert reverse.json()["predictions"] == predictions[::-1]


def test_missing_feature_is_rejected():
    row = {key: value for key, value in ROW.items() if key != "weight_kg"}
    response = post([row])
    assert_validation(response)
    assert any("weight_kg" in item["field"] for item in response.json()["details"])


def test_out_of_range_distance_is_rejected():
    assert_validation(post([{**ROW, "distance_km": 0.0}]))


def test_wrong_type_is_rejected():
    assert_validation(post([{**ROW, "hour": "ten"}]))


def test_invalid_rain_value_is_rejected():
    assert_validation(post([{**ROW, "rain": 2}]))


def test_infeasible_combination_is_rejected():
    response = post([{**ROW, "distance_km": 150.1, "n_stops": 21}])
    assert_validation(response)
    assert "infeasible" in str(response.json()["details"]).lower()


def test_empty_batch_is_rejected():
    assert_validation(post([]))


def test_batch_larger_than_500_is_rejected():
    assert_validation(post([ROW] * 501))


def test_batch_of_500_is_accepted():
    response = post([ROW] * 500)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 500


def test_request_id_is_echoed_on_success():
    response = post([ROW], headers={"X-Request-ID": "student-check-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "student-check-123"
    assert response.json()["request_id"] == "student-check-123"


def test_request_id_is_generated_when_absent():
    response = post([ROW])
    generated = response.headers["X-Request-ID"]
    assert str(uuid.UUID(generated)) == generated
    assert response.json()["request_id"] == generated


def test_request_id_is_echoed_on_validation_error():
    response = post([], headers={"X-Request-ID": "bad-request-123"})
    assert_validation(response)
    assert response.headers["X-Request-ID"] == "bad-request-123"


def test_metrics_have_counter_histogram_and_do_not_count_themselves():
    post([ROW])
    first = client.get("/metrics")
    second = client.get("/metrics")
    assert first.status_code == second.status_code == 200
    assert "eta_requests_total{path=\"/v1/models/eta:predict\",status=\"200\"}" in first.text
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in first.text
    assert "eta_request_latency_seconds_sum" in first.text
    assert "eta_request_latency_seconds_count" in first.text
    assert first.text == second.text
    assert 'path="/metrics"' not in second.text


def test_healthz_reports_loaded_version():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"] == "v1"


def test_inference_failure_returns_generic_500(monkeypatch):
    class FailingModel:
        def predict(self, frame):
            raise RuntimeError("private model detail must not leak")

    monkeypatch.setattr(eta_app, "model", FailingModel())
    response = post([ROW], headers={"X-Request-ID": "failed-inference"})
    assert response.status_code == 500
    assert response.json() == {
        "error": "internal_server_error",
        "request_id": "failed-inference",
    }
    assert response.headers["X-Request-ID"] == "failed-inference"
    assert "private model detail" not in response.text


@pytest.mark.parametrize(
    "changes",
    [
        {"n_stops": 0},
        {"hour": 24},
        {"weekday": 7},
        {"traffic_index": 1.1},
        {"weight_kg": 0},
    ],
)
def test_remaining_range_boundaries(changes):
    assert_validation(post([{**ROW, **changes}]))
