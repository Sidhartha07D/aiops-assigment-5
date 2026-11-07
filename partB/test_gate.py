"""Boundary-aware tests for the pure MLflow promotion decision."""

from copy import deepcopy

from eta_mlflow import decide


PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 8.0, "p95_ms": 4.0}
CAND = {"mae": 9.7, "mae_rush": 12.6, "mae_rain": 8.4, "p95_ms": 10.0}


def test_all_thresholds_exactly_on_boundary_promote():
    result = decide(PROD, CAND)
    assert result["promote"] is True
    assert all(result["criteria"].values())


def test_gain_just_inside_threshold_promotes():
    candidate = {**CAND, "mae": 9.6999}
    assert decide(PROD, candidate)["promote"] is True


def test_gain_just_outside_threshold_blocks():
    candidate = {**CAND, "mae": 9.7001}
    result = decide(PROD, candidate)
    assert result["promote"] is False
    assert result["criteria"]["mae"] is False


def test_latency_just_inside_threshold_promotes():
    candidate = {**CAND, "p95_ms": 9.9999}
    assert decide(PROD, candidate)["promote"] is True


def test_latency_just_outside_threshold_blocks():
    candidate = {**CAND, "p95_ms": 10.0001}
    result = decide(PROD, candidate)
    assert result["promote"] is False
    assert result["criteria"]["p95_ms"] is False


def test_rush_slice_just_outside_threshold_blocks():
    candidate = {**CAND, "mae_rush": 12.6001}
    result = decide(PROD, candidate)
    assert result["promote"] is False
    assert result["criteria"]["mae_rush"] is False


def test_rain_slice_just_outside_threshold_blocks():
    candidate = {**CAND, "mae_rain": 8.4001}
    result = decide(PROD, candidate)
    assert result["promote"] is False
    assert result["criteria"]["mae_rain"] is False


def test_stricter_gain_blocks_and_inputs_remain_unchanged():
    prod, cand = deepcopy(PROD), deepcopy(CAND)
    result = decide(prod, cand, min_rel_gain=0.60)
    assert result["promote"] is False
    assert prod == PROD and cand == CAND
