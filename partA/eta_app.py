"""SwiftParcel ETA REST service.

Run from the assignment root with ``uvicorn eta_app:app --port 8000``.
"""

import os
import threading
import time
import uuid
from collections import Counter
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


MODEL_PATH = os.getenv("ETA_MODEL", "data/eta_v1.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["feature_columns"]
MODEL_VERSION = bundle["version"]

app = FastAPI(title="SwiftParcel ETA")


class EtaInstance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_km: Annotated[float, Field(strict=True, ge=0.1, le=500)]
    n_stops: Annotated[int, Field(strict=True, ge=1, le=60)]
    hour: Annotated[int, Field(strict=True, ge=0, le=23)]
    weekday: Annotated[int, Field(strict=True, ge=0, le=6)]
    traffic_index: Annotated[float, Field(strict=True, ge=0, le=1)]
    rain: Literal[0, 1]
    weight_kg: Annotated[float, Field(strict=True, gt=0, le=1000)]

    @model_validator(mode="after")
    def feasible_delivery(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance and stops are both too high")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: Annotated[list[EtaInstance], Field(min_length=1, max_length=500)]


_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0)
_lock = threading.Lock()
_requests = Counter()
_bucket_counts = Counter()
_latency_sum = 0.0
_latency_count = 0


def _request_id(request: Request) -> str:
    return request.state.request_id


@app.middleware("http")
async def request_tracking(request: Request, call_next):
    global _latency_sum, _latency_count
    request.state.request_id = request.headers.get("x-request-id")
    if request.state.request_id is None:
        request.state.request_id = str(uuid.uuid4())

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        response = JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "request_id": _request_id(request)},
        )

    response.headers["X-Request-ID"] = _request_id(request)
    if request.url.path != "/metrics":
        elapsed = time.perf_counter() - started
        with _lock:
            _requests[(request.url.path, str(response.status_code))] += 1
            _latency_sum += elapsed
            _latency_count += 1
            for boundary in _BUCKETS:
                if elapsed <= boundary:
                    _bucket_counts[boundary] += 1
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    details = [
        {"field": ".".join(str(part) for part in error["loc"]), "msg": error["msg"]}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_failed",
            "details": details,
            "request_id": _request_id(request),
        },
    )


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail), "request_id": _request_id(request)},
    )


@app.post("/v1/models/eta:predict")
def predict(payload: PredictRequest, request: Request):
    rows = [item.model_dump() for item in payload.instances]
    frame = pd.DataFrame.from_records(rows, columns=FEATURES)
    predictions = [round(float(value), 1) for value in model.predict(frame)]
    return {
        "predictions": predictions,
        "model_version": MODEL_VERSION,
        "request_id": _request_id(request),
    }


@app.get("/healthz")
def healthz(request: Request):
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "request_id": _request_id(request),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    with _lock:
        requests = dict(_requests)
        buckets = dict(_bucket_counts)
        latency_sum = _latency_sum
        latency_count = _latency_count

    lines = [
        "# HELP eta_requests_total HTTP requests served by path and status.",
        "# TYPE eta_requests_total counter",
    ]
    for (path, status), count in sorted(requests.items()):
        safe_path = path.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'eta_requests_total{{path="{safe_path}",status="{status}"}} {count}')

    lines.extend(
        [
            "# HELP eta_request_latency_seconds HTTP request latency in seconds.",
            "# TYPE eta_request_latency_seconds histogram",
        ]
    )
    for boundary in _BUCKETS:
        lines.append(
            f'eta_request_latency_seconds_bucket{{le="{boundary:g}"}} '
            f'{buckets.get(boundary, 0)}'
        )
    lines.append(f'eta_request_latency_seconds_bucket{{le="+Inf"}} {latency_count}')
    lines.append(f"eta_request_latency_seconds_sum {latency_sum:.9f}")
    lines.append(f"eta_request_latency_seconds_count {latency_count}")
    return "\n".join(lines) + "\n"
