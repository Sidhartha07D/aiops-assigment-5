"""Ray Serve ETA application with stable customer-level canary routing."""

import hashlib
import time
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray import serve


api = FastAPI(title="SwiftParcel ETA canary")


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
            raise ValueError("infeasible delivery")
        return self


class CanaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: Annotated[str, Field(min_length=1)]
    instances: Annotated[list[EtaInstance], Field(min_length=1, max_length=500)]


def customer_bucket(customer_id: str) -> float:
    """A stable value in [0, 1), independent of Python's salted hash()."""
    digest = hashlib.sha256(customer_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (1 << 64)


@serve.deployment(ray_actor_options={"num_cpus": 0.25}, max_ongoing_requests=4)
class EtaModel:
    def __init__(self, bundle_path: str, extra_work_ms: float = 0.0):
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.features = bundle["feature_columns"]
        self.version = bundle["version"]
        self.extra_work_ms = extra_work_ms

    def predict(self, rows: list[dict]):
        if self.extra_work_ms:
            time.sleep(self.extra_work_ms / 1000)
        frame = pd.DataFrame.from_records(rows, columns=self.features)
        return {
            "predictions": [round(float(value), 1) for value in self.model.predict(frame)],
            "model_version": self.version,
        }


@serve.deployment(ray_actor_options={"num_cpus": 0.25})
@serve.ingress(api)
class EtaRouter:
    def __init__(self, prod, candidate, weight: float):
        self.prod = prod
        self.candidate = candidate
        self.weight = weight

    @api.post("/v1/models/eta:predict")
    async def predict(
        self,
        payload: CanaryRequest,
        x_beta: Annotated[str | None, Header(alias="X-Beta")] = None,
    ):
        force_candidate = x_beta == "1"
        arm = "v2" if force_candidate or customer_bucket(payload.customer_id) < self.weight else "v1"
        handle = self.candidate if arm == "v2" else self.prod
        result = await handle.predict.remote([row.model_dump() for row in payload.instances])
        return {**result, "arm": arm}

    @api.get("/healthz")
    def healthz(self):
        return {"status": "ok", "candidate_weight": self.weight}


def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    """Build an application whose maximum replica CPU reservation is 1.5/2."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")
    if extra_work_ms < 0:
        raise ValueError("extra_work_ms must be nonnegative")

    prod = EtaModel.options(name="EtaProd", num_replicas=2).bind(
        "data/eta_v1.joblib", extra_work_ms
    )
    candidate = EtaModel.options(
        name="EtaCandidate",
        num_replicas="auto",
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 2,
            "upscale_delay_s": 5,
            "downscale_delay_s": 30,
        },
    ).bind("data/eta_v2.joblib", extra_work_ms)
    return EtaRouter.bind(prod, candidate, weight)
