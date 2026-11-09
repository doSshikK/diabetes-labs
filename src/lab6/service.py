from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.metrics import sha256_file
from lab6.config import Lab6Config, ModelVersionConfig, load_config
from lab6.schemas import FeatureRequest, PredictionResponse, PromoteRequest


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ModelService:
    def __init__(self, config: Lab6Config, persist_registry: bool = True) -> None:
        self.config = config
        self.persist_registry = persist_registry
        self._lock = threading.RLock()
        self._models: dict[str, dict[str, Any]] = {}
        self._versions = {item.version: item for item in config.versions}
        self._active_version = config.initial_version
        self._rollback_stack: list[str] = []
        self._history: list[dict[str, Any]] = []
        self._started = time.perf_counter()
        self._requests = 0
        self._successes = 0
        self._failures = 0
        for item in config.versions:
            self._models[item.version] = self._load_checked(item)
        self._record(
            "initialize", None, self._active_version, "configured initial version"
        )
        if self.persist_registry:
            self.save_registry()

    @staticmethod
    def _load_checked(item: ModelVersionConfig) -> dict[str, Any]:
        actual = sha256_file(item.path)
        if actual != item.sha256:
            raise ValueError(
                f"hash mismatch for {item.version}: expected {item.sha256}, got {actual}"
            )
        bundle = joblib.load(item.path)
        required = {"scaler", "model", "features"}
        if not required.issubset(bundle):
            raise ValueError(f"model bundle {item.version} misses required fields")
        return bundle

    def _record(
        self, action: str, previous: str | None, current: str, reason: str
    ) -> None:
        self._history.append(
            {
                "timestamp": utc_now(),
                "action": action,
                "previous": previous,
                "current": current,
                "reason": reason,
            }
        )

    def snapshot_model(self) -> tuple[str, ModelVersionConfig, dict[str, Any]]:
        with self._lock:
            version = self._active_version
            return version, self._versions[version], self._models[version]

    def predict_many(
        self, requests: list[FeatureRequest]
    ) -> tuple[np.ndarray, str, str]:
        version, item, bundle = self.snapshot_model()
        try:
            values = np.asarray(
                [request.values_for(bundle["features"]) for request in requests],
                dtype=float,
            )
            transformed = bundle["scaler"].transform(values)
            predictions = bundle["model"].predict(transformed).astype(float)
            with self._lock:
                self._requests += len(requests)
                self._successes += len(requests)
            return predictions, version, item.sha256
        except Exception:
            with self._lock:
                self._requests += len(requests)
                self._failures += len(requests)
            raise

    def predict(self, request: FeatureRequest) -> PredictionResponse:
        predictions, version, digest = self.predict_many([request])
        return PredictionResponse(
            prediction=float(predictions[0]),
            model_version=version,
            model_sha256=digest,
        )

    def promote(self, version: str, reason: str = "manual promotion") -> str:
        with self._lock:
            if version not in self._versions:
                raise KeyError(version)
            previous = self._active_version
            if version == previous:
                return version
            self._rollback_stack.append(previous)
            self._versions[previous].status = "retired"
            self._versions[version].status = "working"
            self._active_version = version
            self._record("promote", previous, version, reason)
            if self.persist_registry:
                self.save_registry()
            return version

    def rollback(self, reason: str = "manual rollback") -> str:
        with self._lock:
            if not self._rollback_stack:
                raise RuntimeError("rollback history is empty")
            previous = self._active_version
            restored = self._rollback_stack.pop()
            self._versions[previous].status = "canary"
            self._versions[restored].status = "working"
            self._active_version = restored
            self._record("rollback", previous, restored, reason)
            if self.persist_registry:
                self.save_registry()
            return restored

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_version": self._active_version,
                "requests": self._requests,
                "successes": self._successes,
                "failures": self._failures,
                "success_rate": self._successes / max(self._requests, 1),
                "uptime_seconds": time.perf_counter() - self._started,
            }

    def registry_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_version": self._active_version,
                "versions": [
                    item.model_dump(mode="json") for item in self._versions.values()
                ],
                "history": list(self._history),
            }

    def save_registry(self) -> None:
        self.config.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.registry_path.write_text(
            json.dumps(self.registry_payload(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def create_app(
    config: Lab6Config | None = None,
    persist_registry: bool = True,
) -> tuple[FastAPI, ModelService]:
    resolved = config or load_config()
    service = ModelService(resolved, persist_registry=persist_registry)
    app = FastAPI(title="Diabetes model service", version="1.0")
    app.state.model_service = service

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "active_version": service.metrics()["active_version"]}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: FeatureRequest) -> PredictionResponse:
        return service.predict(request)

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return service.metrics()

    @app.get("/models")
    def models() -> dict[str, Any]:
        return service.registry_payload()

    @app.post("/promote")
    def promote(request: PromoteRequest) -> dict[str, str]:
        try:
            active = service.promote(request.version)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="unknown model version"
            ) from error
        return {"active_version": active}

    @app.post("/rollback")
    def rollback() -> dict[str, str]:
        try:
            active = service.rollback()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"active_version": active}

    return app, service


def create_monitoring_app(service: ModelService) -> FastAPI:
    app = FastAPI(title="Diabetes service monitor", version="1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/snapshot")
    def snapshot() -> dict[str, Any]:
        return service.metrics()

    return app


def write_monitoring_artifacts(samples: list[dict[str, Any]], report_dir: Path) -> None:
    frame = pd.DataFrame(samples)
    frame.to_csv(report_dir / "service_metrics.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(frame["sample"], frame["success_rate"], marker="o", label="success rate")
    axis.set_ylim(0.995, 1.0001)
    axis.set_xlabel("Sample")
    axis.set_ylabel("Success rate")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "service_monitor.png", dpi=160)
    plt.close(figure)
