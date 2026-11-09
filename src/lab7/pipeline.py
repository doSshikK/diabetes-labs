from __future__ import annotations

import math
import os
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict, Field

from lab6.config import Lab6Config
from lab6.config import load_config as load_lab6_config
from lab6.schemas import FeatureRequest
from lab6.service import ModelService, create_app
from lab7.config import Lab7Config, load_config


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[FeatureRequest] = Field(min_length=1)
    workers: int = Field(default=1, ge=1)


class BatchResponse(BaseModel):
    predictions: list[float]
    model_version: str
    workers: int
    retried_chunks: int


@dataclass
class BatchResult:
    predictions: list[float]
    model_version: str
    workers: int
    retried_chunks: int


class ParallelBatchPredictor:
    def __init__(self, service: ModelService, max_workers: int) -> None:
        self.service = service
        self.max_workers = min(max_workers, os.cpu_count() or 1)

    def _predict_chunk(
        self,
        chunk_index: int,
        rows: list[FeatureRequest],
        fail_chunk_index: int | None,
    ) -> tuple[int, list[float] | None, str | None]:
        try:
            if chunk_index == fail_chunk_index:
                raise RuntimeError("injected worker failure")
            values, _, _ = self.service.predict_many(rows)
            return chunk_index, values.tolist(), None
        except (RuntimeError, ValueError) as error:
            return chunk_index, None, str(error)

    def predict(
        self,
        rows: list[FeatureRequest],
        workers: int,
        fail_chunk_index: int | None = None,
    ) -> BatchResult:
        if not rows:
            raise ValueError("batch must contain at least one row")
        if workers < 1 or workers > self.max_workers:
            raise ValueError(f"workers must be between 1 and {self.max_workers}")
        version, _, _ = self.service.snapshot_model()
        chunk_size = max(1, math.ceil(len(rows) / workers))
        chunks = [
            rows[index : index + chunk_size]
            for index in range(0, len(rows), chunk_size)
        ]
        results = Parallel(n_jobs=workers, prefer="threads")(
            delayed(self._predict_chunk)(index, chunk, fail_chunk_index)
            for index, chunk in enumerate(chunks)
        )
        retried = 0
        ordered: list[list[float]] = []
        for chunk_index, values, error in sorted(results):
            if error is not None:
                retried += 1
                values = self.service.predict_many(chunks[chunk_index])[0].tolist()
            ordered.append(values or [])
        predictions = [value for chunk in ordered for value in chunk]
        return BatchResult(predictions, version, workers, retried)


def create_batch_app(
    config: Lab7Config | None = None,
    lab6_config: Lab6Config | None = None,
) -> tuple[FastAPI, ParallelBatchPredictor]:
    resolved = config or load_config()
    base_config = lab6_config or load_lab6_config(resolved.lab6_config_path)
    app, service = create_app(base_config, persist_registry=False)
    predictor = ParallelBatchPredictor(service, resolved.max_workers)
    app.title = "Diabetes parallel inference service"
    app.version = "2.0"

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest) -> BatchResponse:
        if len(request.rows) > resolved.max_batch_size:
            raise HTTPException(status_code=413, detail="batch is too large")
        try:
            result = predictor.predict(request.rows, request.workers)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return BatchResponse(
            predictions=result.predictions,
            model_version=result.model_version,
            workers=result.workers,
            retried_chunks=result.retried_chunks,
        )

    return app, predictor
