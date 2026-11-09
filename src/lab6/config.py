from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelVersionConfig(BaseModel):
    version: str = Field(min_length=1)
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, float]
    status: Literal["preparing", "canary", "working", "retired"]


class Lab6Config(BaseModel):
    report_dir: Path
    registry_path: Path
    raw_data_path: Path
    initial_version: str
    feature_min: float
    feature_max: float
    load_requests_per_endpoint: int = Field(ge=1, le=10_000)
    drill_requests: int = Field(ge=100, le=100_000)
    max_load_workers: int = Field(ge=1, le=64)
    versions: list[ModelVersionConfig] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_versions(self) -> Lab6Config:
        names = [item.version for item in self.versions]
        if len(names) != len(set(names)):
            raise ValueError("model versions must be unique")
        if self.initial_version not in names:
            raise ValueError("initial_version must exist in versions")
        if self.feature_min >= self.feature_max:
            raise ValueError("feature_min must be lower than feature_max")
        return self


def load_config(path: str | Path = "configs/lab6.yaml") -> Lab6Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Lab6Config.model_validate(payload)
