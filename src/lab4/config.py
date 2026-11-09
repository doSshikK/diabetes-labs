from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Lab4Config(BaseModel):
    train_path: Path
    test_path: Path
    model_path: Path
    report_dir: Path
    validator_path: Path
    chunk_size: int = Field(gt=0, le=10000)
    distance_quantile: float = Field(gt=0.5, lt=1.0)
    min_model_confidence: float = Field(gt=0.0, lt=1.0)
    covariance_regularization: float = Field(gt=0.0)
    mean_shift_std: float = Field(gt=0.0)
    false_rejection_limit: float = Field(ge=0.0, lt=1.0)
    ood_recall_threshold: float = Field(gt=0.0, le=1.0)
    statistics_tolerance: float = Field(gt=0.0)
    seed: int = Field(ge=0)


def load_config(path: str | Path = "configs/lab4.yaml") -> Lab4Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Lab4Config.model_validate(payload)
