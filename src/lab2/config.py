from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Lab2Config(BaseModel):
    """Validated configuration for the full and chunked ML pipeline."""

    data_path: Path
    processed_dir: Path
    report_dir: Path
    model_path: Path
    scaler: Literal["standard"]
    model: Literal["sgd_regressor"]
    loss: Literal["squared_error"]
    penalty: Literal["elasticnet"]
    seed: int = Field(ge=0)
    chunk_size: int = Field(gt=0, le=10000)
    test_size: float = Field(gt=0.0, lt=0.5)
    validation_size: float = Field(gt=0.0, lt=0.5)
    epochs: int = Field(gt=0, le=10000)
    eta0: float = Field(gt=0.0)
    alpha_grid: list[float] = Field(min_length=1)
    l1_ratio_grid: list[float] = Field(min_length=1)
    elastic_net_max_iter: int = Field(gt=0)
    elastic_net_tol: float = Field(gt=0.0)
    bootstrap_samples: int = Field(ge=100, le=10000)
    ci_level: float = Field(gt=0.5, lt=1.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> Lab2Config:
        if self.test_size + self.validation_size >= 0.8:
            raise ValueError("test_size + validation_size must be less than 0.8")
        if any(value < 0 for value in self.alpha_grid):
            raise ValueError("all alpha_grid values must be non-negative")
        if any(not 0 <= value <= 1 for value in self.l1_ratio_grid):
            raise ValueError("all l1_ratio_grid values must be between 0 and 1")
        return self


def load_config(path: str | Path = "configs/lab2.yaml") -> Lab2Config:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Lab2Config.model_validate(payload)
