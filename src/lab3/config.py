from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class Lab3Config(BaseModel):
    train_path: Path
    validation_path: Path
    test_path: Path
    report_dir: Path
    model_path: Path
    seed: int = Field(ge=0)
    hidden_sizes: list[int] = Field(min_length=1)
    batch_size: int = Field(gt=0, le=4096)
    chunk_size: int = Field(gt=0, le=10000)
    max_epochs: int = Field(gt=0, le=10000)
    patience: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay_grid: list[float] = Field(min_length=1)
    bootstrap_samples: int = Field(ge=100, le=10000)
    ci_level: float = Field(gt=0.5, lt=1.0)
    metric_tolerance: float = Field(gt=0)
    device_tolerance: float = Field(gt=0)
    accelerator_check: bool = False

    @model_validator(mode="after")
    def validate_ranges(self) -> Lab3Config:
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("all hidden_sizes must be positive")
        if any(value < 0 for value in self.weight_decay_grid):
            raise ValueError("all weight_decay values must be non-negative")
        if self.patience > self.max_epochs:
            raise ValueError("patience must not exceed max_epochs")
        return self


def load_config(path: str | Path = "configs/lab3.yaml") -> Lab3Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Lab3Config.model_validate(payload)
