from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class Lab7Config(BaseModel):
    lab6_config_path: Path
    report_dir: Path
    real_data_path: Path
    max_batch_size: int = Field(ge=1, le=100_000)
    max_workers: int = Field(ge=1)
    benchmark_rows: int = Field(ge=100)
    benchmark_workers: list[int] = Field(min_length=2)
    benchmark_warmups: int = Field(ge=1, le=20)
    benchmark_repeats: int = Field(ge=2, le=50)
    window_size: int = Field(ge=100)
    clean_windows: int = Field(ge=2)
    drift_windows: int = Field(ge=2)
    mean_shift_std: float = Field(gt=0.0)
    psi_bins: int = Field(ge=5, le=50)
    psi_threshold: float = Field(gt=0.0)
    ks_threshold: float = Field(gt=0.0, le=1.0)
    csi_threshold: float = Field(gt=0.0)
    feature_alert_fraction: float = Field(gt=0.0, le=1.0)
    max_detection_windows: int = Field(ge=1)
    seed: int

    @model_validator(mode="after")
    def validate_workers(self) -> Lab7Config:
        cpu_count = os.cpu_count() or 1
        if self.max_workers > cpu_count:
            raise ValueError("max_workers must not exceed available CPU cores")
        if any(
            worker < 1 or worker > self.max_workers for worker in self.benchmark_workers
        ):
            raise ValueError("benchmark_workers must be between 1 and max_workers")
        if len(set(self.benchmark_workers)) != len(self.benchmark_workers):
            raise ValueError("benchmark_workers must be unique")
        return self


def load_config(path: str | Path = "configs/lab7.yaml") -> Lab7Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Lab7Config.model_validate(payload)
