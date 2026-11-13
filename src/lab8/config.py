from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class Lab8Config(BaseModel):
    real_data_path: Path
    synthetic_data_path: Path
    report_dir: Path
    model_dir: Path
    seed: int = Field(ge=0)
    test_size: float = Field(gt=0.0, lt=0.5)

    ml_alpha: float = Field(ge=0.0)
    ml_l1_ratio: float = Field(ge=0.0, le=1.0)
    dl_hidden_sizes: list[int] = Field(min_length=1)
    dl_epochs: int = Field(ge=1, le=10_000)
    dl_learning_rate: float = Field(gt=0.0)
    validator_quantile: float = Field(gt=0.5, lt=1.0)

    generated_rows: int = Field(ge=100)
    drift_mean_shift_std: float = Field(gt=0.0)
    psi_bins: int = Field(ge=5, le=100)
    psi_alert_threshold: float = Field(gt=0.0)
    validator_rejection_alert_threshold: float = Field(gt=0.0, le=1.0)
    service_error_rate_alert_threshold: float = Field(gt=0.0, le=1.0)
    alert_deduplication_seconds: int = Field(ge=1)

    minimum_requests_per_stage: int = Field(ge=10_000)
    warmup_duration_seconds: int = Field(ge=300)
    stage_duration_seconds: int = Field(ge=600)
    resource_sample_interval_seconds: float = Field(gt=0.0, le=2.0)
    lab6_p95_seconds: float = Field(gt=0.0)
    latency_limit_multiplier: float = Field(ge=1.0)
    maximum_false_alerts: int = Field(ge=0, le=1)
    maximum_batch_error_rate: float = Field(ge=0.0, le=0.05)
    maximum_health_detection_seconds: int = Field(ge=1, le=30)
    maximum_recovery_seconds: int = Field(ge=1, le=120)

    @model_validator(mode="after")
    def validate_model(self) -> Lab8Config:
        if any(size <= 0 for size in self.dl_hidden_sizes):
            raise ValueError("all dl_hidden_sizes values must be positive")
        if self.model_dir == self.report_dir:
            raise ValueError("model_dir must be a child directory, not report_dir")
        return self


def load_config(path: str | Path = "configs/lab8.yaml") -> Lab8Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Lab8Config.model_validate(payload)
