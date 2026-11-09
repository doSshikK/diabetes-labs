from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Lab5Config(BaseModel):
    real_data_path: Path
    report_dir: Path
    output_dir: Path
    n_samples: int = Field(gt=0, le=1_000_000)
    seed: int = Field(ge=0)
    similar_jitter_scale: float = Field(ge=0.0, le=1.0)
    mean_shift_std: float = Field(ge=-20.0, le=20.0)
    covariance_scale: float = Field(gt=0.0, le=20.0)
    target_noise_std: float = Field(ge=0.0, le=20.0)
    label_flip_fraction: float = Field(ge=0.0, le=1.0)
    psi_bins: int = Field(ge=3, le=100)
    similar_threshold: float = Field(gt=0.0)
    random_psi_threshold: float = Field(gt=0.0)
    similar_required_fraction: float = Field(gt=0.0, le=1.0)
    random_required_fraction: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_output(self) -> Lab5Config:
        if self.real_data_path == self.output_dir:
            raise ValueError("output_dir must differ from real_data_path")
        return self


class GenerationRequest(BaseModel):
    mode: Literal["similar", "random"]
    n_samples: int = Field(gt=0, le=1_000_000)
    seed: int = Field(ge=0)
    mean_shift_std: float = Field(ge=-20.0, le=20.0)
    covariance_scale: float = Field(gt=0.0, le=20.0)
    target_noise_std: float = Field(ge=0.0, le=20.0)
    label_flip_fraction: float = Field(ge=0.0, le=1.0)


def load_config(path: str | Path = "configs/lab5.yaml") -> Lab5Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Lab5Config.model_validate(payload)
