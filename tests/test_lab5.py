from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from lab5.config import GenerationRequest, Lab5Config, load_config
from lab5.generator import (
    csv_bytes,
    distribution_metrics,
    feature_columns,
    generate,
    psi,
    request_from_config,
    run_lab5,
)


@pytest.fixture
def real_data() -> pd.DataFrame:
    return pd.read_csv(load_config().real_data_path)


def test_config_rejects_bad_values() -> None:
    payload = load_config().model_dump(mode="python")
    for update in ({"n_samples": 0}, {"label_flip_fraction": 1.2}, {"psi_bins": 2}):
        with pytest.raises(ValidationError):
            Lab5Config.model_validate({**payload, **update})
    with pytest.raises(ValidationError):
        GenerationRequest.model_validate(
            {
                "mode": "unknown",
                "n_samples": 10,
                "seed": 1,
                "mean_shift_std": 0,
                "covariance_scale": 1,
                "target_noise_std": 0,
                "label_flip_fraction": 0,
            }
        )


def test_same_seed_is_byte_reproducible(real_data: pd.DataFrame) -> None:
    config = load_config()
    request = request_from_config(config, "similar")
    first = csv_bytes(generate(real_data, request, config.similar_jitter_scale))
    second = csv_bytes(generate(real_data, request, config.similar_jitter_scale))
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_modes_have_schema_ranges_and_different_distributions(
    real_data: pd.DataFrame,
) -> None:
    config = load_config()
    similar = generate(
        real_data, request_from_config(config, "similar"), config.similar_jitter_scale
    )
    random = generate(
        real_data, request_from_config(config, "random"), config.similar_jitter_scale
    )
    assert list(similar.columns) == list(real_data.columns)
    assert list(random.columns) == list(real_data.columns)
    for feature in feature_columns(real_data):
        assert (
            similar[feature]
            .between(real_data[feature].min(), real_data[feature].max())
            .all()
        )
        assert (
            random[feature]
            .between(real_data[feature].min(), real_data[feature].max())
            .all()
        )
    metrics = pd.DataFrame(
        distribution_metrics(real_data, random, "random", config.psi_bins)
    )
    assert float(np.mean(metrics["psi"] >= config.random_psi_threshold)) >= 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean_shift_std", 2.0),
        ("covariance_scale", 1.8),
        ("target_noise_std", 0.5),
        ("label_flip_fraction", 0.25),
    ],
)
def test_each_drift_control_changes_output(
    real_data: pd.DataFrame, field: str, value: float
) -> None:
    config = load_config()
    baseline = generate(
        real_data, request_from_config(config, "similar"), config.similar_jitter_scale
    )
    drifted = generate(
        real_data,
        request_from_config(config, "similar", **{field: value}),
        config.similar_jitter_scale,
    )
    assert csv_bytes(baseline) != csv_bytes(drifted)


def test_psi_is_zero_for_same_values() -> None:
    values = np.arange(100, dtype=float)
    assert psi(values, values.copy(), 10) == pytest.approx(0.0)


def test_run_writes_required_artifacts(tmp_path: Path) -> None:
    source = load_config()
    payload = source.model_dump(mode="json")
    payload.update(
        {
            "report_dir": str(tmp_path / "reports"),
            "output_dir": str(tmp_path / "generated"),
            "n_samples": 500,
        }
    )
    config_path = tmp_path / "lab5.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    summary = run_lab5(config_path)

    assert summary["byte_reproducible"] is True
    assert summary["thresholds_passed"] is True
    assert all(result["changed"] for result in summary["drift_controls"].values())
    assert (tmp_path / "reports" / "synthetic_metrics.csv").is_file()
    assert (tmp_path / "reports" / "generator_config.json").is_file()
    assert (tmp_path / "reports" / "run_summary.json").is_file()
    assert (tmp_path / "generated" / "diabetes_similar.csv").is_file()
    assert (tmp_path / "generated" / "diabetes_random.csv").is_file()
