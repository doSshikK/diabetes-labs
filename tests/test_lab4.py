from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from lab4.config import Lab4Config, load_config
from lab4.pipeline import (
    chunk_statistics,
    distances,
    feature_columns,
    full_statistics,
    make_ood,
)


def test_lab4_config_rejects_invalid_values() -> None:
    payload = load_config().model_dump(mode="python")
    for update in (
        {"chunk_size": 0},
        {"distance_quantile": 1.2},
        {"mean_shift_std": 0},
    ):
        with pytest.raises(ValidationError):
            Lab4Config.model_validate({**payload, **update})


def test_chunk_statistics_match_full() -> None:
    config = load_config()
    frame = pd.read_csv(config.train_path)
    features = feature_columns(frame)
    full = full_statistics(frame, features)
    chunks = chunk_statistics(config.train_path, features, config.chunk_size)
    np.testing.assert_allclose(full.mean, chunks.mean, atol=1e-12)
    np.testing.assert_allclose(full.covariance, chunks.covariance, atol=1e-12)


def test_ood_examples_are_far_from_training_mean() -> None:
    config = load_config()
    train = pd.read_csv(config.train_path)
    test = pd.read_csv(config.test_path)
    features = feature_columns(train)
    stats = full_statistics(train, features)
    inverse = np.linalg.pinv(stats.covariance + np.eye(len(features)) * 1e-6)
    threshold = np.quantile(
        distances(train[features].to_numpy(float), stats.mean, inverse), 0.99
    )
    ood = make_ood(test, features, config.mean_shift_std, config.seed)
    scores = distances(ood[features].to_numpy(float), stats.mean, inverse)
    assert float(np.mean(scores > threshold)) >= 0.95


def test_distance_is_non_negative() -> None:
    values = np.array([[0.0, 0.0], [1.0, 2.0]])
    scores = distances(values, np.zeros(2), np.eye(2))
    assert np.all(scores >= 0)
