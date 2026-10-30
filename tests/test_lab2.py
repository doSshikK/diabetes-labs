from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab2.config import Lab2Config, load_config
from lab2.pipeline import assert_train_only_scaler, load_and_split


def test_config_rejects_three_invalid_cases() -> None:
    payload = load_config().model_dump(mode="python")
    cases = [
        {**payload, "chunk_size": 0},
        {**payload, "alpha_grid": [-0.1]},
        {**payload, "l1_ratio_grid": [1.5]},
    ]
    for candidate in cases:
        with pytest.raises(ValidationError):
            Lab2Config.model_validate(candidate)


def test_split_has_no_overlap() -> None:
    train, validation, test = load_and_split(load_config())
    ids = [set(part["row_id"]) for part in (train, validation, test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert len(set.union(*ids)) == 442


def test_leakage_guard_rejects_scaler_fitted_on_all_rows() -> None:
    config = load_config()
    train, _, _ = load_and_split(config)
    all_data = pd.read_csv(config.data_path)
    features = [column for column in all_data.columns if column != "target"]
    leaked_scaler = StandardScaler().fit(all_data[features].to_numpy(dtype=float))
    with pytest.raises(AssertionError, match="data leakage detected"):
        assert_train_only_scaler(leaked_scaler, train, features)


def test_train_only_scaler_is_accepted() -> None:
    config = load_config()
    train, _, _ = load_and_split(config)
    features = [column for column in train.columns if column not in {"row_id", "target"}]
    scaler = StandardScaler().fit(train[features].to_numpy(dtype=float))
    assert_train_only_scaler(scaler, train, features)
    np.testing.assert_allclose(
        scaler.mean_, train[features].to_numpy(dtype=float).mean(axis=0), atol=1e-12
    )
