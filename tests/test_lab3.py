from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.preprocessing import StandardScaler

from lab3.config import Lab3Config, load_config
from lab3.pipeline import MLP, CsvStreamDataset, feature_columns


def test_lab3_config_rejects_invalid_values() -> None:
    payload = load_config().model_dump(mode="python")
    for update in (
        {"batch_size": 0},
        {"weight_decay_grid": [-0.1]},
        {"patience": payload["max_epochs"] + 1},
    ):
        with pytest.raises(ValidationError):
            Lab3Config.model_validate({**payload, **update})


def test_stream_dataset_yields_all_rows_with_bounded_chunks() -> None:
    config = load_config()
    frame = pd.read_csv(config.train_path)
    features = feature_columns(frame)
    scaler = StandardScaler().fit(frame[features].to_numpy(dtype=float))
    dataset = CsvStreamDataset(config.train_path, features, scaler, config.chunk_size)
    rows = list(dataset)
    assert len(rows) == len(frame)
    assert rows[0][0].shape == (len(features),)


def test_mlp_output_shape() -> None:
    model = MLP(10, [32, 16])
    import torch

    assert model(torch.zeros((4, 10))).shape == (4,)


def test_lab3_paths_exist() -> None:
    config = load_config()
    for path in (config.train_path, config.validation_path, config.test_path):
        assert Path(path).exists()
