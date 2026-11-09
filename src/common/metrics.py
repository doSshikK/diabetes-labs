from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator
    return {"mae": mae, "rmse": rmse, "r2": r2}


def bootstrap_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    samples: int,
    level: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    collected = {name: [] for name in ("mae", "rmse", "r2")}
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), len(y_true))
        values = regression_metrics(y_true[indices], y_pred[indices])
        for name, value in values.items():
            collected[name].append(value)
    alpha = (1.0 - level) / 2.0
    point = regression_metrics(y_true, y_pred)
    return {
        name: {
            "value": point[name],
            "ci_low": float(np.quantile(collected[name], alpha)),
            "ci_high": float(np.quantile(collected[name], 1.0 - alpha)),
        }
        for name in point
    }
