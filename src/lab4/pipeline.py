from __future__ import annotations

import json
import os
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import roc_auc_score

from common.metrics import sha256_file
from lab4.config import Lab4Config, load_config

TARGET = "target"
IGNORED = {"row_id", TARGET}


@dataclass
class Statistics:
    count: int
    mean: np.ndarray
    covariance: np.ndarray


@dataclass
class Usage:
    elapsed_seconds: float
    peak_rss_mb: float
    peak_python_mb: float


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in IGNORED]


def full_statistics(frame: pd.DataFrame, features: list[str]) -> Statistics:
    values = frame[features].to_numpy(dtype=float)
    return Statistics(
        len(values), values.mean(axis=0), np.cov(values, rowvar=False, ddof=0)
    )


def merge_statistics(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    values: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    chunk_count = len(values)
    chunk_mean = values.mean(axis=0)
    centered = values - chunk_mean
    chunk_m2 = centered.T @ centered
    if count == 0:
        return chunk_count, chunk_mean, chunk_m2
    total = count + chunk_count
    delta = chunk_mean - mean
    merged_mean = mean + delta * chunk_count / total
    merged_m2 = m2 + chunk_m2 + np.outer(delta, delta) * count * chunk_count / total
    return total, merged_mean, merged_m2


def chunk_statistics(path: Path, features: list[str], chunk_size: int) -> Statistics:
    count = 0
    mean = np.zeros(len(features), dtype=float)
    m2 = np.zeros((len(features), len(features)), dtype=float)
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        count, mean, m2 = merge_statistics(
            count, mean, m2, chunk[features].to_numpy(dtype=float)
        )
    return Statistics(count, mean, m2 / count)


def measure(call: Callable[[], Statistics]) -> tuple[Statistics, Usage]:
    process = psutil.Process()
    before_rss = process.memory_info().rss
    tracemalloc.start()
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = max(before_rss, process.memory_info().rss)
    return result, Usage(elapsed, peak_rss / 1024**2, peak_python / 1024**2)


def distances(values: np.ndarray, mean: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    centered = values - mean
    return np.einsum("ij,jk,ik->i", centered, inverse, centered)


def model_predictions(
    bundle: dict, frame: pd.DataFrame, features: list[str]
) -> np.ndarray:
    values = frame[features].to_numpy(dtype=float)
    transformed = bundle["scaler"].transform(values)
    return bundle["model"].predict(transformed)


def verdicts(
    values: np.ndarray,
    predictions: np.ndarray,
    mean: np.ndarray,
    inverse: np.ndarray,
    distance_threshold: float,
    target_mean: float,
    target_std: float,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance = distances(values, mean, inverse)
    confidence = np.exp(-np.abs(predictions - target_mean) / max(target_std, 1e-12))
    rejected = (distance > distance_threshold) | (confidence < min_confidence)
    distance_risk = distance / distance_threshold
    confidence_risk = -np.log(np.maximum(confidence, 1e-12)) / -np.log(min_confidence)
    risk = np.maximum(distance_risk, confidence_risk)
    return rejected, risk, distance, confidence


def make_ood(
    frame: pd.DataFrame, features: list[str], mean_shift: float, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = frame[features].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-8)
    blocks = []
    labels = []
    shifted = values + mean_shift * scale
    blocks.append(shifted)
    labels.extend(["mean_shift"] * len(values))
    outliers = values.copy()
    outliers[:, :3] = mean[:3] + 12.0 * scale[:3]
    blocks.append(outliers)
    labels.extend(["outliers"] * len(values))
    inverted = -4.0 * values
    blocks.append(inverted)
    labels.extend(["sign_and_scale"] * len(values))
    broken = values.copy()
    for column in range(broken.shape[1]):
        broken[:, column] = rng.permutation(broken[:, column]) * 4.0
    blocks.append(broken)
    labels.extend(["covariance_break"] * len(values))
    output = pd.DataFrame(np.vstack(blocks), columns=features)
    output["scenario"] = labels
    return output


def rejection_metrics(
    expected: np.ndarray, rejected: np.ndarray, risk: np.ndarray
) -> dict[str, float]:
    true_positive = int(np.sum(expected & rejected))
    false_positive = int(np.sum(~expected & rejected))
    false_negative = int(np.sum(expected & ~rejected))
    true_negative = int(np.sum(~expected & ~rejected))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    false_rejection = false_positive / max(false_positive + true_negative, 1)
    return {
        "rejection_precision": precision,
        "rejection_recall": recall,
        "false_rejection_rate": false_rejection,
        "roc_auc": float(roc_auc_score(expected.astype(int), risk)),
    }


def log_run(
    mode: str, usage: Usage, metrics: dict[str, float], config: Lab4Config
) -> str:
    with mlflow.start_run(run_name=f"lab4_{mode}") as run:
        mlflow.log_params(
            {
                "mode": mode,
                "chunk_size": config.chunk_size,
                "distance_quantile": config.distance_quantile,
                "min_model_confidence": config.min_model_confidence,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metric("elapsed_seconds", usage.elapsed_seconds)
        mlflow.log_metric("peak_rss_mb", usage.peak_rss_mb)
        return run.info.run_id


def run_lab4(config_path: str | Path = "configs/lab4.yaml") -> dict:
    config = load_config(config_path)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("diabetes_lab4")
    train = pd.read_csv(config.train_path)
    test = pd.read_csv(config.test_path)
    features = feature_columns(train)
    full, full_usage = measure(lambda: full_statistics(train, features))
    chunks, chunk_usage = measure(
        lambda: chunk_statistics(config.train_path, features, config.chunk_size)
    )
    mean_difference = float(np.max(np.abs(full.mean - chunks.mean)))
    covariance_difference = float(np.max(np.abs(full.covariance - chunks.covariance)))
    regularized = (
        full.covariance + np.eye(len(features)) * config.covariance_regularization
    )
    inverse = np.linalg.pinv(regularized)
    train_values = train[features].to_numpy(dtype=float)
    train_distance = distances(train_values, full.mean, inverse)
    distance_threshold = float(np.quantile(train_distance, config.distance_quantile))
    bundle = joblib.load(config.model_path)
    target_mean = float(train[TARGET].mean())
    target_std = float(train[TARGET].std(ddof=0))
    own_prediction = model_predictions(bundle, test, features)
    own_values = test[features].to_numpy(dtype=float)
    own_rejected, own_risk, own_distance, own_confidence = verdicts(
        own_values,
        own_prediction,
        full.mean,
        inverse,
        distance_threshold,
        target_mean,
        target_std,
        config.min_model_confidence,
    )
    ood = make_ood(test, features, config.mean_shift_std, config.seed)
    ood_prediction = model_predictions(bundle, ood, features)
    ood_rejected, ood_risk, ood_distance, ood_confidence = verdicts(
        ood[features].to_numpy(dtype=float),
        ood_prediction,
        full.mean,
        inverse,
        distance_threshold,
        target_mean,
        target_std,
        config.min_model_confidence,
    )
    expected = np.concatenate(
        (np.zeros(len(test), dtype=bool), np.ones(len(ood), dtype=bool))
    )
    rejected = np.concatenate((own_rejected, ood_rejected))
    risk = np.concatenate((own_risk, ood_risk))
    metrics = rejection_metrics(expected, rejected, risk)
    mode_metrics = {
        **metrics,
        "mean_max_difference": mean_difference,
        "covariance_max_difference": covariance_difference,
    }
    full_run = log_run("full", full_usage, mode_metrics, config)
    chunk_run = log_run("chunks", chunk_usage, mode_metrics, config)
    np.savez_compressed(
        config.validator_path,
        mean=full.mean,
        covariance=full.covariance,
        inverse=inverse,
        distance_threshold=distance_threshold,
        target_mean=target_mean,
        target_std=target_std,
        min_model_confidence=config.min_model_confidence,
        features=np.array(features),
    )
    validator_hash = sha256_file(config.validator_path)
    metric_rows = [
        {"mode": "full", "run_id": full_run, **mode_metrics},
        {"mode": "chunks", "run_id": chunk_run, **mode_metrics},
    ]
    pd.DataFrame(metric_rows).to_csv(
        config.report_dir / "validator_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {"mode": "full", **full_usage.__dict__},
            {"mode": "chunks", **chunk_usage.__dict__},
        ]
    ).to_csv(config.report_dir / "mode_compare.csv", index=False)
    negative = pd.DataFrame(
        {
            "scenario": ood["scenario"],
            "expected": "reject",
            "actual": np.where(ood_rejected, "reject", "accept"),
            "distance": ood_distance,
            "model_confidence": ood_confidence,
        }
    )
    negative.to_csv(config.report_dir / "validator_negative.log", index=False)
    own_table = pd.DataFrame(
        {
            "expected": "accept",
            "actual": np.where(own_rejected, "reject", "accept"),
            "distance": own_distance,
            "model_confidence": own_confidence,
        }
    )
    own_table.to_csv(config.report_dir / "validator_own_inputs.csv", index=False)
    summary = {
        "features": features,
        "distance_threshold": distance_threshold,
        "min_model_confidence": config.min_model_confidence,
        "metrics": metrics,
        "full_run_id": full_run,
        "chunk_run_id": chunk_run,
        "statistics": {
            "mean_max_difference": mean_difference,
            "covariance_max_difference": covariance_difference,
            "tolerance": config.statistics_tolerance,
            "passed": max(mean_difference, covariance_difference)
            <= config.statistics_tolerance,
        },
        "full_usage": full_usage.__dict__,
        "chunk_usage": chunk_usage.__dict__,
        "validator_sha256": validator_hash,
        "all_ood_rejected": bool(np.all(ood_rejected)),
        "thresholds_passed": bool(
            metrics["rejection_recall"] >= config.ood_recall_threshold
            and metrics["false_rejection_rate"] <= config.false_rejection_limit
        ),
    }
    (config.report_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
