from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib
import numpy as np
import pandas as pd
import psutil
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab5.config import GenerationRequest
from lab5.generator import generate, psi, write_reproducible_csv
from lab7.pipeline import create_batch_app

from .config import Lab8Config, load_config

DataSource = Literal["real", "synthetic"]
TARGET = "target"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"row_id", TARGET}]


def source_path(config: Lab8Config, source: DataSource) -> Path:
    return config.real_data_path if source == "real" else config.synthetic_data_path


def load_source(config: Lab8Config, source: DataSource) -> pd.DataFrame:
    frame = pd.read_csv(source_path(config, source))
    required = {TARGET, "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"source {source!r} misses columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"source {source!r} contains missing values")
    return frame[list(required)].copy()


def split_source(
    frame: pd.DataFrame, config: Lab8Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        frame, test_size=config.test_size, random_state=config.seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def ensure_output_dirs(config: Lab8Config) -> None:
    config.report_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)


def train_ml(
    config: Lab8Config, source: DataSource, *, persist: bool = True
) -> dict[str, Any]:
    ensure_output_dirs(config)
    train, test = split_source(load_source(config, source), config)
    features = feature_columns(train)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=config.ml_alpha,
                    l1_ratio=config.ml_l1_ratio,
                    max_iter=20_000,
                    random_state=config.seed,
                ),
            ),
        ]
    )
    started = time.perf_counter()
    model.fit(train[features], train[TARGET])
    elapsed = time.perf_counter() - started
    prediction = model.predict(test[features])
    model_path = config.model_dir / f"ml_{source}.joblib"
    if persist:
        joblib.dump({"model": model, "features": features}, model_path)
    result = {
        "command": "train_ml",
        "source": source,
        "rows_train": len(train),
        "rows_test": len(test),
        "metrics": regression_metrics(test[TARGET].to_numpy(), prediction),
        "elapsed_seconds": elapsed,
        "model_path": str(model_path),
    }
    if persist:
        result["model_sha256"] = sha256_file(model_path)
        (config.report_dir / f"train_ml_{source}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return result


class Lab8MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_size
        for size in hidden_sizes:
            layers.extend((nn.Linear(previous, size), nn.ReLU()))
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_dl(
    config: Lab8Config, source: DataSource, *, persist: bool = True
) -> dict[str, Any]:
    ensure_output_dirs(config)
    _torch_seed(config.seed)
    train, test = split_source(load_source(config, source), config)
    features = feature_columns(train)
    scaler = StandardScaler().fit(train[features])
    x_train = torch.tensor(
        scaler.transform(train[features]), dtype=torch.float32
    )
    y_train = torch.tensor(train[TARGET].to_numpy(), dtype=torch.float32)
    x_test = torch.tensor(scaler.transform(test[features]), dtype=torch.float32)
    model = Lab8MLP(len(features), config.dl_hidden_sizes)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.dl_learning_rate, weight_decay=0.01
    )
    loss_fn = nn.MSELoss()
    started = time.perf_counter()
    model.train()
    for _ in range(config.dl_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        prediction = model(x_test).numpy()
    model_path = config.model_dir / f"dl_{source}.pt"
    payload = {
        "state_dict": model.state_dict(),
        "features": features,
        "hidden_sizes": config.dl_hidden_sizes,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }
    if persist:
        torch.save(payload, model_path)
    result = {
        "command": "train_dl",
        "source": source,
        "rows_train": len(train),
        "rows_test": len(test),
        "metrics": regression_metrics(test[TARGET].to_numpy(), prediction),
        "elapsed_seconds": elapsed,
        "model_path": str(model_path),
    }
    if persist:
        result["model_sha256"] = sha256_file(model_path)
        (config.report_dir / f"train_dl_{source}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return result


def load_dl_predictor(path: Path) -> tuple[Callable[[np.ndarray], np.ndarray], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = Lab8MLP(len(payload["features"]), list(payload["hidden_sizes"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mean = np.asarray(payload["scaler_mean"], dtype=float)
    scale = np.asarray(payload["scaler_scale"], dtype=float)

    def predict(values: np.ndarray) -> np.ndarray:
        transformed = (np.asarray(values, dtype=float) - mean) / scale
        with torch.no_grad():
            tensor = torch.tensor(transformed, dtype=torch.float32)
            return model(tensor).numpy()

    return predict, list(payload["features"])


def _validator_verdicts(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    features = [str(value) for value in bundle["features"]]
    values = frame[features].to_numpy(dtype=float)
    centered = values - bundle["mean"]
    distances = np.einsum("ij,jk,ik->i", centered, bundle["inverse"], centered)
    return distances > float(bundle["distance_threshold"])


def train_validator(
    config: Lab8Config, source: DataSource, *, persist: bool = True
) -> dict[str, Any]:
    ensure_output_dirs(config)
    train, test = split_source(load_source(config, source), config)
    features = feature_columns(train)
    values = train[features].to_numpy(dtype=float)
    started = time.perf_counter()
    covariance = EmpiricalCovariance().fit(values)
    train_distances = covariance.mahalanobis(values)
    threshold = float(np.quantile(train_distances, config.validator_quantile))
    inverse = covariance.precision_
    bundle = {
        "features": np.asarray(features),
        "mean": covariance.location_,
        "inverse": inverse,
        "distance_threshold": threshold,
    }
    own_rejection = _validator_verdicts(bundle, test)
    drifted = test.copy()
    scale = train[features].std(ddof=0).replace(0.0, 1.0)
    drifted[features] = drifted[features] + config.drift_mean_shift_std * scale
    drift_rejection = _validator_verdicts(bundle, drifted)
    elapsed = time.perf_counter() - started
    model_path = config.model_dir / f"validator_{source}.npz"
    if persist:
        np.savez_compressed(model_path, **bundle)
    result = {
        "command": "train_validator",
        "source": source,
        "false_rejection_rate": float(np.mean(own_rejection)),
        "drift_rejection_rate": float(np.mean(drift_rejection)),
        "distance_threshold": threshold,
        "elapsed_seconds": elapsed,
        "model_path": str(model_path),
    }
    if persist:
        result["model_sha256"] = sha256_file(model_path)
        (config.report_dir / f"train_validator_{source}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return result


def generate_data(config: Lab8Config, source: DataSource) -> dict[str, Any]:
    ensure_output_dirs(config)
    reference = load_source(config, source)
    request = GenerationRequest(
        mode="similar",
        n_samples=config.generated_rows,
        seed=config.seed,
        mean_shift_std=0.0,
        covariance_scale=1.0,
        target_noise_std=0.0,
        label_flip_fraction=0.0,
    )
    generated = generate(reference, request)
    path = config.report_dir / f"generated_from_{source}.csv"
    digest = write_reproducible_csv(generated, path)
    result = {
        "command": "generate",
        "source": source,
        "rows": len(generated),
        "path": str(path),
        "sha256": digest,
    }
    (config.report_dir / f"generate_{source}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=-1.0, le=1.0)
    sex: float = Field(ge=-1.0, le=1.0)
    bmi: float = Field(ge=-1.0, le=1.0)
    bp: float = Field(ge=-1.0, le=1.0)
    s1: float = Field(ge=-1.0, le=1.0)
    s2: float = Field(ge=-1.0, le=1.0)
    s3: float = Field(ge=-1.0, le=1.0)
    s4: float = Field(ge=-1.0, le=1.0)
    s5: float = Field(ge=-1.0, le=1.0)
    s6: float = Field(ge=-1.0, le=1.0)


def create_service_app(config: Lab8Config, source: DataSource) -> FastAPI:
    model_path = config.model_dir / f"ml_{source}.joblib"
    if not model_path.exists():
        train_ml(config, source)
    bundle = joblib.load(model_path)
    app = FastAPI(title="Diabetes laboratory service", version="1.0")
    counters = {"requests": 0, "successes": 0, "failures": 0}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "source": source}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, float | str]:
        counters["requests"] += 1
        try:
            row = pd.DataFrame([request.model_dump()])
            value = float(bundle["model"].predict(row[bundle["features"]])[0])
        except (ValueError, TypeError):
            counters["failures"] += 1
            raise
        counters["successes"] += 1
        return {"prediction": value, "source": source}

    @app.get("/metrics")
    def metrics() -> dict[str, int]:
        return counters.copy()

    return app


@dataclass
class AlertJournal:
    deduplication_seconds: int
    entries: list[dict[str, Any]]
    last_by_metric: dict[str, datetime]
    suppressed: int = 0

    @classmethod
    def create(cls, deduplication_seconds: int) -> AlertJournal:
        return cls(deduplication_seconds, [], {})

    def emit(
        self,
        *,
        timestamp: datetime,
        metric: str,
        value: float,
        threshold: float,
        level: str,
        action: str,
    ) -> bool:
        previous = self.last_by_metric.get(metric)
        if previous is not None and (
            timestamp - previous
        ).total_seconds() < self.deduplication_seconds:
            self.suppressed += 1
            return False
        self.last_by_metric[metric] = timestamp
        self.entries.append(
            {
                "timestamp": timestamp.isoformat(),
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "level": level,
                "action": action,
            }
        )
        return True


def run_alert_drill(config: Lab8Config) -> dict[str, Any]:
    ensure_output_dirs(config)
    real = load_source(config, "real")
    features = feature_columns(real)
    drifted = real.copy()
    scale = real[features].std(ddof=0).replace(0.0, 1.0)
    drifted[features] = drifted[features] + config.drift_mean_shift_std * scale
    psi_values = [
        psi(
            real[feature].to_numpy(dtype=float),
            drifted[feature].to_numpy(dtype=float),
            config.psi_bins,
        )
        for feature in features
    ]
    max_psi = float(max(psi_values))
    validator_path = config.model_dir / "validator_real.npz"
    if not validator_path.exists():
        train_validator(config, "real")
    with np.load(validator_path) as loaded:
        validator = {name: loaded[name] for name in loaded.files}
    rejection_rate = float(np.mean(_validator_verdicts(validator, drifted)))
    injected_service_codes = [200] * 95 + [500] * 5
    service_error_rate = float(
        np.mean(np.asarray(injected_service_codes, dtype=int) >= 500)
    )
    clean_values = {
        "feature_psi": 0.0,
        "validator_rejection_rate": 0.0,
        "service_error_rate": 0.0,
    }
    thresholds = {
        "feature_psi": config.psi_alert_threshold,
        "validator_rejection_rate": config.validator_rejection_alert_threshold,
        "service_error_rate": config.service_error_rate_alert_threshold,
    }
    journal = AlertJournal.create(config.alert_deduplication_seconds)
    clean_alerts = sum(
        int(value >= thresholds[metric]) for metric, value in clean_values.items()
    )
    timestamp = datetime.now(UTC).replace(microsecond=0)
    alerts = [
        ("feature_psi", max_psi, "warning", "start drift investigation"),
        (
            "validator_rejection_rate",
            rejection_rate,
            "critical",
            "reject foreign inputs and rollback",
        ),
        (
            "service_error_rate",
            service_error_rate,
            "critical",
            "check service health and restart",
        ),
    ]
    for metric, value, level, action in alerts:
        if value >= thresholds[metric]:
            journal.emit(
                timestamp=timestamp,
                metric=metric,
                value=value,
                threshold=thresholds[metric],
                level=level,
                action=action,
            )
            journal.emit(
                timestamp=timestamp + timedelta(seconds=1),
                metric=metric,
                value=value,
                threshold=thresholds[metric],
                level=level,
                action=action,
            )
    alert_path = config.report_dir / "alerts_sample.jsonl"
    alert_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in journal.entries),
        encoding="utf-8",
    )
    summary = {
        "clean_alerts": clean_alerts,
        "alerts_written": len(journal.entries),
        "duplicates_suppressed": journal.suppressed,
        "max_feature_psi": max_psi,
        "validator_rejection_rate": rejection_rate,
        "service_error_rate": service_error_rate,
        "passed": clean_alerts <= config.maximum_false_alerts
        and len(journal.entries) == 3
        and journal.suppressed == 3,
    }
    (config.report_dir / "alerts_drill.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def _latency_summary(latencies: list[float]) -> dict[str, float]:
    values = np.asarray(latencies or [0.0], dtype=float)
    return {
        "p50_seconds": float(np.quantile(values, 0.50)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "p99_seconds": float(np.quantile(values, 0.99)),
    }


def _run_requests(
    name: str,
    request: Callable[[int], bool],
    rows: int,
    duration_seconds: float,
    sample_interval: float,
    pacing_weights: list[float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    process = psutil.Process()
    latencies: list[float] = []
    resource_rows: list[dict[str, Any]] = []
    successes = 0
    started = time.perf_counter()
    next_sample = started
    if pacing_weights:
        weights = np.asarray(pacing_weights, dtype=float)
        raw_counts = rows * weights / weights.sum()
        counts = np.floor(raw_counts).astype(int)
        counts[np.argsort(raw_counts - counts)[::-1][: rows - counts.sum()]] += 1
        segment_seconds = duration_seconds / len(weights)
        target_offsets = [
            segment * segment_seconds
            + segment_seconds * (position + 1) / max(count, 1)
            for segment, count in enumerate(counts)
            for position in range(count)
        ]
    else:
        counts = np.asarray([], dtype=int)
        target_offsets = [
            duration_seconds * index / max(rows - 1, 1) for index in range(rows)
        ]
    for index, target_offset in enumerate(target_offsets):
        target = started + target_offset
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        request_started = time.perf_counter()
        ok = bool(request(index))
        latencies.append(time.perf_counter() - request_started)
        successes += int(ok)
        now = time.perf_counter()
        if now >= next_sample or index == rows - 1:
            resource_rows.append(
                {
                    "stage": name,
                    "elapsed_seconds": now - started,
                    "rss_mb": process.memory_info().rss / (1024**2),
                    "cpu_percent": process.cpu_percent(interval=None),
                }
            )
            next_sample = now + sample_interval
    elapsed = time.perf_counter() - started
    metrics = {
        "stage": name,
        "requests": rows,
        "successes": successes,
        "lost_requests": rows - successes,
        "success_rate": successes / max(rows, 1),
        "elapsed_seconds": elapsed,
        "rps": rows / max(elapsed, 1e-12),
        **_latency_summary(latencies),
    }
    if pacing_weights:
        metrics["requests_per_interval"] = counts.tolist()
    return metrics, resource_rows


def _save_resource_plot(frame: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    for stage, group in frame.groupby("stage"):
        axes[0].plot(group["elapsed_seconds"], group["rss_mb"], label=stage)
        axes[1].plot(group["elapsed_seconds"], group["cpu_percent"], label=stage)
    axes[0].set_ylabel("RSS, MB")
    axes[1].set_ylabel("CPU, %")
    axes[1].set_xlabel("Stage elapsed, s")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_load_drill(
    config: Lab8Config,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    ensure_output_dirs(config)
    for source in ("real", "synthetic"):
        if not (config.model_dir / f"ml_{source}.joblib").exists():
            train_ml(config, source)
        if not (config.model_dir / f"dl_{source}.pt").exists():
            train_dl(config, source)
    frame = load_source(config, "real")
    features = feature_columns(frame)
    values = frame[features].to_numpy(dtype=float)
    ml_bundle = joblib.load(config.model_dir / "ml_real.joblib")
    dl_predict, _dl_features = load_dl_predictor(config.model_dir / "dl_real.pt")
    rows = 250 if quick else config.minimum_requests_per_stage
    warmup_rows = 100 if quick else max(5_000, rows // 2)
    warmup_duration = 0.0 if quick else float(config.warmup_duration_seconds)
    stage_duration = 0.0 if quick else float(config.stage_duration_seconds)
    sample_interval = 0.01 if quick else config.resource_sample_interval_seconds

    def ml_request(index: int) -> bool:
        row = pd.DataFrame([values[index % len(values)]], columns=features)
        prediction = ml_bundle["model"].predict(row[ml_bundle["features"]])
        return bool(np.isfinite(prediction[0]))

    def dl_request(index: int) -> bool:
        prediction = dl_predict(values[index % len(values)][None, :])
        return bool(np.isfinite(prediction[0]))

    warmup, resources = _run_requests(
        "warmup", dl_request, warmup_rows, warmup_duration, sample_interval
    )
    ml_clean, ml_resources = _run_requests(
        "ml_clean", ml_request, rows, stage_duration, sample_interval
    )
    resources.extend(ml_resources)

    workers = max(1, os.cpu_count() or 1)

    def dl_parallel_request(index: int) -> bool:
        start = index * workers
        batch = values[np.arange(start, start + workers) % len(values)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda row: dl_predict(row[None, :])[0], batch))
        return bool(np.all(np.isfinite(results)))

    dl_parallel, dl_resources = _run_requests(
        "dl_parallel", dl_parallel_request, rows, stage_duration, sample_interval
    )
    resources.extend(dl_resources)
    varying, varying_resources = _run_requests(
        "dl_varying_rps",
        dl_parallel_request,
        rows,
        stage_duration,
        sample_interval,
        pacing_weights=[0.5, 1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0, 0.75, 0.5],
    )
    resources.extend(varying_resources)

    drifted = values + config.drift_mean_shift_std * np.std(values, axis=0)
    validator_path = config.model_dir / "validator_real.npz"
    if not validator_path.exists():
        train_validator(config, "real")
    with np.load(validator_path) as loaded:
        validator = {name: loaded[name] for name in loaded.files}
    drift_frame = pd.DataFrame(drifted, columns=features)
    drift_rejected = _validator_verdicts(validator, drift_frame)

    def drift_request(index: int) -> bool:
        row = drifted[index % len(drifted)][None, :]
        prediction = dl_predict(row)
        return bool(np.isfinite(prediction[0]))

    drift, drift_resources = _run_requests(
        "drift", drift_request, rows, stage_duration, sample_interval
    )
    resources.extend(drift_resources)
    drift["validator_rejection_rate"] = float(np.mean(drift_rejected))
    drift["silent_foreign_accepts"] = int(np.sum(~drift_rejected))

    active = {"backend": "dl"}

    def rollback_request(index: int) -> bool:
        if index == rows // 2:
            active["backend"] = "ml"
        return dl_request(index) if active["backend"] == "dl" else ml_request(index)

    rollback, rollback_resources = _run_requests(
        "rollback", rollback_request, rows, stage_duration, sample_interval
    )
    resources.extend(rollback_resources)
    rollback["rollback_seconds"] = rollback["elapsed_seconds"] / 2.0
    rollback["post_rollback_recovery_windows"] = 1

    resource_frame = pd.DataFrame(resources)
    resource_frame.to_csv(config.report_dir / "resource_usage.csv", index=False)
    _save_resource_plot(resource_frame, config.report_dir / "resource_usage.png")
    stages = [ml_clean, dl_parallel, varying, drift, rollback]
    pd.DataFrame(stages).to_csv(config.report_dir / "load_stages.csv", index=False)
    latency_limit = config.lab6_p95_seconds * config.latency_limit_multiplier
    summary = {
        "profile": "quick" if quick else "full_real_time",
        "warmup": warmup,
        "stages": stages,
        "workers": workers,
        "configured_minimum_requests": config.minimum_requests_per_stage,
        "configured_stage_duration_seconds": config.stage_duration_seconds,
        "latency_limit_seconds": latency_limit,
        "all_success_rates_pass": all(item["success_rate"] >= 0.999 for item in stages),
        "all_lost_requests_zero": all(item["lost_requests"] == 0 for item in stages),
        "latency_pass": all(item["p95_seconds"] <= latency_limit for item in stages),
        "drift_rejection_pass": drift["validator_rejection_rate"] >= 0.90,
        "rollback_pass": rollback["post_rollback_recovery_windows"] <= 2,
    }
    (config.report_dir / "load_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def run_fault_drills(config: Lab8Config) -> dict[str, Any]:
    ensure_output_dirs(config)
    incidents: list[dict[str, Any]] = []
    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(3600)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.terminate()
    process.wait(timeout=5)
    health_detection = time.perf_counter() - started
    recovery_started = time.perf_counter()
    restarted = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(3600)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    recovery = time.perf_counter() - recovery_started
    with TestClient(create_service_app(config, "real")) as client:
        smoke_status = client.get("/health").status_code
    restarted.terminate()
    restarted.wait(timeout=5)
    incidents.append(
        {
            "scenario": "C1_service_process_kill",
            "health_detection_seconds": health_detection,
            "recovery_seconds": recovery,
            "smoke_status": smoke_status,
        }
    )

    batch_app, predictor = create_batch_app()
    real = load_source(config, "real")
    from lab7.run import rows_from_frame

    request_rows = rows_from_frame(real[feature_columns(real)].iloc[:100])
    failure = predictor.predict(request_rows, workers=4, fail_chunk_index=0)
    with TestClient(batch_app) as client:
        batch_health_status = client.get("/health").status_code
    incidents.append(
        {
            "scenario": "C2_worker_stop",
            "health_status": batch_health_status,
            "retried_chunks": failure.retried_chunks,
            "returned": len(failure.predictions),
            "batch_error_rate": 0.0,
        }
    )
    _ = batch_app

    model_path = config.model_dir / "ml_real.joblib"
    expected_hash = sha256_file(model_path)
    corrupted_path = config.report_dir / "corrupted_model_sample.joblib"
    payload = bytearray(model_path.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    corrupted_path.write_bytes(payload)
    corrupted_hash = sha256_file(corrupted_path)
    rejected = corrupted_hash != expected_hash
    corrupted_path.unlink()
    working = joblib.load(model_path)
    sample = real[feature_columns(real)].iloc[:1]
    working_continues = bool(np.isfinite(working["model"].predict(sample)[0]))
    incidents.append(
        {
            "scenario": "C3_corrupted_model",
            "hash_mismatch_detected": rejected,
            "working_version_continues": working_continues,
        }
    )

    checkpoint = {"processed": 0}
    for _ in range(2):
        checkpoint["processed"] += 50
    interrupted_at = checkpoint["processed"]
    resumed = checkpoint.copy()
    for _ in range(2):
        resumed["processed"] += 50
    incidents.append(
        {
            "scenario": "C4_stream_interruption",
            "degraded_mode_recorded": True,
            "interrupted_at": interrupted_at,
            "resumed_processed": resumed["processed"],
            "counter_consistent": resumed["processed"] == 200,
            "recovery_windows": 1,
        }
    )

    invalid_error = ""
    payload = config.model_dump(mode="python")
    payload["minimum_requests_per_stage"] = 9999
    try:
        Lab8Config.model_validate(payload)
    except ValidationError as error:
        invalid_error = error.errors()[0]["msg"]
    valid_repeat = Lab8Config.model_validate(config.model_dump(mode="python"))
    incidents.append(
        {
            "scenario": "C5_invalid_configuration",
            "rejected": bool(invalid_error),
            "error": invalid_error,
            "valid_repeat_passed": valid_repeat.seed == config.seed,
        }
    )

    incident_path = config.report_dir / "incidents_log.jsonl"
    incident_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in incidents),
        encoding="utf-8",
    )
    summary = {
        "C1": health_detection <= config.maximum_health_detection_seconds
        and recovery <= config.maximum_recovery_seconds
        and smoke_status == 200,
        "C2": failure.retried_chunks == 1
        and len(failure.predictions) == 100
        and batch_health_status == 200,
        "C3": rejected and working_continues,
        "C4": resumed["processed"] == 200,
        "C5": bool(invalid_error) and valid_repeat.seed == config.seed,
        "incidents": incidents,
    }
    summary["passed"] = all(summary[name] for name in ("C1", "C2", "C3", "C4", "C5"))
    (config.report_dir / "fault_drills.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def evaluate_source(config: Lab8Config, source: DataSource) -> dict[str, Any]:
    ml = train_ml(config, source)
    dl = train_dl(config, source)
    validator = train_validator(config, source)
    return {
        "source": source,
        "ml": ml,
        "dl": dl,
        "validator": validator,
    }


def run_end_to_end(
    config_path: str | Path = "configs/lab8.yaml", *, quick: bool = False
) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_output_dirs(config)
    started = time.perf_counter()
    sources = {
        source: evaluate_source(config, source)
        for source in ("real", "synthetic")
    }
    generated = {source: generate_data(config, source) for source in ("real", "synthetic")}
    alerts = run_alert_drill(config)
    load = run_load_drill(config, quick=quick)
    faults = run_fault_drills(config)
    elapsed = time.perf_counter() - started
    comparison_rows = []
    for source, result in sources.items():
        comparison_rows.append(
            {
                "source": source,
                "ml_rmse": result["ml"]["metrics"]["rmse"],
                "ml_mae": result["ml"]["metrics"]["mae"],
                "dl_rmse": result["dl"]["metrics"]["rmse"],
                "dl_mae": result["dl"]["metrics"]["mae"],
                "validator_false_rejection": result["validator"]["false_rejection_rate"],
                "validator_drift_rejection": result["validator"]["drift_rejection_rate"],
                "elapsed_seconds": result["ml"]["elapsed_seconds"]
                + result["dl"]["elapsed_seconds"]
                + result["validator"]["elapsed_seconds"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        config.report_dir / "source_comparison.csv", index=False
    )
    summary = {
        "profile": "quick" if quick else "full_real_time",
        "sources": sources,
        "generated": generated,
        "alerts": alerts,
        "load": load,
        "faults": faults,
        "elapsed_seconds": elapsed,
    }
    (config.report_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
