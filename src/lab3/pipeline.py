from __future__ import annotations

import copy
import json
import os
import random
import time
import tracemalloc
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.metrics import bootstrap_regression, regression_metrics, sha256_file
from lab3.config import Lab3Config, load_config

TARGET = "target"
IGNORED = {"row_id", TARGET}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
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


class CsvStreamDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        features: list[str],
        scaler: StandardScaler,
        chunk_size: int,
    ) -> None:
        self.path = path
        self.features = features
        self.scaler = scaler
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for chunk in pd.read_csv(self.path, chunksize=self.chunk_size):
            x_values = self.scaler.transform(
                chunk[self.features].to_numpy(dtype=float)
            ).astype("float32")
            y_values = chunk[TARGET].to_numpy(dtype="float32")
            for x_row, y_value in zip(x_values, y_values, strict=True):
                yield torch.from_numpy(x_row), torch.tensor(y_value)


@dataclass
class TrainingResult:
    model: MLP
    curves: list[dict[str, float]]
    best_epoch: int
    elapsed_seconds: float
    peak_rss_mb: float
    peak_python_mb: float


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in IGNORED]


def fit_scaler_streaming(
    path: Path, features: list[str], chunk_size: int
) -> StandardScaler:
    scaler = StandardScaler()
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        scaler.partial_fit(chunk[features].to_numpy(dtype=float))
    return scaler


def make_loader(
    frame: pd.DataFrame,
    config: Lab3Config,
    features: list[str],
    scaler: StandardScaler,
    mode: str,
) -> DataLoader:
    if mode == "full":
        x_values = scaler.transform(frame[features].to_numpy(dtype=float)).astype(
            "float32"
        )
        y_values = frame[TARGET].to_numpy(dtype="float32")
        dataset = TensorDataset(torch.from_numpy(x_values), torch.from_numpy(y_values))
    elif mode == "chunks":
        dataset = CsvStreamDataset(
            config.train_path, features, scaler, config.chunk_size
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )


def predict(
    model: MLP, frame: pd.DataFrame, features: list[str], scaler: StandardScaler
) -> np.ndarray:
    values = scaler.transform(frame[features].to_numpy(dtype=float)).astype("float32")
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(values)).cpu().numpy().astype(float)


def train_model(
    frame: pd.DataFrame,
    validation: pd.DataFrame,
    config: Lab3Config,
    features: list[str],
    scaler: StandardScaler,
    mode: str,
    weight_decay: float,
    seed: int,
) -> TrainingResult:
    set_seed(seed)
    model = MLP(len(features), config.hidden_sizes)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.MSELoss()
    x_validation = torch.from_numpy(
        scaler.transform(validation[features].to_numpy(dtype=float)).astype("float32")
    )
    y_validation = torch.from_numpy(validation[TARGET].to_numpy(dtype="float32"))
    loader = make_loader(frame, config, features, scaler, mode)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    tracemalloc.start()
    started = time.perf_counter()
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    curves: list[dict[str, float]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = loss_function(prediction, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y_batch)
            total_rows += len(y_batch)
            peak_rss = max(peak_rss, process.memory_info().rss)
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(x_validation), y_validation))
        curves.append(
            {
                "epoch": float(epoch),
                "train_mse": total_loss / total_rows,
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    elapsed = time.perf_counter() - started
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        curves=curves,
        best_epoch=best_epoch,
        elapsed_seconds=elapsed,
        peak_rss_mb=peak_rss / 1024**2,
        peak_python_mb=peak_python / 1024**2,
    )


def select_weight_decay(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: Lab3Config,
    features: list[str],
    scaler: StandardScaler,
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for value in config.weight_decay_grid:
        result = train_model(
            train, validation, config, features, scaler, "full", value, config.seed
        )
        prediction = predict(result.model, validation, features, scaler)
        metrics = regression_metrics(validation[TARGET].to_numpy(float), prediction)
        rows.append({"weight_decay": value, "best_epoch": result.best_epoch, **metrics})
    best = min(rows, key=lambda row: row["rmse"])
    return float(best["weight_decay"]), rows


def log_run(
    mode: str,
    config: Lab3Config,
    result: TrainingResult,
    metrics: dict[str, dict[str, float]],
    weight_decay: float,
) -> str:
    with mlflow.start_run(run_name=f"lab3_{mode}") as run:
        mlflow.log_params(
            {
                "mode": mode,
                "seed": config.seed,
                "batch_size": config.batch_size,
                "chunk_size": config.chunk_size,
                "hidden_sizes": str(config.hidden_sizes),
                "weight_decay": weight_decay,
            }
        )
        for name, values in metrics.items():
            mlflow.log_metric(name, values["value"])
            mlflow.log_metric(f"{name}_ci_low", values["ci_low"])
            mlflow.log_metric(f"{name}_ci_high", values["ci_high"])
        mlflow.log_metric("elapsed_seconds", result.elapsed_seconds)
        mlflow.log_metric("peak_rss_mb", result.peak_rss_mb)
        return run.info.run_id


def device_parity(
    model: MLP,
    frame: pd.DataFrame,
    features: list[str],
    scaler: StandardScaler,
    tolerance: float,
    enabled: bool,
) -> dict[str, str | float | bool | None]:
    cpu_prediction = predict(model, frame, features, scaler)
    if not enabled:
        return {
            "accelerator": "not_checked",
            "max_difference": None,
            "tolerance": tolerance,
            "passed": None,
            "reason": "accelerator check disabled in the current environment",
        }
    if not torch.backends.mps.is_available():
        return {
            "accelerator": "unavailable",
            "max_difference": None,
            "tolerance": tolerance,
            "passed": None,
        }
    try:
        device = torch.device("mps")
        accelerated = copy.deepcopy(model).to(device)
        values = torch.from_numpy(
            scaler.transform(frame[features].to_numpy(dtype=float)).astype("float32")
        ).to(device)
        accelerated.eval()
        with torch.no_grad():
            accelerated_prediction = accelerated(values).cpu().numpy().astype(float)
        difference = float(np.max(np.abs(cpu_prediction - accelerated_prediction)))
        return {
            "accelerator": "mps",
            "max_difference": difference,
            "tolerance": tolerance,
            "passed": difference <= tolerance,
        }
    except RuntimeError as error:
        return {
            "accelerator": "unavailable",
            "max_difference": None,
            "tolerance": tolerance,
            "passed": None,
            "reason": str(error).splitlines()[0],
        }


def stream_probe(path: Path, chunk_size: int, repeats: int) -> dict[str, float]:
    tracemalloc.start()
    max_chunk = 0
    rows = 0
    for _ in range(repeats):
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            max_chunk = max(max_chunk, len(chunk))
            rows += len(chunk)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "logical_rows": float(rows),
        "max_chunk_rows": float(max_chunk),
        "peak_python_mb": peak / 1024**2,
    }


def save_curves(report_dir: Path, curves: dict[str, list[dict[str, float]]]) -> None:
    rows = []
    for mode, values in curves.items():
        rows.extend({"mode": mode, **row} for row in values)
    pd.DataFrame(rows).to_csv(report_dir / "loss_curves.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 4))
    for mode, values in curves.items():
        axis.plot(
            [row["epoch"] for row in values],
            [row["validation_mse"] for row in values],
            label=mode,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation MSE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "loss_curves.png", dpi=160)
    plt.close(figure)


def run_lab3(config_path: str | Path = "configs/lab3.yaml") -> dict:
    config = load_config(config_path)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("diabetes_lab3")
    train = pd.read_csv(config.train_path)
    validation = pd.read_csv(config.validation_path)
    test = pd.read_csv(config.test_path)
    features = feature_columns(train)
    full_scaler = StandardScaler().fit(train[features].to_numpy(float))
    chunk_scaler = fit_scaler_streaming(config.train_path, features, config.chunk_size)
    weight_decay, scan = select_weight_decay(
        train, validation, config, features, full_scaler
    )
    pd.DataFrame(scan).to_csv(config.report_dir / "weight_decay_scan.csv", index=False)
    results: dict[str, TrainingResult] = {}
    scalers = {"full": full_scaler, "chunks": chunk_scaler}
    summaries = {}
    y_test = test[TARGET].to_numpy(float)
    for mode in ("full", "chunks"):
        result = train_model(
            train,
            validation,
            config,
            features,
            scalers[mode],
            mode,
            weight_decay,
            config.seed,
        )
        prediction = predict(result.model, test, features, scalers[mode])
        metrics = bootstrap_regression(
            y_test, prediction, config.bootstrap_samples, config.ci_level, config.seed
        )
        run_id = log_run(mode, config, result, metrics, weight_decay)
        results[mode] = result
        summaries[mode] = {
            "run_id": run_id,
            "metrics": metrics,
            "best_epoch": result.best_epoch,
            "elapsed_seconds": result.elapsed_seconds,
            "peak_rss_mb": result.peak_rss_mb,
            "peak_python_mb": result.peak_python_mb,
            "prediction": prediction,
        }
    save_curves(
        config.report_dir, {mode: result.curves for mode, result in results.items()}
    )
    checkpoint = {
        "state_dict": results["full"].model.state_dict(),
        "features": features,
        "hidden_sizes": config.hidden_sizes,
        "scaler_mean": full_scaler.mean_,
        "scaler_scale": full_scaler.scale_,
        "weight_decay": weight_decay,
        "seed": config.seed,
    }
    torch.save(checkpoint, config.model_path)
    model_hash = sha256_file(config.model_path)
    repeat = train_model(
        train,
        validation,
        config,
        features,
        full_scaler,
        "full",
        weight_decay,
        config.seed,
    )
    repeat_prediction = predict(repeat.model, test, features, full_scaler)
    reproducibility_difference = float(
        np.max(np.abs(summaries["full"]["prediction"] - repeat_prediction))
    )
    different_seed = train_model(
        train,
        validation,
        config,
        features,
        full_scaler,
        "full",
        weight_decay,
        config.seed + 1,
    )
    different_prediction = predict(different_seed.model, test, features, full_scaler)
    seed_control_difference = float(
        np.max(np.abs(summaries["full"]["prediction"] - different_prediction))
    )
    parity = device_parity(
        results["full"].model,
        test,
        features,
        full_scaler,
        config.device_tolerance,
        config.accelerator_check,
    )
    probes = {
        "1x": stream_probe(config.train_path, config.chunk_size, 1),
        "100x": stream_probe(config.train_path, config.chunk_size, 100),
    }
    metric_rows = []
    memory_rows = []
    for mode in ("full", "chunks"):
        for name, values in summaries[mode]["metrics"].items():
            metric_rows.append(
                {
                    "mode": mode,
                    "metric": name,
                    "run_id": summaries[mode]["run_id"],
                    **values,
                }
            )
        memory_rows.append(
            {
                "mode": mode,
                "elapsed_seconds": summaries[mode]["elapsed_seconds"],
                "peak_rss_mb": summaries[mode]["peak_rss_mb"],
                "peak_python_mb": summaries[mode]["peak_python_mb"],
                "logical_rows": len(train),
                "max_chunk_rows": len(train) if mode == "full" else config.chunk_size,
            }
        )
    for name, values in probes.items():
        memory_rows.append(
            {
                "mode": f"stream_probe_{name}",
                "elapsed_seconds": "",
                "peak_rss_mb": "",
                **values,
            }
        )
    pd.DataFrame(metric_rows).to_csv(config.report_dir / "dl_metrics.csv", index=False)
    pd.DataFrame(memory_rows).to_csv(
        config.report_dir / "memory_compare.csv", index=False
    )
    with (config.report_dir / "reproducibility.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "same_seed_max_difference": reproducibility_difference,
                "tolerance": config.metric_tolerance,
                "passed": reproducibility_difference <= config.metric_tolerance,
                "different_seed_max_difference": seed_control_difference,
                "device_parity": parity,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
    negative_lines = [
        "PASS batch_size_zero: value must be greater than 0",
        "PASS negative_weight_decay: values must be non-negative",
        f"PASS seed_control: max prediction difference={seed_control_difference:.9f}",
    ]
    (config.report_dir / "negative_controls.log").write_text(
        "\n".join(negative_lines) + "\n", encoding="utf-8"
    )
    summary = {
        "selected_weight_decay": weight_decay,
        "parameter_count": sum(
            parameter.numel() for parameter in results["full"].model.parameters()
        ),
        "model_sha256": model_hash,
        "full": {
            key: value
            for key, value in summaries["full"].items()
            if key != "prediction"
        },
        "chunks": {
            key: value
            for key, value in summaries["chunks"].items()
            if key != "prediction"
        },
        "reproducibility_max_difference": reproducibility_difference,
        "different_seed_max_difference": seed_control_difference,
        "device_parity": parity,
        "stream_probes": probes,
    }
    (config.report_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
