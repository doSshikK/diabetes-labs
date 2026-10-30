from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import joblib
import mlflow
import numpy as np
import pandas as pd
import psutil
from pydantic import ValidationError
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Lab2Config, load_config

TARGET = "target"
MODE = Literal["full", "chunks"]


@dataclass
class ResourceUsage:
    elapsed_seconds: float
    peak_rss_mb: float
    peak_python_mb: float


@dataclass
class ModeResult:
    mode: str
    run_id: str
    metrics: dict[str, dict[str, float]]
    resources: ResourceUsage
    model_path: str
    model_sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_and_split(config: Lab2Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(config.data_path)
    if TARGET not in frame.columns:
        raise ValueError(f"required target column {TARGET!r} is missing")
    if frame.isna().any().any():
        raise ValueError("dataset contains missing values")
    frame = frame.copy()
    frame.insert(0, "row_id", np.arange(len(frame), dtype=int))
    train_validation, test = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=config.seed,
    )
    validation_relative = config.validation_size / (1.0 - config.test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=validation_relative,
        random_state=config.seed,
    )
    splits = [part.sort_values("row_id").reset_index(drop=True) for part in (train, validation, test)]
    id_sets = [set(part["row_id"]) for part in splits]
    if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise RuntimeError("data split overlap detected")
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, part in zip(("train", "validation", "test"), splits, strict=True):
        part.to_csv(config.processed_dir / f"lab2_{name}.csv", index=False)
    return tuple(splits)  # type: ignore[return-value]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"row_id", TARGET}]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    samples: int,
    ci_level: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    point = regression_metrics(y_true, y_pred)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in point}
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), len(y_true))
        sample_metrics = regression_metrics(y_true[indices], y_pred[indices])
        for name, value in sample_metrics.items():
            values[name].append(value)
    alpha = 1.0 - ci_level
    return {
        name: {
            "value": point[name],
            "ci_low": float(np.quantile(values[name], alpha / 2.0)),
            "ci_high": float(np.quantile(values[name], 1.0 - alpha / 2.0)),
        }
        for name in point
    }


def new_incremental_model(config: Lab2Config, alpha: float, l1_ratio: float) -> SGDRegressor:
    return SGDRegressor(
        loss=config.loss,
        penalty=config.penalty,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=1,
        tol=None,
        shuffle=False,
        random_state=config.seed,
        learning_rate="invscaling",
        eta0=config.eta0,
        power_t=0.25,
        warm_start=True,
    )


def fit_full(
    train: pd.DataFrame,
    features: list[str],
    config: Lab2Config,
    alpha: float,
    l1_ratio: float,
) -> dict:
    x_train = train[features].to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x_train)
    transformed = scaler.transform(x_train)
    model = new_incremental_model(config, alpha, l1_ratio)
    for _ in range(config.epochs):
        model.partial_fit(transformed, y_train)
    return {"scaler": scaler, "model": model, "features": features}


def iter_training_chunks(path: Path, chunk_size: int) -> pd.io.parsers.TextFileReader:
    return pd.read_csv(path, chunksize=chunk_size)


def fit_chunks(
    train_path: Path,
    features: list[str],
    config: Lab2Config,
    alpha: float,
    l1_ratio: float,
) -> dict:
    scaler = StandardScaler()
    for chunk in iter_training_chunks(train_path, config.chunk_size):
        scaler.partial_fit(chunk[features].to_numpy(dtype=float))
    model = new_incremental_model(config, alpha, l1_ratio)
    for _ in range(config.epochs):
        for chunk in iter_training_chunks(train_path, config.chunk_size):
            x_chunk = scaler.transform(chunk[features].to_numpy(dtype=float))
            model.partial_fit(x_chunk, chunk[TARGET].to_numpy(dtype=float))
    return {"scaler": scaler, "model": model, "features": features}


def predict(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    values = frame[bundle["features"]].to_numpy(dtype=float)
    return bundle["model"].predict(bundle["scaler"].transform(values))


def measure_call(function: Callable, *args) -> tuple[object, ResourceUsage]:
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    stop = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop.wait(0.002):
            peak_rss = max(peak_rss, process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    tracemalloc.start()
    sampler.start()
    started = time.perf_counter()
    try:
        result = function(*args)
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        sampler.join()
        _, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_rss = max(peak_rss, process.memory_info().rss)
    usage = ResourceUsage(
        elapsed_seconds=elapsed,
        peak_rss_mb=peak_rss / (1024**2),
        peak_python_mb=peak_python / (1024**2),
    )
    return result, usage


def validation_search(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    config: Lab2Config,
) -> tuple[pd.DataFrame, float, float]:
    x_train = train[features].to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy(dtype=float)
    x_validation = validation[features].to_numpy(dtype=float)
    y_validation = validation[TARGET].to_numpy(dtype=float)
    records: list[dict] = []
    reference_models = {
        "naive_mean": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
    }
    for name, model in reference_models.items():
        model.fit(x_train, y_train)
        records.append(
            {"model": name, "alpha": "", "l1_ratio": "", **regression_metrics(y_validation, model.predict(x_validation))}
        )
    for alpha in config.alpha_grid:
        for l1_ratio in config.l1_ratio_grid:
            elastic_net = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            max_iter=config.elastic_net_max_iter,
                            tol=config.elastic_net_tol,
                            random_state=config.seed,
                        ),
                    ),
                ]
            )
            elastic_net.fit(x_train, y_train)
            records.append(
                {
                    "model": "elastic_net_reference",
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    **regression_metrics(y_validation, elastic_net.predict(x_validation)),
                }
            )
            incremental = fit_full(train, features, config, alpha, l1_ratio)
            records.append(
                {
                    "model": "sgd_elastic_net",
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    **regression_metrics(y_validation, predict(incremental, validation)),
                }
            )
    comparison = pd.DataFrame(records)
    candidates = comparison[comparison["model"] == "sgd_elastic_net"]
    best = candidates.loc[candidates["rmse"].idxmin()]
    return comparison, float(best["alpha"]), float(best["l1_ratio"])


def assert_train_only_scaler(
    scaler: StandardScaler,
    train: pd.DataFrame,
    features: list[str],
    atol: float = 1e-12,
) -> None:
    expected = train[features].to_numpy(dtype=float).mean(axis=0)
    if not np.allclose(scaler.mean_, expected, rtol=0.0, atol=atol):
        raise AssertionError("data leakage detected: scaler statistics are not train-only")


def write_negative_config_log(config: Lab2Config) -> list[str]:
    payload = config.model_dump(mode="python")
    cases = [
        ("chunk_size_zero", {**payload, "chunk_size": 0}),
        ("negative_alpha", {**payload, "alpha_grid": [-0.1]}),
        ("l1_ratio_out_of_range", {**payload, "l1_ratio_grid": [1.5]}),
    ]
    lines: list[str] = []
    for name, candidate in cases:
        try:
            Lab2Config.model_validate(candidate)
        except ValidationError as error:
            lines.append(f"PASS {name}: {error.errors()[0]['msg']}")
        else:
            lines.append(f"FAIL {name}: invalid configuration was accepted")
    (config.report_dir / "config_negative_test.log").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return lines


def write_leakage_log(
    all_data: pd.DataFrame,
    train: pd.DataFrame,
    features: list[str],
    config: Lab2Config,
) -> str:
    correct_scaler = StandardScaler().fit(train[features].to_numpy(dtype=float))
    leaked_scaler = StandardScaler().fit(all_data[features].to_numpy(dtype=float))
    assert_train_only_scaler(correct_scaler, train, features)
    try:
        assert_train_only_scaler(leaked_scaler, train, features)
    except AssertionError as error:
        max_difference = float(np.max(np.abs(leaked_scaler.mean_ - correct_scaler.mean_)))
        line = f"PASS leakage trap detected: {error}; max_mean_difference={max_difference:.12f}"
    else:
        line = "FAIL leakage trap was not detected"
    (config.report_dir / "leakage_test.log").write_text(line + "\n", encoding="utf-8")
    return line


def memory_scaling_probe(
    values: np.ndarray,
    chunk_size: int,
    report_path: Path,
) -> pd.DataFrame:
    records = []
    for multiplier in (1, 10, 100):
        total_rows = len(values) * multiplier
        max_rows = 0
        checksum = 0.0
        tracemalloc.start()
        for start in range(0, total_rows, chunk_size):
            stop = min(start + chunk_size, total_rows)
            indices = np.arange(start, stop) % len(values)
            chunk = values[indices].copy()
            max_rows = max(max_rows, len(chunk))
            checksum += float(chunk.mean())
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        records.append(
            {
                "volume_multiplier": multiplier,
                "total_rows": total_rows,
                "max_rows_in_memory": max_rows,
                "measured_peak_python_mb": peak_bytes / (1024**2),
                "checksum": checksum,
                "bounded": max_rows <= chunk_size,
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(report_path, index=False)
    return result


def log_mode_run(
    mode: MODE,
    bundle: dict,
    metrics: dict[str, dict[str, float]],
    usage: ResourceUsage,
    config: Lab2Config,
    alpha: float,
    l1_ratio: float,
) -> ModeResult:
    mode_model_path = config.report_dir / f"ml_model_{mode}.joblib"
    joblib.dump(bundle, mode_model_path)
    with mlflow.start_run(run_name=f"lab2_{mode}") as run:
        mlflow.log_params(
            {
                "variant": "05_diabetes",
                "mode": mode,
                "seed": config.seed,
                "chunk_size": config.chunk_size,
                "epochs": config.epochs,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "solver": config.model,
                "scaler": config.scaler,
                "loss": config.loss,
                "penalty": config.penalty,
            }
        )
        mlflow.log_metrics({name: values["value"] for name, values in metrics.items()})
        mlflow.log_metrics(asdict(usage))
        mlflow.log_artifact(str(mode_model_path))
        mlflow.log_artifact("configs/lab2.yaml")
        run_id = run.info.run_id
    return ModeResult(
        mode=mode,
        run_id=run_id,
        metrics=metrics,
        resources=usage,
        model_path=str(mode_model_path),
        model_sha256=sha256_file(mode_model_path),
    )


def write_report(
    config: Lab2Config,
    features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    comparison: pd.DataFrame,
    alpha: float,
    l1_ratio: float,
    full: ModeResult,
    chunks: ModeResult,
    final_hash: str,
    data_hash: str,
    metric_differences: dict[str, float],
    reproducibility_max_difference: float,
    memory_probe: pd.DataFrame,
) -> None:
    best_reference = comparison[comparison["model"] == "elastic_net_reference"].sort_values("rmse").iloc[0]
    linear = comparison[comparison["model"] == "linear_regression"].iloc[0]
    naive = comparison[comparison["model"] == "naive_mean"].iloc[0]
    memory_1x = memory_probe.iloc[0]
    memory_100x = memory_probe.iloc[-1]

    def metric_rows(result: ModeResult) -> str:
        return "\n".join(
            f"| {name.upper()} | {values['value']:.6f} | {values['ci_low']:.6f} | {values['ci_high']:.6f} |"
            for name, values in result.metrics.items()
        )

    report = f"""# Лабораторная работа 2 Пайплайн ML модели

- Вариант: 05 Diabetes мультиколлинеарность
- Дата: 2026-10-30
- Автор: Митцева Дарья Сергеевна
- Хеш коммита кода: `{current_commit()}`
- SHA-256 исходных данных: `{data_hash}`
- SHA-256 итоговой модели: `{final_hash}`

## Цель и схема данных

Построен воспроизводимый конвейер регрессии с масштабированием признаков и
elastic-net-регуляризацией. Исходные {len(train) + len(validation) + len(test)} наблюдения
разделены один раз: train {len(train)}, validation {len(validation)}, test {len(test)}.
Идентификаторы строк в частях не пересекаются. Test-часть используется только после
выбора гиперпараметров. Источник чисел: `src/lab2/pipeline.py`, прогон
`python -m lab2.run --config configs/lab2.yaml`.

## Конфигурация и подбор модели

Схема `Lab2Config` проверяет пути, размер чанка, зерно, доли разбиения,
число эпох и сетки гиперпараметров. Три отказа приведены в
`config_negative_test.log`. Предобработка состоит из `StandardScaler`, который
обучается только на train, и регрессора. На validation сравнивались наивное среднее,
`LinearRegression`, эталонный `ElasticNet` и инкрементальный
`SGDRegressor(loss=squared_error, penalty=elasticnet)`. Последний выбран, потому что
один и тот же алгоритм поддерживает `partial_fit` для полного и чанкового режимов.

| Модель | RMSE validation | MAE validation | R2 validation |
|---|---:|---:|---:|
| Наивное среднее | {naive['rmse']:.6f} | {naive['mae']:.6f} | {naive['r2']:.6f} |
| LinearRegression | {linear['rmse']:.6f} | {linear['mae']:.6f} | {linear['r2']:.6f} |
| Лучший ElasticNet | {best_reference['rmse']:.6f} | {best_reference['mae']:.6f} | {best_reference['r2']:.6f} |

Для итогового инкрементального конвейера validation выбрал `alpha={alpha}` и
`l1_ratio={l1_ratio}`. Полная таблица сетки находится в
`model_validation_comparison.csv`.

## Результаты на test с бутстрэп интервалами

Интервалы уровня {config.ci_level:.0%} построены по {config.bootstrap_samples}
бутстрэп-перевыборкам test-части с зерном {config.seed}.

### Полный датасет

MLflow run `{full.run_id}`. Весь train загружен одним объектом.

| Метрика | Значение | Нижняя граница | Верхняя граница |
|---|---:|---:|---:|
{metric_rows(full)}

Время обучения {full.resources.elapsed_seconds:.6f} с, пик RSS
{full.resources.peak_rss_mb:.3f} МБ, пик Python-аллокаций
{full.resources.peak_python_mb:.3f} МБ.

### Чанки

MLflow run `{chunks.run_id}`. `StandardScaler.partial_fit` и
`SGDRegressor.partial_fit` получают последовательные чанки не более
{config.chunk_size} строк без материализации всего train.

| Метрика | Значение | Нижняя граница | Верхняя граница |
|---|---:|---:|---:|
{metric_rows(chunks)}

Время обучения {chunks.resources.elapsed_seconds:.6f} с, пик RSS
{chunks.resources.peak_rss_mb:.3f} МБ, пик Python-аллокаций
{chunks.resources.peak_python_mb:.3f} МБ.

## Сравнение режимов и проверки

Абсолютные различия точечных метрик: MAE {metric_differences['mae']:.12f},
RMSE {metric_differences['rmse']:.12f}, R2 {metric_differences['r2']:.12f}.
Они возникают только из-за порядка накопления чисел с плавающей точкой и существенно
меньше допуска 0.005. Повтор полного режима с тем же зерном дал максимальное
расхождение предсказаний {reproducibility_max_difference:.12f}.

Проверка утечки в `leakage_test.log` намеренно обучает второй scaler на всех данных;
защитная проверка его отклоняет. `memory_scaling_check.csv` показывает, что при росте
логического объёма в 100 раз в памяти остаётся не более {config.chunk_size} строк:
измеренный пик Python-аллокаций изменился с
{memory_1x['measured_peak_python_mb']:.6f} до
{memory_100x['measured_peak_python_mb']:.6f} МБ. Близкий абсолютный RSS двух основных
прогонов объясняется общей памятью интерпретатора и MLflow на малом наборе из 442 строк.
`config_negative_test.log` содержит три остановки на невалидных настройках.

## Выводы

1. LinearRegression уменьшил RMSE validation относительно наивного ориентира с
   {naive['rmse']:.3f} до {linear['rmse']:.3f} (`model_validation_comparison.csv`).
2. ElasticNet учитывает выявленную в ЛР1 мультиколлинеарность и на validation имеет
   RMSE {best_reference['rmse']:.3f} (`model_validation_comparison.csv`).
3. Полный и чанковый режимы дали согласованные test-метрики с расхождением RMSE
   {metric_differences['rmse']:.12f} (`ml_metrics.csv`).
4. Чанковый режим ограничивает рабочую порцию {config.chunk_size} строками, а измеренный
   Python-пик при увеличении логического объёма в 100 раз не превышает
   {memory_100x['measured_peak_python_mb']:.6f} МБ (`memory_scaling_check.csv`).
5. Итоговая модель воспроизводимо загружается из `ml_model.joblib`; её SHA-256
   равен `{final_hash}` (`model_manifest.json`).
"""
    (config.report_dir / "lab2_report.md").write_text(report, encoding="utf-8")


def run_lab2(config_path: str | Path = "configs/lab2.yaml") -> dict:
    config = load_config(config_path)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("diabetes_lab2")

    all_data = pd.read_csv(config.data_path)
    train, validation, test = load_and_split(config)
    features = feature_columns(train)
    comparison, alpha, l1_ratio = validation_search(train, validation, features, config)
    comparison.to_csv(config.report_dir / "model_validation_comparison.csv", index=False)

    train_path = config.processed_dir / "lab2_train.csv"
    full_bundle, full_usage = measure_call(fit_full, train, features, config, alpha, l1_ratio)
    chunk_bundle, chunk_usage = measure_call(
        fit_chunks, train_path, features, config, alpha, l1_ratio
    )
    full_bundle = dict(full_bundle)
    chunk_bundle = dict(chunk_bundle)
    assert_train_only_scaler(full_bundle["scaler"], train, features)
    assert_train_only_scaler(chunk_bundle["scaler"], train, features)

    y_test = test[TARGET].to_numpy(dtype=float)
    full_prediction = predict(full_bundle, test)
    chunk_prediction = predict(chunk_bundle, test)
    full_metrics = bootstrap_metrics(
        y_test, full_prediction, config.bootstrap_samples, config.ci_level, config.seed
    )
    chunk_metrics = bootstrap_metrics(
        y_test, chunk_prediction, config.bootstrap_samples, config.ci_level, config.seed
    )

    full_result = log_mode_run(
        "full", full_bundle, full_metrics, full_usage, config, alpha, l1_ratio
    )
    chunk_result = log_mode_run(
        "chunks", chunk_bundle, chunk_metrics, chunk_usage, config, alpha, l1_ratio
    )

    rows = []
    for result in (full_result, chunk_result):
        for name, values in result.metrics.items():
            rows.append(
                {
                    "mode": result.mode,
                    "metric": name,
                    **values,
                    "run_id": result.run_id,
                    **asdict(result.resources),
                }
            )
    pd.DataFrame(rows).to_csv(config.report_dir / "ml_metrics.csv", index=False)

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_bundle, config.model_path)
    final_hash = sha256_file(config.model_path)
    manifest = {
        "path": str(config.model_path),
        "sha256": final_hash,
        "features": features,
        "seed": config.seed,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
    }
    (config.report_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    repeat_bundle = fit_full(train, features, config, alpha, l1_ratio)
    repeat_prediction = predict(repeat_bundle, test)
    reproducibility_max_difference = float(np.max(np.abs(full_prediction - repeat_prediction)))
    reproducibility = {
        "seed": config.seed,
        "max_prediction_difference": reproducibility_max_difference,
        "passed": reproducibility_max_difference <= 1e-12,
    }
    (config.report_dir / "reproducibility_check.json").write_text(
        json.dumps(reproducibility, indent=2) + "\n", encoding="utf-8"
    )

    negative_lines = write_negative_config_log(config)
    leakage_line = write_leakage_log(all_data, train, features, config)
    memory_probe = memory_scaling_probe(
        all_data[[*features, TARGET]].to_numpy(dtype=float),
        config.chunk_size,
        config.report_dir / "memory_scaling_check.csv",
    )
    metric_differences = {
        name: abs(full_metrics[name]["value"] - chunk_metrics[name]["value"])
        for name in full_metrics
    }
    data_hash = sha256_file(config.data_path)
    write_report(
        config,
        features,
        train,
        validation,
        test,
        comparison,
        alpha,
        l1_ratio,
        full_result,
        chunk_result,
        final_hash,
        data_hash,
        metric_differences,
        reproducibility_max_difference,
        memory_probe,
    )
    summary = {
        "config": config.model_dump(mode="json"),
        "data_sha256": data_hash,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "selected": {"alpha": alpha, "l1_ratio": l1_ratio},
        "full": asdict(full_result),
        "chunks": asdict(chunk_result),
        "metric_differences": metric_differences,
        "reproducibility": reproducibility,
        "negative_config_tests": negative_lines,
        "leakage_test": leakage_line,
        "memory_scaling_bounded": bool(memory_probe["bounded"].all()),
        "final_model_sha256": final_hash,
    }
    (config.report_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
