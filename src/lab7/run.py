from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from fastapi.testclient import TestClient
from scipy.stats import ks_2samp

from lab5.config import GenerationRequest
from lab5.generator import feature_columns, generate, psi
from lab6.schemas import FeatureRequest
from lab7.config import Lab7Config, load_config
from lab7.pipeline import ParallelBatchPredictor, create_batch_app


def rows_from_frame(frame: pd.DataFrame) -> list[FeatureRequest]:
    return [
        FeatureRequest.model_validate(row) for row in frame.to_dict(orient="records")
    ]


def repeat_features(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    features = feature_columns(frame)
    indices = np.arange(rows) % len(frame)
    return frame.iloc[indices][features].reset_index(drop=True)


def benchmark(
    predictor: ParallelBatchPredictor,
    rows: list[FeatureRequest],
    config: Lab7Config,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for workers in config.benchmark_workers:
        for _ in range(config.benchmark_warmups):
            predictor.predict(rows, workers)
        latencies = []
        for repeat in range(1, config.benchmark_repeats + 1):
            started = time.perf_counter()
            result = predictor.predict(rows, workers)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            records.append(
                {
                    "workers": workers,
                    "repeat": repeat,
                    "rows": len(rows),
                    "latency_seconds": elapsed,
                    "throughput_rows_per_second": len(rows) / elapsed,
                    "retried_chunks": result.retried_chunks,
                }
            )
        records.append(
            {
                "workers": workers,
                "repeat": "mean",
                "rows": len(rows),
                "latency_seconds": float(np.mean(latencies)),
                "throughput_rows_per_second": len(rows) / float(np.mean(latencies)),
                "retried_chunks": 0,
            }
        )
    return pd.DataFrame(records)


def repeated_reference(real: pd.DataFrame, windows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = [
        real.iloc[rng.permutation(len(real))].reset_index(drop=True)
        for _ in range(windows)
    ]
    return pd.concat(parts, ignore_index=True)


def predict_frame(predictor: ParallelBatchPredictor, frame: pd.DataFrame) -> np.ndarray:
    rows = rows_from_frame(frame[feature_columns(frame)])
    return np.asarray(predictor.predict(rows, workers=1).predictions)


def window_metrics(
    reference: pd.DataFrame,
    stream: pd.DataFrame,
    predictor: ParallelBatchPredictor,
    scenario: str,
    drift_start_window: int | None,
    config: Lab7Config,
) -> pd.DataFrame:
    features = feature_columns(reference)
    reference_predictions = predict_frame(predictor, reference)
    rows: list[dict[str, Any]] = []
    for window_index, start in enumerate(range(0, len(stream), config.window_size)):
        window = stream.iloc[start : start + config.window_size]
        if len(window) < config.window_size:
            continue
        exceeded = 0
        feature_values: dict[str, tuple[float, float]] = {}
        for feature in features:
            reference_values = reference[feature].to_numpy(dtype=float)
            current_values = window[feature].to_numpy(dtype=float)
            psi_value = psi(reference_values, current_values, config.psi_bins)
            ks_value = float(ks_2samp(reference_values, current_values).statistic)
            feature_values[feature] = (psi_value, ks_value)
            exceeded += int(
                psi_value >= config.psi_threshold and ks_value >= config.ks_threshold
            )
        current_predictions = predict_frame(predictor, window)
        csi = psi(reference_predictions, current_predictions, config.psi_bins)
        exceeded_fraction = exceeded / len(features)
        alert = (
            exceeded_fraction >= config.feature_alert_fraction
            or csi >= config.csi_threshold
        )
        for feature, (psi_value, ks_value) in feature_values.items():
            rows.append(
                {
                    "scenario": scenario,
                    "window": window_index,
                    "drift_expected": drift_start_window is not None
                    and window_index >= drift_start_window,
                    "feature": feature,
                    "psi": psi_value,
                    "ks": ks_value,
                    "csi_prediction": csi,
                    "exceeded_feature_fraction": exceeded_fraction,
                    "alert": alert,
                }
            )
    return pd.DataFrame(rows)


def first_alert_window(metrics: pd.DataFrame) -> int | None:
    alerts = metrics.loc[metrics["alert"], "window"]
    return None if alerts.empty else int(alerts.min())


def run_drift_monitoring(
    predictor: ParallelBatchPredictor,
    real: pd.DataFrame,
    config: Lab7Config,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    clean = repeated_reference(real, config.clean_windows, config.seed)
    request = GenerationRequest(
        mode="similar",
        n_samples=config.window_size * config.drift_windows,
        seed=config.seed,
        mean_shift_std=config.mean_shift_std,
        covariance_scale=1.0,
        target_noise_std=0.0,
        label_flip_fraction=0.0,
    )
    drifted = generate(real, request)
    drift_stream = pd.concat(
        [repeated_reference(real, 1, config.seed + 1), drifted], ignore_index=True
    )
    clean_metrics = window_metrics(real, clean, predictor, "clean", None, config)
    drift_metrics = window_metrics(
        real, drift_stream, predictor, "mean_shift", 1, config
    )
    metrics = pd.concat([clean_metrics, drift_metrics], ignore_index=True)
    clean_alert = first_alert_window(clean_metrics)
    drift_alert = first_alert_window(drift_metrics)
    detection_delay = None if drift_alert is None else drift_alert - 1
    summary = {
        "clean_false_alerts": int(
            clean_metrics.groupby("window")["alert"].first().sum()
        ),
        "first_drift_alert_window": drift_alert,
        "detection_delay_windows": detection_delay,
        "detected_by_deadline": drift_alert is not None
        and detection_delay is not None
        and detection_delay < config.max_detection_windows,
        "clean_passed": clean_alert is None,
    }
    return metrics, summary, drifted.iloc[: config.window_size].copy()


def save_evidently_report(
    reference: pd.DataFrame, current: pd.DataFrame, report_dir: Path
) -> None:
    snapshot = Report([DataDriftPreset()]).run(
        reference_data=reference,
        current_data=current,
    )
    snapshot.save_html(str(report_dir / "evidently_report.html"))
    snapshot.save_json(str(report_dir / "evidently_report.json"))


def run_lab7(config_path: str | Path = "configs/lab7.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    app, predictor = create_batch_app(config)
    real = pd.read_csv(config.real_data_path)
    benchmark_frame = repeat_features(real, config.benchmark_rows)
    benchmark_rows = rows_from_frame(benchmark_frame)
    benchmark_frame_result = benchmark(predictor, benchmark_rows, config)
    benchmark_frame_result.to_csv(
        config.report_dir / "parallel_benchmark.csv", index=False
    )

    with TestClient(app) as client:
        sample = [row.model_dump() for row in benchmark_rows[:20]]
        batch_response = client.post(
            "/predict/batch", json={"rows": sample, "workers": config.max_workers}
        )

    failure_control = predictor.predict(
        benchmark_rows[:100], workers=config.max_workers, fail_chunk_index=0
    )
    failure_summary = {
        "injected_failures": 1,
        "retried_chunks": failure_control.retried_chunks,
        "predictions_returned": len(failure_control.predictions),
        "service_survived": len(failure_control.predictions) == 100,
    }
    (config.report_dir / "worker_failure.json").write_text(
        json.dumps(failure_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    drift_metrics, drift_summary, detected_window = run_drift_monitoring(
        predictor, real, config
    )
    drift_metrics.to_csv(config.report_dir / "drift_metrics.csv", index=False)
    save_evidently_report(real, detected_window, config.report_dir)

    means = benchmark_frame_result[benchmark_frame_result["repeat"] == "mean"].copy()
    single_latency = float(means.loc[means["workers"] == 1, "latency_seconds"].iloc[0])
    means["speedup_vs_one_worker"] = single_latency / means["latency_seconds"]
    benchmark_summary = means.to_dict(orient="records")
    summary = {
        "batch_endpoint_code": batch_response.status_code,
        "batch_predictions": len(batch_response.json().get("predictions", [])),
        "cpu_count": __import__("os").cpu_count(),
        "benchmark": benchmark_summary,
        "worker_failure_control": failure_summary,
        "drift": drift_summary,
    }
    (config.report_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run laboratory work 7")
    parser.add_argument("--config", default="configs/lab7.yaml")
    args = parser.parse_args()
    print(json.dumps(run_lab7(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
