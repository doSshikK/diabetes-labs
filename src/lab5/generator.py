from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from lab5.config import GenerationRequest, Lab5Config, load_config

TARGET = "target"


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != TARGET]


def apply_drift(
    frame: pd.DataFrame,
    real: pd.DataFrame,
    request: GenerationRequest,
    rng: np.random.Generator,
) -> pd.DataFrame:
    output = frame.copy()
    features = feature_columns(real)
    means = real[features].mean()
    scales = real[features].std(ddof=0).replace(0.0, 1.0)
    output[features] = means + (output[features] - means) * request.covariance_scale
    output[features] = output[features] + request.mean_shift_std * scales
    if request.target_noise_std > 0:
        noise = rng.normal(
            0.0, real[TARGET].std(ddof=0) * request.target_noise_std, len(output)
        )
        output[TARGET] = output[TARGET] + noise
    flip_count = round(len(output) * request.label_flip_fraction)
    if flip_count:
        indices = rng.choice(len(output), size=flip_count, replace=False)
        target_mean = float(real[TARGET].mean())
        output.loc[indices, TARGET] = 2.0 * target_mean - output.loc[indices, TARGET]
    return output


def generate_similar(
    real: pd.DataFrame,
    request: GenerationRequest,
    jitter_scale: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(request.seed)
    features = feature_columns(real)
    indices = rng.integers(0, len(real), size=request.n_samples)
    output = real.iloc[indices].reset_index(drop=True).copy()
    covariance = real[features].cov(ddof=0).to_numpy(dtype=float)
    jitter = rng.multivariate_normal(
        np.zeros(len(features)), covariance * jitter_scale**2, request.n_samples
    )
    output[features] = output[features].to_numpy(dtype=float) + jitter
    for feature in features:
        output[feature] = output[feature].clip(real[feature].min(), real[feature].max())
    return apply_drift(output, real, request, rng)


def generate_random(real: pd.DataFrame, request: GenerationRequest) -> pd.DataFrame:
    rng = np.random.default_rng(request.seed)
    features = feature_columns(real)
    data = {
        feature: rng.uniform(
            real[feature].min(), real[feature].max(), request.n_samples
        )
        for feature in features
    }
    data[TARGET] = rng.uniform(
        real[TARGET].min(), real[TARGET].max(), request.n_samples
    )
    return apply_drift(pd.DataFrame(data), real, request, rng)


def generate(
    real: pd.DataFrame,
    request: GenerationRequest,
    jitter_scale: float = 0.03,
) -> pd.DataFrame:
    if request.mode == "similar":
        return generate_similar(real, request, jitter_scale)
    if request.mode == "random":
        return generate_random(real, request)
    raise ValueError(f"unsupported mode: {request.mode}")


def psi(reference: np.ndarray, candidate: np.ndarray, bins: int) -> float:
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        lower = float(np.min(reference)) - 1e-9
        upper = float(np.max(reference)) + 1e-9
        edges = np.linspace(lower, upper, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_count, _ = np.histogram(reference, bins=edges)
    candidate_count, _ = np.histogram(candidate, bins=edges)
    reference_share = np.maximum(reference_count / len(reference), 1e-6)
    candidate_share = np.maximum(candidate_count / len(candidate), 1e-6)
    return float(
        np.sum(
            (candidate_share - reference_share)
            * np.log(candidate_share / reference_share)
        )
    )


def distribution_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    mode: str,
    bins: int,
) -> list[dict[str, float | str]]:
    rows = []
    for feature in feature_columns(real):
        reference = real[feature].to_numpy(dtype=float)
        candidate = synthetic[feature].to_numpy(dtype=float)
        rows.append(
            {
                "mode": mode,
                "feature": feature,
                "real_mean": float(np.mean(reference)),
                "synthetic_mean": float(np.mean(candidate)),
                "psi": psi(reference, candidate, bins),
                "ks": float(ks_2samp(reference, candidate).statistic),
            }
        )
    return rows


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.10g", lineterminator="\n").encode(
        "utf-8"
    )


def write_reproducible_csv(frame: pd.DataFrame, path: Path) -> str:
    payload = csv_bytes(frame)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def request_from_config(
    config: Lab5Config, mode: str, **updates: float
) -> GenerationRequest:
    payload = {
        "mode": mode,
        "n_samples": config.n_samples,
        "seed": config.seed,
        "mean_shift_std": config.mean_shift_std,
        "covariance_scale": config.covariance_scale,
        "target_noise_std": config.target_noise_std,
        "label_flip_fraction": config.label_flip_fraction,
        **updates,
    }
    return GenerationRequest.model_validate(payload)


def run_lab5(config_path: str | Path = "configs/lab5.yaml") -> dict:
    config = load_config(config_path)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    real = pd.read_csv(config.real_data_path)
    requests = {
        "similar": request_from_config(config, "similar"),
        "random": request_from_config(config, "random"),
    }
    generated = {
        mode: generate(real, request, config.similar_jitter_scale)
        for mode, request in requests.items()
    }
    hashes = {
        mode: write_reproducible_csv(frame, config.output_dir / f"diabetes_{mode}.csv")
        for mode, frame in generated.items()
    }
    repeated = generate(real, requests["similar"], config.similar_jitter_scale)
    repeat_hash = hashlib.sha256(csv_bytes(repeated)).hexdigest()
    metrics = []
    for mode, frame in generated.items():
        metrics.extend(distribution_metrics(real, frame, mode, config.psi_bins))
    metrics_frame = pd.DataFrame(metrics)
    similar = metrics_frame[metrics_frame["mode"] == "similar"]
    random = metrics_frame[metrics_frame["mode"] == "random"]
    similar_pass_fraction = float(
        np.mean(
            (similar["psi"] <= config.similar_threshold)
            | (similar["ks"] <= config.similar_threshold)
        )
    )
    random_pass_fraction = float(np.mean(random["psi"] >= config.random_psi_threshold))
    metrics_frame["threshold"] = np.where(
        metrics_frame["mode"] == "similar",
        config.similar_threshold,
        config.random_psi_threshold,
    )
    metrics_frame["passed"] = np.where(
        metrics_frame["mode"] == "similar",
        (metrics_frame["psi"] <= config.similar_threshold)
        | (metrics_frame["ks"] <= config.similar_threshold),
        metrics_frame["psi"] >= config.random_psi_threshold,
    )
    metrics_frame.to_csv(config.report_dir / "synthetic_metrics.csv", index=False)
    config_payload = config.model_dump(mode="json")
    config_payload["requests"] = {
        mode: request.model_dump(mode="json") for mode, request in requests.items()
    }
    (config.report_dir / "generator_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    drift_requests = {
        "mean_shift": request_from_config(config, "similar", mean_shift_std=2.0),
        "covariance": request_from_config(config, "similar", covariance_scale=1.8),
        "target_noise": request_from_config(config, "similar", target_noise_std=0.5),
        "label_flip": request_from_config(config, "similar", label_flip_fraction=0.25),
    }
    drift_summary = {}
    baseline = generated["similar"]
    for name, request in drift_requests.items():
        drifted = generate(real, request, config.similar_jitter_scale)
        feature_delta = float(
            np.max(
                np.abs(
                    drifted[feature_columns(real)].mean().to_numpy()
                    - baseline[feature_columns(real)].mean().to_numpy()
                )
            )
        )
        target_delta = float(abs(drifted[TARGET].std() - baseline[TARGET].std()))
        drift_summary[name] = {
            "feature_mean_max_difference": feature_delta,
            "target_std_difference": target_delta,
            "changed": max(feature_delta, target_delta) > 1e-8,
        }
    summary = {
        "hashes": hashes,
        "repeat_hash": repeat_hash,
        "byte_reproducible": repeat_hash == hashes["similar"],
        "similar_pass_fraction": similar_pass_fraction,
        "similar_required_fraction": config.similar_required_fraction,
        "random_pass_fraction": random_pass_fraction,
        "random_required_fraction": config.random_required_fraction,
        "thresholds_passed": bool(
            similar_pass_fraction >= config.similar_required_fraction
            and random_pass_fraction >= config.random_required_fraction
        ),
        "drift_controls": drift_summary,
    }
    (config.report_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
