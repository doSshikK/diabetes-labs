from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from lab6.config import Lab6Config, load_config
from lab6.service import create_app, create_monitoring_app, write_monitoring_artifacts


def valid_payload(config: Lab6Config) -> dict[str, float]:
    frame = pd.read_csv(config.raw_data_path)
    return {
        name: float(frame.iloc[0][name])
        for name in ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    }


def run_contracts(
    client: TestClient, payload: dict[str, float]
) -> list[dict[str, Any]]:
    cases = [
        ("valid", payload, 200),
        ("missing_field", {k: v for k, v in payload.items() if k != "bmi"}, 422),
        ("wrong_type", {**payload, "bmi": "invalid"}, 422),
        ("out_of_range", {**payload, "bmi": 10.0}, 422),
    ]
    rows = []
    for name, body, expected in cases:
        response = client.post("/predict", json=body)
        rows.append(
            {
                "case": name,
                "expected_code": expected,
                "actual_code": response.status_code,
                "passed": response.status_code == expected,
                "detail": response.json().get("detail", "ok"),
            }
        )
    return rows


def run_endpoint_load(
    client: TestClient,
    payload: dict[str, float],
    config: Lab6Config,
) -> list[dict[str, Any]]:
    calls = {
        "/health": lambda: client.get("/health"),
        "/predict": lambda: client.post("/predict", json=payload),
        "/metrics": lambda: client.get("/metrics"),
        "/models": lambda: client.get("/models"),
    }
    rows = []
    for endpoint, call in calls.items():
        started = time.perf_counter()
        codes = [call().status_code for _ in range(config.load_requests_per_endpoint)]
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "endpoint": endpoint,
                "requests": len(codes),
                "successes": sum(code < 400 for code in codes),
                "errors": sum(code >= 400 for code in codes),
                "elapsed_seconds": elapsed,
            }
        )
    started = time.perf_counter()
    promote_codes = []
    rollback_codes = []
    for _ in range(config.load_requests_per_endpoint):
        promote_codes.append(
            client.post("/promote", json={"version": "v2-chunks"}).status_code
        )
        rollback_codes.append(client.post("/rollback").status_code)
    elapsed = time.perf_counter() - started
    for endpoint, codes in (("/promote", promote_codes), ("/rollback", rollback_codes)):
        rows.append(
            {
                "endpoint": endpoint,
                "requests": len(codes),
                "successes": sum(code < 400 for code in codes),
                "errors": sum(code >= 400 for code in codes),
                "elapsed_seconds": elapsed / 2,
            }
        )
    return rows


def run_switch_drill(
    client: TestClient,
    payload: dict[str, float],
    config: Lab6Config,
) -> dict[str, Any]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.max_load_workers) as pool:
        futures = [
            pool.submit(client.post, "/predict", json=payload)
            for _ in range(config.drill_requests)
        ]
        promoted = client.post("/promote", json={"version": "v2-chunks"})
        rolled_back = client.post("/rollback")
        responses = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    successes = sum(response.status_code == 200 for response in responses)
    return {
        "requests": len(responses),
        "successes": successes,
        "lost_requests": len(responses) - successes,
        "success_rate": successes / len(responses),
        "promote_code": promoted.status_code,
        "rollback_code": rolled_back.status_code,
        "elapsed_seconds": elapsed,
    }


def run_lab6(config_path: str | Path = "configs/lab6.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    app, service = create_app(config)
    monitor_app = create_monitoring_app(service)
    payload = valid_payload(config)
    samples = []
    with TestClient(app) as client, TestClient(monitor_app) as monitor:
        contracts = run_contracts(client, payload)
        samples.append({"sample": 0, **monitor.get("/snapshot").json()})
        endpoint_load = run_endpoint_load(client, payload, config)
        samples.append({"sample": 1, **monitor.get("/snapshot").json()})
        drill = run_switch_drill(client, payload, config)
        samples.append({"sample": 2, **monitor.get("/snapshot").json()})
        monitor_health = monitor.get("/health").status_code

    pd.DataFrame(contracts).to_csv(
        config.report_dir / "contract_cases.csv", index=False
    )
    pd.DataFrame(endpoint_load).to_csv(
        config.report_dir / "endpoint_load.csv", index=False
    )
    write_monitoring_artifacts(samples, config.report_dir)
    service.save_registry()
    summary = {
        "contracts_passed": all(item["passed"] for item in contracts),
        "endpoint_load": endpoint_load,
        "switch_drill": drill,
        "monitor_health_code": monitor_health,
        "final_metrics": service.metrics(),
        "registry_path": str(config.registry_path),
    }
    (config.report_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run laboratory work 6")
    parser.add_argument("--config", default="configs/lab6.yaml")
    args = parser.parse_args()
    print(json.dumps(run_lab6(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
