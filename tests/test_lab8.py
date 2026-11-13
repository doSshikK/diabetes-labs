from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from lab8.config import Lab8Config, load_config
from lab8.manifest import verify_manifest
from lab8.pipeline import (
    create_service_app,
    load_source,
    run_alert_drill,
    run_fault_drills,
    train_dl,
    train_ml,
    train_validator,
)


def temporary_config(tmp_path: Path) -> Lab8Config:
    config = load_config("configs/lab8.yaml")
    return config.model_copy(
        update={
            "report_dir": tmp_path / "reports",
            "model_dir": tmp_path / "reports" / "models",
            "dl_epochs": 3,
            "generated_rows": 100,
        }
    )


def test_config_rejects_short_load_stage() -> None:
    payload = load_config("configs/lab8.yaml").model_dump(mode="python")
    payload["stage_duration_seconds"] = 599
    with pytest.raises(ValidationError, match="greater than or equal to 600"):
        Lab8Config.model_validate(payload)


def test_sources_have_expected_schema() -> None:
    config = load_config("configs/lab8.yaml")
    for source in ("real", "synthetic"):
        frame = load_source(config, source)
        assert len(frame) >= 442
        assert "target" in frame
        assert not frame.isna().any().any()


def test_unified_training_commands(tmp_path: Path) -> None:
    config = temporary_config(tmp_path)
    for source in ("real", "synthetic"):
        ml = train_ml(config, source)
        dl = train_dl(config, source)
        validator = train_validator(config, source)
        assert np.isfinite(ml["metrics"]["rmse"])
        assert np.isfinite(dl["metrics"]["rmse"])
        assert 0.0 <= validator["false_rejection_rate"] <= 1.0
        assert Path(ml["model_path"]).exists()
        assert Path(dl["model_path"]).exists()
        assert Path(validator["model_path"]).exists()


def test_service_contract(tmp_path: Path) -> None:
    config = temporary_config(tmp_path)
    train_ml(config, "real")
    frame = pd.read_csv(config.real_data_path)
    payload = frame.drop(columns="target").iloc[0].to_dict()
    with TestClient(create_service_app(config, "real")) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json=payload).status_code == 200
        assert client.post("/predict", json={}).status_code == 422


def test_alerts_have_required_fields_and_are_deduplicated(tmp_path: Path) -> None:
    config = temporary_config(tmp_path)
    train_validator(config, "real")
    result = run_alert_drill(config)
    assert result["passed"] is True
    lines = (config.report_dir / "alerts_sample.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 3
    required = {"timestamp", "metric", "value", "threshold", "level", "action"}
    assert all(required <= json.loads(line).keys() for line in lines)


def test_fault_drills_pass(tmp_path: Path) -> None:
    config = temporary_config(tmp_path)
    train_ml(config, "real")
    result = run_fault_drills(config)
    assert result["passed"] is True
    assert all(result[name] is True for name in ("C1", "C2", "C3", "C4", "C5"))


def test_artifact_manifest_detects_changes(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    (root / ".git").mkdir(parents=True)
    (root / "reports" / "LAB8").mkdir(parents=True)
    (root / "result.txt").write_text("stable\n", encoding="utf-8")
    output = root / "reports" / "LAB8" / "artifacts_manifest.json"
    payload = {
        "variant": "05 Diabetes",
        "code_commit": "test",
        "file_count": 1,
        "files": [
            {
                "path": "result.txt",
                "size_bytes": (root / "result.txt").stat().st_size,
                "sha256": __import__("hashlib").sha256(b"stable\n").hexdigest(),
            }
        ],
    }
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_manifest(root, output) == []
    (root / "result.txt").write_text("changed\n", encoding="utf-8")
    assert verify_manifest(root, output) == ["size mismatch: result.txt"]
