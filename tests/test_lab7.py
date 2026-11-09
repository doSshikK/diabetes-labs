from __future__ import annotations

import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from lab7.config import Lab7Config, load_config
from lab7.pipeline import create_batch_app
from lab7.run import repeat_features, rows_from_frame, run_drift_monitoring


@pytest.fixture
def batch_context():
    config = load_config()
    app, predictor = create_batch_app(config)
    real = pd.read_csv(config.real_data_path)
    rows = rows_from_frame(repeat_features(real, 100))
    return config, app, predictor, real, rows


def test_single_and_parallel_predictions_match(batch_context) -> None:
    config, _, predictor, _, rows = batch_context
    single = predictor.predict(rows, 1)
    parallel = predictor.predict(rows, config.max_workers)
    assert parallel.predictions == pytest.approx(single.predictions)


def test_batch_endpoint_and_validation(batch_context) -> None:
    config, app, _, _, rows = batch_context
    payload = [row.model_dump() for row in rows[:10]]
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch", json={"rows": payload, "workers": config.max_workers}
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 10
        assert (
            client.post("/predict/batch", json={"rows": [], "workers": 1}).status_code
            == 422
        )
        assert (
            client.post(
                "/predict/batch",
                json={"rows": payload, "workers": config.max_workers + 1},
            ).status_code
            == 422
        )


def test_worker_failure_is_retried(batch_context) -> None:
    config, _, predictor, _, rows = batch_context
    result = predictor.predict(rows, config.max_workers, fail_chunk_index=0)
    assert result.retried_chunks == 1
    assert len(result.predictions) == len(rows)


def test_drift_controls(batch_context) -> None:
    config, _, predictor, real, _ = batch_context
    _, summary, _ = run_drift_monitoring(predictor, real, config)
    assert summary["clean_passed"]
    assert summary["detected_by_deadline"]


def test_bad_worker_config_is_rejected() -> None:
    payload = copy.deepcopy(load_config().model_dump(mode="python"))
    payload["benchmark_workers"] = [1, payload["max_workers"] + 1]
    with pytest.raises(ValidationError):
        Lab7Config.model_validate(payload)
