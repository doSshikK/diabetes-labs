from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from lab6.config import Lab6Config, load_config
from lab6.run import valid_payload
from lab6.service import ModelService, create_app


@pytest.fixture
def client_and_service() -> tuple[TestClient, ModelService]:
    config = load_config()
    app, service = create_app(config, persist_registry=False)
    return TestClient(app), service


def test_contract_validation(
    client_and_service: tuple[TestClient, ModelService],
) -> None:
    client, _ = client_and_service
    payload = valid_payload(load_config())
    assert client.post("/predict", json=payload).status_code == 200
    assert client.post("/predict", json={**payload, "bmi": "bad"}).status_code == 422
    assert client.post("/predict", json={**payload, "bmi": 5.0}).status_code == 422
    payload.pop("bmi")
    assert client.post("/predict", json=payload).status_code == 422


def test_endpoints_promote_and_rollback(
    client_and_service: tuple[TestClient, ModelService],
) -> None:
    client, _ = client_and_service
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert len(client.get("/models").json()["versions"]) >= 2
    assert (
        client.post("/promote", json={"version": "v2-chunks"}).json()["active_version"]
        == "v2-chunks"
    )
    assert client.post("/rollback").json()["active_version"] == "v1-full"


def test_hash_mismatch_is_rejected() -> None:
    payload = copy.deepcopy(load_config().model_dump(mode="python"))
    payload["versions"][0]["sha256"] = "0" * 64
    config = Lab6Config.model_validate(payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        ModelService(config, persist_registry=False)


def test_bad_config_is_rejected() -> None:
    payload = load_config().model_dump(mode="python")
    with pytest.raises(ValidationError):
        Lab6Config.model_validate({**payload, "initial_version": "unknown"})
