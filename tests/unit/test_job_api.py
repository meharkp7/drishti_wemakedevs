"""
Unit tests for the DRISHTI analysis-job HTTP API.
"""

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.jobs import get_job_service, router
from backend.services.job_repository import LocalFileJobRepository
from backend.services.job_service import JobService


@pytest.fixture
def client(tmp_path):
    repository = LocalFileJobRepository(
        Path(tmp_path)
    )

    service = JobService(repository)

    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_job_service] = (
        lambda: service
    )

    yield TestClient(app)

    app.dependency_overrides.clear()


def _create_payload():
    return {
        "aoi": {
            "type": "Polygon",
            "coordinates": [],
        },
        "imagery_source": {
            "type": "local",
            "path": "tests/fixtures/demo_aoi.png",
        },
    }


def _create_payload_with_metadata():
    return {
        "aoi": {
            "type": "Polygon",
            "coordinates": [
                [
                    [77.10, 28.60],
                    [77.11, 28.60],
                    [77.11, 28.61],
                    [77.10, 28.61],
                    [77.10, 28.60],
                ]
            ],
        },
        "imagery_source": {
            "type": "local",
            "path": "tests/fixtures/demo_aoi.png",
        },
        "metadata": {
            "purpose": "urban_change_analysis",
            "requested_by": "api_test",
        },
    }


def test_create_job(client):
    response = client.post(
        "/jobs",
        json=_create_payload(),
    )

    assert response.status_code == 201

    payload = response.json()

    assert "job_id" in payload
    assert payload["state"] == "queued"
    assert payload["revision"] == 0
    assert len(payload["transitions"]) == 1

    assert payload["request"] == _create_payload()


def test_create_job_preserves_metadata(client):
    request = _create_payload_with_metadata()

    response = client.post(
        "/jobs",
        json=request,
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["request"] == request
    assert payload["request"]["metadata"] == {
        "purpose": "urban_change_analysis",
        "requested_by": "api_test",
    }


def test_get_job(client):
    request = _create_payload_with_metadata()

    create_response = client.post(
        "/jobs",
        json=request,
    )

    assert create_response.status_code == 201

    job_id = create_response.json()["job_id"]

    response = client.get(
        f"/jobs/{job_id}"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["job_id"] == job_id
    assert payload["state"] == "queued"
    assert payload["revision"] == 0
    assert payload["request"] == request


def test_get_unknown_job_returns_404(client):
    job_id = uuid4()

    response = client.get(
        f"/jobs/{job_id}"
    )

    assert response.status_code == 404


def test_cancel_job(client):
    request = _create_payload_with_metadata()

    create_response = client.post(
        "/jobs",
        json=request,
    )

    assert create_response.status_code == 201

    job_id = create_response.json()["job_id"]

    response = client.post(
        f"/jobs/{job_id}/cancel"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["job_id"] == job_id
    assert payload["state"] == "cancelled"
    assert payload["revision"] == 1
    assert len(payload["transitions"]) == 2

    # The original analysis request must remain unchanged
    # throughout lifecycle transitions.
    assert payload["request"] == request


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"aoi": {}},
        {"imagery_source": {}},
        {"aoi": [], "imagery_source": {}},
        {"aoi": {}, "imagery_source": []},
    ],
)
def test_create_job_rejects_invalid_payload(
    client,
    payload,
):
    response = client.post(
        "/jobs",
        json=payload,
    )

    assert response.status_code == 422