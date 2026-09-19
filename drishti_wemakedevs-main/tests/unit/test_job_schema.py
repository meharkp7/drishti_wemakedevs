"""
Unit tests for DRISHTI analysis-job serialization schemas.
"""

from datetime import datetime, timedelta, timezone
import json

import pytest

from backend.core.job_state import (
    AnalysisJob,
    AnalysisJobRequest,
    JobState,
)
from backend.schemas.job import (
    AnalysisJobSchema,
    JobSchemaError,
)


def _time(minutes: int) -> datetime:
    """Return a deterministic timezone-aware test timestamp."""
    return datetime(
        2026,
        1,
        1,
        12,
        0,
        tzinfo=timezone.utc,
    ) + timedelta(minutes=minutes)


def _job_request() -> AnalysisJobRequest:
    """Return a deterministic analysis request for schema tests."""
    return AnalysisJobRequest(
        aoi={
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
        imagery_source={
            "type": "local",
            "path": "tests/fixtures/demo_aoi.png",
        },
        metadata={
            "source": "unit-test",
        },
    )


def _completed_job() -> AnalysisJob:
    """Create a deterministic completed analysis job."""
    job = AnalysisJob.create(
        request=_job_request(),
        now=_time(0),
    )

    job.transition(
        JobState.PREPARING,
        reason="preparation_started",
        now=_time(1),
    )

    job.transition(
        JobState.FETCHING_IMAGERY,
        reason="imagery_fetch_started",
        now=_time(2),
    )

    job.transition(
        JobState.TILING,
        reason="tiling_started",
        now=_time(3),
    )

    job.transition(
        JobState.INFERENCE,
        reason="inference_started",
        now=_time(4),
    )

    job.transition(
        JobState.POSTPROCESSING,
        reason="postprocessing_started",
        now=_time(5),
    )

    job.transition(
        JobState.COMPLETED,
        reason="analysis_completed",
        now=_time(6),
    )

    return job


def test_job_schema_serializes_job():
    job = _completed_job()

    schema = AnalysisJobSchema.from_job(job)

    payload = schema.to_dict()

    assert payload["job_id"] == str(job.job_id)
    assert payload["state"] == "completed"
    assert payload["revision"] == 6

    assert payload["request"]["aoi"]["type"] == "Polygon"
    assert (
        payload["request"]["imagery_source"]["type"]
        == "local"
    )
    assert payload["request"]["metadata"]["source"] == "unit-test"

    assert len(payload["transitions"]) == 7
    assert payload["transitions"][0]["from_state"] is None
    assert payload["transitions"][0]["to_state"] == "queued"
    assert payload["transitions"][-1]["to_state"] == "completed"


def test_schema_round_trip_preserves_job():
    original = _completed_job()

    schema = AnalysisJobSchema.from_job(original)

    payload = schema.to_dict()

    restored_schema = AnalysisJobSchema.from_dict(
        payload
    )

    restored = restored_schema.to_job()

    assert restored.job_id == original.job_id
    assert restored.state == original.state
    assert restored.created_at == original.created_at
    assert restored.updated_at == original.updated_at
    assert restored.request == original.request
    assert restored.revision == original.revision
    assert restored.transitions == original.transitions


def test_schema_output_is_json_safe():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_json_dict()

    encoded = json.dumps(
        payload,
        sort_keys=True,
    )

    decoded = json.loads(encoded)

    assert decoded["job_id"] == str(job.job_id)
    assert decoded["state"] == "completed"
    assert decoded["revision"] == 6
    assert decoded["request"]["aoi"]["type"] == "Polygon"


def test_missing_required_field_is_rejected():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    del payload["job_id"]

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_invalid_job_id_is_rejected():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["job_id"] = "not-a-uuid"

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_invalid_state_is_rejected():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["state"] = "not-a-real-state"

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_naive_timestamp_is_rejected():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["created_at"] = (
        "2026-01-01T12:00:00"
    )

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_inconsistent_final_state_is_rejected():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["state"] = "failed"

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_empty_transition_history_is_rejected():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["transitions"] = []
    payload["revision"] = 0

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_invalid_first_transition_is_rejected():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["transitions"][0]["to_state"] = (
        "preparing"
    )

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_missing_request_is_rejected():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    del payload["request"]

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_invalid_request_aoi_is_rejected():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["request"]["aoi"] = []

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_invalid_request_imagery_source_is_rejected():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["request"]["imagery_source"] = []

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_revision_must_match_transition_history():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["revision"] = 99

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_revision_cannot_be_negative():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["revision"] = -1

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)


def test_revision_cannot_be_boolean():
    job = _completed_job()

    payload = AnalysisJobSchema.from_job(
        job
    ).to_dict()

    payload["revision"] = True

    with pytest.raises(JobSchemaError):
        AnalysisJobSchema.from_dict(payload)