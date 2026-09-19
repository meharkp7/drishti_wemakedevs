from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backend.core.job_state import (
    AnalysisJobRequest,
    JobState,
)
from backend.services.job_repository import (
    JobConcurrencyError,
    JobNotFoundError,
    LocalFileJobRepository,
)
from backend.services.job_service import (
    JobService,
    JobServiceError,
    JobTransitionServiceError,
)


def _time(second: int) -> datetime:
    return datetime(
        2026,
        9,
        17,
        12,
        0,
        second,
        tzinfo=timezone.utc,
    )


def _request() -> AnalysisJobRequest:
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
            "crs": "EPSG:4326",
        },
        imagery_source={
            "type": "local_raster",
            "path": "tests/fixtures/demo_aoi.tif",
        },
        metadata={
            "purpose": "urban_change_analysis",
            "requested_by": "test",
        },
    )


def _service(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    return JobService(repository), repository


def test_create_job_persists_queued_job(tmp_path):
    service, repository = _service(tmp_path)

    job_id = uuid4()
    request = _request()

    job = service.create_job(
        request=request,
        job_id=job_id,
        now=_time(0),
    )

    assert job.job_id == job_id
    assert job.state is JobState.QUEUED
    assert job.revision == 0
    assert job.request == request
    assert repository.exists(job_id)


def test_create_job_persists_request_and_revision(tmp_path):
    service, _ = _service(tmp_path)

    request = _request()

    job = service.create_job(
        request=request,
        now=_time(0),
    )

    restored = service.get_job(job.job_id)

    assert restored.request == request
    assert restored.revision == 0
    assert restored.state is JobState.QUEUED
    assert len(restored.transitions) == 1


def test_get_job_returns_persisted_job(tmp_path):
    service, _ = _service(tmp_path)

    request = _request()

    original = service.create_job(
        request=request,
        now=_time(0),
    )

    restored = service.get_job(original.job_id)

    assert restored.job_id == original.job_id
    assert restored.state is JobState.QUEUED
    assert restored.transitions == original.transitions
    assert restored.request == original.request
    assert restored.revision == original.revision


def test_full_lifecycle_is_persisted(tmp_path):
    service, _ = _service(tmp_path)

    job = service.create_job(
        request=_request(),
        now=_time(0),
    )

    service.prepare(
        job.job_id,
        now=_time(1),
    )
    service.start_imagery_fetch(
        job.job_id,
        now=_time(2),
    )
    service.start_tiling(
        job.job_id,
        now=_time(3),
    )
    service.start_inference(
        job.job_id,
        now=_time(4),
    )
    service.start_postprocessing(
        job.job_id,
        now=_time(5),
    )
    service.complete(
        job.job_id,
        now=_time(6),
    )

    restored = service.get_job(job.job_id)

    assert restored.state is JobState.COMPLETED
    assert restored.is_terminal
    assert len(restored.transitions) == 7
    assert restored.revision == 6
    assert restored.updated_at == _time(6)
    assert restored.request == job.request


def test_invalid_transition_is_rejected(tmp_path):
    service, _ = _service(tmp_path)

    job = service.create_job(
        request=_request(),
        now=_time(0),
    )

    with pytest.raises(JobTransitionServiceError):
        service.start_inference(
            job.job_id,
            now=_time(1),
        )

    restored = service.get_job(job.job_id)

    assert restored.state is JobState.QUEUED
    assert restored.revision == 0
    assert len(restored.transitions) == 1


def test_failure_is_persisted(tmp_path):
    service, _ = _service(tmp_path)

    job = service.create_job(
        request=_request(),
        now=_time(0),
    )

    service.prepare(
        job.job_id,
        now=_time(1),
    )

    service.fail(
        job.job_id,
        reason="imagery_provider_failure",
        now=_time(2),
    )

    restored = service.get_job(job.job_id)

    assert restored.state is JobState.FAILED
    assert restored.is_terminal
    assert restored.transitions[-1].reason == (
        "imagery_provider_failure"
    )


def test_cancellation_is_persisted(tmp_path):
    service, _ = _service(tmp_path)

    job = service.create_job(
        request=_request(),
        now=_time(0),
    )

    service.cancel(
        job.job_id,
        reason="user_requested",
        now=_time(1),
    )

    restored = service.get_job(job.job_id)

    assert restored.state is JobState.CANCELLED
    assert restored.is_terminal
    assert restored.transitions[-1].reason == "user_requested"


def test_terminal_job_cannot_be_advanced(tmp_path):
    service, _ = _service(tmp_path)

    job = service.create_job(
        request=_request(),
        now=_time(0),
    )

    service.cancel(
        job.job_id,
        reason="user_requested",
        now=_time(1),
    )

    with pytest.raises(JobTransitionServiceError):
        service.prepare(
            job.job_id,
            now=_time(2),
        )

    restored = service.get_job(job.job_id)

    assert restored.state is JobState.CANCELLED
    assert len(restored.transitions) == 2
    assert restored.revision == 1


def test_missing_job_is_propagated(tmp_path):
    service, _ = _service(tmp_path)

    missing_id = uuid4()

    with pytest.raises(JobNotFoundError):
        service.get_job(missing_id)


def test_duplicate_job_creation_is_wrapped(tmp_path):
    service, _ = _service(tmp_path)

    job_id = uuid4()
    request = _request()

    service.create_job(
        request=request,
        job_id=job_id,
        now=_time(0),
    )

    with pytest.raises(JobServiceError):
        service.create_job(
            request=request,
            job_id=job_id,
            now=_time(1),
        )


def test_transition_reason_is_preserved(tmp_path):
    service, _ = _service(tmp_path)

    job = service.create_job(
        request=_request(),
        now=_time(0),
    )

    transition = service.prepare(
        job.job_id,
        now=_time(1),
    )

    assert transition.from_state is JobState.QUEUED
    assert transition.to_state is JobState.PREPARING
    assert transition.reason == "preparation_started"

    restored = service.get_job(job.job_id)

    assert restored.transitions[-1].reason == (
        "preparation_started"
    )


def test_transition_uses_expected_revision(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    service = JobService(repository)

    job = service.create_job(
        request=_request(),
    )

    assert job.revision == 0

    service.prepare(job.job_id)

    persisted = service.get_job(job.job_id)

    assert persisted.state == JobState.PREPARING
    assert persisted.revision == 1
    assert persisted.request == job.request


def test_transition_rejects_stale_revision(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    service = JobService(repository)

    job = service.create_job(
        request=_request(),
    )

    stale_job = service.get_job(job.job_id)

    service.prepare(job.job_id)

    assert service.get_job(job.job_id).revision == 1

    stale_job.transition(
        JobState.PREPARING,
        reason="stale_worker_transition",
    )

    with pytest.raises(JobConcurrencyError):
        repository.save(
            stale_job,
            expected_revision=0,
        )

    persisted = service.get_job(job.job_id)

    assert persisted.state == JobState.PREPARING
    assert persisted.revision == 1
    assert persisted.request == job.request


class ConflictRepository:
    def __init__(self, delegate):
        self._delegate = delegate

    def create(self, job):
        return self._delegate.create(job)

    def get(self, job_id):
        return self._delegate.get(job_id)

    def exists(self, job_id):
        return self._delegate.exists(job_id)

    def save(self, job, *, expected_revision=None):
        raise JobConcurrencyError(
            "simulated concurrent modification"
        )


def test_service_surfaces_concurrency_conflict(tmp_path):
    repository = LocalFileJobRepository(tmp_path)

    base_service = JobService(repository)
    job = base_service.create_job(
        request=_request(),
    )

    conflict_service = JobService(
        ConflictRepository(repository)
    )

    with pytest.raises(JobTransitionServiceError) as exc_info:
        conflict_service.prepare(job.job_id)

    assert "Concurrent modification detected" in str(
        exc_info.value
    )