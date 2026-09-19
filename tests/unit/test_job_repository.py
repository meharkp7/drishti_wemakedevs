import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backend.core.job_state import (
    AnalysisJob,
    AnalysisJobRequest,
    JobState,
)
from backend.services.job_repository import (
    JobConcurrencyError,
    JobNotFoundError,
    JobPersistenceError,
    LocalFileJobRepository,
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
        },
        imagery_source={
            "type": "local",
            "path": "tests/fixtures/demo_aoi.png",
        },
        metadata={
            "source": "repository-test",
        },
    )


def _job() -> AnalysisJob:
    job = AnalysisJob.create(
        request=_request(),
        job_id=uuid4(),
        now=_time(0),
    )

    job.transition(
        JobState.PREPARING,
        reason="prepare",
        now=_time(1),
    )

    return job


def test_create_and_get_round_trip(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    original = _job()

    repository.create(original)

    assert repository.exists(original.job_id)

    restored = repository.get(original.job_id)

    assert restored.job_id == original.job_id
    assert restored.state is original.state
    assert restored.created_at == original.created_at
    assert restored.updated_at == original.updated_at
    assert restored.request == original.request
    assert restored.revision == original.revision
    assert restored.transitions == original.transitions


def test_create_writes_canonical_json_file(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    job = _job()

    repository.create(job)

    path = tmp_path / f"{job.job_id}.json"

    assert path.exists()
    assert path.is_file()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    assert payload["job_id"] == str(job.job_id)
    assert payload["state"] == "preparing"
    assert payload["revision"] == 1
    assert "request" in payload
    assert payload["request"]["aoi"]["type"] == "Polygon"
    assert (
        payload["request"]["imagery_source"]["type"]
        == "local"
    )
    assert isinstance(payload["transitions"], list)


def test_duplicate_create_is_rejected(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    job = _job()

    repository.create(job)

    with pytest.raises(JobPersistenceError):
        repository.create(job)


def test_missing_job_is_rejected(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    job_id = uuid4()

    with pytest.raises(JobNotFoundError):
        repository.get(job_id)


def test_save_updates_existing_job(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    job = _job()

    repository.create(job)

    expected_revision = job.revision

    job.transition(
        JobState.FETCHING_IMAGERY,
        reason="imagery_ready",
        now=_time(2),
    )

    repository.save(
        job,
        expected_revision=expected_revision,
    )

    restored = repository.get(job.job_id)

    assert restored.state is JobState.FETCHING_IMAGERY
    assert restored.revision == 2
    assert len(restored.transitions) == 3
    assert restored.updated_at == _time(2)


def test_save_missing_job_is_rejected(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    job = _job()

    with pytest.raises(JobNotFoundError):
        repository.save(job)


def test_corrupt_json_is_rejected(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    job = _job()

    path = tmp_path / f"{job.job_id}.json"

    path.write_text(
        "{ definitely not valid json",
        encoding="utf-8",
    )

    with pytest.raises(JobPersistenceError):
        repository.get(job.job_id)


def test_invalid_persisted_schema_is_rejected(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    job = _job()

    path = tmp_path / f"{job.job_id}.json"

    path.write_text(
        json.dumps(
            {
                "job_id": str(job.job_id),
                "state": "not_a_real_state",
                "created_at": _time(0).isoformat(),
                "updated_at": _time(1).isoformat(),
                "request": {
                    "aoi": {
                        "type": "Polygon",
                        "coordinates": [],
                    },
                    "imagery_source": {
                        "type": "local",
                        "path": "tests/fixtures/demo_aoi.png",
                    },
                },
                "revision": 1,
                "transitions": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(JobPersistenceError):
        repository.get(job.job_id)


def test_persisted_job_id_mismatch_is_rejected(tmp_path):
    repository = LocalFileJobRepository(tmp_path)

    requested_id = uuid4()
    different_job = _job()

    path = tmp_path / f"{requested_id}.json"

    payload = {
        "job_id": str(different_job.job_id),
        "state": different_job.state.value,
        "created_at": different_job.created_at.isoformat(),
        "updated_at": different_job.updated_at.isoformat(),
        "request": {
            "aoi": different_job.request.aoi,
            "imagery_source": different_job.request.imagery_source,
            "metadata": different_job.request.metadata,
        },
        "revision": different_job.revision,
        "transitions": [
            {
                "from_state": (
                    transition.from_state.value
                    if transition.from_state is not None
                    else None
                ),
                "to_state": transition.to_state.value,
                "timestamp": transition.timestamp.isoformat(),
                "reason": transition.reason,
            }
            for transition in different_job.transitions
        ],
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(JobPersistenceError):
        repository.get(requested_id)


def test_atomic_write_does_not_leave_temporary_files(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    job = _job()

    repository.create(job)

    temporary_files = list(
        tmp_path.glob(f".{job.job_id}.*.tmp")
    )

    assert temporary_files == []


def test_exists_returns_false_for_missing_job(tmp_path):
    repository = LocalFileJobRepository(tmp_path)

    assert repository.exists(uuid4()) is False


def test_save_rejects_stale_revision(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    original = _job()

    repository.create(original)

    stale_job = repository.get(original.job_id)

    current_job = repository.get(original.job_id)

    current_expected_revision = current_job.revision

    current_job.transition(
        JobState.FETCHING_IMAGERY,
        reason="current_worker",
        now=_time(2),
    )

    repository.save(
        current_job,
        expected_revision=current_expected_revision,
    )

    assert repository.get(
        original.job_id
    ).revision == 2

    stale_job.transition(
        JobState.FETCHING_IMAGERY,
        reason="stale_worker",
        now=_time(2),
    )

    with pytest.raises(JobConcurrencyError):
        repository.save(
            stale_job,
            expected_revision=1,
        )

    persisted = repository.get(original.job_id)

    assert persisted.state is JobState.FETCHING_IMAGERY
    assert persisted.revision == 2
    assert persisted.transitions[-1].reason == "current_worker"


def test_save_rejects_invalid_new_revision(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    job = _job()

    repository.create(job)

    with pytest.raises(JobConcurrencyError):
        repository.save(
            job,
            expected_revision=0,
        )


def test_save_without_expected_revision_preserves_legacy_repository_contract(
    tmp_path,
):
    repository = LocalFileJobRepository(tmp_path)
    job = _job()

    repository.create(job)

    job.transition(
        JobState.FETCHING_IMAGERY,
        reason="legacy_save",
        now=_time(2),
    )

    repository.save(job)

    restored = repository.get(job.job_id)

    assert restored.state is JobState.FETCHING_IMAGERY
    assert restored.revision == 2


def test_request_is_persisted_without_mutation(tmp_path):
    repository = LocalFileJobRepository(tmp_path)
    job = _job()

    original_request = job.request

    repository.create(job)

    restored = repository.get(job.job_id)

    assert restored.request == original_request
    assert restored.request is not original_request

def _concurrent_save_worker(root_dir, job_id_text, ready, go, results):
    repository = LocalFileJobRepository(root_dir)
    job_id = __import__("uuid").UUID(job_id_text)
    job = repository.get(job_id)
    ready.put("loaded")
    go.wait(10)
    try:
        job.transition(
            JobState.FETCHING_IMAGERY,
            reason="concurrent_worker",
            now=_time(2),
        )
        repository.save(job, expected_revision=1)
        results.put("saved")
    except JobConcurrencyError:
        results.put("conflict")
    except Exception as exc:
        results.put(f"error:{type(exc).__name__}:{exc}")


def test_process_concurrency_allows_only_one_compare_and_swap_winner(tmp_path):
    import multiprocessing as mp

    repository = LocalFileJobRepository(tmp_path)
    job = _job()
    repository.create(job)

    ctx = mp.get_context("spawn")
    ready = ctx.Queue()
    results = ctx.Queue()
    go = ctx.Event()

    processes = [
        ctx.Process(
            target=_concurrent_save_worker,
            args=(str(tmp_path), str(job.job_id), ready, go, results),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()

    assert ready.get(timeout=10) == "loaded"
    assert ready.get(timeout=10) == "loaded"
    go.set()

    outcomes = [results.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sorted(outcomes) == ["conflict", "saved"]

    persisted = repository.get(job.job_id)
    assert persisted.revision == 2
    assert persisted.state is JobState.FETCHING_IMAGERY
    assert persisted.transitions[-1].reason == "concurrent_worker"
