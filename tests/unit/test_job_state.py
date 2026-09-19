from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backend.core.job_state import (
    AnalysisJob,
    InvalidJobTransitionError,
    JobState,
    TerminalJobStateError,
    AnalysisJobRequest,
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


def test_new_job_starts_queued():
    job = AnalysisJob.create(
        job_id=uuid4(),
        now=_time(0),
        request=_job_request(),
    )

    assert job.state is JobState.QUEUED
    assert not job.is_terminal
    assert job.created_at == _time(0)
    assert job.updated_at == _time(0)

    assert len(job.transitions) == 1
    assert job.transitions[0].from_state is None
    assert job.transitions[0].to_state is JobState.QUEUED
    assert job.transitions[0].reason == "job_created"


def test_happy_path_transition_sequence():
    job = AnalysisJob.create(now=_time(0), request=_job_request())

    expected_states = [
        JobState.PREPARING,
        JobState.FETCHING_IMAGERY,
        JobState.TILING,
        JobState.INFERENCE,
        JobState.POSTPROCESSING,
        JobState.COMPLETED,
    ]

    for second, state in enumerate(expected_states, start=1):
        assert job.can_transition_to(state)

        job.transition(
            state,
            reason=f"entered_{state.value}",
            now=_time(second),
        )

        assert job.state is state
        assert job.updated_at == _time(second)

    assert job.is_terminal
    assert len(job.transitions) == 7


def test_invalid_transition_is_rejected():
    job = AnalysisJob.create(now=_time(0), request=_job_request())

    with pytest.raises(InvalidJobTransitionError):
        job.transition(
            JobState.INFERENCE,
            now=_time(1),
        )

    assert job.state is JobState.QUEUED
    assert len(job.transitions) == 1


def test_terminal_state_cannot_be_modified():
    job = AnalysisJob.create(now=_time(0), request=_job_request())

    job.transition(
        JobState.PREPARING,
        now=_time(1),
    )
    job.transition(
        JobState.FAILED,
        reason="imagery_unavailable",
        now=_time(2),
    )

    assert job.is_terminal

    with pytest.raises(TerminalJobStateError):
        job.transition(
            JobState.COMPLETED,
            now=_time(3),
        )

    assert job.state is JobState.FAILED
    assert len(job.transitions) == 3


def test_cancelled_job_is_terminal():
    job = AnalysisJob.create(now=_time(0), request=_job_request())

    job.cancel(
        reason="user_requested",
        now=_time(1),
    )

    assert job.state is JobState.CANCELLED
    assert job.is_terminal
    assert job.transitions[-1].reason == "user_requested"


def test_fail_helper_transitions_to_failed():
    job = AnalysisJob.create(now=_time(0), request=_job_request())

    transition = job.fail(
        reason="inference_error",
        now=_time(1),
    )

    assert transition.from_state is JobState.QUEUED
    assert transition.to_state is JobState.FAILED
    assert transition.reason == "inference_error"
    assert job.state is JobState.FAILED


def test_transition_history_is_immutable():
    job = AnalysisJob.create(now=_time(0), request=_job_request())

    job.transition(
        JobState.PREPARING,
        now=_time(1),
    )

    first_history = job.transitions

    job.transition(
        JobState.FETCHING_IMAGERY,
        now=_time(2),
    )

    assert len(first_history) == 2
    assert len(job.transitions) == 3
    assert first_history[0].to_state is JobState.QUEUED
    assert first_history[1].to_state is JobState.PREPARING


def test_transition_timestamp_cannot_go_backwards():
    job = AnalysisJob.create(now=_time(5), request=_job_request())

    with pytest.raises(ValueError):
        job.transition(
            JobState.PREPARING,
            now=_time(4),
        )

    assert job.state is JobState.QUEUED


def test_naive_timestamp_is_rejected():
    naive_time = datetime(
        2026,
        9,
        17,
        12,
        0,
        0,
    )

    with pytest.raises(ValueError):
        AnalysisJob.create(now=naive_time, request=_job_request())


def test_all_terminal_states_have_no_outgoing_transitions():
    for terminal_state in (
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.CANCELLED,
    ):
        job = AnalysisJob.create(now=_time(0), request=_job_request())

        job.transition(
            JobState.PREPARING,
            now=_time(1),
        )
        job.transition(
            terminal_state
            if terminal_state is not JobState.COMPLETED
            else JobState.FETCHING_IMAGERY,
            now=_time(2),
        )

        if terminal_state is JobState.COMPLETED:
            job.transition(
                JobState.TILING,
                now=_time(3),
            )
            job.transition(
                JobState.INFERENCE,
                now=_time(4),
            )
            job.transition(
                JobState.POSTPROCESSING,
                now=_time(5),
            )
            job.transition(
                JobState.COMPLETED,
                now=_time(6),
            )

        assert job.is_terminal
        assert not job.can_transition_to(JobState.QUEUED)

def _job_request():
    return AnalysisJobRequest(
        aoi={
            "type": "Polygon",
            "coordinates": [],
        },
        imagery_source={
            "type": "local",
            "path": "tests/fixtures/demo_aoi.png",
        },
        metadata={
            "source": "unit-test",
        },
    )

def test_job_request_is_persisted_on_job():
    request = _job_request()

    job = AnalysisJob.create(
        request=request,
    )

    assert job.request == request
    assert job.request.aoi == request.aoi
    assert job.request.imagery_source == request.imagery_source
    assert job.request.metadata == request.metadata

def test_job_request_is_immutable():
    request = _job_request()

    with pytest.raises(
        (AttributeError, TypeError, ValueError)
    ):
        request.aoi = {}

def test_job_request_requires_aoi_and_imagery_source():
    with pytest.raises(ValueError):
        AnalysisJobRequest(
            aoi=[],
            imagery_source={},
        )

    with pytest.raises(ValueError):
        AnalysisJobRequest(
            aoi={},
            imagery_source=[],
        )