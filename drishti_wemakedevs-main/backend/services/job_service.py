"""
Application service for DRISHTI analysis jobs.

JobService is the orchestration boundary between callers such as APIs,
workers, and the lower-level job state machine/repository.

Responsibilities:
- create jobs
- retrieve jobs
- perform validated lifecycle transitions
- persist every mutation
- provide a single application-facing job API

The service intentionally does not execute geospatial or ML work.
Workers will perform that work and use this service to report lifecycle
progress.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from backend.core.job_state import (
    AnalysisJob,
    AnalysisJobRequest,
    JobState,
    JobTransition,
)
from backend.services.job_repository import (
    JobRepository,
    JobNotFoundError,
    JobConcurrencyError,
)


class JobServiceError(RuntimeError):
    """Base exception for job-service failures."""


class JobTransitionServiceError(JobServiceError):
    """Raised when a requested job transition cannot be completed."""


class JobService:
    """
    Application-level service for analysis-job lifecycle management.

    The repository is injected so the service remains independent of
    the persistence implementation.
    """

    def __init__(self, repository: JobRepository) -> None:
        self._repository = repository

    def create_job(
        self,
        *,
        request: AnalysisJobRequest,
        job_id: Optional[UUID] = None,
        now=None,
    ) -> AnalysisJob:
        """
        Create and persist a new queued analysis job.

        The analysis request is part of the job's durable domain state and
        therefore must be supplied at creation time.
        """
        job = AnalysisJob.create(
            job_id=job_id,
            now=now,
            request=request,
        )

        try:
            self._repository.create(job)
        except Exception as exc:
            raise JobServiceError(
                f"Unable to create job: {job.job_id}"
            ) from exc

        return job

    def get_job(self, job_id: UUID) -> AnalysisJob:
        """Retrieve a persisted analysis job."""
        try:
            return self._repository.get(job_id)
        except JobNotFoundError:
            raise
        except Exception as exc:
            raise JobServiceError(
                f"Unable to retrieve job: {job_id}"
            ) from exc

    def transition(
        self,
        job_id: UUID,
        target: JobState,
        *,
        reason: Optional[str] = None,
        now=None,
    ) -> JobTransition:
        """
        Transition a job using optimistic concurrency control.

        The persisted revision is captured before mutation. The state
        machine increments the revision exactly once, and the repository
        verifies that the persisted job still has the expected revision
        before accepting the write.
        """
        job = self.get_job(job_id)

        expected_revision = job.revision

        try:
            transition = job.transition(
                target,
                reason=reason,
                now=now,
            )
        except Exception as exc:
            raise JobTransitionServiceError(
                f"Unable to transition job {job_id} "
                f"to '{target.value}'."
            ) from exc

        try:
            self._repository.save(
                job,
                expected_revision=expected_revision,
            )
        except JobConcurrencyError as exc:
            raise JobTransitionServiceError(
                f"Concurrent modification detected for job "
                f"{job_id}. The job changed after it was loaded."
            ) from exc
        except Exception as exc:
            raise JobTransitionServiceError(
                f"Job {job_id} transitioned in memory but could not "
                f"be persisted."
            ) from exc

        return transition

    def prepare(
        self,
        job_id: UUID,
        *,
        now=None,
    ) -> JobTransition:
        """Move a queued job into preparation."""
        return self.transition(
            job_id,
            JobState.PREPARING,
            reason="preparation_started",
            now=now,
        )

    def start_imagery_fetch(
        self,
        job_id: UUID,
        *,
        now=None,
    ) -> JobTransition:
        """Move a preparing job into imagery acquisition."""
        return self.transition(
            job_id,
            JobState.FETCHING_IMAGERY,
            reason="imagery_fetch_started",
            now=now,
        )

    def start_tiling(
        self,
        job_id: UUID,
        *,
        now=None,
    ) -> JobTransition:
        """Move a job into raster tiling."""
        return self.transition(
            job_id,
            JobState.TILING,
            reason="tiling_started",
            now=now,
        )

    def start_inference(
        self,
        job_id: UUID,
        *,
        now=None,
    ) -> JobTransition:
        """Move a job into model inference."""
        return self.transition(
            job_id,
            JobState.INFERENCE,
            reason="inference_started",
            now=now,
        )

    def start_postprocessing(
        self,
        job_id: UUID,
        *,
        now=None,
    ) -> JobTransition:
        """Move a job into postprocessing."""
        return self.transition(
            job_id,
            JobState.POSTPROCESSING,
            reason="postprocessing_started",
            now=now,
        )

    def complete(
        self,
        job_id: UUID,
        *,
        now=None,
    ) -> JobTransition:
        """Mark a job as successfully completed."""
        return self.transition(
            job_id,
            JobState.COMPLETED,
            reason="analysis_completed",
            now=now,
        )

    def fail(
        self,
        job_id: UUID,
        *,
        reason: Optional[str] = None,
        now=None,
    ) -> JobTransition:
        """Mark a job as failed."""
        return self.transition(
            job_id,
            JobState.FAILED,
            reason=reason,
            now=now,
        )

    def cancel(
        self,
        job_id: UUID,
        *,
        reason: Optional[str] = None,
        now=None,
    ) -> JobTransition:
        """Cancel a job."""
        return self.transition(
            job_id,
            JobState.CANCELLED,
            reason=reason,
            now=now,
        )