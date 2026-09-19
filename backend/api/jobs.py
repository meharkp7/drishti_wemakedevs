"""
HTTP API for DRISHTI analysis jobs.

The API layer is intentionally thin:

    HTTP request
        -> request validation
        -> JobService
        -> HTTP response

It does not execute imagery, tiling, inference, or postprocessing.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from backend.schemas.job_api import (
    AnalysisJobResponse,
    CreateAnalysisJobRequest,
    JobAPIError,
)
from backend.services.job_repository import (
    JobNotFoundError,
    LocalFileJobRepository,
)
from backend.services.job_service import (
    JobService,
    JobServiceError,
    JobTransitionServiceError,
)


router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
)


def get_job_service() -> JobService:
    """
    Construct the default job service.

    This dependency is intentionally replaceable in tests and later
    application wiring.
    """
    repository = LocalFileJobRepository(
        Path("storage/jobs")
    )

    return JobService(repository)


@router.post(
    "",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
def create_job(
    request: dict,
    service: JobService = Depends(get_job_service),
) -> dict:
    """Create a new queued analysis job."""

    try:
        api_request = CreateAnalysisJobRequest.from_dict(request)
        domain_request = api_request.to_request()
    except JobAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        job = service.create_job(
            request=domain_request,
        )
    except JobServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return AnalysisJobResponse.from_job(job).to_dict()


@router.get(
    "/{job_id}",
    response_model=dict,
)
def get_job(
    job_id: UUID,
    service: JobService = Depends(get_job_service),
) -> dict:
    """Retrieve an analysis job by UUID."""

    try:
        job = service.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except JobServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return AnalysisJobResponse.from_job(job).to_dict()


@router.post(
    "/{job_id}/cancel",
    response_model=dict,
)
def cancel_job(
    job_id: UUID,
    service: JobService = Depends(get_job_service),
) -> dict:
    """Cancel a queued/in-progress analysis job."""

    try:
        service.cancel(
            job_id,
            reason="cancelled_by_api",
        )
        job = service.get_job(job_id)

    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except JobTransitionServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except JobServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return AnalysisJobResponse.from_job(job).to_dict()