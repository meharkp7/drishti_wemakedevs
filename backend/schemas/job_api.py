"""
Public API schemas for DRISHTI analysis jobs.

These schemas define the HTTP-facing representation of jobs.
They are intentionally separate from the persistence schema so that
API evolution does not directly change the storage contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID

from backend.core.job_state import AnalysisJob, AnalysisJobRequest
from backend.schemas.job import AnalysisJobSchema


class JobAPIError(ValueError):
    """Raised when an API job payload is invalid."""


@dataclass(frozen=True)
class CreateAnalysisJobRequest:
    """
    Request payload for creating an analysis job.

    ``aoi`` and ``imagery_source`` are kept as structured dictionaries
    at this boundary. Their domain-specific validation belongs to the
    geospatial and imagery layers respectively.
    """

    aoi: Dict[str, Any]
    imagery_source: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.aoi, dict):
            raise JobAPIError("aoi must be an object.")

        if not isinstance(self.imagery_source, dict):
            raise JobAPIError(
                "imagery_source must be an object."
            )

        if self.metadata is not None and not isinstance(
            self.metadata,
            dict,
        ):
            raise JobAPIError("metadata must be an object.")

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
    ) -> "CreateAnalysisJobRequest":
        """Parse and validate an incoming API payload."""
        if not isinstance(payload, dict):
            raise JobAPIError("Request payload must be an object.")

        required = {"aoi", "imagery_source"}
        missing = required - payload.keys()

        if missing:
            raise JobAPIError(
                "Missing required fields: "
                + ", ".join(sorted(missing))
            )

        return cls(
            aoi=payload["aoi"],
            imagery_source=payload["imagery_source"],
            metadata=payload.get("metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the API request back to a JSON-compatible dictionary."""
        payload: Dict[str, Any] = {
            "aoi": self.aoi,
            "imagery_source": self.imagery_source,
        }

        if self.metadata is not None:
            payload["metadata"] = self.metadata

        return payload

    def to_request(self) -> AnalysisJobRequest:
        """
        Convert the API representation into the domain request.

        This is the explicit boundary between HTTP/API schemas and the
        domain state machine.
        """
        return AnalysisJobRequest(
            aoi=dict(self.aoi),
            imagery_source=dict(self.imagery_source),
            metadata=(
                dict(self.metadata)
                if self.metadata is not None
                else None
            ),
        )


@dataclass(frozen=True)
class AnalysisJobResponse:
    """Public representation of an analysis job."""

    job_id: str
    state: str
    revision: int
    created_at: str
    updated_at: str
    request: Dict[str, Any]
    transitions: tuple[Dict[str, Any], ...]

    @classmethod
    def from_job(
        cls,
        job: AnalysisJob,
    ) -> "AnalysisJobResponse":
        """Build the public API response from a domain job."""
        schema = AnalysisJobSchema.from_job(job)
        payload = schema.to_json_dict()

        return cls(
            job_id=payload["job_id"],
            state=payload["state"],
            revision=payload["revision"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            request=dict(payload["request"]),
            transitions=tuple(
                payload["transitions"]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the public response."""
        return {
            "job_id": self.job_id,
            "state": self.state,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request": dict(self.request),
            "transitions": [
                dict(transition)
                for transition in self.transitions
            ],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
    ) -> "AnalysisJobResponse":
        """Validate and reconstruct an API response object."""
        if not isinstance(payload, dict):
            raise JobAPIError("Job response must be an object.")

        required = {
            "job_id",
            "state",
            "revision",
            "created_at",
            "updated_at",
            "request",
            "transitions",
        }

        missing = required - payload.keys()

        if missing:
            raise JobAPIError(
                "Missing required fields: "
                + ", ".join(sorted(missing))
            )

        try:
            UUID(payload["job_id"])
        except (ValueError, TypeError, AttributeError) as exc:
            raise JobAPIError(
                "job_id must be a valid UUID."
            ) from exc

        if not isinstance(payload["state"], str):
            raise JobAPIError("state must be a string.")

        if (
            isinstance(payload["revision"], bool)
            or not isinstance(payload["revision"], int)
            or payload["revision"] < 0
        ):
            raise JobAPIError(
                "revision must be a non-negative integer."
            )

        if not isinstance(payload["created_at"], str):
            raise JobAPIError("created_at must be a string.")

        if not isinstance(payload["updated_at"], str):
            raise JobAPIError("updated_at must be a string.")

        if not isinstance(payload["request"], dict):
            raise JobAPIError("request must be an object.")

        # Validate the nested request using the same API contract
        # used for incoming create requests.
        try:
            CreateAnalysisJobRequest.from_dict(payload["request"])
        except JobAPIError as exc:
            raise JobAPIError(
                f"Invalid request: {exc}"
            ) from exc

        if not isinstance(payload["transitions"], list):
            raise JobAPIError("transitions must be a list.")

        normalized_transitions = []

        for item in payload["transitions"]:
            if not isinstance(item, dict):
                raise JobAPIError(
                    "Each transition must be an object."
                )
            normalized_transitions.append(dict(item))

        return cls(
            job_id=payload["job_id"],
            state=payload["state"],
            revision=payload["revision"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            request=dict(payload["request"]),
            transitions=tuple(normalized_transitions),
        )