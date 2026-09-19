"""
Serializable schemas for DRISHTI analysis jobs.

The schema layer converts domain job objects into JSON-safe structures
for persistence and API boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Dict, Optional
from uuid import UUID

from backend.core.job_state import (
    AnalysisJob,
    AnalysisJobRequest,
    JobState,
    JobTransition,
)


class JobSchemaError(ValueError):
    """Raised when a persisted job schema is invalid."""


@dataclass(frozen=True)
class JobTransitionSchema:
    """Serializable representation of a job transition."""

    from_state: Optional[str]
    to_state: str
    timestamp: str
    reason: Optional[str] = None

    @classmethod
    def from_transition(
        cls,
        transition: JobTransition,
    ) -> "JobTransitionSchema":
        return cls(
            from_state=(
                transition.from_state.value
                if transition.from_state is not None
                else None
            ),
            to_state=transition.to_state.value,
            timestamp=transition.timestamp.isoformat(),
            reason=transition.reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
    ) -> "JobTransitionSchema":
        if not isinstance(payload, dict):
            raise JobSchemaError(
                "Transition must be an object."
            )

        required = {
            "from_state",
            "to_state",
            "timestamp",
            "reason",
        }

        missing = required - payload.keys()

        if missing:
            raise JobSchemaError(
                "Missing transition fields: "
                + ", ".join(sorted(missing))
            )

        from_state = payload["from_state"]
        to_state = payload["to_state"]
        timestamp = payload["timestamp"]
        reason = payload["reason"]

        if from_state is not None and not isinstance(
            from_state,
            str,
        ):
            raise JobSchemaError(
                "Transition from_state must be a string or null."
            )

        if not isinstance(to_state, str):
            raise JobSchemaError(
                "Transition to_state must be a string."
            )

        if not isinstance(timestamp, str):
            raise JobSchemaError(
                "Transition timestamp must be a string."
            )

        if reason is not None and not isinstance(
            reason,
            str,
        ):
            raise JobSchemaError(
                "Transition reason must be a string or null."
            )

        try:
            parsed_timestamp = datetime.fromisoformat(
                timestamp
            )
        except ValueError as exc:
            raise JobSchemaError(
                "Transition timestamp must be valid ISO-8601."
            ) from exc

        if parsed_timestamp.tzinfo is None:
            raise JobSchemaError(
                "Transition timestamp must be timezone-aware."
            )

        try:
            JobState(to_state)
        except ValueError as exc:
            raise JobSchemaError(
                f"Unknown transition target state: {to_state}"
            ) from exc

        if from_state is not None:
            try:
                JobState(from_state)
            except ValueError as exc:
                raise JobSchemaError(
                    f"Unknown transition source state: "
                    f"{from_state}"
                ) from exc

        return cls(
            from_state=from_state,
            to_state=to_state,
            timestamp=timestamp,
            reason=reason,
        )

    def to_transition(self) -> JobTransition:
        return JobTransition(
            from_state=(
                JobState(self.from_state)
                if self.from_state is not None
                else None
            ),
            to_state=JobState(self.to_state),
            timestamp=datetime.fromisoformat(
                self.timestamp
            ),
            reason=self.reason,
        )


@dataclass(frozen=True)
class AnalysisJobRequestSchema:
    """Serializable representation of an analysis job request."""

    aoi: Dict[str, Any]
    imagery_source: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_request(
        cls,
        request: AnalysisJobRequest,
    ) -> "AnalysisJobRequestSchema":
        return cls(
            aoi=dict(request.aoi),
            imagery_source=dict(request.imagery_source),
            metadata=(
                dict(request.metadata)
                if request.metadata is not None
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "aoi": dict(self.aoi),
            "imagery_source": dict(self.imagery_source),
        }

        if self.metadata is not None:
            payload["metadata"] = dict(self.metadata)

        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
    ) -> "AnalysisJobRequestSchema":
        if not isinstance(payload, dict):
            raise JobSchemaError(
                "request must be an object."
            )

        required = {
            "aoi",
            "imagery_source",
        }

        missing = required - payload.keys()

        if missing:
            raise JobSchemaError(
                "Missing required request fields: "
                + ", ".join(sorted(missing))
            )

        if not isinstance(payload["aoi"], dict):
            raise JobSchemaError(
                "request.aoi must be an object."
            )

        if not isinstance(
            payload["imagery_source"],
            dict,
        ):
            raise JobSchemaError(
                "request.imagery_source must be an object."
            )

        metadata = payload.get("metadata")

        if metadata is not None and not isinstance(
            metadata,
            dict,
        ):
            raise JobSchemaError(
                "request.metadata must be an object."
            )

        return cls(
            aoi=dict(payload["aoi"]),
            imagery_source=dict(
                payload["imagery_source"]
            ),
            metadata=(
                dict(metadata)
                if metadata is not None
                else None
            ),
        )

    def to_request(self) -> AnalysisJobRequest:
        return AnalysisJobRequest(
            aoi=dict(self.aoi),
            imagery_source=dict(
                self.imagery_source
            ),
            metadata=(
                dict(self.metadata)
                if self.metadata is not None
                else None
            ),
        )


@dataclass(frozen=True)
class AnalysisJobSchema:
    """Serializable representation of an AnalysisJob."""

    job_id: str
    state: str
    created_at: str
    updated_at: str
    request: AnalysisJobRequestSchema
    revision: int
    transitions: tuple[JobTransitionSchema, ...]

    @classmethod
    def from_job(
        cls,
        job: AnalysisJob,
    ) -> "AnalysisJobSchema":
        return cls(
            job_id=str(job.job_id),
            state=job.state.value,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            request=AnalysisJobRequestSchema.from_request(
                job.request
            ),
            revision=job.revision,
            transitions=tuple(
                JobTransitionSchema.from_transition(
                    transition
                )
                for transition in job.transitions
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request": self.request.to_dict(),
            "revision": self.revision,
            "transitions": [
                transition.to_dict()
                for transition in self.transitions
            ],
        }

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-safe representation.

        Kept separate from ``to_dict`` so callers can clearly
        distinguish persistence/API serialization from the schema
        object itself.
        """
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
    ) -> "AnalysisJobSchema":
        if not isinstance(payload, dict):
            raise JobSchemaError(
                "Job payload must be an object."
            )

        required = {
            "job_id",
            "state",
            "created_at",
            "updated_at",
            "request",
            "revision",
            "transitions",
        }

        missing = required - payload.keys()

        if missing:
            raise JobSchemaError(
                "Missing job fields: "
                + ", ".join(sorted(missing))
            )

        try:
            UUID(payload["job_id"])
        except (
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            raise JobSchemaError(
                "job_id must be a valid UUID."
            ) from exc

        state = payload["state"]

        if not isinstance(state, str):
            raise JobSchemaError(
                "state must be a string."
            )

        try:
            JobState(state)
        except ValueError as exc:
            raise JobSchemaError(
                f"Unknown job state: {state}"
            ) from exc

        created_at = payload["created_at"]
        updated_at = payload["updated_at"]

        if not isinstance(created_at, str):
            raise JobSchemaError(
                "created_at must be a string."
            )

        if not isinstance(updated_at, str):
            raise JobSchemaError(
                "updated_at must be a string."
            )

        try:
            parsed_created = datetime.fromisoformat(
                created_at
            )
            parsed_updated = datetime.fromisoformat(
                updated_at
            )
        except ValueError as exc:
            raise JobSchemaError(
                "created_at and updated_at must be valid ISO-8601."
            ) from exc

        if (
            parsed_created.tzinfo is None
            or parsed_updated.tzinfo is None
        ):
            raise JobSchemaError(
                "created_at and updated_at must be "
                "timezone-aware."
            )

        revision = payload["revision"]

        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
        ):
            raise JobSchemaError(
                "revision must be a non-negative integer."
            )

        if not isinstance(payload["transitions"], list):
            raise JobSchemaError(
                "transitions must be a list."
            )

        transitions = tuple(
            JobTransitionSchema.from_dict(item)
            for item in payload["transitions"]
        )

        if not transitions:
            raise JobSchemaError(
                "Job must contain at least one transition."
            )

        if transitions[0].from_state is not None:
            raise JobSchemaError(
                "First job transition must have "
                "from_state=null."
            )

        if transitions[0].to_state != JobState.QUEUED.value:
            raise JobSchemaError(
                "First job transition must enter queued state."
            )

        if transitions[-1].to_state != state:
            raise JobSchemaError(
                "Final transition does not match job state."
            )

        if revision != len(transitions) - 1:
            raise JobSchemaError(
                "Job revision must equal "
                "len(transitions) - 1."
            )

        return cls(
            job_id=payload["job_id"],
            state=state,
            created_at=created_at,
            updated_at=updated_at,
            request=AnalysisJobRequestSchema.from_dict(
                payload["request"]
            ),
            revision=revision,
            transitions=transitions,
        )

    def to_job(self) -> AnalysisJob:
        return AnalysisJob(
            job_id=UUID(self.job_id),
            state=JobState(self.state),
            created_at=datetime.fromisoformat(
                self.created_at
            ),
            updated_at=datetime.fromisoformat(
                self.updated_at
            ),
            request=self.request.to_request(),
            revision=self.revision,
            transitions=tuple(
                transition.to_transition()
                for transition in self.transitions
            ),
        )