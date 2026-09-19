"""
Production-grade analysis job state machine for DRISHTI.

The job state machine provides a strict lifecycle for asynchronous
geospatial analysis jobs.

Design goals:
- explicit states instead of arbitrary strings
- deterministic transition rules
- terminal-state protection
- immutable transition history records
- timestamps for auditability
- retry-safe semantics
- clear failure and cancellation paths

The state machine itself does not perform work. Workers/orchestrators
perform the work and advance the state through this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple
from uuid import UUID, uuid4


class JobState(str, Enum):
    """Canonical lifecycle states for a DRISHTI analysis job."""

    QUEUED = "queued"
    PREPARING = "preparing"
    FETCHING_IMAGERY = "fetching_imagery"
    TILING = "tiling"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStateError(RuntimeError):
    """Base exception for invalid job-state operations."""


class InvalidJobTransitionError(JobStateError):
    """Raised when a requested state transition is not permitted."""


class TerminalJobStateError(JobStateError):
    """Raised when attempting to mutate a terminal job."""


TERMINAL_STATES: FrozenSet[JobState] = frozenset(
    {
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.CANCELLED,
    }
)


ALLOWED_TRANSITIONS: Dict[JobState, FrozenSet[JobState]] = {
    JobState.QUEUED: frozenset(
        {
            JobState.PREPARING,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.PREPARING: frozenset(
        {
            JobState.FETCHING_IMAGERY,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.FETCHING_IMAGERY: frozenset(
        {
            JobState.TILING,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.TILING: frozenset(
        {
            JobState.INFERENCE,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.INFERENCE: frozenset(
        {
            JobState.POSTPROCESSING,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.POSTPROCESSING: frozenset(
        {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
}


@dataclass(frozen=True)
class JobTransition:
    """
    Immutable audit record for one state transition.
    """

    from_state: Optional[JobState]
    to_state: JobState
    timestamp: datetime
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "Job transition timestamp must be timezone-aware."
            )


@dataclass
class AnalysisJob:
    """
    Mutable state-machine object representing one analysis job.

    ``revision`` is incremented after every successful state
    transition and is used by persistence layers for optimistic
    concurrency control.
    """

    job_id: UUID
    state: JobState
    created_at: datetime
    updated_at: datetime
    request: AnalysisJobRequest
    revision: int = 0
    transitions: Tuple[JobTransition, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        request: AnalysisJobRequest,
        job_id: Optional[UUID] = None,
        now: Optional[datetime] = None,
    ) -> "AnalysisJob":
        """Create a new job in the QUEUED state."""
        timestamp = now or datetime.now(timezone.utc)

        if timestamp.tzinfo is None:
            raise ValueError("Job timestamp must be timezone-aware.")

        return cls(
            job_id=job_id or uuid4(),
            state=JobState.QUEUED,
            created_at=timestamp,
            updated_at=timestamp,
            request=request,
            revision=0,
            transitions=(
                JobTransition(
                    from_state=None,
                    to_state=JobState.QUEUED,
                    timestamp=timestamp,
                    reason="job_created",
                ),
            ),
        )

    @property
    def is_terminal(self) -> bool:
        """Return whether the job can no longer transition."""
        return self.state in TERMINAL_STATES

    def can_transition_to(self, target: JobState) -> bool:
        """Return whether ``target`` is a valid next state."""
        if self.is_terminal:
            return False

        return target in ALLOWED_TRANSITIONS[self.state]

    def transition(
        self,
        target: JobState,
        *,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> JobTransition:
        """
        Advance the job to ``target`` if the transition is valid.

        Returns
        -------
        JobTransition
            The immutable audit record created by the transition.
        """
        if not isinstance(target, JobState):
            raise TypeError(
                "target must be an instance of JobState."
            )

        if self.is_terminal:
            raise TerminalJobStateError(
                f"Job {self.job_id} is already in terminal state "
                f"'{self.state.value}'."
            )

        if not self.can_transition_to(target):
            raise InvalidJobTransitionError(
                f"Invalid job transition: "
                f"'{self.state.value}' -> '{target.value}'."
            )

        timestamp = now or datetime.now(timezone.utc)

        if timestamp.tzinfo is None:
            raise ValueError(
                "Job transition timestamp must be timezone-aware."
            )

        if timestamp < self.updated_at:
            raise ValueError(
                "Transition timestamp cannot precede the previous "
                "job update timestamp."
            )

        transition = JobTransition(
            from_state=self.state,
            to_state=target,
            timestamp=timestamp,
            reason=reason,
        )

        self.state = target
        self.updated_at = timestamp
        self.revision += 1
        self.transitions = (*self.transitions, transition)

        return transition

    def fail(
        self,
        *,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> JobTransition:
        """Transition the job to FAILED."""
        return self.transition(
            JobState.FAILED,
            reason=reason,
            now=now,
        )

    def cancel(
        self,
        *,
        reason: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> JobTransition:
        """Transition the job to CANCELLED."""
        return self.transition(
            JobState.CANCELLED,
            reason=reason,
            now=now,
        )

@dataclass(frozen=True)
class AnalysisJobRequest:
    """
    Immutable request specification attached to an analysis job.

    The job state machine does not interpret these fields. They describe
    what the job was requested to analyze and are passed downstream to
    the appropriate geospatial/imagery services.
    """

    aoi: Dict[str, object]
    imagery_source: Dict[str, object]
    metadata: Optional[Dict[str, object]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.aoi, dict):
            raise ValueError("aoi must be a dictionary.")

        if not isinstance(self.imagery_source, dict):
            raise ValueError(
                "imagery_source must be a dictionary."
            )

        if self.metadata is not None and not isinstance(
            self.metadata,
            dict,
        ):
            raise ValueError(
                "metadata must be a dictionary."
            )